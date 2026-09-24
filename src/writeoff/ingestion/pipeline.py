"""fetch -> parse -> normalize -> chunk -> contextualize -> embed -> upsert (spec section 1).

Idempotency: chunk ids are deterministic and each chunk carries a content hash. Before any
paid API call, the pipeline asks the store what it already holds for the document.
Chunks with the same id, hash and parent (and a context summary, when summaries are
enabled) are kept as they are, so re-ingesting an unchanged document makes no API calls
and no writes. Chunks that disappeared from the source are deleted, and a new tax year
is a different document, so it gets new rows instead of overwriting.
"""

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID

from writeoff.chunking.chunker import ChunkingConfig, SourceContext, chunk_document
from writeoff.chunking.context import ContextSummarizer
from writeoff.ingestion.interfaces import DocumentFetcher
from writeoff.ingestion.parsers import parse
from writeoff.ingestion.registry import SourceEntry
from writeoff.ingestion.tree import render
from writeoff.models import (
    Chunk,
    ChunkLevel,
    Document,
    EmbeddedChunk,
    StoredChunkState,
    UpsertStats,
    make_document_id,
    sha256_hex,
)
from writeoff.retrieval.interfaces import Embedder, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    source_id: str
    tax_year: int
    document: Document
    chunks: list[Chunk]
    stats: UpsertStats | None  # None on a dry run
    summarized: int = 0
    embedded: int = 0


class IngestionPipeline:
    def __init__(
        self,
        fetcher: DocumentFetcher,
        *,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        summarizer: ContextSummarizer | None = None,
        config: ChunkingConfig | None = None,
        summary_concurrency: int = 8,
    ) -> None:
        if (store is None) != (embedder is None):
            raise ValueError(
                "store and embedder must be given together (or neither, for a dry run)"
            )
        self._fetcher = fetcher
        self._store = store
        self._embedder = embedder
        self._summarizer = summarizer
        self._config = config or ChunkingConfig()
        self._semaphore = asyncio.Semaphore(summary_concurrency)

    async def ingest(self, entry: SourceEntry, tax_year: int) -> IngestResult:
        spec = entry.spec_for(tax_year)
        fetched = await self._fetcher.fetch(spec)
        parsed = parse(entry.editions[tax_year].parser, fetched.content)
        text = f"{parsed.title}\n\n{render(parsed.root, include_heading=False)}"
        document = Document(
            id=make_document_id(str(spec.source_url), tax_year),
            source_url=spec.source_url,
            title=parsed.title,
            doc_type=spec.doc_type,
            tax_year=tax_year,
            effective_date=spec.effective_date,
            retrieved_at=fetched.retrieved_at,
            text=text,
            content_hash=sha256_hex(text),
        )
        context = SourceContext(
            document=document,
            citation_root=entry.citation_root,
            style=entry.citation_style,
            entity_types=entry.entity_types,
            entity_overrides=entry.entity_overrides,
        )
        chunks = chunk_document(parsed, context, self._config)
        if self._store is None or self._embedder is None:
            return IngestResult(entry.id, tax_year, document, chunks, stats=None)

        existing = await self._store.existing_chunks(document.id)
        keep = [c.id for c in chunks if self._is_unchanged(c, existing)]
        kept = set(keep)
        changed = [c for c in chunks if c.id not in kept]
        children = [c for c in changed if c.level is ChunkLevel.CHILD]

        summarized = 0
        if self._summarizer is not None and children:
            parents = {c.id: c for c in chunks if c.level is ChunkLevel.PARENT}
            summaries = await asyncio.gather(
                *(
                    self._summarize(document.title, parents[c.parent_id].text, c)
                    for c in children
                    if c.parent_id is not None
                )
            )
            by_id = dict(zip((c.id for c in children), summaries, strict=True))
            children = [c.model_copy(update={"context_summary": by_id[c.id]}) for c in children]
            summarized = len(children)

        vectors = (
            await self._embedder.embed_documents([c.embedding_text for c in children])
            if children
            else []
        )
        embedded = [
            EmbeddedChunk(chunk=c, embedding=tuple(v))
            for c, v in zip(children, vectors, strict=True)
        ]
        embedded += [EmbeddedChunk(chunk=c) for c in changed if c.level is ChunkLevel.PARENT]
        stats = await self._store.sync_document(document, embedded, keep)
        logger.info("%s %d: %s", entry.id, tax_year, stats)
        final = {e.chunk.id: e.chunk for e in embedded}
        return IngestResult(
            entry.id,
            tax_year,
            document,
            [final.get(c.id, c) for c in chunks],
            stats,
            summarized=summarized,
            embedded=len(vectors),
        )

    def _is_unchanged(self, chunk: Chunk, existing: Mapping[UUID, StoredChunkState]) -> bool:
        state = existing.get(chunk.id)
        if (
            state is None
            or state.content_hash != chunk.content_hash
            or state.parent_id != chunk.parent_id
        ):
            return False
        needs_summary = self._summarizer is not None and chunk.level is ChunkLevel.CHILD
        return state.has_context_summary or not needs_summary

    async def _summarize(self, title: str, section: str, chunk: Chunk) -> str:
        assert self._summarizer is not None  # noqa: S101 - narrowed by the caller
        async with self._semaphore:
            return await self._summarizer.summarize(
                document_title=title, section_text=section, chunk_text=chunk.text
            )
