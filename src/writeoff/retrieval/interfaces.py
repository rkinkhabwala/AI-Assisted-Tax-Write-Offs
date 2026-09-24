"""Abstract interfaces for the retrieval stack.

pgvector is the primary VectorStore; the interface exists so Qdrant or ChromaDB can be
swapped in. Likewise Embedder and Reranker hide Voyage vs. local bge models. Everything is
async because callers (FastAPI, the Agent SDK tool handlers) run on an event loop.
"""

from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from uuid import UUID

from writeoff.models import (
    Chunk,
    Document,
    EmbeddedChunk,
    RerankScore,
    ScoredChunk,
    SearchFilters,
    StoredChunkState,
    UpsertStats,
)


class VectorStore(ABC):
    """Persistent store for chunks, supporting dense and lexical search."""

    @abstractmethod
    async def existing_chunks(self, document_id: UUID) -> dict[UUID, StoredChunkState]:
        """State of every stored chunk of a document, keyed by chunk id."""

    @abstractmethod
    async def sync_document(
        self, document: Document, changed: Sequence[EmbeddedChunk], keep: Collection[UUID]
    ) -> UpsertStats:
        """Make the store hold exactly `changed` + `keep` for this document, atomically.

        Upserts the document row and every chunk in `changed`, leaves chunks in `keep`
        untouched, and deletes the document's other chunks (sections that disappeared).
        """

    @abstractmethod
    async def dense_search(
        self, embedding: Sequence[float], k: int, filters: SearchFilters
    ) -> list[ScoredChunk]:
        """Top-k CHILD chunks by cosine similarity, best first."""

    @abstractmethod
    async def lexical_search(self, query: str, k: int, filters: SearchFilters) -> list[ScoredChunk]:
        """Top-k CHILD chunks by full-text relevance, best first.

        `query` uses web-search syntax: words are ANDed, "quoted phrases" match as
        phrases, and `or` separates alternatives.
        """

    @abstractmethod
    async def get_by_citation(self, citation_path: str, tax_year: int) -> list[Chunk]:
        """Chunks whose citation_path equals or falls under `citation_path`, parents first."""

    @abstractmethod
    async def get_enclosing(self, citation_path: str, tax_year: int) -> list[Chunk]:
        """Chunks with the longest citation_path that contains `citation_path`.

        For a citation deeper than anything stored ("IRC § 280A(c)(1)(A)" when the
        chunk is "IRC § 280A(c)(1)"), these are the chunks that hold it.
        """

    @abstractmethod
    async def get_chunks(self, ids: Sequence[UUID]) -> list[Chunk]:
        """Fetch chunks by id (used to load parents for child hits). Missing ids are skipped."""


class Embedder(ABC):
    """Turns text into fixed-size vectors. Documents and queries may embed differently."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier recorded alongside evaluation runs."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector length; must equal the chunks.embedding column dimension."""

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for indexing, preserving input order."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""


class Reranker(ABC):
    """Re-scores candidate passages against a query with a cross-encoder."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier recorded alongside evaluation runs."""

    @abstractmethod
    async def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[RerankScore]:
        """Return up to top_k scores, best first, each pointing back into `documents`."""
