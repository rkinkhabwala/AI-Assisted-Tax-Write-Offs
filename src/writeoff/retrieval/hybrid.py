"""Hybrid retrieval (spec section 3): citation fast path + dense + lexical -> RRF -> rerank.

    query ──► extract citations ──► statutory citations fetched directly (fast path)
          ├─► rewrite to tax terms ─┬─► dense:   embed(query + terms), HNSW cosine, top 40
          │                         └─► lexical: OR of query words, terms, citations, top 40
          └─► RRF(k=60) ──► top 40 ──► Voyage rerank against the original query ──► top 8
              ──► citation hits first ──► attach each child's parent section

Parameter choices, measured by the phase 4 retrieval evals (evals/retrieval_runs.csv):
- dense_k = lexical_k = 40 and rerank_depth = 40, final_k = 8: the spec's "top ~40 down
  to top ~8". Depths 20 and 80 scored the same. 40 is kept so `min_primary` has
  statutes to draw on. The fused candidates contain 88% of labeled targets. The reranker
  is the largest single contributor (without it, Recall@10 falls from 0.82 to 0.63).
- Equal RRF weights: tilting to 1.3 either way lowered Recall@10.
- min_primary = 2: guarantees statute or regulation authority in the list. Publications
  paraphrase the law and tend to outrank it, and IRC targets had the lowest recall. The
  guarantee raised Recall@10 by 2.1 points without query rewriting, and never hurt.
- citation_k = 3: an explicit "§ 162" can match dozens of chunks. The reranker keeps the
  three most relevant, and the rest of the list stays open for context the user didn't name.
- weak_threshold = 0.6: answerable eval queries' top rerank scores start at 0.66, and
  out-of-scope ones sit mostly below 0.48. A false "weak" only costs the agent a retry.
"""

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from writeoff.models import (
    Chunk,
    ChunkLevel,
    DocType,
    RetrievalResult,
    ScoredChunk,
    SearchFilters,
)
from writeoff.privacy import redact
from writeoff.retrieval.citations import CitationRef, extract_citations
from writeoff.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from writeoff.retrieval.interfaces import Embedder, Reranker, VectorStore
from writeoff.retrieval.rewrite import QueryRewriteError, QueryRewriter

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z0-9§][A-Za-z0-9§().-]*")
# Keyword search ORs terms together, so filler words would match nearly everything.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "this",
        "to",
        "us",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    dense_k: int = 40
    lexical_k: int = 40
    rrf_k: int = DEFAULT_RRF_K
    rerank_depth: int = 40
    final_k: int = 8
    citation_k: int = 3
    weak_threshold: float = 0.6
    # Ablation and tuning switches for the retrieval evals. The defaults are production.
    use_dense: bool = True
    use_lexical: bool = True
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    rerank: bool = True
    # Guarantee at least this many statute or regulation results in the final list.
    min_primary: int = 2

    def __post_init__(self) -> None:
        if min(self.dense_k, self.lexical_k, self.rerank_depth, self.final_k) < 1:
            raise ValueError("retrieval depths must be positive")
        if not 0 <= self.citation_k <= self.final_k:
            raise ValueError("citation_k must be between 0 and final_k")
        if not (self.use_dense or self.use_lexical):
            raise ValueError("enable at least one of dense and lexical search")
        if not 0 <= self.min_primary <= self.final_k:
            raise ValueError("min_primary must be between 0 and final_k")


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    rewritten_terms: list[str]
    citations: list[CitationRef]
    results: list[RetrievalResult]
    weak: bool  # no results, or the best rerank score is below the configured threshold


_PRIMARY = frozenset({DocType.IRC, DocType.TREASURY_REGULATION})
_Scored = tuple[Chunk, float | None]


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        rewriter: QueryRewriter,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._rewriter = rewriter
        self.config = config or RetrievalConfig()

    async def search(self, query: str, filters: SearchFilters) -> SearchResponse:
        cfg = self.config
        refs = extract_citations(query)
        terms = await self._rewrite(query)
        logger.info(
            "search query=%r rewritten=%r citations=%r",
            redact(query),
            terms,
            [r.citation for r in refs],
        )
        dense, lexical, cited = await asyncio.gather(
            self._dense(query, terms, filters),
            self._lexical(query, terms, refs, filters),
            self._citation_chunks(refs, filters.tax_year),
        )
        dense_rank = {h.chunk.id: h.rank for h in dense}
        lexical_rank = {h.chunk.id: h.rank for h in lexical}
        chunks = {h.chunk.id: h.chunk for h in (*dense, *lexical)}
        fused = reciprocal_rank_fusion(
            {"dense": [h.chunk.id for h in dense], "lexical": [h.chunk.id for h in lexical]},
            k=cfg.rrf_k,
            weights={"dense": cfg.dense_weight, "lexical": cfg.lexical_weight},
        )
        cited_ids = {c.id for c in cited}
        candidates = [(cid, s) for cid, s in fused if cid not in cited_ids][: cfg.rerank_depth]
        fused_score = dict(candidates)

        cited_hits = (await self._order(query, cited))[: cfg.citation_k]
        ordered = await self._order(query, [chunks[cid] for cid, _ in candidates])
        search_hits = _ensure_primary(
            ordered,
            cfg.final_k - len(cited_hits),
            cfg.min_primary - sum(c.doc_type in _PRIMARY for c, _ in cited_hits),
        )

        parents = await self._parents([c for c, _ in (*cited_hits, *search_hits)])
        results: list[RetrievalResult] = []
        for (chunk, score), via_citation in [
            *((hit, True) for hit in cited_hits),
            *((hit, False) for hit in search_hits),
        ]:
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    parent=parents.get(chunk.parent_id),
                    rank=len(results) + 1,
                    citation_lookup=via_citation,
                    rerank_score=score,
                    fused_score=0.0 if via_citation else fused_score[chunk.id],
                    dense_rank=dense_rank.get(chunk.id),
                    lexical_rank=lexical_rank.get(chunk.id),
                )
            )
        if cfg.rerank:
            best = max((r.rerank_score or 0.0 for r in results), default=0.0)
            weak = best < cfg.weak_threshold
        else:
            weak = not results
        return SearchResponse(query, terms, refs, results, weak=weak)

    async def _dense(
        self, query: str, terms: Sequence[str], filters: SearchFilters
    ) -> list[ScoredChunk]:
        if not self.config.use_dense:
            return []
        text = f"{query}\n{'; '.join(terms)}" if terms else query
        vector = await self._embedder.embed_query(text)
        return await self._store.dense_search(vector, self.config.dense_k, filters)

    async def _lexical(
        self, query: str, terms: Sequence[str], refs: Sequence[CitationRef], filters: SearchFilters
    ) -> list[ScoredChunk]:
        if not self.config.use_lexical:
            return []
        q = lexical_query(query, terms, refs)
        return await self._store.lexical_search(q, self.config.lexical_k, filters) if q else []

    async def lookup_citation(self, citation: str, tax_year: int) -> list[Chunk]:
        """Chunks for an exact citation (the `get_citation` tool): the provision and
        everything under it, or else the deepest stored provision that contains it.
        Parents come first."""
        found = await self._store.get_by_citation(citation, tax_year)
        if not found:
            found = await self._store.get_enclosing(citation, tax_year)
        parent_ids = [c.parent_id for c in found if c.parent_id is not None]
        have = {c.id for c in found}
        missing = [pid for pid in dict.fromkeys(parent_ids) if pid not in have]
        enclosing_parents = await self._store.get_chunks(missing) if missing else []
        return [*enclosing_parents, *found]

    async def _rewrite(self, query: str) -> list[str]:
        try:
            return await self._rewriter.rewrite(query)
        except QueryRewriteError as exc:
            logger.warning("query rewrite failed, searching without it: %s", exc)
            return []

    async def _citation_chunks(self, refs: Sequence[CitationRef], tax_year: int) -> list[Chunk]:
        chunks: dict[UUID, Chunk] = {}
        for ref in refs:
            if not ref.kind.is_statutory:
                continue
            found = await self._store.get_by_citation(ref.citation, tax_year)
            if not any(c.level is ChunkLevel.CHILD for c in found):
                found = await self._store.get_enclosing(ref.citation, tax_year)
            for chunk in found:
                if chunk.level is ChunkLevel.CHILD:
                    chunks.setdefault(chunk.id, chunk)
        return list(chunks.values())

    async def _order(self, query: str, chunks: Sequence[Chunk]) -> list[_Scored]:
        """All `chunks`, best first: by reranker score, or unchanged when reranking is off."""
        if not chunks:
            return []
        if not self.config.rerank:
            return [(c, None) for c in chunks]
        scores = await self._reranker.rerank(query, [c.embedding_text for c in chunks], len(chunks))
        return [(chunks[s.index], s.score) for s in scores]

    async def _parents(self, chunks: Sequence[Chunk]) -> dict[UUID | None, Chunk]:
        ids = list(dict.fromkeys(c.parent_id for c in chunks if c.parent_id is not None))
        return {p.id: p for p in await self._store.get_chunks(ids)} if ids else {}


def _ensure_primary(ordered: Sequence[_Scored], k: int, min_primary: int) -> list[_Scored]:
    """Top `k` of `ordered`, swapping in the best-ranked statute or regulation results
    from lower down when fewer than `min_primary` made the cut. Swapped-in results replace
    the lowest-ranked non-primary ones and stay at the end, so scores remain descending."""
    top = list(ordered[: max(k, 0)])
    missing = min_primary - sum(c.doc_type in _PRIMARY for c, _ in top)
    if missing <= 0:
        return top
    extra = [hit for hit in ordered[len(top) :] if hit[0].doc_type in _PRIMARY][:missing]
    for hit in extra:
        drop = max((i for i, (c, _) in enumerate(top) if c.doc_type not in _PRIMARY), default=None)
        if drop is None:
            break
        top.pop(drop)
        top.append(hit)
    return top


def lexical_query(query: str, terms: Sequence[str], refs: Sequence[CitationRef]) -> str:
    """Web-search syntax OR-ing the query's content words, rewritten phrases and citations.

    Postgres web-search syntax ANDs bare words, and requiring every word of a
    conversational question (or of eight expanded phrases) to appear would find almost
    nothing, so alternatives are ORed and ts_rank_cd rewards passages that match many.
    """
    words = [w for w in _WORD.findall(query) if w.lower() not in _STOPWORDS and len(w) > 1]
    phrases = [*terms, *(r.citation for r in refs)]
    parts = [
        *dict.fromkeys(words),
        *(f'"{p.replace(chr(34), "")}"' for p in dict.fromkeys(phrases)),
    ]
    return " or ".join(parts)
