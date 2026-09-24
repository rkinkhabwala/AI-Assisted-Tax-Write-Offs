"""Indexes and retrieval configurations the evals compare.

An index variant is a separate Postgres database built from the same cached sources with
different index-time choices (chunking, context summaries, embedding model). A retrieval
variant changes only query-time parameters and runs against any index.

Chunking variants are built *without* context summaries: new chunks would need new
(paid) summaries, and comparing them against `noctx` isolates the chunking effect.
`noctx` vs `prod` measures what contextual retrieval itself adds.
"""

from dataclasses import dataclass, field

from psycopg.conninfo import make_conninfo

from writeoff.chunking.chunker import ChunkingConfig
from writeoff.config import Settings
from writeoff.retrieval.hybrid import RetrievalConfig


@dataclass(frozen=True, slots=True)
class IndexVariant:
    name: str
    description: str
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    contextual: bool = True
    embedding_model: str = "voyage-4"

    def conninfo(self, settings: Settings) -> str:
        base = str(settings.database_url)
        return (
            base
            if self.name == "prod"
            else make_conninfo(base, dbname=f"writeoff_eval_{self.name}")
        )


INDEX_VARIANTS: dict[str, IndexVariant] = {
    v.name: v
    for v in (
        IndexVariant("prod", "production index: default chunking, context summaries, voyage-4"),
        IndexVariant("noctx", "default chunking, no context summaries", contextual=False),
        IndexVariant(
            "law2",
            "default chunking, context summaries, voyage-law-2",
            embedding_model="voyage-law-2",
        ),
        IndexVariant(
            "merge_noctx",
            "statutory siblings merged, no summaries",
            contextual=False,
            chunking=ChunkingConfig(merge_statutory_siblings=True),
        ),
        IndexVariant(
            "small_noctx",
            "children 150-500 tokens, no summaries",
            contextual=False,
            chunking=ChunkingConfig(target_min=150, target_max=500),
        ),
        IndexVariant(
            "large_noctx",
            "children 500-1,100 tokens, no summaries",
            contextual=False,
            chunking=ChunkingConfig(target_min=500, target_max=1100),
        ),
    )
}

# Query-time variants. Each is compared against `baseline` (the production config, with
# final_k raised to 10 so Recall@10 can be measured). RRF weights must stay below
# (k + depth) / (k + 1) = 100 / 61, about 1.64. At or above that, every item from the heavier list
# outranks every item from the other and the "hybrid" degenerates to one retriever.
RETRIEVAL_VARIANTS: dict[str, RetrievalConfig] = {
    "baseline": RetrievalConfig(final_k=10),
    "dense_only": RetrievalConfig(final_k=10, use_lexical=False),
    "lexical_only": RetrievalConfig(final_k=10, use_dense=False),
    "no_rerank": RetrievalConfig(final_k=10, rerank=False),
    "dense_1.3": RetrievalConfig(final_k=10, dense_weight=1.3),
    "lexical_1.3": RetrievalConfig(final_k=10, lexical_weight=1.3),
    "rerank_depth_20": RetrievalConfig(final_k=10, rerank_depth=20),
    "rerank_depth_80": RetrievalConfig(final_k=10, dense_k=80, lexical_k=80, rerank_depth=80),
    "min_primary_0": RetrievalConfig(final_k=10, min_primary=0),
    "min_primary_1": RetrievalConfig(final_k=10, min_primary=1),
    # Diagnostic: the fused candidate list without reranking, to measure the recall ceiling.
    "candidates_40": RetrievalConfig(final_k=40, rerank=False),
}
