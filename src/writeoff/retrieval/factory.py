"""Build a `HybridRetriever` from settings (used by the search CLI and the agent tools)."""

import anthropic
import httpx

from writeoff.config import Settings
from writeoff.retrieval.hybrid import HybridRetriever, RetrievalConfig
from writeoff.retrieval.pgvector_store import PgVectorStore
from writeoff.retrieval.rewrite import ClaudeQueryRewriter, NoRewrite, QueryRewriter
from writeoff.retrieval.voyage import VoyageEmbedder, VoyageReranker


class RetrieverConfigError(RuntimeError):
    """Required API keys are missing."""


def build_retriever(
    settings: Settings,
    http: httpx.AsyncClient,
    *,
    rewrite: bool = True,
    config: RetrievalConfig | None = None,
) -> HybridRetriever:
    if settings.voyage_api_key is None:
        raise RetrieverConfigError("VOYAGE_API_KEY is not set")
    voyage_key = settings.voyage_api_key.get_secret_value()
    rewriter: QueryRewriter = NoRewrite()
    if rewrite:
        if settings.anthropic_api_key is None:
            raise RetrieverConfigError("ANTHROPIC_API_KEY is not set (or disable query rewriting)")
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        rewriter = ClaudeQueryRewriter(client, settings.rewrite_model)
    return HybridRetriever(
        store=PgVectorStore(str(settings.database_url)),
        embedder=VoyageEmbedder(
            http, voyage_key, model=settings.embedding_model, dimension=settings.embedding_dimension
        ),
        reranker=VoyageReranker(http, voyage_key, model=settings.reranker_model),
        rewriter=rewriter,
        config=config,
    )
