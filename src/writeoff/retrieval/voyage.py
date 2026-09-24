"""Voyage AI embeddings and reranking over its REST API (https://api.voyageai.com/v1)."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx

from writeoff.chunking.tokens import estimate_tokens
from writeoff.models import RerankScore
from writeoff.retrieval.interfaces import Embedder, Reranker

logger = logging.getLogger(__name__)

VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"
_RETRYABLE = frozenset({429, 500, 502, 503, 504})
_MAX_DELAY = 60.0  # Voyage rate limits are per minute; never wait longer than one window

Sleep = Callable[[float], Awaitable[None]]


class VoyageError(RuntimeError):
    """A Voyage API call failed or returned something unusable."""


class EmbeddingError(VoyageError):
    """The embeddings endpoint failed or returned something unusable."""


class RerankError(VoyageError):
    """The rerank endpoint failed or returned something unusable."""


class _VoyageAPI:
    """POST with bearer auth, retrying 429/5xx and transport errors with capped backoff."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        *,
        max_attempts: int,
        backoff: float,
        sleep: Sleep,
    ) -> None:
        self._client = client
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._max_attempts = max_attempts
        self._backoff = backoff
        self._sleep = sleep

    async def post(self, url: str, payload: dict[str, Any], error: type[VoyageError]) -> Any:
        for attempt in range(1, self._max_attempts + 1):
            delay = min(self._backoff * 2 ** (attempt - 1), _MAX_DELAY)
            try:
                response = await self._client.post(url, json=payload, headers=self._headers)
            except httpx.TransportError as exc:
                if attempt == self._max_attempts:
                    raise error(f"Voyage request failed: {exc!r}") from exc
                await self._sleep(delay)
                continue
            if response.status_code in _RETRYABLE and attempt < self._max_attempts:
                logger.warning("Voyage HTTP %d, retrying in %.1fs", response.status_code, delay)
                await self._sleep(delay)
                continue
            if response.status_code != httpx.codes.OK:
                raise error(f"Voyage HTTP {response.status_code}: {response.text[:300]}")
            return response.json()
        raise error("Voyage: retries exhausted")  # pragma: no cover


class VoyageEmbedder(Embedder):
    """Batches by count and (conservatively estimated) tokens.

    Voyage allows 1,000 texts and 320k tokens per request for voyage-4. The defaults stay
    well under both so one oversized batch never fails a whole document.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        *,
        model: str,
        dimension: int,
        max_batch_texts: int = 128,
        max_batch_tokens: int = 100_000,
        max_attempts: int = 8,
        backoff: float = 2.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._api = _VoyageAPI(
            client, api_key, max_attempts=max_attempts, backoff=backoff, sleep=sleep
        )
        self._model = model
        self._dimension = dimension
        self._max_texts = max_batch_texts
        self._max_tokens = max_batch_tokens

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in self._batches(texts):
            vectors.extend(await self._embed(batch, "document"))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "query"))[0]

    def _batches(self, texts: Sequence[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current: list[str] = []
        tokens = 0
        for text in texts:
            size = estimate_tokens(text)
            if current and (len(current) >= self._max_texts or tokens + size > self._max_tokens):
                batches.append(current)
                current, tokens = [], 0
            current.append(text)
            tokens += size
        if current:
            batches.append(current)
        return batches

    async def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        payload = {
            "input": texts,
            "model": self._model,
            "input_type": input_type,
            "output_dimension": self._dimension,
            "truncation": False,  # fail loudly rather than silently embed a truncated text
        }
        body = await self._api.post(VOYAGE_EMBEDDINGS_URL, payload, EmbeddingError)
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            got = len(data) if isinstance(data, list) else "no"
            raise EmbeddingError(f"Voyage returned {got} embeddings for {len(texts)} inputs")
        vectors = [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
        if any(len(v) != self._dimension for v in vectors):
            raise EmbeddingError(f"Voyage returned vectors not of dimension {self._dimension}")
        return [[float(x) for x in v] for v in vectors]


class VoyageReranker(Reranker):
    """Cross-encoder reranking (rerank-2.5: up to 1,000 documents, 600k total tokens).

    Long passages are truncated by Voyage to fit, which is acceptable for scoring (the
    full text still reaches the agent); this is why truncation stays on here but not for
    embeddings.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        *,
        model: str,
        max_attempts: int = 8,
        backoff: float = 2.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._api = _VoyageAPI(
            client, api_key, max_attempts=max_attempts, backoff=backoff, sleep=sleep
        )
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[RerankScore]:
        if not documents:
            return []
        payload = {
            "query": query,
            "documents": list(documents),
            "model": self._model,
            "top_k": min(top_k, len(documents)),
        }
        body = await self._api.post(VOYAGE_RERANK_URL, payload, RerankError)
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise RerankError("Voyage rerank returned no data")
        scores = [
            RerankScore(index=int(d["index"]), score=float(d["relevance_score"])) for d in data
        ]
        if any(s.index >= len(documents) for s in scores):
            raise RerankError("Voyage rerank returned an out-of-range index")
        return sorted(scores, key=lambda s: s.score, reverse=True)
