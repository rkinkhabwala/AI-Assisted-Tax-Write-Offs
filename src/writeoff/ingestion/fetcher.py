"""Document fetching: polite HTTP with retries, plus an on-disk cache of raw bytes.

The cache makes re-runs cheap and reproducible: re-ingesting reads the bytes fetched
earlier (with their original `retrieved_at`) unless `refresh` is requested. The same URL
serving several tax years (statutes, D7) is fetched once.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

from writeoff.ingestion.interfaces import DocumentFetcher
from writeoff.ingestion.registry import ALLOWED_HOSTS
from writeoff.models import FetchedDocument, SourceFormat, SourceSpec

logger = logging.getLogger(__name__)

_RETRYABLE = frozenset({408, 425, 429, 500, 502, 503, 504})
_EXPECTED_TYPES = {
    SourceFormat.HTML: ("text/html", "application/xhtml+xml", "application/xml", "text/xml"),
    SourceFormat.PDF: ("application/pdf",),
}


class FetchError(RuntimeError):
    """A document could not be fetched (after retries) or came back in the wrong format."""


class HttpDocumentFetcher(DocumentFetcher):
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        min_interval: float = 1.0,
        max_attempts: int = 4,
        backoff: float = 2.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._min_interval = min_interval
        self._max_attempts = max_attempts
        self._backoff = backoff
        self._sleep = sleep
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}

    def supports(self, spec: SourceSpec) -> bool:
        return spec.source_url.scheme == "https" and spec.source_url.host in ALLOWED_HOSTS

    async def fetch(self, spec: SourceSpec) -> FetchedDocument:
        if not self.supports(spec):
            raise FetchError(f"refusing to fetch {spec.source_url}: host not allowed")
        url = str(spec.source_url)
        host = spec.source_url.host or ""
        for attempt in range(1, self._max_attempts + 1):
            await self._throttle(host)
            try:
                response = await self._client.get(url)
            except httpx.TransportError as exc:
                if attempt == self._max_attempts:
                    raise FetchError(f"{url}: {exc!r}") from exc
                await self._sleep(self._backoff * 2 ** (attempt - 1))
                continue
            if response.status_code in _RETRYABLE and attempt < self._max_attempts:
                delay = _retry_after(response) or self._backoff * 2 ** (attempt - 1)
                logger.warning("%s: HTTP %d, retrying in %.1fs", url, response.status_code, delay)
                await self._sleep(delay)
                continue
            if response.status_code != httpx.codes.OK:
                raise FetchError(f"{url}: HTTP {response.status_code}")
            return _to_document(spec, response)
        raise FetchError(f"{url}: gave up after {self._max_attempts} attempts")  # pragma: no cover

    async def _throttle(self, host: str) -> None:
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._last_request.get(host, -1e9) + self._min_interval - time.monotonic()
            if wait > 0:
                await self._sleep(wait)
            self._last_request[host] = time.monotonic()


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after", "")
    return float(value) if value.isdigit() else None


def _to_document(spec: SourceSpec, response: httpx.Response) -> FetchedDocument:
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if not content_type.startswith(_EXPECTED_TYPES[spec.format]):
        raise FetchError(f"{spec.source_url}: expected {spec.format}, got {content_type!r}")
    if not response.content:
        raise FetchError(f"{spec.source_url}: empty response body")
    return FetchedDocument(
        spec=spec,
        content=response.content,
        content_type=content_type,
        retrieved_at=datetime.now(UTC),
    )


class CachingFetcher(DocumentFetcher):
    """Wraps a fetcher with a content cache keyed by URL."""

    def __init__(self, inner: DocumentFetcher, cache_dir: Path, *, refresh: bool = False) -> None:
        self._inner = inner
        self._dir = cache_dir
        self._refresh = refresh

    def supports(self, spec: SourceSpec) -> bool:
        return self._inner.supports(spec)

    def _paths(self, spec: SourceSpec) -> tuple[Path, Path]:
        key = hashlib.sha256(str(spec.source_url).encode()).hexdigest()[:32]
        return self._dir / f"{key}.body", self._dir / f"{key}.json"

    async def fetch(self, spec: SourceSpec) -> FetchedDocument:
        body_path, meta_path = self._paths(spec)
        if not self._refresh and body_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return FetchedDocument(
                spec=spec,
                content=body_path.read_bytes(),
                content_type=meta["content_type"],
                retrieved_at=datetime.fromisoformat(meta["retrieved_at"]),
            )
        document = await self._inner.fetch(spec)
        self._dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(body_path, document.content)
        meta = {
            "url": str(spec.source_url),
            "content_type": document.content_type,
            "retrieved_at": document.retrieved_at.isoformat(),
        }
        _atomic_write(meta_path, json.dumps(meta, indent=2).encode())
        return document


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
