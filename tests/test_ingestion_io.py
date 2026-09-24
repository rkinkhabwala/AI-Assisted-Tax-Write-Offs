"""Registry, fetcher, Voyage embedder and Claude summarizer, all offline (mocked HTTP)."""

import json
from collections.abc import Callable
from pathlib import Path

import anthropic
import httpx
import httpx2
import pytest
from pydantic import HttpUrl

from writeoff.chunking.context import (
    CachingSummarizer,
    ClaudeContextSummarizer,
    ContextSummarizer,
    ContextSummaryError,
)
from writeoff.config import Settings
from writeoff.ingestion.fetcher import CachingFetcher, FetchError, HttpDocumentFetcher
from writeoff.ingestion.parsers import ParserName
from writeoff.ingestion.registry import RegistryError, load_registry
from writeoff.models import DocType, SourceFormat, SourceSpec
from writeoff.retrieval.voyage import EmbeddingError, VoyageEmbedder

REPO = Path(__file__).resolve().parents[1]

# --- registry ------------------------------------------------------------------------


def test_shipped_registry() -> None:
    registry = load_registry(REPO / "data" / "sources.yaml")
    by_id = {s.id: s for s in registry.sources}
    assert len(by_id) == len(registry.sources)
    for year in Settings().supported_tax_years:
        assert registry.select(year)
    # Statutes and regulations are snapshotted for every supported year (D7).
    for source in registry.sources:
        if source.doc_type in {DocType.IRC, DocType.TREASURY_REGULATION}:
            assert set(source.editions) == set(Settings().supported_tax_years), source.id
    assert by_id["pub-15-b"].editions[2025].parser is ParserName.PDF
    assert by_id["irc-162"].entity_overrides["IRC § 162(l)"]
    assert "reg-1.263_a-1" in by_id  # distinct sections, distinct ids
    assert "reg-1.263A-1" in by_id


def _registry(tmp_path: Path, url: str, source_id: str = "x") -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": source_id,
                        "title": "T",
                        "citation_root": "Pub 1",
                        "doc_type": "irs_publication",
                        "editions": {"2025": {"url": url, "parser": "irs_html"}},
                    }
                ]
            }
        )
    )
    return path


@pytest.mark.parametrize("url", ["http://www.irs.gov/p1", "https://evil.example.com/p1"])
def test_registry_rejects_disallowed_urls(tmp_path: Path, url: str) -> None:
    with pytest.raises(RegistryError, match="not an https URL"):
        load_registry(_registry(tmp_path, url))


def test_registry_select(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path, "https://www.irs.gov/publications/p1"))
    assert [s.id for s in registry.select(2025)] == ["x"]
    assert registry.select(2026) == []
    with pytest.raises(KeyError, match="unknown source ids"):
        registry.select(2025, frozenset({"nope"}))


# --- fetcher -------------------------------------------------------------------------

SPEC = SourceSpec(
    source_url=HttpUrl("https://www.irs.gov/publications/p463"),
    title="Pub 463",
    doc_type=DocType.IRS_PUBLICATION,
    tax_year=2025,
    format=SourceFormat.HTML,
)


class _Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _fetcher(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    min_interval: float = 1.0,
    max_attempts: int = 4,
) -> tuple[HttpDocumentFetcher, _Sleeps]:
    sleeps = _Sleeps()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = HttpDocumentFetcher(
        client, sleep=sleeps, min_interval=min_interval, max_attempts=max_attempts
    )
    return fetcher, sleeps


def _html(
    status: int = 200, body: bytes = b"<html>ok</html>", headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(
        status,
        content=body,
        headers={"content-type": "text/html; charset=UTF-8", **(headers or {})},
    )


async def test_fetch_success() -> None:
    fetcher, _ = _fetcher(lambda r: _html())
    doc = await fetcher.fetch(SPEC)
    assert doc.content == b"<html>ok</html>"
    assert doc.content_type == "text/html"
    assert doc.retrieved_at.tzinfo is not None


async def test_fetch_retries_then_succeeds() -> None:
    responses = iter([_html(503), _html(429, headers={"retry-after": "7"}), _html()])
    fetcher, sleeps = _fetcher(lambda r: next(responses), min_interval=0)
    assert (await fetcher.fetch(SPEC)).content == b"<html>ok</html>"
    assert 7.0 in sleeps.calls  # honours Retry-After


async def test_fetch_gives_up_after_max_attempts() -> None:
    fetcher, _ = _fetcher(lambda r: _html(503), min_interval=0, max_attempts=3)
    with pytest.raises(FetchError, match="HTTP 503"):
        await fetcher.fetch(SPEC)


async def test_fetch_does_not_retry_404() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _html(404)

    fetcher, _ = _fetcher(handler, min_interval=0)
    with pytest.raises(FetchError, match="HTTP 404"):
        await fetcher.fetch(SPEC)
    assert len(calls) == 1


async def test_fetch_rejects_wrong_content_type() -> None:
    pdf_spec = SourceSpec.model_validate(SPEC.model_dump() | {"format": SourceFormat.PDF})
    fetcher, _ = _fetcher(lambda r: _html())  # an HTML error page where a PDF was expected
    with pytest.raises(FetchError, match="expected pdf"):
        await fetcher.fetch(pdf_spec)


async def test_fetch_refuses_disallowed_host() -> None:
    spec = SourceSpec.model_validate(SPEC.model_dump() | {"source_url": "https://example.com/x"})
    fetcher, _ = _fetcher(lambda r: _html())
    with pytest.raises(FetchError, match="not allowed"):
        await fetcher.fetch(spec)


async def test_fetch_throttles_per_host() -> None:
    fetcher, sleeps = _fetcher(lambda r: _html(), min_interval=5.0)
    await fetcher.fetch(SPEC)
    await fetcher.fetch(SPEC)
    assert sleeps.calls
    assert sleeps.calls[-1] > 4.0


async def test_caching_fetcher(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _html(body=f"<html>v{len(calls)}</html>".encode())

    inner, _ = _fetcher(handler, min_interval=0)
    first = await CachingFetcher(inner, tmp_path).fetch(SPEC)
    second = await CachingFetcher(inner, tmp_path).fetch(SPEC)
    assert (first.content, second.content) == (b"<html>v1</html>", b"<html>v1</html>")
    assert second.retrieved_at == first.retrieved_at  # original fetch time is preserved
    refreshed = await CachingFetcher(inner, tmp_path, refresh=True).fetch(SPEC)
    assert refreshed.content == b"<html>v2</html>"
    assert len(calls) == 2


# --- Voyage embedder -----------------------------------------------------------------


def _voyage(
    handler: Callable[[httpx.Request], httpx.Response], *, max_batch_texts: int = 128
) -> VoyageEmbedder:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return VoyageEmbedder(
        client,
        "test-key",
        model="voyage-4",
        dimension=3,
        sleep=_Sleeps(),
        max_batch_texts=max_batch_texts,
    )


def _echo_embeddings(
    requests: list[dict[str, object]],
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append({**body, "auth": request.headers["authorization"]})
        data = [
            {"index": i, "embedding": [float(len(t)), 0.0, 1.0]}
            for i, t in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(data)), "model": "voyage-4"})

    return handler


async def test_voyage_batches_and_preserves_order() -> None:
    requests: list[dict[str, object]] = []
    embedder = _voyage(_echo_embeddings(requests), max_batch_texts=2)
    vectors = await embedder.embed_documents(["a", "bb", "ccc"])
    assert vectors == [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]
    assert [r["input"] for r in requests] == [["a", "bb"], ["ccc"]]
    assert requests[0] | {"input": None} == {
        "input": None,
        "model": "voyage-4",
        "input_type": "document",
        "output_dimension": 3,
        "truncation": False,
        "auth": "Bearer test-key",
    }


async def test_voyage_query_embedding() -> None:
    requests: list[dict[str, object]] = []
    assert await _voyage(_echo_embeddings(requests)).embed_query("home office") == [11.0, 0.0, 1.0]
    assert requests[0]["input_type"] == "query"


async def test_voyage_retries_rate_limit() -> None:
    requests: list[dict[str, object]] = []
    ok = _echo_embeddings(requests)
    responses = iter([httpx.Response(429), None])

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses) or ok(request)

    assert await _voyage(handler).embed_documents(["x"]) == [[1.0, 0.0, 1.0]]


async def test_voyage_errors() -> None:
    with pytest.raises(EmbeddingError, match="HTTP 400"):
        await _voyage(lambda r: httpx.Response(400, text="bad")).embed_documents(["x"])
    wrong_dim = {"data": [{"index": 0, "embedding": [1.0]}]}
    with pytest.raises(EmbeddingError, match="dimension"):
        await _voyage(lambda r: httpx.Response(200, json=wrong_dim)).embed_documents(["x"])
    with pytest.raises(EmbeddingError, match="for 2 inputs"):
        await _voyage(lambda r: httpx.Response(200, json=wrong_dim)).embed_documents(["x", "y"])


# --- Claude context summarizer -------------------------------------------------------


def _claude(
    stop_reason: str, text: str, captured: list[dict[str, object]]
) -> ClaudeContextSummarizer:
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": text}] if text else [],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    client = anthropic.AsyncAnthropic(
        api_key="test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return ClaudeContextSummarizer(client, "claude-haiku-4-5")


async def test_summarizer_sends_section_and_passage() -> None:
    captured: list[dict[str, object]] = []
    summary = await _claude(
        "end_turn", " From IRC § 280A(c)(1) on home offices. ", captured
    ).summarize(
        document_title="IRC § 280A",
        section_text="(c) Exceptions ...",
        chunk_text="(A) principal place",
    )
    assert summary == "From IRC § 280A(c)(1) on home offices."
    request = captured[0]
    assert request["model"] == "claude-haiku-4-5"
    content = request["messages"][0]["content"]  # type: ignore[index]
    assert "<section>\n(c) Exceptions ...\n</section>" in content
    assert "<passage>\n(A) principal place\n</passage>" in content


@pytest.mark.parametrize(
    ("stop_reason", "text", "match"), [("refusal", "", "declined"), ("end_turn", "", "empty")]
)
async def test_summarizer_errors(stop_reason: str, text: str, match: str) -> None:
    with pytest.raises(ContextSummaryError, match=match):
        await _claude(stop_reason, text, []).summarize(
            document_title="T", section_text="s", chunk_text="c"
        )


class _CountingSummarizer(ContextSummarizer):
    def __init__(self) -> None:
        self.calls = 0

    async def summarize(self, *, document_title: str, section_text: str, chunk_text: str) -> str:
        self.calls += 1
        return f"summary {self.calls} of {chunk_text}"


async def test_caching_summarizer_persists_and_reuses(tmp_path: Path) -> None:
    inner = _CountingSummarizer()
    cache = CachingSummarizer(inner, tmp_path, model="m1")
    first = await cache.summarize(document_title="T", section_text="s", chunk_text="c")
    again = await CachingSummarizer(inner, tmp_path, model="m1").summarize(
        document_title="T", section_text="s", chunk_text="c"
    )
    assert first == again == "summary 1 of c"
    assert inner.calls == 1  # survives a new instance, i.e. a new run
    await cache.summarize(document_title="T", section_text="s changed", chunk_text="c")
    await CachingSummarizer(inner, tmp_path, model="m2").summarize(
        document_title="T", section_text="s", chunk_text="c"
    )
    assert inner.calls == 3  # changed input or model -> fresh summary
