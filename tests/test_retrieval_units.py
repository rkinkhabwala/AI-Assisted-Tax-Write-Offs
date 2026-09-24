"""Retrieval building blocks that need no database: citations, RRF, query building,
rewriter parsing, the Voyage reranker and PII redaction."""

import json
from collections.abc import Callable

import anthropic
import httpx
import httpx2
import pytest

from writeoff.privacy import redact
from writeoff.retrieval.citations import CitationKind, extract_citations
from writeoff.retrieval.fusion import reciprocal_rank_fusion
from writeoff.retrieval.hybrid import RetrievalConfig, lexical_query
from writeoff.retrieval.rewrite import ClaudeQueryRewriter, QueryRewriteError, parse_terms
from writeoff.retrieval.voyage import RerankError, VoyageReranker

# --- citations -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What does § 179(b)(1) limit?", ["IRC § 179(b)(1)"]),
        ("Section 179 limit", ["IRC § 179"]),
        ("is sec. 280A(c)(1) met", ["IRC § 280A(c)(1)"]),
        ("IRC 162(a) and 26 U.S.C. § 274(n)(1)", ["IRC § 162(a)", "IRC § 274(n)(1)"]),
        ("§280a home office", ["IRC § 280A"]),
        ("Treas. Reg. § 1.263(a)-3(h) safe harbor", ["Treas. Reg. § 1.263(a)-3(h)"]),
        ("see Reg. 1.274-5T and § 1.162-5", ["Treas. Reg. § 1.274-5T", "Treas. Reg. § 1.162-5"]),
        ("Pub 946 or Publication 15-B", ["Pub 946", "Pub 15-B"]),
        (
            "do I file Form 8829 or Form 1120-S with Schedule C",
            [
                "Instructions for Form 8829",
                "Instructions for Form 1120-S",
                "Instructions for Schedule C",
            ],
        ),
        ("I spent 179 dollars on 2 laptops in 2025", []),  # bare numbers are not citations
        ("§ 179 and section 179", ["IRC § 179"]),  # de-duplicated
    ],
)
def test_extract_citations(query: str, expected: list[str]) -> None:
    assert [r.citation for r in extract_citations(query)] == expected


def test_citation_kinds() -> None:
    kinds = {
        r.citation: r.kind for r in extract_citations("§ 179, Reg. 1.179-2, Pub 946, Form 4562")
    }
    assert kinds == {
        "IRC § 179": CitationKind.IRC,
        "Treas. Reg. § 1.179-2": CitationKind.REGULATION,
        "Pub 946": CitationKind.PUBLICATION,
        "Instructions for Form 4562": CitationKind.FORM,
    }
    assert CitationKind.IRC.is_statutory
    assert not CitationKind.FORM.is_statutory


# --- fusion --------------------------------------------------------------------------


def test_rrf_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "lexical": ["c", "d", "b"]}, k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 63)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert [item for item, _ in fused][:2] == ["c", "b"]  # found by both beats a lone first place


def test_weighted_rrf() -> None:
    fused = dict(
        reciprocal_rank_fusion({"dense": ["a"], "lexical": ["b"]}, k=60, weights={"dense": 2.0})
    )
    assert fused["a"] == pytest.approx(2 / 61)
    assert fused["b"] == pytest.approx(1 / 61)  # unnamed rankings weigh 1


def test_rrf_validates_k() -> None:
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion({"x": ["a"]}, k=0)


# --- query building ------------------------------------------------------------------


def test_lexical_query_ors_words_phrases_and_citations() -> None:
    query = "Can I deduct my home office under § 280A?"
    q = lexical_query(query, ["listed property", 'the "quoted" term'], extract_citations(query))
    parts = q.split(" or ")
    assert "Can" not in parts  # stopwords dropped
    assert "my" not in parts
    assert {"deduct", "home", "office"} <= set(parts)
    assert '"listed property"' in parts
    assert '"the quoted term"' in parts  # inner quotes can't break the syntax
    assert '"IRC § 280A"' in parts


def test_retrieval_config_validation() -> None:
    with pytest.raises(ValueError, match="citation_k"):
        RetrievalConfig(final_k=2, citation_k=3)
    with pytest.raises(ValueError, match="positive"):
        RetrievalConfig(dense_k=0)
    with pytest.raises(ValueError, match="at least one"):
        RetrievalConfig(use_dense=False, use_lexical=False)
    with pytest.raises(ValueError, match="min_primary"):
        RetrievalConfig(final_k=2, citation_k=0, min_primary=3)


# --- rewriter ------------------------------------------------------------------------


def test_parse_terms() -> None:
    text = (
        '- vehicle expense deduction\n2. "listed property"\n* § 280F\n\nListed Property\n'
        + "x" * 90
    )
    assert parse_terms(text, 8) == ["vehicle expense deduction", "listed property", "§ 280F"]
    assert parse_terms("a\nb\nc", 2) == ["a", "b"]


def _claude(handler: Callable[[httpx2.Request], httpx2.Response]) -> ClaudeQueryRewriter:
    client = anthropic.AsyncAnthropic(
        api_key="test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return ClaudeQueryRewriter(client, "claude-haiku-4-5")


def _message(text: str, stop_reason: str = "end_turn") -> httpx2.Response:
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


async def test_claude_rewriter_returns_terms() -> None:
    sent: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return _message("vehicle expense deduction\nlisted property\n§ 280F")

    terms = await _claude(handler).rewrite("can I write off my truck")
    assert terms == ["vehicle expense deduction", "listed property", "§ 280F"]
    assert sent[0]["messages"] == [{"role": "user", "content": "can I write off my truck"}]


@pytest.mark.parametrize(
    "response",
    [_message("", "refusal"), _message(""), httpx2.Response(500, json={"type": "error"})],
)
async def test_claude_rewriter_failures_raise_rewrite_error(response: httpx2.Response) -> None:
    with pytest.raises(QueryRewriteError):
        await _claude(lambda r: response).rewrite("q")


# --- Voyage reranker -----------------------------------------------------------------


class _NoSleep:
    async def __call__(self, seconds: float) -> None:
        return None


def _reranker(handler: Callable[[httpx.Request], httpx.Response]) -> VoyageReranker:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return VoyageReranker(client, "key", model="rerank-2.5", sleep=_NoSleep())


async def test_voyage_rerank_orders_by_score() -> None:
    sent: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "relevance_score": 0.2},
                    {"index": 2, "relevance_score": 0.9},
                ]
            },
        )

    scores = await _reranker(handler).rerank("q", ["a", "b", "c"], top_k=2)
    assert [(s.index, s.score) for s in scores] == [(2, 0.9), (0, 0.2)]
    assert sent[0] == {
        "query": "q",
        "documents": ["a", "b", "c"],
        "model": "rerank-2.5",
        "top_k": 2,
    }


async def test_voyage_rerank_edge_cases() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected for an empty candidate list")

    assert await _reranker(fail).rerank("q", [], top_k=5) == []
    bad_index = {"data": [{"index": 7, "relevance_score": 0.5}]}
    with pytest.raises(RerankError, match="out-of-range"):
        await _reranker(lambda r: httpx.Response(200, json=bad_index)).rerank("q", ["a"], 1)
    with pytest.raises(RerankError, match="HTTP 401"):
        await _reranker(lambda r: httpx.Response(401, text="no")).rerank("q", ["a"], 1)


# --- privacy -------------------------------------------------------------------------


def test_redact_taxpayer_ids() -> None:
    text = "My SSN is 123-45-6789 (also 123456789), EIN 12-3456789; I spent $1,500 in 2025."
    assert redact(text) == "My SSN is [SSN] (also [SSN]), EIN [EIN]; I spent $1,500 in 2025."
