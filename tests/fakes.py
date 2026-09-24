"""Test doubles shared by the DB-backed pipeline and retrieval tests."""

import hashlib
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from writeoff.config import EMBEDDING_DIMENSION
from writeoff.ingestion.interfaces import DocumentFetcher
from writeoff.ingestion.registry import SourceEntry
from writeoff.models import FetchedDocument, RerankScore, SourceSpec
from writeoff.retrieval.interfaces import Embedder, Reranker
from writeoff.retrieval.rewrite import QueryRewriteError, QueryRewriter

FIXTURES = Path(__file__).parent / "fixtures"
MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"
URL_280A = "https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title26-section280A&num=0&edition=prelim"
URL_P463 = "https://www.irs.gov/publications/p463"

IRC_280A = SourceEntry.model_validate(
    {
        "id": "irc-280a",
        "title": "26 U.S.C. § 280A",
        "citation_root": "IRC § 280A",
        "doc_type": "irc",
        "entity_types": ["sole_prop", "partnership", "s_corp"],
        "editions": {
            2025: {"url": URL_280A, "parser": "uscode_html"},
            2026: {"url": URL_280A, "parser": "uscode_html"},
        },
    }
)
PUB_463 = SourceEntry.model_validate(
    {
        "id": "pub-463",
        "title": "Publication 463",
        "citation_root": "Pub 463",
        "doc_type": "irs_publication",
        "editions": {2025: {"url": URL_P463, "parser": "irs_html"}},
    }
)


class FixtureFetcher(DocumentFetcher):
    def __init__(self) -> None:
        self.bodies = {
            URL_280A: (FIXTURES / "uscode_280A.html").read_bytes(),
            URL_P463: (FIXTURES / "irs_p463_excerpt.html").read_bytes(),
        }

    def supports(self, spec: SourceSpec) -> bool:
        return str(spec.source_url) in self.bodies

    async def fetch(self, spec: SourceSpec) -> FetchedDocument:
        return FetchedDocument(
            spec=spec,
            content=self.bodies[str(spec.source_url)],
            content_type="text/html",
            retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
        )


def fake_vector(text: str) -> list[float]:
    """Deterministic unit vector; texts sharing words land near each other."""
    vec = [0.0] * EMBEDDING_DIMENSION
    for word in text.lower().split():
        vec[int(hashlib.sha256(word.encode()).hexdigest(), 16) % EMBEDDING_DIMENSION] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeEmbedder(Embedder):
    def __init__(self) -> None:
        self.texts: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [fake_vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return fake_vector(text)


class FakeReranker(Reranker):
    """Scores by the share of query words present in the document (0..1)."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    @property
    def model_name(self) -> str:
        return "fake-rerank"

    async def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[RerankScore]:
        words = set(re.findall(r"\w+", query.lower()))
        scores = []
        for i, doc in enumerate(documents):
            present = set(re.findall(r"\w+", doc.lower()))
            scores.append(
                RerankScore(index=i, score=self.scale * len(words & present) / max(len(words), 1))
            )
        return sorted(scores, key=lambda s: s.score, reverse=True)[:top_k]


class FakeRewriter(QueryRewriter):
    def __init__(self, terms: list[str] | None = None, *, fail: bool = False) -> None:
        self.terms = terms or []
        self.fail = fail

    async def rewrite(self, query: str) -> list[str]:
        if self.fail:
            raise QueryRewriteError("simulated outage")
        return list(self.terms)
