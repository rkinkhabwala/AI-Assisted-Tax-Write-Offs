"""Pipeline + PgVectorStore against the docker Postgres: idempotency, versioning, search."""

from pathlib import Path

import psycopg
import pytest

from tests.fakes import (
    IRC_280A,
    PUB_463,
    URL_280A,
    FakeEmbedder,
    FixtureFetcher,
    fake_vector,
)
from writeoff.chunking.context import ContextSummarizer
from writeoff.db.migrate import apply_migrations
from writeoff.ingestion.pipeline import IngestionPipeline
from writeoff.models import (
    ChunkLevel,
    DocType,
    EntityType,
    SearchFilters,
)
from writeoff.retrieval.pgvector_store import PgVectorStore

pytestmark = pytest.mark.db

MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


class FakeSummarizer(ContextSummarizer):
    def __init__(self) -> None:
        self.calls = 0

    async def summarize(self, *, document_title: str, section_text: str, chunk_text: str) -> str:
        self.calls += 1
        return f"From {document_title}."


@pytest.fixture
def store(fresh_db: str) -> PgVectorStore:
    apply_migrations(fresh_db, MIGRATIONS)
    return PgVectorStore(fresh_db)


def _count(conninfo: str, sql: str) -> int:
    with psycopg.connect(conninfo) as conn:
        row = conn.execute(sql.encode()).fetchone()
    assert row is not None
    return int(row[0])


async def test_first_ingest_writes_everything(store: PgVectorStore, fresh_db: str) -> None:
    embedder, summarizer = FakeEmbedder(), FakeSummarizer()
    pipeline = IngestionPipeline(
        FixtureFetcher(), store=store, embedder=embedder, summarizer=summarizer
    )
    result = await pipeline.ingest(IRC_280A, 2025)
    children = [c for c in result.chunks if c.level is ChunkLevel.CHILD]
    assert result.stats is not None
    assert result.stats.inserted == len(result.chunks)
    assert (result.stats.updated, result.stats.unchanged, result.stats.deleted) == (0, 0, 0)
    assert summarizer.calls == len(children) == result.embedded
    assert _count(fresh_db, "SELECT count(*) FROM chunks") == len(result.chunks)
    assert (
        _count(fresh_db, "SELECT count(*) FROM chunks WHERE level = 'child' AND embedding IS NULL")
        == 0
    )
    assert (
        _count(
            fresh_db, "SELECT count(*) FROM chunks WHERE level = 'parent' AND embedding IS NOT NULL"
        )
        == 0
    )
    # The context summary is embedded but not stored in the display text.
    assert all(t.startswith("From IRC § 280A") for t in embedder.texts)
    assert _count(fresh_db, "SELECT count(*) FROM chunks WHERE text LIKE 'From IRC%'") == 0


async def test_reingesting_unchanged_document_is_a_no_op(store: PgVectorStore) -> None:
    await IngestionPipeline(
        FixtureFetcher(), store=store, embedder=FakeEmbedder(), summarizer=FakeSummarizer()
    ).ingest(IRC_280A, 2025)
    embedder, summarizer = FakeEmbedder(), FakeSummarizer()
    again = await IngestionPipeline(
        FixtureFetcher(), store=store, embedder=embedder, summarizer=summarizer
    ).ingest(IRC_280A, 2025)
    assert again.stats is not None
    assert (again.stats.inserted, again.stats.updated, again.stats.deleted) == (0, 0, 0)
    assert again.stats.unchanged == len(again.chunks)
    assert embedder.texts == []  # no paid API calls
    assert summarizer.calls == 0


async def test_changed_source_updates_and_prunes(store: PgVectorStore, fresh_db: str) -> None:
    fetcher = FixtureFetcher()
    await IngestionPipeline(fetcher, store=store, embedder=FakeEmbedder()).ingest(IRC_280A, 2025)
    before = _count(fresh_db, "SELECT count(*) FROM chunks")
    # Amend (b) and repeal (a): its anchor, heading and text disappear from the page.
    html = fetcher.bodies[URL_280A].decode()
    html = html.replace(
        "without regard to its connection with his trade or business",
        "without regard to its connection with a trade or business",
    )
    start = html.index('<a name="substructure-location_a"></a>')
    html = html[:start] + html[html.index('<a name="substructure-location_b"></a>') :]
    fetcher.bodies[URL_280A] = html.encode()
    embedder = FakeEmbedder()
    result = await IngestionPipeline(fetcher, store=store, embedder=embedder).ingest(IRC_280A, 2025)
    assert result.stats is not None
    assert result.stats.updated + result.stats.inserted >= 1
    assert result.stats.unchanged > 0  # untouched provisions are not re-embedded
    assert len(embedder.texts) < len([c for c in result.chunks if c.level is ChunkLevel.CHILD])
    assert _count(fresh_db, "SELECT count(*) FROM chunks") == len(result.chunks) <= before
    assert _count(fresh_db, "SELECT count(*) FROM chunks WHERE text LIKE '%with his trade%'") == 0


async def test_new_tax_year_adds_rows_without_overwriting(
    store: PgVectorStore, fresh_db: str
) -> None:
    pipeline = IngestionPipeline(FixtureFetcher(), store=store, embedder=FakeEmbedder())
    first = await pipeline.ingest(IRC_280A, 2025)
    second = await pipeline.ingest(IRC_280A, 2026)
    assert second.stats is not None
    assert second.stats.inserted == len(second.chunks)
    assert first.document.id != second.document.id
    assert {c.id for c in first.chunks}.isdisjoint(c.id for c in second.chunks)
    assert _count(fresh_db, "SELECT count(*) FROM documents") == 2
    assert _count(fresh_db, "SELECT count(DISTINCT tax_year) FROM chunks") == 2


async def test_enabling_summaries_later_backfills_them(store: PgVectorStore, fresh_db: str) -> None:
    await IngestionPipeline(FixtureFetcher(), store=store, embedder=FakeEmbedder()).ingest(
        IRC_280A, 2025
    )
    summarizer = FakeSummarizer()
    result = await IngestionPipeline(
        FixtureFetcher(), store=store, embedder=FakeEmbedder(), summarizer=summarizer
    ).ingest(IRC_280A, 2025)
    assert summarizer.calls == len([c for c in result.chunks if c.level is ChunkLevel.CHILD])
    assert (
        _count(
            fresh_db, "SELECT count(*) FROM chunks WHERE level='child' AND context_summary IS NULL"
        )
        == 0
    )


async def test_dry_run_needs_no_store() -> None:
    result = await IngestionPipeline(FixtureFetcher()).ingest(PUB_463, 2025)
    assert result.stats is None
    assert result.chunks
    with pytest.raises(ValueError, match="together"):
        IngestionPipeline(FixtureFetcher(), store=PgVectorStore("postgresql://unused"))


# --- store search --------------------------------------------------------------------


async def _loaded(store: PgVectorStore) -> None:
    pipeline = IngestionPipeline(FixtureFetcher(), store=store, embedder=FakeEmbedder())
    await pipeline.ingest(IRC_280A, 2025)
    await pipeline.ingest(PUB_463, 2025)


async def test_dense_search_with_filters(store: PgVectorStore) -> None:
    await _loaded(store)
    query = fake_vector("principal place of business exclusively used regular basis dwelling unit")
    hits = await store.dense_search(query, 3, SearchFilters(tax_year=2025))
    assert hits[0].chunk.citation_path == "IRC § 280A(c)(1)"
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score >= hits[1].score
    only_pubs = await store.dense_search(
        query, 5, SearchFilters(tax_year=2025, doc_types=frozenset({DocType.IRS_PUBLICATION}))
    )
    assert only_pubs
    assert all(h.chunk.doc_type is DocType.IRS_PUBLICATION for h in only_pubs)
    assert await store.dense_search(query, 5, SearchFilters(tax_year=2026)) == []


async def test_entity_filter_keeps_untagged_chunks(store: PgVectorStore) -> None:
    await _loaded(store)
    query = fake_vector("business use of home travel expenses")
    c_corp = await store.dense_search(
        query, 50, SearchFilters(tax_year=2025, entity_type=EntityType.C_CORP)
    )
    assert c_corp
    assert all(
        h.chunk.doc_type is DocType.IRS_PUBLICATION for h in c_corp
    )  # § 280A is tagged pass-through only
    sole_prop = await store.dense_search(
        query, 200, SearchFilters(tax_year=2025, entity_type=EntityType.SOLE_PROP)
    )
    assert {h.chunk.doc_type for h in sole_prop} == {DocType.IRC, DocType.IRS_PUBLICATION}


async def test_lexical_search_matches_exact_statutory_tokens(store: PgVectorStore) -> None:
    await _loaded(store)
    hits = await store.lexical_search("280A(c)(1)", 5, SearchFilters(tax_year=2025))
    assert hits
    assert hits[0].chunk.citation_path.startswith("IRC § 280A(c)")
    prose = await store.lexical_search(
        "exclusive use regular basis", 5, SearchFilters(tax_year=2025)
    )
    assert any(h.chunk.citation_path == "IRC § 280A(c)(1)" for h in prose)


async def test_citation_lookup_and_get_chunks(store: PgVectorStore) -> None:
    await _loaded(store)
    found = await store.get_by_citation("IRC § 280A(c)", 2025)
    assert found
    assert all(c.citation_path.startswith("IRC § 280A(c)") for c in found)
    whole = await store.get_by_citation("IRC § 280A", 2025)
    levels = [c.level for c in whole]
    assert levels == sorted(levels, key=lambda lvl: lvl is ChunkLevel.CHILD)  # parents first
    assert ChunkLevel.PARENT in levels
    assert await store.get_by_citation("IRC § 280", 2025) == []  # prefix must end at a boundary
    child = next(c for c in found if c.level is ChunkLevel.CHILD)
    assert child.parent_id is not None
    fetched = await store.get_chunks([child.parent_id, child.id])
    assert [c.id for c in fetched] == [child.parent_id, child.id]
