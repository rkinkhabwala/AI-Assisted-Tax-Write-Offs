"""HybridRetriever end to end against the docker Postgres, with deterministic fakes for
the embedder, reranker and rewriter."""

import json
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from tests.fakes import (
    IRC_280A,
    MIGRATIONS,
    PUB_463,
    FakeEmbedder,
    FakeReranker,
    FakeRewriter,
    FixtureFetcher,
)
from writeoff.agent.store import AgentStore, SessionState, ToolCallRecord
from writeoff.agent.tools import ToolRuntime
from writeoff.agent.verifier import VerificationReport
from writeoff.db.migrate import apply_migrations
from writeoff.evals.dataset import RetrievalCase, Target
from writeoff.evals.retrieval import append_csv, check_baseline, run_eval, save_baseline
from writeoff.evals.variants import INDEX_VARIANTS
from writeoff.ingestion.pipeline import IngestionPipeline
from writeoff.models import ChunkLevel, DocType, EntityType, SearchFilters
from writeoff.retention import PurgeResult, purge
from writeoff.retrieval.hybrid import HybridRetriever, RetrievalConfig
from writeoff.retrieval.pgvector_store import PgVectorStore
from writeoff.tax_parameters import TaxParameters
from writeoff.tools.law import get_citation, search_tax_law

pytestmark = pytest.mark.db
REPO = Path(__file__).resolve().parents[1]
YEAR_2025 = SearchFilters(tax_year=2025)


@pytest.fixture
async def store(fresh_db: str) -> PgVectorStore:
    apply_migrations(fresh_db, Path(MIGRATIONS))
    store = PgVectorStore(fresh_db)
    pipeline = IngestionPipeline(FixtureFetcher(), store=store, embedder=FakeEmbedder())
    await pipeline.ingest(IRC_280A, 2025)
    await pipeline.ingest(PUB_463, 2025)
    return store


def _retriever(
    store: PgVectorStore,
    rewriter: FakeRewriter | None = None,
    reranker: FakeReranker | None = None,
    config: RetrievalConfig | None = None,
) -> HybridRetriever:
    return HybridRetriever(
        store,
        FakeEmbedder(),
        reranker or FakeReranker(),
        rewriter or FakeRewriter(),
        config,
    )


async def test_search_fuses_reranks_and_attaches_parents(store: PgVectorStore) -> None:
    response = await _retriever(store).search(
        "portion of the dwelling unit exclusively used on a regular basis "
        "as principal place of business",
        YEAR_2025,
    )
    results = response.results
    assert results[0].chunk.citation_path == "IRC § 280A(c)(1)"
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert len(results) <= 8
    for r in results:
        assert r.chunk.level is ChunkLevel.CHILD
        assert r.parent is not None
        assert r.parent.id == r.chunk.parent_id
        assert not r.citation_lookup
        assert r.fused_score > 0
        assert r.dense_rank is not None or r.lexical_rank is not None
    scores = [r.rerank_score or 0 for r in results]
    assert scores == sorted(scores, reverse=True)
    assert not response.weak


async def test_lexical_leg_finds_exact_statutory_tokens(store: PgVectorStore) -> None:
    response = await _retriever(store).search("280A(c)(5) limitation", YEAR_2025)
    hit = next(r for r in response.results if r.chunk.citation_path == "IRC § 280A(c)(5)")
    assert hit.lexical_rank == 1


async def test_citation_fast_path_comes_first(store: PgVectorStore) -> None:
    response = await _retriever(store).search(
        "What does § 280A(c)(1) require for travel?", YEAR_2025
    )
    assert [c.citation for c in response.citations] == ["IRC § 280A(c)(1)"]
    first = response.results[0]
    assert first.citation_lookup
    assert first.chunk.citation_path == "IRC § 280A(c)(1)"
    assert first.fused_score == 0.0
    assert first.rerank_score is not None
    assert sum(r.citation_lookup for r in response.results) <= 3
    assert len({r.chunk.id for r in response.results}) == len(response.results)  # no duplicates


async def test_citation_deeper_than_any_chunk_uses_enclosing_provision(
    store: PgVectorStore,
) -> None:
    response = await _retriever(store).search("explain § 280A(c)(1)(A)", YEAR_2025)
    assert response.results[0].citation_lookup
    assert response.results[0].chunk.citation_path == "IRC § 280A(c)(1)"


async def test_rewrite_terms_are_used_and_failures_are_tolerated(store: PgVectorStore) -> None:
    rewritten = await _retriever(store, FakeRewriter(["dwelling unit"])).search(
        "work from my house", YEAR_2025
    )
    assert rewritten.rewritten_terms == ["dwelling unit"]
    assert rewritten.results
    failed = await _retriever(store, FakeRewriter(fail=True)).search(
        "dwelling unit rules", YEAR_2025
    )
    assert failed.rewritten_terms == []
    assert failed.results  # search still works without the rewrite


async def test_weak_results_and_filters(store: PgVectorStore) -> None:
    low = await _retriever(store, reranker=FakeReranker(scale=0.1)).search(
        "dwelling unit", YEAR_2025
    )
    assert low.results
    assert low.weak
    empty = await _retriever(store).search("dwelling unit", SearchFilters(tax_year=2026))
    assert empty.results == []
    assert empty.weak
    pubs = await _retriever(store).search(
        "travel expenses away from home",
        SearchFilters(tax_year=2025, doc_types=frozenset({DocType.IRS_PUBLICATION})),
    )
    assert pubs.results
    assert all(r.chunk.doc_type is DocType.IRS_PUBLICATION for r in pubs.results)


async def test_final_k_is_respected(store: PgVectorStore) -> None:
    response = await _retriever(store, config=RetrievalConfig(final_k=3, citation_k=1)).search(
        "§ 280A travel expenses home office", YEAR_2025
    )
    assert len(response.results) == 3
    assert sum(r.citation_lookup for r in response.results) == 1


async def test_lookup_citation_returns_enclosing_parent_first(store: PgVectorStore) -> None:
    retriever = _retriever(store)
    chunks = await retriever.lookup_citation("IRC § 280A(c)(4)", 2025)
    assert chunks[0].level is ChunkLevel.PARENT  # the grouped § 280A parent holding (c)(4)
    assert any(c.citation_path == "IRC § 280A(c)(4)" for c in chunks)
    deeper = await retriever.lookup_citation("IRC § 280A(c)(4)(B)(ii)", 2025)
    assert any(c.citation_path == "IRC § 280A(c)(4)" for c in deeper)
    assert await retriever.lookup_citation("IRC § 9999", 2025) == []


async def test_store_get_enclosing(store: PgVectorStore) -> None:
    enclosing = await store.get_enclosing("IRC § 280A(c)(1)(A)", 2025)
    assert {c.citation_path for c in enclosing} == {"IRC § 280A(c)(1)"}
    headings = await store.get_enclosing(
        "Pub 463, ch. 1, Traveling Away From Home, Tax Home, Nowhere", 2025
    )
    assert headings
    assert all(
        "Pub 463, ch. 1, Traveling Away From Home, Tax Home".startswith(c.citation_path)
        for c in headings
    )
    assert await store.get_enclosing("IRC § 28", 2025) == []


async def test_ablation_modes(store: PgVectorStore) -> None:
    query = "dwelling unit exclusively used principal place of business"
    for config in (
        RetrievalConfig(use_lexical=False),
        RetrievalConfig(use_dense=False),
        RetrievalConfig(rerank=False),
        RetrievalConfig(dense_weight=2.0, lexical_weight=0.5),
    ):
        response = await _retriever(store, config=config).search(query, YEAR_2025)
        assert response.results, config
        if not config.use_lexical:
            assert all(r.lexical_rank is None for r in response.results)
        if not config.use_dense:
            assert all(r.dense_rank is None for r in response.results)
        if not config.rerank:
            assert all(r.rerank_score is None for r in response.results)
            fused = [r.fused_score for r in response.results]
            assert fused == sorted(fused, reverse=True)


async def test_min_primary_swaps_in_a_statute(store: PgVectorStore) -> None:
    query = "travel expenses away from home tax home meals lodging dwelling unit"
    plain = await _retriever(store, config=RetrievalConfig(final_k=4, min_primary=0)).search(
        query, YEAR_2025
    )
    assert all(r.chunk.doc_type is DocType.IRS_PUBLICATION for r in plain.results)
    guaranteed = await _retriever(store, config=RetrievalConfig(final_k=4, min_primary=1)).search(
        query, YEAR_2025
    )
    assert len(guaranteed.results) == 4
    assert guaranteed.results[-1].chunk.doc_type is DocType.IRC
    assert [r.chunk.id for r in guaranteed.results[:3]] == [r.chunk.id for r in plain.results[:3]]


# --- the eval harness end to end, on the fixture index ---------------------------------


async def test_run_eval_end_to_end(store: PgVectorStore, tmp_path: Path) -> None:
    cases = [
        RetrievalCase(
            id="ho",
            category="home_office",
            query="dwelling unit exclusively used as principal place of business",
            relevant=(Target(citation="IRC § 280A(c)(1)"),),
        ),
        RetrievalCase(
            id="oos", category="out_of_scope", query="chocolate chip cookies", out_of_scope=True
        ),
    ]
    report = await run_eval(
        _retriever(store),
        cases,
        index=INDEX_VARIANTS["prod"],
        retrieval_name="test",
        rewrite=False,
        reranker_model="fake",
    )
    assert report.summary()["recall_at_10"] == 1.0
    assert report.unresolved == {}
    assert report.by_doc_type()["irc"]["targets"] == 1
    data = report.to_json()
    assert data["cases"][0]["retrieved"][0]["relevant"] is True

    csv_path = tmp_path / "runs.csv"
    append_csv(report, csv_path)
    append_csv(report, csv_path)
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("run_at,run_id,index")
    (tmp_path / "foreign.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="unexpected header"):
        append_csv(report, tmp_path / "foreign.csv")

    baseline = tmp_path / "baseline.json"
    assert check_baseline(report, baseline)[0]  # no baseline yet: gate passes
    save_baseline(report, baseline)
    assert check_baseline(report, baseline)[0]
    worse = json.loads(baseline.read_text()) | {"recall_at_10": 1.2}
    baseline.write_text(json.dumps(worse))
    ok, message = check_baseline(report, baseline)
    assert not ok
    assert "REGRESSION" in message


# --- agent tools backed by the store ---------------------------------------------------


async def test_search_tool_dedupes_sections(store: PgVectorStore) -> None:
    result = await search_tax_law(
        _retriever(store), "dwelling unit exclusively used as principal place of business", 2025
    )
    assert result.passages
    assert result.passages[0].citation == "IRC § 280A(c)(1)"
    section_ids = {p.section_id for p in result.passages if p.section_id}
    assert section_ids == set(result.sections)  # every referenced section included once
    assert all(s.text for s in result.sections.values())
    assert result.guidance is None


async def test_search_tool_weak_guidance(store: PgVectorStore) -> None:
    result = await search_tax_law(_retriever(store), "dwelling unit", 2026)
    assert result.weak
    assert result.guidance is not None
    assert "don't cover" in result.guidance


async def test_get_citation_tool(store: PgVectorStore) -> None:
    found = await get_citation(_retriever(store), "§ 280A(c)(4)", 2025)
    assert found.found
    assert found.resolved == "IRC § 280A(c)(4)"
    assert "(4) Use in providing day care services" in found.passages[0].text
    missing = await get_citation(_retriever(store), "§ 9999", 2025)
    assert not missing.found
    assert missing.guidance is not None


# --- agent: weak-retrieval retry limit and persistence ---------------------------------


async def test_weak_search_retry_limit(store: PgVectorStore) -> None:
    runtime = ToolRuntime(
        TaxParameters(REPO / "data" / "tax_parameters", (2025, 2026)),
        _retriever(store),
        max_weak_retries=2,
    )
    runtime.begin(uuid4())
    search = next(t for t in runtime.tools() if t.name == "search_tax_law")
    guidance = []
    for _ in range(3):
        out = await search.handler({"query": "dwelling unit", "tax_year": 2026})
        guidance.append(json.loads(out["content"][0]["text"])["guidance"])
    assert guidance[0] == guidance[1] != guidance[2]  # the third weak search hits the limit
    assert "Retry limit reached" in guidance[2]


async def test_agent_store_sessions_traces_and_retention(fresh_db: str) -> None:
    apply_migrations(fresh_db, Path(MIGRATIONS))
    agent_store = AgentStore(fresh_db)
    session = SessionState(uuid4(), EntityType.S_CORP, 2025, {"industry": "consulting"}, "sdk-1")
    await agent_store.save_session(session)
    assert await agent_store.load_session(session.session_id) == session
    assert (await agent_store.load_session(uuid4())).entity_type is None
    request_id = uuid4()
    await agent_store.start_request(request_id, session.session_id, "q [SSN]", "1.0.0", "m")
    chunk = uuid4()
    await agent_store.record_tool_call(
        request_id,
        ToolCallRecord("t1", "search_tax_law", {"query": "meals"}, "ok", 12, 3400, (chunk,)),
    )
    await agent_store.finish_request(
        request_id,
        status="complete",
        stop_reason="end_turn",
        num_turns=3,
        tool_calls=1,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.02,
        latency_ms=900,
    )
    stored = await agent_store.request(request_id)
    assert stored is not None
    assert stored.status == "complete"
    assert stored.tool_calls == 1
    calls = await agent_store.tool_calls(request_id)
    assert calls[0].chunk_ids == (chunk,)
    report = VerificationReport(status="revised", rounds=2, untraced_numbers=["$75"])
    await agent_store.record_verification(request_id, report)
    async with await psycopg.AsyncConnection.connect(fresh_db) as conn:
        row = await (
            await conn.execute(
                "SELECT verification_status, verification_rounds, untraced_numbers "
                "FROM agent_requests WHERE request_id = %s",
                (request_id,),
            )
        ).fetchone()
    assert row == ("revised", 2, ["$75"])
    assert await agent_store.purge_traces(older_than_days=30) == 0
    assert await agent_store.purge_traces(older_than_days=0) == 1
    assert await agent_store.tool_calls(request_id) == []
    assert await agent_store.find_session(uuid4()) is None
    assert await agent_store.find_session(session.session_id) == session
    assert await agent_store.purge_sessions(idle_days=30) == 0
    assert await agent_store.purge_sessions(idle_days=0) == 1
    assert await agent_store.find_session(session.session_id) is None


async def test_retention_purges_old_traces_and_idle_sessions(fresh_db: str) -> None:
    apply_migrations(fresh_db, Path(MIGRATIONS))
    agent_store = AgentStore(fresh_db)
    old, recent = SessionState(uuid4()), SessionState(uuid4())
    for s in (old, recent):
        await agent_store.save_session(s)
        await agent_store.start_request(uuid4(), s.session_id, "q", "1.0.0", "m")
    async with await psycopg.AsyncConnection.connect(fresh_db) as conn:
        await conn.execute(
            "UPDATE agent_sessions SET updated_at = now() - interval '40 days' "
            "WHERE session_id = %s",
            (old.session_id,),
        )
        await conn.execute(
            "UPDATE agent_requests SET created_at = now() - interval '40 days' "
            "WHERE session_id = %s",
            (old.session_id,),
        )
    assert await purge(agent_store, 0, 0) == PurgeResult(0, 0)  # 0 skips both steps
    assert await purge(agent_store, 30, 0) == PurgeResult(1, 0)
    assert await agent_store.find_session(old.session_id) is not None
    assert await purge(agent_store, 30, 30) == PurgeResult(0, 1)
    assert await agent_store.find_session(old.session_id) is None
    assert await agent_store.find_session(recent.session_id) is not None


async def test_search_and_citation_tools_record_evidence(store: PgVectorStore) -> None:
    runtime = ToolRuntime(
        TaxParameters(REPO / "data" / "tax_parameters", (2025, 2026)), _retriever(store)
    )
    runtime.begin(uuid4())
    tools = {t.name: t for t in runtime.tools()}
    await tools["search_tax_law"].handler(
        {"query": "dwelling unit principal place of business", "tax_year": 2025}
    )
    await tools["get_citation"].handler({"citation": "§ 280A(c)(4)", "tax_year": 2025})
    citations = runtime.state.evidence.citations
    assert "IRC § 280A(c)(1)" in citations
    assert any(c.startswith("IRC § 280A") for c in citations)
    # Every recorded passage carries its document, for the UI's Sources panel.
    documents = runtime.state.evidence.documents
    assert set(documents) == set(citations)
    assert all(d.url.startswith("http") and d.title for d in documents.values())
