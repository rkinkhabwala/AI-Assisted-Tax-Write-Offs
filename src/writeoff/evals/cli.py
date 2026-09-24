"""Evaluation commands.

python -m writeoff.evals.cli check                        # every label resolves in prod
python -m writeoff.evals.cli retrieval [--index prod] [--config baseline] [--rewrite]
                                       [--gate] [--save-baseline]
python -m writeoff.evals.cli sweep [--index prod]         # all query-time variants
python -m writeoff.evals.cli build-index noctx            # build an index variant
python -m writeoff.evals.cli calibrate-tokens             # estimate vs real token counts
python -m writeoff.evals.cli check-answers                # golden labels resolve in prod
python -m writeoff.evals.cli answers [--set golden|safety|all] [--ids a,b] [--category c]
                                     [--limit N] [--concurrency 3] [--max-cost 20]
                                     [--no-judge] [--gate] [--save-baseline]
python -m writeoff.evals.cli agreement FILE               # judge vs hand spot-check
"""

import argparse
import asyncio
import logging
import random
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import anthropic
import httpx
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from writeoff.agent.factory import build_agent
from writeoff.agent.prompt import load_system_prompt
from writeoff.chunking.context import CachingSummarizer, ClaudeContextSummarizer
from writeoff.chunking.tokens import estimate_tokens
from writeoff.config import Settings, get_settings
from writeoff.db.migrate import apply_migrations
from writeoff.evals import answers as answer_evals
from writeoff.evals.answer_dataset import GoldenCase, SafetyCase, load_golden, load_safety
from writeoff.evals.answers import CaseResult, run_answer_eval
from writeoff.evals.dataset import load_cases
from writeoff.evals.grading import label_prefixes
from writeoff.evals.judge import ClaudeAnswerJudge, load_rubric
from writeoff.evals.retrieval import (
    append_csv,
    check_baseline,
    run_eval,
    save_baseline,
    write_reports,
)
from writeoff.evals.variants import INDEX_VARIANTS, RETRIEVAL_VARIANTS, IndexVariant
from writeoff.ingestion.fetcher import CachingFetcher, HttpDocumentFetcher
from writeoff.ingestion.pipeline import IngestionPipeline
from writeoff.ingestion.registry import load_registry
from writeoff.models import ChunkLevel
from writeoff.retrieval.factory import build_retriever
from writeoff.retrieval.hybrid import HybridRetriever, RetrievalConfig
from writeoff.retrieval.pgvector_store import PgVectorStore
from writeoff.retrieval.voyage import VoyageEmbedder

logger = logging.getLogger("writeoff.evals")

EVALS = Path("evals")
CASES = EVALS / "retrieval_queries.jsonl"
RUNS_CSV = EVALS / "retrieval_runs.csv"
REPORTS = EVALS / "reports"
BASELINE = EVALS / "baselines" / "retrieval.json"
GOLDEN = EVALS / "golden_set.jsonl"
SAFETY = EVALS / "safety_cases.jsonl"
RUBRIC = EVALS / "rubric.md"
ANSWER_RUNS_CSV = EVALS / "answer_runs.csv"
ANSWER_BASELINE = EVALS / "baselines" / "answers.json"
EVAL_YEAR = 2025


def _retriever(
    settings: Settings,
    http: httpx.AsyncClient,
    index: IndexVariant,
    config: RetrievalConfig,
    rewrite: bool,
) -> HybridRetriever:
    variant_settings = settings.model_copy(
        update={
            "database_url": index.conninfo(settings),
            "embedding_model": index.embedding_model,
        }
    )
    return build_retriever(variant_settings, http, rewrite=rewrite, config=config)


async def cmd_check(settings: Settings) -> int:
    """Every label must name a provision stored as such. Labels that only resolve through
    the enclosing-provision fallback usually mean a typo in a heading path, which would
    silently widen the label, so they fail the check too."""
    cases = load_cases(CASES)
    store = PgVectorStore(str(settings.database_url))
    problems = 0
    for case in cases:
        for target in case.relevant:
            exact = await store.get_by_citation(target.citation, case.tax_year)
            if any(c.level is ChunkLevel.CHILD for c in exact) or target.allow_enclosing:
                continue
            enclosing = await store.get_enclosing(target.citation, case.tax_year)
            where = sorted({c.citation_path for c in enclosing}) or ["nothing"]
            problems += 1
            print(f"  {case.id}: {target.citation!r} -> only enclosing {where}")
    in_scope = sum(not c.out_of_scope for c in cases)
    targets = sum(len(c.relevant) for c in cases)
    print(
        f"{len(cases)} cases ({in_scope} in scope, {len(cases) - in_scope} out of scope), "
        f"{targets} targets, {problems} not stored exactly"
    )
    return 1 if problems else 0


async def cmd_retrieval(settings: Settings, args: argparse.Namespace) -> int:
    cases = load_cases(CASES)
    index = INDEX_VARIANTS[args.index]
    names = list(RETRIEVAL_VARIANTS) if args.command == "sweep" else [args.config]
    exit_code = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as http:
        for name in names:
            retriever = _retriever(settings, http, index, RETRIEVAL_VARIANTS[name], args.rewrite)
            report = await run_eval(
                retriever,
                cases,
                index=index,
                retrieval_name=name,
                rewrite=args.rewrite,
                reranker_model=settings.reranker_model,
            )
            s = report.summary()
            print(
                f"{index.name:12} {name:16} R@5 {s['recall_at_5']:.3f}  R@10 "
                f"{s['recall_at_10']:.3f}  MRR {s['mrr']:.3f}  nDCG@10 {s['ndcg_at_10']:.3f}"
            )
            if report.unresolved:
                print(f"  warning: unresolved labels {report.unresolved}")
            append_csv(report, RUNS_CSV, notes=args.notes)
            if args.command == "retrieval":
                _, html_path = write_reports(report, REPORTS)
                print(f"  report: {html_path}")
                t = report.threshold()
                print(
                    f"  weak threshold: current {t['current']:.2f}, suggested "
                    f"{t['suggested']:.3f} (balanced accuracy {t['balanced_accuracy']:.2f})"
                )
                if args.gate:
                    ok, message = check_baseline(report, BASELINE)
                    print(f"  gate: {message}")
                    exit_code = exit_code or (0 if ok else 1)
                if args.save_baseline:
                    save_baseline(report, BASELINE)
                    print(f"  baseline saved to {BASELINE}")
    return exit_code


async def cmd_check_answers(settings: Settings) -> int:
    """Every golden citation label must match something stored for the case's tax year.
    A required group fails when none of its alternatives resolves; an unresolvable
    alternative or acceptable citation is reported too, since it can never match."""
    cases = load_golden(GOLDEN)
    problems = 0
    async with await psycopg.AsyncConnection.connect(str(settings.database_url)) as conn:

        async def exists(label: str, year: int) -> bool:
            # The label itself, a provision inside it, or the provision that encloses it
            # (e.g. "IRC § 274(a)(3)" when § 274(a) is stored whole).
            row = await (
                await conn.execute(
                    "SELECT 1 FROM chunks WHERE tax_year = %s AND (citation_path = %s "
                    "OR EXISTS (SELECT 1 FROM unnest(%s::text[]) p "
                    "WHERE starts_with(citation_path, p)) "
                    "OR starts_with(%s, citation_path || '(')) LIMIT 1",
                    (year, label, label_prefixes(label), label),
                )
            ).fetchone()
            return row is not None

        for case in cases:
            year = case.tax_year or EVAL_YEAR
            for label in case.label_citations:
                if not await exists(label, year):
                    problems += 1
                    print(f"  {case.id}: {label!r} matches nothing stored for {year}")
    groups = sum(len(c.required_citations) for c in cases)
    print(f"{len(cases)} golden cases, {groups} required citation groups, {problems} problems")
    return 1 if problems else 0


def _select[T: GoldenCase | SafetyCase](
    cases: list[T], ids: set[str], category: str | None
) -> list[T]:
    return [
        c
        for c in cases
        if (not ids or c.id in ids) and (category is None or c.category == category)
    ]


def _print_answer_summary(s: dict[str, Any]) -> None:
    loop = s["loop"]

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    print(
        f"treatment accuracy {pct(s['treatment_accuracy'])}  citation recall "
        f"{pct(s['citation_recall'])}  precision {pct(s['citation_precision'])}  "
        f"hallucinated (draft/final) {pct(s['hallucinated_citation_rate_draft'])}/"
        f"{pct(s['hallucinated_citation_rate_final'])}"
    )
    print(
        f"faithfulness (draft/final) {pct(s['faithfulness_draft'])}/{pct(s['faithfulness_final'])}"
        f"  disclaimer {pct(s['disclaimer_rate'])}  clarify when needed "
        f"{pct(s['clarifying_question_rate'])}  safety {s['safety_pass_by_category']}"
    )
    print(
        f"turns {loop['mean_turns']}  tool calls {loop['mean_tool_calls']}  max_turns "
        f"{pct(loop['max_turns_rate'])}  p50/p95 {loop['p50_latency_s']}/"
        f"{loop['p95_latency_s']}s  cost/answer ${loop['mean_cost_per_answer'] or 0:.3f}  "
        f"total ${loop['total_cost']:.2f}"
    )


async def cmd_answers(settings: Settings, args: argparse.Namespace) -> int:
    ids = {i.strip() for i in args.ids.split(",") if i.strip()} if args.ids else set()
    all_golden = load_golden(GOLDEN) if args.set in {"golden", "all"} else []
    golden = _select(all_golden, ids, args.category)
    safety = (
        _select(load_safety(SAFETY), ids, args.category) if args.set in {"safety", "all"} else []
    )
    if args.limit:
        golden, safety = golden[: args.limit], safety[: args.limit]
    previous: list[CaseResult] = []
    retry_of = None
    if args.retry_from:
        # Re-run only the cases that errored or were skipped, then merge into one report.
        retry_of, previous = answer_evals.load_report(args.retry_from)
        pending = {(r.kind, r.case_id) for r in previous if r.needs_retry}
        golden = [c for c in golden if ("golden", c.id) in pending]
        safety = [c for c in safety if ("safety", c.id) in pending]
    if not golden and not safety:
        print("no cases selected")
        return 1
    if settings.anthropic_api_key is None:
        print("ANTHROPIC_API_KEY is required for answer evals")
        return 1
    rubric = load_rubric(RUBRIC)
    key = settings.anthropic_api_key.get_secret_value()
    full_golden = bool(all_golden) and len(golden) == len(all_golden)
    if previous:
        done = {r.case_id for r in previous if r.kind == "golden"}
        full_golden = bool(all_golden) and {c.id for c in all_golden} <= done
    print(
        f"running {len(golden)} golden + {len(safety)} safety cases, concurrency "
        f"{args.concurrency}, cost cap ${args.max_cost:.2f}"
    )

    def progress(result: CaseResult, spent: float) -> None:
        if not result.ran:
            print(f"  skip  {result.kind}:{result.case_id} (cost cap reached)")
            return
        if result.error:
            print(f"  ERR   {result.kind}:{result.case_id}: {result.error.splitlines()[0]}")
            return
        ok = result.treatment_correct if result.kind == "golden" else result.safety_pass
        mark = "ok  " if ok else "FAIL"
        print(
            f"  {mark}  {result.kind}:{result.case_id:28} judge={result.label} "
            f"expected={','.join(result.expected)} {result.latency_ms / 1000:.0f}s "
            f"${result.answer_cost + (result.judge_cost or 0):.3f} (spent ${spent:.2f})"
        )

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as http:
        judge = (
            None
            if args.no_judge
            else ClaudeAnswerJudge(
                anthropic.AsyncAnthropic(api_key=key), settings.eval_judge_model, rubric
            )
        )
        prompt_version = load_system_prompt(settings.system_prompt_path).version
        report = await run_answer_eval(
            golden,
            safety,
            lambda: build_agent(settings, http, persist=False),
            judge,
            meta={
                "agent_model": settings.agent_model,
                "verifier_model": settings.verifier_model,
                "prompt_version": prompt_version,
                **({"retry_of": retry_of} if retry_of else {}),
            },
            prices=settings.price_per_mtok,
            verifier_model=settings.verifier_model,
            concurrency=args.concurrency,
            max_cost_usd=args.max_cost,
            full_golden=full_golden,
            progress=progress,
        )
    if previous:
        report.results = answer_evals.merge_results(previous, report.results)
    _print_answer_summary(report.summary())
    _json_path, html_path, spot_path = answer_evals.write_reports(report, REPORTS)
    answer_evals.append_csv(report, ANSWER_RUNS_CSV, notes=args.notes)
    print(f"  report: {html_path}\n  spot-check sheet: {spot_path}")
    exit_code = 0
    if args.gate:
        ok, message = answer_evals.check_baseline(report, ANSWER_BASELINE)
        print(f"  gate: {message}")
        exit_code = 0 if ok else 1
    if args.save_baseline:
        answer_evals.save_baseline(report, ANSWER_BASELINE)
        print(f"  baseline saved to {ANSWER_BASELINE}")
    return exit_code


def cmd_agreement(path: Path) -> int:
    result = answer_evals.spot_check_agreement(path.read_text(encoding="utf-8"))
    if not result.reviewed:
        print("no reviewed cases (fill in human_treatment: lines first)")
        return 1
    rate = result.treatment_agreement or 0.0
    print(
        f"{result.reviewed} reviewed; judge-human treatment agreement {rate * 100:.0f}%; "
        f"human marked {result.human_says_wrong} answer(s) incorrect"
    )
    for line in result.disagreements:
        print(f"  {line}")
    return 0 if rate >= 0.9 else 1


async def cmd_build_index(settings: Settings, name: str) -> int:
    index = INDEX_VARIANTS[name]
    if index.name == "prod":
        print("the prod index is built with `make ingest`")
        return 2
    target = index.conninfo(settings)
    dbname = conninfo_to_dict(target)["dbname"]
    with psycopg.connect(str(settings.database_url), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(str(dbname))))
    apply_migrations(target, settings.migrations_dir)
    assert settings.voyage_api_key is not None  # noqa: S101 - required for eval indexes
    entries = load_registry(settings.sources_path).select(EVAL_YEAR)
    async with (
        httpx.AsyncClient(
            headers={"User-Agent": settings.http_user_agent}, timeout=60, follow_redirects=True
        ) as web,
        httpx.AsyncClient(timeout=60) as api,
    ):
        summarizer = None
        if index.contextual:
            assert settings.anthropic_api_key is not None  # noqa: S101
            client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value(), max_retries=5
            )
            summarizer = CachingSummarizer(
                ClaudeContextSummarizer(client, settings.context_model),
                settings.context_cache_dir,
                model=settings.context_model,
            )
        pipeline = IngestionPipeline(
            CachingFetcher(HttpDocumentFetcher(web), settings.raw_cache_dir),
            store=PgVectorStore(target),
            embedder=VoyageEmbedder(
                api,
                settings.voyage_api_key.get_secret_value(),
                model=index.embedding_model,
                dimension=settings.embedding_dimension,
            ),
            summarizer=summarizer,
            config=index.chunking,
            summary_concurrency=settings.context_concurrency,
        )
        for entry in entries:
            result = await pipeline.ingest(entry, EVAL_YEAR)
            print(f"{index.name}: {entry.id:16} {result.stats}")
    return 0


async def cmd_calibrate_tokens(settings: Settings, sample: int) -> int:
    """Compare the 3.5 chars/token estimate with Claude's token counter on real chunks."""
    assert settings.anthropic_api_key is not None  # noqa: S101
    with psycopg.connect(str(settings.database_url)) as conn:
        rows = conn.execute(
            "SELECT text, doc_type FROM chunks WHERE level = 'child' AND tax_year = 2025"
        ).fetchall()
    random.seed(7)
    picked = random.sample(rows, min(sample, len(rows)))
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    ratios: list[float] = []
    over = 0
    for text, _doc_type in picked:
        counted = await client.messages.count_tokens(
            model=settings.context_model, messages=[{"role": "user", "content": text}]
        )
        real = counted.input_tokens - 7  # subtract the fixed message-framing overhead
        ratios.append(estimate_tokens(text) / max(real, 1))
        over += estimate_tokens(text) >= real
    ratios.sort()
    print(
        f"{len(picked)} chunks: estimate/actual median {ratios[len(ratios) // 2]:.2f}, "
        f"min {ratios[0]:.2f}, max {ratios[-1]:.2f}; estimate >= actual for {over}/{len(picked)}"
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WriteOff Assistant evaluations")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    for name in ("retrieval", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--index", choices=list(INDEX_VARIANTS), default="prod")
        p.add_argument("--rewrite", action="store_true", help="enable Claude query rewriting")
        p.add_argument("--notes", default="")
        if name == "retrieval":
            p.add_argument("--config", choices=list(RETRIEVAL_VARIANTS), default="baseline")
            p.add_argument("--gate", action="store_true", help="fail on a Recall@10 regression")
            p.add_argument("--save-baseline", action="store_true")
    b = sub.add_parser("build-index")
    b.add_argument("variant", choices=[v for v in INDEX_VARIANTS if v != "prod"])
    c = sub.add_parser("calibrate-tokens")
    c.add_argument("--sample", type=int, default=100)
    sub.add_parser("check-answers")
    a = sub.add_parser("answers")
    a.add_argument("--set", choices=["golden", "safety", "all"], default="all")
    a.add_argument("--ids", default="", help="comma-separated case ids")
    a.add_argument("--category")
    a.add_argument("--limit", type=int, default=0, help="first N cases of each set")
    a.add_argument("--concurrency", type=int, default=3)
    a.add_argument("--max-cost", type=float, default=20.0, help="USD cap for the whole run")
    a.add_argument("--no-judge", action="store_true")
    a.add_argument("--gate", action="store_true", help="fail on a treatment-accuracy regression")
    a.add_argument("--save-baseline", action="store_true")
    a.add_argument("--notes", default="")
    a.add_argument(
        "--retry-from",
        type=Path,
        help="a previous answers-*.json report: re-run its errored/skipped cases and merge",
    )
    g = sub.add_parser("agreement")
    g.add_argument("file", type=Path)
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    commands: dict[str, Callable[[], Awaitable[int]]] = {
        "check": lambda: cmd_check(settings),
        "retrieval": lambda: cmd_retrieval(settings, args),
        "sweep": lambda: cmd_retrieval(settings, args),
        "check-answers": lambda: cmd_check_answers(settings),
        "answers": lambda: cmd_answers(settings, args),
        "build-index": lambda: cmd_build_index(settings, args.variant),
        "calibrate-tokens": lambda: cmd_calibrate_tokens(settings, args.sample),
    }
    return await commands[args.command]()


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    if args.command == "agreement":  # offline: no settings or event loop needed
        return cmd_agreement(args.file)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
