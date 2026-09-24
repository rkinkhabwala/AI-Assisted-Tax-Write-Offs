"""Run the retrieval eval: search every case, resolve labels, score, and report.

Outputs per run: `evals/reports/retrieval-<run>.json` and `.html`, and one row appended
to `evals/retrieval_runs.csv`. `check_baseline` implements the regression gate (spec
section 7c): Recall@10 may not drop more than 3 points below the saved baseline.
"""

import asyncio
import csv
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from writeoff.evals.dataset import RetrievalCase
from writeoff.evals.html import render_html
from writeoff.evals.metrics import CaseScores, best_threshold, mean, score_case
from writeoff.evals.variants import IndexVariant
from writeoff.models import ChunkLevel, SearchFilters
from writeoff.retrieval.hybrid import HybridRetriever, RetrievalConfig

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "run_at",
    "run_id",
    "index",
    "embedding_model",
    "reranker_model",
    "contextual",
    "chunk_target_min",
    "chunk_target_max",
    "chunk_overlap",
    "merge_statutory",
    "retrieval",
    "rewrite",
    "dense_k",
    "lexical_k",
    "rrf_k",
    "dense_weight",
    "lexical_weight",
    "rerank",
    "rerank_depth",
    "final_k",
    "min_primary",
    "queries",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
    "weak_threshold_suggested",
    "notes",
]
REGRESSION_TOLERANCE = 0.03  # spec 7c: fail if Recall@10 drops more than 3 points


@dataclass(slots=True)
class CaseResult:
    case: RetrievalCase
    scores: CaseScores | None
    retrieved: list[dict[str, Any]]
    top_rerank: float | None
    weak: bool


@dataclass(slots=True)
class EvalReport:
    run_id: str
    index: IndexVariant
    retrieval_name: str
    retrieval: RetrievalConfig
    rewrite: bool
    reranker_model: str
    results: list[CaseResult]
    unresolved: dict[str, list[str]] = field(default_factory=dict)

    @property
    def scored(self) -> list[CaseResult]:
        return [r for r in self.results if r.scores is not None]

    def summary(self) -> dict[str, float]:
        s = [r.scores for r in self.scored if r.scores is not None]
        return {
            "recall_at_5": mean([x.recall_at_5 for x in s]),
            "recall_at_10": mean([x.recall_at_10 for x in s]),
            "mrr": mean([x.mrr for x in s]),
            "ndcg_at_10": mean([x.ndcg_at_10 for x in s]),
        }

    def by_doc_type(self) -> dict[str, dict[str, float]]:
        """Target-level recall grouped by the labeled target's document type."""
        found5: dict[str, list[float]] = defaultdict(list)
        found10: dict[str, list[float]] = defaultdict(list)
        for r in self.scored:
            assert r.scores is not None  # noqa: S101 - narrowed by `scored`
            for t in r.case.relevant:
                rank = r.scores.found.get(t.citation)
                found5[t.doc_type.value].append(float(rank is not None and rank <= 5))
                found10[t.doc_type.value].append(float(rank is not None and rank <= 10))
        return {
            dt: {
                "targets": len(found10[dt]),
                "recall_at_5": mean(found5[dt]),
                "recall_at_10": mean(found10[dt]),
            }
            for dt in sorted(found10)
        }

    def by_category(self) -> dict[str, dict[str, float]]:
        groups: dict[str, list[CaseScores]] = defaultdict(list)
        for r in self.scored:
            if r.scores is not None:
                groups[r.case.category].append(r.scores)
        return {
            cat: {
                "cases": len(s),
                "recall_at_10": mean([x.recall_at_10 for x in s]),
                "mrr": mean([x.mrr for x in s]),
            }
            for cat, s in sorted(groups.items())
        }

    def threshold(self) -> dict[str, Any]:
        in_scope = [
            r.top_rerank
            for r in self.results
            if not r.case.out_of_scope and r.top_rerank is not None
        ]
        oos = [
            r.top_rerank for r in self.results if r.case.out_of_scope and r.top_rerank is not None
        ]
        value, accuracy = best_threshold(in_scope, oos)
        return {
            "suggested": value,
            "balanced_accuracy": accuracy,
            "current": self.retrieval.weak_threshold,
            "in_scope_top_scores": sorted(in_scope),
            "out_of_scope_top_scores": sorted(oos),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "index": asdict(self.index),
            "retrieval_name": self.retrieval_name,
            "retrieval": asdict(self.retrieval),
            "rewrite": self.rewrite,
            "reranker_model": self.reranker_model,
            "summary": self.summary(),
            "by_doc_type": self.by_doc_type(),
            "by_category": self.by_category(),
            "weak_threshold": self.threshold(),
            "unresolved_targets": self.unresolved,
            "cases": [
                {
                    "id": r.case.id,
                    "query": r.case.query,
                    "category": r.case.category,
                    "out_of_scope": r.case.out_of_scope,
                    "targets": [t.model_dump() for t in r.case.relevant],
                    "scores": None if r.scores is None else asdict(r.scores),
                    "top_rerank": r.top_rerank,
                    "weak": r.weak,
                    "retrieved": r.retrieved,
                }
                for r in self.results
            ],
        }


async def resolve_targets(
    retriever: HybridRetriever, cases: list[RetrievalCase]
) -> tuple[dict[tuple[int, str], frozenset[UUID]], dict[str, list[str]]]:
    """Each target citation -> the child chunk ids holding it in the index under test."""
    resolved: dict[tuple[int, str], frozenset[UUID]] = {}
    unresolved: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        for target in case.relevant:
            key = (case.tax_year, target.citation)
            if key not in resolved:
                chunks = await retriever.lookup_citation(target.citation, case.tax_year)
                resolved[key] = frozenset(c.id for c in chunks if c.level is ChunkLevel.CHILD)
            if not resolved[key]:
                unresolved[case.id].append(target.citation)
    return resolved, dict(unresolved)


async def run_eval(
    retriever: HybridRetriever,
    cases: list[RetrievalCase],
    *,
    index: IndexVariant,
    retrieval_name: str,
    rewrite: bool,
    reranker_model: str,
    concurrency: int = 4,
) -> EvalReport:
    resolved, unresolved = await resolve_targets(retriever, cases)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case: RetrievalCase) -> CaseResult:
        filters = SearchFilters(tax_year=case.tax_year, entity_type=case.entity_type)
        async with semaphore:
            response = await retriever.search(case.query, filters)
        ids = [r.chunk.id for r in response.results]
        scores = None
        if case.relevant:
            per_target = {t.citation: resolved[(case.tax_year, t.citation)] for t in case.relevant}
            scores = score_case(ids, case.relevant, per_target)
        retrieved = [
            {
                "rank": r.rank,
                "citation": r.chunk.citation_path,
                "doc_type": r.chunk.doc_type.value,
                "rerank": r.rerank_score,
                "rrf": r.fused_score,
                "dense_rank": r.dense_rank,
                "lexical_rank": r.lexical_rank,
                "via_citation": r.citation_lookup,
                "relevant": any(
                    r.chunk.id in resolved.get((case.tax_year, t.citation), frozenset())
                    for t in case.relevant
                ),
            }
            for r in response.results
        ]
        top = max(
            (r.rerank_score for r in response.results if r.rerank_score is not None), default=None
        )
        return CaseResult(case, scores, retrieved, top, response.weak)

    results = await asyncio.gather(*(one(c) for c in cases))
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{index.name}-{retrieval_name}"
    return EvalReport(
        run_id,
        index,
        retrieval_name,
        retriever.config,
        rewrite,
        reranker_model,
        list(results),
        unresolved,
    )


def append_csv(report: EvalReport, path: Path, notes: str = "") -> None:
    s = report.summary()
    cfg, chunking = report.retrieval, report.index.chunking
    row = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": report.run_id,
        "index": report.index.name,
        "embedding_model": report.index.embedding_model,
        "reranker_model": report.reranker_model,
        "contextual": report.index.contextual,
        "chunk_target_min": chunking.target_min,
        "chunk_target_max": chunking.target_max,
        "chunk_overlap": chunking.overlap,
        "merge_statutory": chunking.merge_statutory_siblings,
        "retrieval": report.retrieval_name,
        "rewrite": report.rewrite,
        "dense_k": cfg.dense_k,
        "lexical_k": cfg.lexical_k,
        "rrf_k": cfg.rrf_k,
        "dense_weight": cfg.dense_weight,
        "lexical_weight": cfg.lexical_weight,
        "rerank": cfg.rerank,
        "rerank_depth": cfg.rerank_depth,
        "final_k": cfg.final_k,
        "min_primary": cfg.min_primary,
        "queries": len(report.scored),
        **{k: f"{v:.4f}" for k, v in s.items()},
        "weak_threshold_suggested": f"{report.threshold()['suggested']:.3f}",
        "notes": notes,
    }
    new = not path.exists() or path.read_text(encoding="utf-8").strip() == ""
    header_ok = not new and path.read_text(encoding="utf-8").splitlines()[0] == ",".join(CSV_FIELDS)
    if not new and not header_ok:
        raise ValueError(f"{path} has an unexpected header; move it aside to start a new log")
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def write_reports(report: EvalReport, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    data = report.to_json()
    json_path = directory / f"retrieval-{report.run_id}.json"
    html_path = directory / f"retrieval-{report.run_id}.html"
    json_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    html_path.write_text(render_html(data), encoding="utf-8")
    return json_path, html_path


def check_baseline(report: EvalReport, baseline_path: Path) -> tuple[bool, str]:
    """Regression gate: Recall@10 must stay within 3 points of the saved baseline."""
    if not baseline_path.exists():
        return True, f"no baseline at {baseline_path}; skipping the regression gate"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    before, now = float(baseline["recall_at_10"]), report.summary()["recall_at_10"]
    ok = now >= before - REGRESSION_TOLERANCE
    verdict = "ok" if ok else "REGRESSION"
    return ok, f"Recall@10 {now:.3f} vs baseline {before:.3f} ({baseline['run_id']}): {verdict}"


def save_baseline(report: EvalReport, baseline_path: Path) -> None:
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"run_id": report.run_id, **report.summary()}
    baseline_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
