"""The retrieval eval harness: dataset rules, metric math, reports and the regression gate."""

import json
import math
from pathlib import Path
from uuid import uuid4

import pytest

from writeoff.evals.dataset import DatasetError, Target, doc_type_of, load_cases
from writeoff.evals.html import render_html
from writeoff.evals.metrics import best_threshold, score_case
from writeoff.models import DocType

REPO = Path(__file__).resolve().parents[1]

# --- dataset -------------------------------------------------------------------------


def test_shipped_dataset() -> None:
    cases = load_cases(REPO / "evals" / "retrieval_queries.jsonl")
    in_scope = [c for c in cases if not c.out_of_scope]
    assert len(in_scope) >= 60  # spec 7a: 60+ labeled queries
    assert sum(c.out_of_scope for c in cases) >= 5
    covered = {t.doc_type for c in in_scope for t in c.relevant}
    assert covered == set(DocType)  # every document type is evaluated
    assert len({c.category for c in in_scope}) >= 10


def _write(tmp_path: Path, *rows: dict[str, object]) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def _case(**overrides: object) -> dict[str, object]:
    case: dict[str, object] = {
        "id": "a",
        "category": "c",
        "query": "q",
        "relevant": [{"citation": "IRC § 162(a)"}],
    }
    case.update(overrides)
    return case


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_case(), _case()], "duplicate case ids"),
        ([_case(out_of_scope=True)], "out-of-scope cases have no targets"),
        ([_case(relevant=[])], "in-scope cases need some"),
        ([_case(relevant=[{"citation": "Blog post 7"}])], "document type"),
        ([_case(relevant=[{"citation": "IRC § 1", "grade": 3}])], "grade"),
        ([_case(relevant=[{"citation": "IRC § 1"}, {"citation": "IRC § 1"}])], "duplicate target"),
    ],
)
def test_dataset_validation(tmp_path: Path, rows: list[dict[str, object]], message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        load_cases(_write(tmp_path, *rows))


def test_doc_type_of() -> None:
    assert doc_type_of("IRC § 280A(c)(1)") is DocType.IRC
    assert doc_type_of("Treas. Reg. § 1.162-5(a)") is DocType.TREASURY_REGULATION
    assert doc_type_of("Pub 463, ch. 2") is DocType.IRS_PUBLICATION
    assert doc_type_of("Instructions for Form 8829") is DocType.FORM_INSTRUCTIONS


# --- metrics -------------------------------------------------------------------------


def test_score_case() -> None:
    a1, a2, b1, noise = uuid4(), uuid4(), uuid4(), [uuid4() for _ in range(10)]
    targets = [Target(citation="A", grade=2), Target(citation="B", grade=1), Target(citation="C")]
    resolved = {"A": frozenset({a1, a2}), "B": frozenset({b1}), "C": frozenset({uuid4()})}
    retrieved = [noise[0], a1, a2, noise[1], noise[2], noise[3], b1, *noise[4:]]
    s = score_case(retrieved, targets, resolved)
    assert s.found == {"A": 2, "B": 7}  # A credited once, at its first chunk
    assert s.recall_at_5 == pytest.approx(1 / 3)
    assert s.recall_at_10 == pytest.approx(2 / 3)
    assert s.mrr == pytest.approx(1 / 2)
    dcg = 3 / math.log2(3) + 1 / math.log2(8)
    idcg = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert s.ndcg_at_10 == pytest.approx(dcg / idcg)


def test_score_case_no_hits() -> None:
    s = score_case([uuid4()], [Target(citation="A")], {"A": frozenset({uuid4()})})
    assert (s.recall_at_10, s.mrr, s.ndcg_at_10) == (0.0, 0.0, 0.0)


def test_best_threshold() -> None:
    threshold, accuracy = best_threshold([0.7, 0.8, 0.9], [0.2, 0.4, 0.75])
    assert threshold == pytest.approx(0.7)
    assert accuracy == pytest.approx((1 + 2 / 3) / 2)
    assert best_threshold([], [0.1]) == (0.0, 0.0)


# --- reports and gate ----------------------------------------------------------------


def _report_data() -> dict[str, object]:
    return {
        "run_id": "r1",
        "index": {"name": "prod", "description": "d"},
        "retrieval_name": "base",
        "rewrite": False,
        "reranker_model": "rerank-2.5",
        "summary": {"recall_at_5": 0.5, "recall_at_10": 0.75, "mrr": 0.6, "ndcg_at_10": 0.55},
        "by_doc_type": {"irc": {"targets": 2, "recall_at_5": 0.5, "recall_at_10": 1.0}},
        "by_category": {"meals": {"cases": 1, "recall_at_10": 0.75, "mrr": 0.6}},
        "weak_threshold": {"current": 0.6, "suggested": 0.66, "balanced_accuracy": 1.0},
        "unresolved_targets": {},
        "cases": [
            {
                "id": "x",
                "query": "<script>alert(1)</script>",
                "category": "meals",
                "out_of_scope": False,
                "targets": [{"citation": "IRC § 274(n)", "grade": 2}],
                "scores": {"recall_at_10": 1.0, "found": {"IRC § 274(n)": 1}},
                "retrieved": [
                    {"rank": 1, "rerank": 0.9, "relevant": True, "citation": "IRC § 274(n)"}
                ],
            }
        ],
    }


def test_render_html_escapes_and_summarizes() -> None:
    html = render_html(_report_data())
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "Recall@10" in html
    assert "75.0" in html
    assert "prefers-color-scheme: dark" in html
