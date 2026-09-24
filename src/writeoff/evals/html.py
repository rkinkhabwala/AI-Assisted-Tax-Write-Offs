"""Self-contained HTML rendering of a retrieval eval report (no external assets)."""

from html import escape
from typing import Any

_STYLE = """
:root { --bg:#fff; --fg:#1b1f24; --muted:#5b6470; --line:#d8dde3;
        --hit:#e6f4ea; --miss:#fdecea; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171a; --fg:#e6e9ec; --muted:#9aa4ae; --line:#2c3238;
          --hit:#17311f; --miss:#3a1d1b; }
}
body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:24px; }
h1 { font-size:20px; } h2 { font-size:16px; margin-top:28px; }
table { border-collapse:collapse; margin:8px 0; }
td, th { border:1px solid var(--line); padding:4px 8px; text-align:left; vertical-align:top; }
th { color:var(--muted); font-weight:600; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.hit { background:var(--hit); } .miss { background:var(--miss); }
details { margin:6px 0; } summary { cursor:pointer; }
.muted { color:var(--muted); }
"""


def _pct(value: float) -> str:
    return f"{value * 100:.1f}"


def _table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"num" if i >= numeric_from else ""}">{cell}</td>'
            for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def render_html(data: dict[str, Any]) -> str:
    s = data["summary"]
    index = data["index"]
    parts = [
        f"<h1>Retrieval eval — {escape(data['run_id'])}</h1>",
        f"<p class='muted'>index <b>{escape(index['name'])}</b> ({escape(index['description'])}); "
        f"retrieval <b>{escape(data['retrieval_name'])}</b>; rewrite {data['rewrite']}; "
        f"reranker {escape(data['reranker_model'])}</p>",
        _table(
            ["metric", "value"],
            [
                ["Recall@5", _pct(s["recall_at_5"])],
                ["Recall@10", _pct(s["recall_at_10"])],
                ["MRR", f"{s['mrr']:.3f}"],
                ["nDCG@10", f"{s['ndcg_at_10']:.3f}"],
            ],
        ),
        "<h2>By target document type</h2>",
        _table(
            ["doc type", "targets", "Recall@5", "Recall@10"],
            [
                [escape(dt), str(v["targets"]), _pct(v["recall_at_5"]), _pct(v["recall_at_10"])]
                for dt, v in data["by_doc_type"].items()
            ],
        ),
        "<h2>By category</h2>",
        _table(
            ["category", "cases", "Recall@10", "MRR"],
            [
                [escape(c), str(v["cases"]), _pct(v["recall_at_10"]), f"{v['mrr']:.3f}"]
                for c, v in data["by_category"].items()
            ],
        ),
    ]
    t = data["weak_threshold"]
    parts.append(
        f"<h2>Weak-result threshold</h2><p>current {t['current']:.2f}; suggested "
        f"{t['suggested']:.3f} (balanced accuracy {t['balanced_accuracy']:.2f} separating "
        f"in-scope from out-of-scope queries by top rerank score)</p>"
    )
    if data["unresolved_targets"]:
        items = "".join(
            f"<li>{escape(k)}: {escape(', '.join(v))}</li>"
            for k, v in data["unresolved_targets"].items()
        )
        parts.append(f"<h2>Unresolved labels</h2><ul>{items}</ul>")
    parts.append("<h2>Cases</h2>")
    for case in data["cases"]:
        scores = case["scores"]
        label = "out of scope" if case["out_of_scope"] else f"R@10 {_pct(scores['recall_at_10'])}"
        cls = "hit" if case["out_of_scope"] or scores["recall_at_10"] == 1 else "miss"
        targets = (
            ", ".join(
                f"{escape(t['citation'])} (g{t['grade']}"
                f"{'' if not scores else ', rank ' + str(scores['found'].get(t['citation'], '—'))})"
                for t in case["targets"]
            )
            or "—"
        )
        rows = [
            [
                str(r["rank"]),
                f"{r['rerank']:.3f}" if r["rerank"] is not None else "—",
                "✓" if r["relevant"] else "",
                escape(r["citation"]),
            ]
            for r in case["retrieved"]
        ]
        parts.append(
            f"<details><summary class='{cls}'>[{escape(case['category'])}] "
            f"{escape(case['query'])} — {label}</summary>"
            f"<p class='muted'>targets: {targets}</p>"
            f"{_table(['#', 'rerank', 'hit', 'citation'], rows, numeric_from=0)}</details>"
        )
    body = "\n".join(parts)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Retrieval eval</title><style>{_STYLE}</style></head><body>{body}</body></html>"
    )


def _maybe_pct(value: float | None) -> str:
    return "—" if value is None else _pct(value)


def _maybe(value: float | None, fmt: str = "{:.2f}") -> str:
    return "—" if value is None else fmt.format(value)


def render_answers_html(data: dict[str, Any]) -> str:
    """The answer + safety eval report."""
    s, meta = data["summary"], data["meta"]
    loop = s["loop"]
    meta_line = ", ".join(f"{escape(str(k))} <b>{escape(str(v))}</b>" for k, v in meta.items())
    parts = [
        f"<h1>Answer eval — {escape(data['run_id'])}</h1>",
        f"<p class='muted'>{meta_line}</p>",
        f"<p class='muted'>{s['golden_cases']} golden + {s['safety_cases']} safety cases; "
        f"{s['skipped']} skipped (cost cap); {s['errors']} agent errors; "
        f"{s['judge_errors']} judge errors; full golden run: {data['full_golden']}</p>",
        "<h2>Answer quality (golden set)</h2>",
        _table(
            ["metric", "value"],
            [
                ["Treatment accuracy (%)", _maybe_pct(s["treatment_accuracy"])],
                ["Citation recall (%)", _maybe_pct(s["citation_recall"])],
                ["Citation precision (%, lower bound)", _maybe_pct(s["citation_precision"])],
                [
                    "Hallucinated citations, draft (% of cites)",
                    _maybe_pct(s["hallucinated_citation_rate_draft"]),
                ],
                [
                    "Hallucinated citations, final (% of cites)",
                    _maybe_pct(s["hallucinated_citation_rate_final"]),
                ],
                ["Faithfulness, draft (% claims supported)", _maybe_pct(s["faithfulness_draft"])],
                ["Faithfulness, final (% claims supported)", _maybe_pct(s["faithfulness_final"])],
                [
                    "Answers with untraced figures (%)",
                    _maybe_pct(s["answers_with_untraced_figures"]),
                ],
                ["Disclaimer present (%)", _maybe_pct(s["disclaimer_rate"])],
                ["Clarifying question when needed (%)", _maybe_pct(s["clarifying_question_rate"])],
                [
                    "Clarifying question when not needed (%)",
                    _maybe_pct(s["unneeded_clarification_rate"]),
                ],
                ["Required text present (%)", _maybe_pct(s["required_text_pass_rate"])],
                ["Forbidden text violations", str(s["forbidden_text_violations"])],
            ],
        ),
        "<h2>Rubric means (0-2)</h2>",
        _table(["dimension", "mean"], [[k, _maybe(v)] for k, v in s["rubric_means"].items()]),
        "<h2>Treatment accuracy by category</h2>",
        _table(
            ["category", "accuracy (%)"],
            [[escape(k), _maybe_pct(v)] for k, v in s["treatment_accuracy_by_category"].items()],
        ),
        "<h2>Safety (pass rate)</h2>",
        _table(
            ["category", "pass (%)"],
            [[escape(k), _maybe_pct(v)] for k, v in s["safety_pass_by_category"].items()],
        ),
        "<h2>Loop behavior and cost</h2>",
        _table(
            ["metric", "value"],
            [
                ["Mean turns", _maybe(loop["mean_turns"], "{:.1f}")],
                ["Mean tool calls", _maybe(loop["mean_tool_calls"], "{:.1f}")],
                ["Hit max_turns (%)", _maybe_pct(loop["max_turns_rate"])],
                ["Hit cost budget (%)", _maybe_pct(loop["budget_stop_rate"])],
                ["Partial answers (%)", _maybe_pct(loop["partial_rate"])],
                ["p50 latency (s)", _maybe(loop["p50_latency_s"], "{:.0f}")],
                ["p95 latency (s)", _maybe(loop["p95_latency_s"], "{:.0f}")],
                ["Mean cost per answer (USD)", _maybe(loop["mean_cost_per_answer"], "{:.3f}")],
                ["  of which agent", _maybe(loop["mean_agent_cost"], "{:.3f}")],
                ["  of which verifier", _maybe(loop["mean_verifier_cost"], "{:.3f}")],
                ["Judge cost, total (USD)", _maybe(loop["judge_cost_total"])],
                ["Run cost, total (USD)", _maybe(loop["total_cost"])],
            ],
        ),
        "<h2>Cases</h2>",
    ]
    for r in data["results"]:
        ok = r["correct"]
        cls = "hit" if ok else "miss"
        label = escape(str(r["label"]))
        cites = r["citations"] or {}
        detail = [
            f"<p><b>expected</b> {escape(', '.join(r['expected']))} · <b>judge</b> {label} · "
            f"status {escape(r['status'])} · turns {r['num_turns']} · tools {r['tool_calls']} · "
            f"{r['latency_ms'] / 1000:.0f}s · verification "
            f"{escape(str(r['verification_status']))}</p>",
        ]
        if cites:
            detail.append(
                f"<p class='muted'>recall {_maybe_pct(cites['recall'])} · precision "
                f"{_maybe_pct(cites['precision'])} · hallucinated (draft) "
                f"{escape(', '.join(cites['hallucinated_draft']) or 'none')}</p>"
            )
        if r["classification"]:
            detail.append(f"<p class='muted'>judge: {escape(r['classification']['rationale'])}</p>")
        if r["scores"]:
            sc = r["scores"]
            detail.append(
                f"<p class='muted'>scores: correctness {sc['correctness']}, completeness "
                f"{sc['completeness']}, clarification {sc['clarification']}, format "
                f"{sc['format']} — {escape(sc['rationale'])}</p>"
            )
        for key in ("error", "judge_error"):
            if r[key]:
                detail.append(f"<p class='miss'>{key}: {escape(r[key])}</p>")
        if r["forbidden_text"] or r["missing_required_text"]:
            detail.append(
                f"<p class='miss'>forbidden {escape(str(r['forbidden_text']))}; missing "
                f"{escape(str(r['missing_required_text']))}</p>"
            )
        detail.append(
            f"<pre style='white-space:pre-wrap'>{escape(r['answer'] or '(no answer)')}</pre>"
        )
        parts.append(
            f"<details><summary class='{cls}'>{escape(r['kind'])} · {escape(r['id'])} "
            f"({escape(r['category'])}): {label}</summary>{''.join(detail)}</details>"
        )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Answer eval {escape(data['run_id'])}</title><style>{_STYLE}</style></head>"
        f"<body>{''.join(parts)}</body></html>"
    )
