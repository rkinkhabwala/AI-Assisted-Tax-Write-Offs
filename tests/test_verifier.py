"""Grounding verifier: deterministic checks, the verify-revise loop, and the Claude judge."""

import json
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import anthropic
import httpx2
import pytest

from tests.fakes import FakeEmbedder, FakeReranker, FakeRewriter
from writeoff.agent.evidence import Evidence, all_numbers, money_figures
from writeoff.agent.prompt import DISCLAIMER
from writeoff.agent.tools import ToolRuntime
from writeoff.agent.verifier import (
    ClaimCheck,
    ClaudeJudge,
    Findings,
    Judge,
    Judgement,
    JudgeOutput,
    Label,
    Verifier,
    VerifierError,
    apply_edits,
    citation_tags,
    cited_evidence,
    deterministic_findings,
    needs_verification,
    resolve_tag,
    strip_unsupported,
    tidy,
)
from writeoff.retrieval.hybrid import HybridRetriever
from writeoff.retrieval.pgvector_store import PgVectorStore
from writeoff.tax_parameters import TaxParameters

PARAMS = TaxParameters(Path(__file__).parent / "fixtures" / "tax_parameters", (2025,))
MEALS = "IRC § 274(n)"
PUB = "Pub 463, ch. 2, 50% Limit"


def _evidence() -> Evidence:
    ev = Evidence()
    ev.add_passage(MEALS, "(n) Only 50 percent of meal expenses allowed as deduction ...")
    ev.add_passage(
        PUB, "In general, you can deduct only 50% of your business-related meal expenses."
    )
    ev.add_parameter(
        "standard_mileage_rate_business",
        "0.70 usd_per_mile (source: https://www.irs.gov/publications/p463)",
    )
    return ev


def _answer(body: str) -> str:
    return f"**Short answer**\n\n{body}\n\n{DISCLAIMER}"


# --- deterministic checks --------------------------------------------------------------


def test_citation_tags() -> None:
    text = (
        "Meals are 50% [IRC § 274(n)(1)]; see [Pub 463, ch. 2; Treas. Reg. § 1.274-12(a)] "
        "and [note]."
    )
    assert citation_tags(text) == ["IRC § 274(n)(1)", "Pub 463, ch. 2", "Treas. Reg. § 1.274-12(a)"]


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("IRC § 274(n)", MEALS),  # exact
        ("IRC § 274(n)(1)", MEALS),  # sub-provision inside a retrieved section
        ("§ 274(n)(2)", MEALS),  # abbreviated form
        ("IRC § 274", MEALS),  # broader cite of a retrieved provision
        ("Pub 463, ch. 2", PUB),
        ("IRC § 274(n)(1), (2)", MEALS),  # several sub-provisions in one tag
        ("IRC § 162(a)", None),
        ("IRC § 27", None),  # a prefix that isn't a provision boundary
    ],
)
def test_resolve_tag(tag: str, expected: str | None) -> None:
    assert resolve_tag(tag, [MEALS, PUB]) == expected


def test_money_figures_and_numbers() -> None:
    assert money_figures("Deduct $2,500,000, or 50% of it, at 70 cents a mile; § 179(b)(1).") == [
        ("$2,500,000", Decimal(2500000)),
        ("50%", Decimal(50)),
        ("70 cents", Decimal("0.7")),
    ]
    assert {Decimal("0.7"), Decimal(70), Decimal(2025)} <= all_numbers("70 cents per mile in 2025")


def test_deterministic_findings() -> None:
    draft = _answer(
        "Meals are 50% deductible [IRC § 274(n)(1)] at up to $75 per meal [IRC § 162(a)]. "
        "Your $2,000 lunch bill counts. Mileage is 70 cents [Pub 463, ch. 2]."
    )
    findings = deterministic_findings(
        draft, _evidence(), question="I spent $2,000 on client lunches."
    )
    assert findings.unknown_citations == ["IRC § 162(a)"]
    assert findings.untraced_numbers == [
        "$75"
    ]  # $2,000 is the user's own fact; 70 cents = 0.70 parameter


def test_needs_verification_and_cited_evidence() -> None:
    assert not needs_verification("Is the office used only for business?")
    assert needs_verification(_answer("Yes."))
    blocks = cited_evidence("Meals [IRC § 274(n)(1)].", _evidence())
    assert f'citation="{MEALS}"' in blocks
    assert PUB not in blocks  # uncited passages are not sent
    assert "standard_mileage_rate_business" in blocks


# --- the loop, with a scripted judge ------------------------------------------------------


class ScriptedJudge(Judge):
    def __init__(self, *outputs: JudgeOutput | Exception) -> None:
        self.outputs = list(outputs)
        self.seen: list[str] = []

    async def judge(
        self, question: str, draft: str, evidence: str, findings: Findings
    ) -> Judgement:
        self.seen.append(draft)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return Judgement(output, 100, 20)


def _claim(text: str, label: Label, replacement: str | None = None) -> ClaimCheck:
    return ClaimCheck(
        claim=text, citations=[MEALS], label=label, reason="r", replacement=replacement
    )


GOOD = "Meals are 50% deductible [IRC § 274(n)]."
BAD = "Meals are fully deductible at conferences [IRC § 274(n)]."
NOTE = "I couldn't confirm the conference rule from my sources."


async def test_verified_on_first_pass() -> None:
    draft = _answer(GOOD)
    judge = ScriptedJudge(JudgeOutput(claims=[_claim(GOOD, "SUPPORTED")], unconfirmed=[]))
    text, report = await Verifier(judge).verify(draft, "q", _evidence())
    assert text == draft
    assert (report.status, report.rounds, report.faithfulness) == ("verified", 1, 1.0)


async def test_edits_applied_once_then_verified() -> None:
    draft = _answer(f"{GOOD}\n{BAD}")
    judge = ScriptedJudge(
        JudgeOutput(
            claims=[_claim(GOOD, "SUPPORTED"), _claim(BAD, "UNSUPPORTED", NOTE)],
            unconfirmed=["conference meals"],
        ),
        JudgeOutput(claims=[], unconfirmed=[]),  # the note itself is not a claim
    )
    text, report = await Verifier(judge).verify(draft, "q", _evidence())
    assert report.status == "revised"
    assert report.rounds == 2
    assert judge.seen == [draft, NOTE]  # round 2 checks only the edited sentence
    assert GOOD in text
    assert NOTE in text
    assert BAD not in text
    assert "conference meals" in text  # what was narrowed is disclosed
    assert text.rstrip().endswith(DISCLAIMER)


async def test_dropped_claim_needs_no_second_judge_call() -> None:
    draft = _answer(f"{GOOD}\n{BAD}")
    judge = ScriptedJudge(JudgeOutput(claims=[_claim(BAD, "UNSUPPORTED")], unconfirmed=["x"]))
    text, report = await Verifier(judge).verify(draft, "q", _evidence())
    assert report.status == "revised"
    assert len(judge.seen) == 1
    assert BAD not in text
    assert GOOD in text


async def test_second_failure_strips_and_discloses() -> None:
    draft = _answer(f"{GOOD}\n{BAD}")
    still_wrong = "Meals at conferences are 80% deductible [IRC § 274(n)]."
    judge = ScriptedJudge(
        JudgeOutput(
            claims=[_claim(BAD, "UNSUPPORTED", still_wrong)], unconfirmed=["conference meals"]
        ),
        JudgeOutput(claims=[_claim(still_wrong, "UNSUPPORTED")], unconfirmed=[]),
    )
    text, report = await Verifier(judge).verify(draft, "q", _evidence())
    assert report.status == "partially_verified"
    assert GOOD in text
    assert "80%" not in text
    assert BAD not in text
    assert "conference meals" in text
    assert text.rstrip().endswith(DISCLAIMER)


async def test_deterministic_findings_override_a_lenient_judge() -> None:
    capped = "Meals are capped at $75 [IRC § 274(n)]."
    draft = _answer(f"{GOOD}\n{capped}")
    lenient = JudgeOutput(
        claims=[_claim(GOOD, "SUPPORTED"), _claim(capped, "SUPPORTED")], unconfirmed=[]
    )
    text, report = await Verifier(ScriptedJudge(lenient)).verify(draft, "q", _evidence())
    assert report.status == "partially_verified"
    assert report.untraced_numbers == ["$75"]
    assert "$75" not in text
    assert GOOD in text


async def test_judge_failure_degrades_gracefully() -> None:
    draft = _answer(GOOD)
    text, report = await Verifier(ScriptedJudge(VerifierError("down"))).verify(
        draft, "q", _evidence()
    )
    assert report.status == "error"
    assert GOOD in text
    assert "verification was unavailable" in text
    assert text.rstrip().endswith(DISCLAIMER)


async def test_clarifying_question_is_not_verified() -> None:
    judge = ScriptedJudge()
    question = "Which entity type is your business?"
    text, report = await Verifier(judge).verify(question, "q", Evidence())
    assert text == question
    assert report.status == "skipped"
    assert judge.seen == []


def test_strip_unsupported_keeps_other_lines() -> None:
    text = strip_unsupported(_answer(f"{GOOD}\n{BAD}"), [_claim(BAD, "UNSUPPORTED")], [])
    assert GOOD in text
    assert BAD not in text


# --- Claude judge (mocked API) -------------------------------------------------------------


def _claude(payload: dict[str, object] | None, stop_reason: str = "end_turn") -> ClaudeJudge:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert body["thinking"] == {"type": "disabled"}
        assert "<evidence>" in body["messages"][0]["content"]
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [{"type": "text", "text": json.dumps(payload)}] if payload else [],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 1200, "output_tokens": 300},
            },
        )

    client = anthropic.AsyncAnthropic(
        api_key="test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return ClaudeJudge(client, "claude-sonnet-5")


async def test_claude_judge_parses_structured_output() -> None:
    payload: dict[str, object] = {
        "claims": [
            {
                "claim": GOOD,
                "citations": [MEALS],
                "label": "SUPPORTED",
                "reason": "says so",
                "replacement": None,
            }
        ],
        "unconfirmed": [],
    }
    judgement = await _claude(payload).judge("q", _answer(GOOD), "<passage/>", Findings([], []))
    assert judgement.output.claims[0].label == "SUPPORTED"
    assert (judgement.input_tokens, judgement.output_tokens) == (1200, 300)


async def test_claude_judge_refusal_raises() -> None:
    with pytest.raises(VerifierError):
        await _claude(None, "refusal").judge("q", "d", "e", Findings([], []))


# --- evidence capture in the tool runtime --------------------------------------------------


async def test_runtime_records_parameters_and_calculations() -> None:
    retriever = HybridRetriever(
        PgVectorStore("postgresql://unused"), FakeEmbedder(), FakeReranker(), FakeRewriter()
    )
    runtime = ToolRuntime(PARAMS, retriever)
    runtime.begin(uuid4())
    tools = {t.name: t for t in runtime.tools()}
    await tools["get_tax_parameter"].handler(
        {"name": "business_meals_deduction_pct", "tax_year": 2025}
    )
    await tools["calc_vehicle"].handler(
        {"tax_year": 2025, "method": "standard_mileage", "total_miles": 1000, "business_miles": 500}
    )
    evidence = runtime.state.evidence
    assert "business_meals_deduction_pct" in evidence.parameters
    assert "standard_mileage_rate_business" in evidence.parameters  # recorded via the calculator
    assert evidence.calculations[0].startswith("calc_vehicle:")
    assert evidence.contains_number(Decimal(350))  # 500 miles x 0.70


def test_evidence_keeps_every_text_for_a_shared_citation() -> None:
    ev = Evidence()
    ev.add_passage("IRC § 274", "(a) Entertainment ... first grouped parent")
    ev.add_passage("IRC § 274", "(d) Substantiation required ... second grouped parent")
    ev.add_passage(
        "IRC § 274", "(d) Substantiation required ... second grouped parent"
    )  # duplicate
    ev.add_passage("IRC § 274(n)", "(n) Only 50 percent of meal expenses ...")
    ev.add_passage("Pub 463, ch. 2", "unrelated publication text")
    assert len(ev.passages["IRC § 274"]) == 2
    blocks = cited_evidence("Records are required [IRC § 274(d)].", ev)
    assert "first grouped parent" in blocks  # both § 274 parents reach the judge
    assert "second grouped parent" in blocks
    assert "(n) Only 50 percent" not in blocks  # a sibling provision is not related to (d)
    assert "unrelated publication" not in blocks
    broad = cited_evidence("Meals rules [IRC § 274].", ev)
    assert "(n) Only 50 percent" in broad  # a broad cite brings the passages under it


def test_abbreviated_heading_citation_resolves() -> None:
    full = "Pub 463, ch. 5, How To Prove Expenses, What Are Adequate Records?"
    assert resolve_tag("Pub 463, ch. 5, What Are Adequate Records?", [full]) == full
    assert resolve_tag("Pub 463, ch. 6, What Are Adequate Records?", [full]) is None
    assert resolve_tag("Pub 334, ch. 5, What Are Adequate Records?", [full]) is None


def test_tidy_cleans_up_after_edits() -> None:
    text = (
        "**Conditions**\n\n- Suits are personal [IRC § 262].. More.\n- \n-   Keep receipts.\n"
        "1.\n- Paper [Pub 334, ch. 8]. [Pub 334, ch. 8]. Wait...\n"
    )
    assert tidy(text) == (
        "**Conditions**\n\n- Suits are personal [IRC § 262]. More.\n- Keep receipts.\n"
        "- Paper [Pub 334, ch. 8]. Wait...\n"
    )


def test_dropping_a_whole_bullet_leaves_no_empty_item() -> None:
    draft = f"**Explanation**\n\n- {GOOD}\n- {BAD}\n\n{DISCLAIMER}"
    claims = [ClaimCheck(claim=BAD, citations=[MEALS], label="UNSUPPORTED", reason="no")]
    edited = apply_edits(draft, claims)
    assert "- \n" not in edited
    assert edited.count("\n- ") == 1
