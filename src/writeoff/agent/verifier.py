"""Grounding verifier (spec sections 5-6).

The draft is checked against the evidence the agent actually retrieved:

1. Deterministic checks. Every bracketed citation must resolve to a retrieved passage
   (a sub-provision of a retrieved section, or a broader cite of a retrieved provision,
   also resolves). Every dollar figure and percentage must appear in the evidence or in the
   user's own question.
2. A separate Claude call (the judge) sees only the user's question, the draft and the
   *cited* evidence (not the conversation). It labels each claim SUPPORTED /
   PARTIALLY_SUPPORTED / UNSUPPORTED against its cited passage and writes a revised
   answer that keeps supported claims, narrows partly supported ones and drops the rest.
3. If the draft fails, the revision is verified once more. If that also fails, the
   revision is stripped to what is supported and states what couldn't be confirmed.

The question is shared with the judge only so facts the user stated ("a $2,000 laptop")
aren't mistaken for unsupported claims.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from writeoff.agent.evidence import Evidence, all_numbers, money_figures
from writeoff.agent.prompt import DISCLAIMER
from writeoff.tools.law import normalize_citation

Label = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]
VerificationStatus = Literal["verified", "revised", "partially_verified", "skipped", "error"]

_TAG = re.compile(r"\[([^\[\]\n]{2,250})\]")
_CITATION_START = ("IRC", "§", "Treas", "Reg", "Pub", "Instructions", "26 U.S.C", "Sec")
MAX_EVIDENCE_CHARS = 150_000


class ClaimCheck(BaseModel):
    claim: str = Field(description="The claim, quoted verbatim from the answer")
    citations: list[str] = Field(description="Citation tags attached to the claim")
    label: Label
    reason: str = Field(description="At most 25 words: what the evidence does or doesn't say")
    replacement: str | None = Field(
        default=None,
        description="For PARTIALLY_SUPPORTED or UNSUPPORTED claims: text to put in place of the "
        "quoted claim, narrowed to what the evidence supports with its citation, or an empty "
        "string to drop it. Null for SUPPORTED claims.",
    )


class JudgeOutput(BaseModel):
    claims: list[ClaimCheck]
    unconfirmed: list[str] = Field(description="Short descriptions of what couldn't be confirmed")


@dataclass(frozen=True, slots=True)
class Findings:
    unknown_citations: list[str]
    untraced_numbers: list[str]


@dataclass(slots=True)
class VerificationReport:
    status: VerificationStatus
    rounds: int = 0
    claims: list[ClaimCheck] = field(default_factory=list)
    draft_claims: list[ClaimCheck] = field(default_factory=list)  # round 1, before edits
    unknown_citations: list[str] = field(default_factory=list)
    untraced_numbers: list[str] = field(default_factory=list)
    unconfirmed: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    def count(self, label: Label) -> int:
        return sum(c.label == label for c in self.claims)

    @property
    def faithfulness(self) -> float | None:
        """Share of the final answer's checked claims that are fully supported."""
        return self.count("SUPPORTED") / len(self.claims) if self.claims else None

    @property
    def draft_faithfulness(self) -> float | None:
        """Share of the agent's draft claims that were fully supported (before edits)."""
        if not self.draft_claims:
            return None
        return sum(c.label == "SUPPORTED" for c in self.draft_claims) / len(self.draft_claims)


# --- deterministic checks --------------------------------------------------------------


def citation_tags(text: str) -> list[str]:
    tags: list[str] = []
    for match in _TAG.finditer(text):
        for raw in match.group(1).split(";"):
            part = raw.strip()
            if part.startswith(_CITATION_START):
                tags.append(part)
    return tags


def _within(inner: str, outer: str) -> bool:
    return (
        inner == outer
        or inner.startswith((outer + "(", outer + ", "))
        or _abbreviates(inner, outer)
    )


def _abbreviates(tag: str, citation: str) -> bool:
    """A heading-style tag that skips intermediate headings of a retrieved path, e.g.
    "Pub 463, ch. 5, What Are Adequate Records?" for
    "Pub 463, ch. 5, How To Prove Expenses, What Are Adequate Records?". Same document,
    same final heading, the rest in order."""
    tag_parts, cite_parts = tag.split(", "), citation.split(", ")
    if len(tag_parts) < 2 or len(tag_parts) >= len(cite_parts) or tag_parts[0] != cite_parts[0]:
        return False
    if tag_parts[-1] != cite_parts[-1]:
        return False
    remaining = iter(cite_parts[1:-1])
    return all(part in remaining for part in tag_parts[1:-1])


def resolve_tag(tag: str, citations: list[str]) -> str | None:
    """The retrieved citation a tag refers to: the same provision, one it sits inside,
    or (for a broader cite) one under it. None if nothing retrieved matches."""
    candidates = [normalize_citation(tag)]
    head = re.split(r",\s*\(", candidates[0])[0]  # "IRC § 179(a), (b)(1)" -> "IRC § 179(a)"
    if head != candidates[0]:
        candidates.append(head)
    for candidate in candidates:
        for citation in citations:
            if _within(candidate, citation) or _within(citation, candidate):
                return citation
    return None


def deterministic_findings(draft: str, evidence: Evidence, question: str) -> Findings:
    unknown = [
        t for t in dict.fromkeys(citation_tags(draft)) if resolve_tag(t, evidence.citations) is None
    ]
    stated = all_numbers(question)
    untraced = [
        raw
        for raw, value in money_figures(_strip_tags(draft))
        if not evidence.contains_number(value) and value not in stated
    ]
    return Findings(unknown, list(dict.fromkeys(untraced)))


def _strip_tags(text: str) -> str:
    return _TAG.sub(" ", text)


def needs_verification(draft: str) -> bool:
    """Full answers are verified; a lone clarifying question has no claims to check."""
    return DISCLAIMER in draft or bool(citation_tags(draft)) or bool(money_figures(draft))


def related_citations(tag: str, citations: list[str]) -> list[str]:
    """Every retrieved citation a tag could draw on: the provision itself, the sections
    containing it, and the passages under it."""
    candidate = normalize_citation(tag)
    forms = {candidate, re.split(r",\s*\(", candidate)[0]}
    return [c for c in citations if any(_within(f, c) or _within(c, f) for f in forms)]


def cited_citations(text: str, evidence: Evidence) -> list[str]:
    """Retrieved citations the text's tags refer to, in order of first mention."""
    wanted: list[str] = []
    for tag in citation_tags(text):
        for citation in related_citations(tag, evidence.citations):
            if citation not in wanted:
                wanted.append(citation)
    return wanted


def cited_evidence(draft: str, evidence: Evidence) -> str:
    """The evidence blocks the judge sees: cited passages, then parameters and calculations."""
    wanted = cited_citations(draft, evidence)
    blocks = [
        f'<passage citation="{c}">\n{text}\n</passage>'
        for c in wanted
        for text in evidence.passages[c]
    ]
    blocks += [f'<parameter name="{n}">{d}</parameter>' for n, d in evidence.parameters.items()]
    blocks += [f"<calculation>{c}</calculation>" for c in evidence.calculations]
    text = "\n".join(blocks)
    return text[:MAX_EVIDENCE_CHARS]


# --- the judge ---------------------------------------------------------------------------


JUDGE_SYSTEM = """\
You check a tax answer against its evidence. You see the user's question, the draft \
answer, and the evidence the answer cites: passages from the law and IRS guidance, \
tax parameters, and calculator results. You don't see the conversation that produced the \
draft.

For each sentence or bullet that states a tax rule, limit, requirement, number, or a \
conclusion about deductibility, record a claim (quote it verbatim) and label it:
- SUPPORTED: the cited evidence says it.
- PARTIALLY_SUPPORTED: the evidence supports part of it, or it overstates or oversimplifies.
- UNSUPPORTED: no citation, the cited evidence doesn't say it, or a number doesn't \
appear in the evidence or the question.
Facts the user stated in the question are not claims. Headings, the disclaimer, and \
advice to keep records or consult a professional are not claims.

For every claim that is not SUPPORTED, give a `replacement`: the claim narrowed to \
exactly what the evidence supports (keep a valid citation), or an empty string to drop \
it. Don't write a sentence saying a point couldn't be confirmed; the answer adds one short \
note itself. Never add a fact, number or citation that isn't in the evidence. Keep \
reasons short.

List in `unconfirmed` a short description of each point you dropped or narrowed."""


@dataclass(frozen=True, slots=True)
class Judgement:
    output: JudgeOutput
    input_tokens: int
    output_tokens: int


class Judge(ABC):
    @abstractmethod
    async def judge(
        self, question: str, draft: str, evidence: str, findings: Findings
    ) -> Judgement:
        """Label the draft's claims against the evidence and propose a revision."""


class VerifierError(RuntimeError):
    """The judge could not produce a verdict."""


class ClaudeJudge(Judge):
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        model: str,
        *,
        max_tokens: int = 16000,
        thinking: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        # Labeling claims against quoted evidence doesn't need extended reasoning; with
        # thinking on, live verification took about two minutes per answer.
        self._thinking = thinking

    async def judge(
        self, question: str, draft: str, evidence: str, findings: Findings
    ) -> Judgement:
        notes = ""
        if findings.unknown_citations:
            notes += f"\nCitations not found in the evidence: {findings.unknown_citations}"
        if findings.untraced_numbers:
            notes += f"\nFigures not found in the evidence or question: {findings.untraced_numbers}"
        user = (
            f"<question>\n{question}\n</question>\n<draft>\n{draft}\n</draft>\n"
            f"<evidence>\n{evidence or '(no evidence was retrieved)'}\n</evidence>"
            + (f"\n<automatic_findings>{notes}\n</automatic_findings>" if notes else "")
        )
        try:
            response = await self._client.messages.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user}],
                output_format=JudgeOutput,
                thinking={"type": "adaptive"} if self._thinking else {"type": "disabled"},
            )
        except anthropic.APIError as exc:
            raise VerifierError(f"verifier request failed: {exc}") from exc
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise VerifierError(
                f"verifier returned no verdict (stop_reason={response.stop_reason})"
            )
        usage = response.usage
        return Judgement(response.parsed_output, usage.input_tokens, usage.output_tokens)


# --- the loop ----------------------------------------------------------------------------


def _passes(output: JudgeOutput, findings: Findings) -> bool:
    return (
        all(c.label == "SUPPORTED" for c in output.claims)
        and not findings.unknown_citations
        and not findings.untraced_numbers
    )


_EMPTY_ITEM = re.compile(r"^[ \t]*(?:[-*\u2022]|\d+\.)[ \t]*$\n?", re.M)
_DOUBLE_PERIOD = re.compile(r"(?<!\.)\.\.(?!\.)")
_REPEATED_TAG = re.compile(r"(\[[^\[\]\n]+\])\.?[ \t]+\1")
_BULLET_GAP = re.compile(r"^([ \t]*[-*\u2022])[ \t]{2,}", re.M)


def tidy(text: str) -> str:
    """Clean up what removing or replacing sentences leaves behind: empty list items,
    doubled periods, a citation repeated back to back, and stray spaces after bullets."""
    text = _EMPTY_ITEM.sub("", text)
    text = _DOUBLE_PERIOD.sub(".", text)
    text = _REPEATED_TAG.sub(r"\1", text)
    text = _BULLET_GAP.sub(r"\1 ", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def apply_edits(text: str, claims: list[ClaimCheck]) -> str:
    """Put each non-supported claim's replacement in place of the quoted claim; an
    unsupported claim with no replacement is dropped."""
    edited = text
    for claim in claims:
        if claim.label == "SUPPORTED":
            continue
        quoted = claim.claim.strip()
        replacement = claim.replacement
        if replacement is None and claim.label == "UNSUPPORTED":
            replacement = ""
        if quoted and replacement is not None and quoted in edited:
            edited = edited.replace(quoted, replacement.strip(), 1)
    return tidy(edited)


UNCONFIRMED_NOTE = "_Some points couldn't be confirmed from my sources and were left out._"


def _with_note(text: str, unconfirmed: list[str]) -> str:
    """One short note in the answer. The specifics (`report.unconfirmed`) are written for
    review, not for the user, and are returned by the API separately."""
    if not unconfirmed:
        return text
    body = text.replace(DISCLAIMER, "").rstrip()
    return f"{body}\n\n{UNCONFIRMED_NOTE}\n\n{DISCLAIMER}"


def strip_unsupported(
    text: str,
    claims: list[ClaimCheck],
    unconfirmed: list[str],
    findings: Findings | None = None,
) -> str:
    """Remove lines holding unsupported claims, untraced figures or unresolvable citations,
    and say what couldn't be confirmed."""
    markers = [c.claim.strip() for c in claims if c.label == "UNSUPPORTED" and c.claim.strip()]
    if findings is not None:
        markers += [f"[{tag}]" for tag in findings.unknown_citations]
        markers += findings.untraced_numbers
    kept = text
    for marker in markers:
        kept = "\n".join(line for line in kept.split("\n") if marker not in line)
    return _with_note(_ensure_disclaimer(tidy(kept)), unconfirmed)


class Verifier:
    def __init__(self, judge: Judge) -> None:
        self._judge = judge

    async def verify(
        self, draft: str, question: str, evidence: Evidence
    ) -> tuple[str, VerificationReport]:
        """Verify; apply the judge's edits; re-verify only the edited sentences; then keep
        only what is supported and say what couldn't be confirmed."""
        if not needs_verification(draft):
            return draft, VerificationReport(status="skipped")
        report = VerificationReport(status="verified")
        try:
            findings = deterministic_findings(draft, evidence, question)
            first = await self._judge.judge(
                question, draft, cited_evidence(draft, evidence), findings
            )
            self._tally(report, first, findings, rounds=1)
            report.claims = first.output.claims
            report.draft_claims = first.output.claims
            if _passes(first.output, findings):
                return draft, report
            report.unconfirmed = list(first.output.unconfirmed)
            text = _ensure_disclaimer(apply_edits(draft, first.output.claims))

            # Round 2: only the replacement sentences are new; the rest was checked.
            edited = [
                c.replacement.strip()
                for c in first.output.claims
                if c.label != "SUPPORTED" and c.replacement and c.replacement.strip()
            ]
            findings = deterministic_findings(text, evidence, question)
            second_claims: list[ClaimCheck] = []
            if edited:
                snippet = "\n".join(edited)
                second = await self._judge.judge(
                    question, snippet, cited_evidence(snippet, evidence), findings
                )
                self._tally(report, second, findings, rounds=2)
                second_claims = second.output.claims
                report.unconfirmed = list(
                    dict.fromkeys([*report.unconfirmed, *second.output.unconfirmed])
                )
            report.rounds = 2
            supported = [c for c in first.output.claims if c.label == "SUPPORTED"]
            report.claims = supported + second_claims
            report.unknown_citations = findings.unknown_citations
            report.untraced_numbers = findings.untraced_numbers
            clean = not (findings.unknown_citations or findings.untraced_numbers)
            if clean and all(c.label == "SUPPORTED" for c in second_claims):
                report.status = "revised"
                return _with_note(text, report.unconfirmed), report
            report.status = "partially_verified"
            text = apply_edits(text, second_claims)
            return strip_unsupported(text, second_claims, report.unconfirmed, findings), report
        except VerifierError as exc:
            report.status = "error"
            report.error = str(exc)
            note = "_Automatic source verification was unavailable for this answer._"
            body = draft.replace(DISCLAIMER, "").rstrip()
            return f"{body}\n\n{note}\n\n{DISCLAIMER}", report

    @staticmethod
    def _tally(
        report: VerificationReport, judgement: Judgement, findings: Findings, *, rounds: int
    ) -> None:
        report.rounds = rounds
        report.input_tokens += judgement.input_tokens
        report.output_tokens += judgement.output_tokens
        report.unknown_citations = findings.unknown_citations
        report.untraced_numbers = findings.untraced_numbers


def _ensure_disclaimer(text: str) -> str:
    return text if DISCLAIMER in text else f"{text.rstrip()}\n\n{DISCLAIMER}"
