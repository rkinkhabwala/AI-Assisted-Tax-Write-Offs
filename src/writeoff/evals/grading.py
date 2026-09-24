"""Deterministic answer grading: citations, disclaimer and required or forbidden text.

A cited tag matches a label when one sits inside the other ("IRC § 274(n)(1)" matches
the label "IRC § 274(n)" and vice versa), using the same resolution as the verifier.
Labels may also name a whole regulation part: "Treas. Reg. § 1.274" covers
"Treas. Reg. § 1.274-12(a)", and "Treas. Reg. § 1.263(a)" covers "§ 1.263(a)-3(d)".
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from writeoff.agent.prompt import DISCLAIMER
from writeoff.agent.verifier import citation_tags, related_citations, resolve_tag
from writeoff.evals.answer_dataset import GoldenCase
from writeoff.tools.law import normalize_citation

# A regulation label with no section number after the part, e.g. "Treas. Reg. § 1.274".
_REG_PART = re.compile(r"^Treas\. Reg\. § \d+\.[0-9A-Za-z()]+$")

# The disclaimer's core, so a reflowed or re-punctuated disclaimer still counts.
_DISCLAIMER_CORE = re.compile(r"not\s+tax\s+or\s+legal\s+advice", re.IGNORECASE)


def cited(text: str) -> list[str]:
    """Distinct citation tags in the text, in order of first appearance."""
    return list(dict.fromkeys(citation_tags(text)))


def label_prefixes(label: str) -> list[str]:
    """Prefixes that a citation inside `label` starts with."""
    prefixes = [label + "(", label + ", "]
    if _REG_PART.match(label):
        prefixes.append(label + "-")
    return prefixes


def matches(tag: str, label: str) -> bool:
    if related_citations(tag, [label]):
        return True
    return bool(_REG_PART.match(label)) and normalize_citation(tag).startswith(label + "-")


def citation_recall(tags: Iterable[str], groups: Iterable[Iterable[str]]) -> float | None:
    """Share of required groups satisfied by some cited tag. None when nothing is required."""
    groups = [list(g) for g in groups]
    if not groups:
        return None
    tags = list(tags)
    hit = sum(any(matches(t, label) for t in tags for label in g) for g in groups)
    return hit / len(groups)


def citation_precision(tags: Iterable[str], labels: Iterable[str]) -> float | None:
    """Share of cited tags that are on-topic for the case. None when nothing was cited or
    the case labels no authorities (e.g. an expected clarifying question). A lower bound:
    labels list the expected authorities, not every relevant one."""
    tags, labels = list(tags), list(labels)
    if not tags or not labels:
        return None
    return sum(any(matches(t, label) for label in labels) for t in tags) / len(tags)


def hallucinated(tags: Iterable[str], retrieved: list[str]) -> list[str]:
    """Cited tags that resolve to nothing the agent retrieved in this request."""
    return [t for t in tags if resolve_tag(t, retrieved) is None]


def has_disclaimer(text: str) -> bool:
    return DISCLAIMER in text or bool(_DISCLAIMER_CORE.search(text))


def pattern_hits(text: str, patterns: Iterable[str]) -> list[str]:
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


@dataclass(frozen=True, slots=True)
class CitationGrades:
    cited_final: list[str]
    cited_draft: list[str]
    recall: float | None
    precision: float | None
    hallucinated_final: list[str]
    hallucinated_draft: list[str]


def grade_citations(
    case: GoldenCase, final: str, draft: str | None, retrieved: list[str]
) -> CitationGrades:
    """Recall and precision are graded on the final answer the user sees. Hallucinated
    citations are counted on both the draft (what the agent wrote) and the final answer
    (what survived the verifier)."""
    final_tags = cited(final)
    draft_tags = cited(draft) if draft is not None else final_tags
    return CitationGrades(
        cited_final=final_tags,
        cited_draft=draft_tags,
        recall=citation_recall(final_tags, case.required_citations),
        precision=citation_precision(final_tags, case.label_citations),
        hallucinated_final=hallucinated(final_tags, retrieved),
        hallucinated_draft=hallucinated(draft_tags, retrieved),
    )
