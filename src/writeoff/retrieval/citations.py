"""Detect explicit citations in a query ("§ 179(b)", "Reg. 1.274-12", "Pub 946", "Form 8829").

Citations are rendered in the same form the chunker writes to `citation_path`, so they can
be looked up directly:

- IRC and Treasury Regulation citations drive the citation fast path: the named
  provision is fetched by citation before semantic search.
- Publication and form mentions only add keyword terms. Fetching an entire publication
  would swamp the results, and hybrid search already ranks its sections well once the
  document name is a search term.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

_SUB = r"(?:\((?:[0-9]{1,3}|[a-z]{1,4}|[A-Z]{1,4})\))*"
# Regulations: 1.162-5, 1.263(a)-3, 1.274-5T, optionally with paragraphs (h)(1)(i).
_REG = re.compile(
    r"(?:\bTreas(?:ury)?\.?\s*Reg(?:ulation)?s?\.?\s*|\bReg(?:ulation)?s?\.?\s*)?(?:§\s*)?"
    rf"\b(?P<num>1\.\d{{1,4}}[A-Z]?(?:\([a-z]\))?-\d{{1,3}}T?)(?P<sub>{_SUB})"
)
# IRC: needs an explicit marker (§, "section", "sec.", "IRC", "26 U.S.C.") so ordinary
# numbers in a question never look like code sections.
_IRC = re.compile(
    r"(?:\bI\.?R\.?C\.?\s*(?:§§?|sec(?:tion)?\.?)?|\b26\s*U\.?S\.?C\.?\s*§?|§§?|\bsec(?:tion)?s?\.?)"
    rf"\s*(?P<num>\d{{1,4}}[A-Z]?)(?![\d.])(?P<sub>{_SUB})",
    re.IGNORECASE,
)
_PUB = re.compile(r"\bPub(?:lication)?\.?\s*(?P<num>\d{1,4}(?:-[A-Z])?)\b", re.IGNORECASE)
_FORM = re.compile(r"\bForm\s+(?P<num>\d{3,4}(?:-[A-Z]{1,2})?)\b", re.IGNORECASE)
_SCHEDULE_C = re.compile(r"\bSchedule\s+C\b", re.IGNORECASE)


class CitationKind(StrEnum):
    IRC = "irc"
    REGULATION = "regulation"
    PUBLICATION = "publication"
    FORM = "form"

    @property
    def is_statutory(self) -> bool:
        return self in {CitationKind.IRC, CitationKind.REGULATION}


@dataclass(frozen=True, slots=True)
class CitationRef:
    kind: CitationKind
    citation: str  # citation_path prefix, e.g. "IRC § 179(b)" or "Pub 946"
    matched: str  # the text as it appeared in the query


def extract_citations(query: str) -> list[CitationRef]:
    """All citations in `query`, in order of appearance, without duplicates."""
    found: list[tuple[int, CitationRef]] = []
    taken: list[range] = []

    def add(start: int, end: int, ref: CitationRef) -> None:
        if not any(start < r.stop and end > r.start for r in taken):  # no overlap
            taken.append(range(start, end))
            found.append((start, ref))

    for m in _REG.finditer(query):
        add(
            m.start(),
            m.end(),
            CitationRef(CitationKind.REGULATION, f"Treas. Reg. § {m['num']}{m['sub']}", m.group(0)),
        )
    for m in _IRC.finditer(query):
        add(
            m.start(),
            m.end(),
            CitationRef(CitationKind.IRC, f"IRC § {m['num'].upper()}{m['sub']}", m.group(0)),
        )
    for m in _PUB.finditer(query):
        add(
            m.start(),
            m.end(),
            CitationRef(CitationKind.PUBLICATION, f"Pub {m['num'].upper()}", m.group(0)),
        )
    for m in _FORM.finditer(query):
        add(
            m.start(),
            m.end(),
            CitationRef(CitationKind.FORM, f"Instructions for Form {m['num'].upper()}", m.group(0)),
        )
    for m in _SCHEDULE_C.finditer(query):
        add(
            m.start(),
            m.end(),
            CitationRef(CitationKind.FORM, "Instructions for Schedule C", m.group(0)),
        )
    unique: dict[str, CitationRef] = {}
    for _, ref in sorted(found, key=lambda item: item[0]):
        unique.setdefault(ref.citation, ref)
    return list(unique.values())
