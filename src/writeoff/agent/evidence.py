"""What the agent actually saw during one request: the ground truth for verification.

Recorded by the tool runtime as tools return. Passages and sections are keyed by
citation; parameters and calculator results keep their sources. The verifier checks the
draft against this, and nothing else.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

# Dollar amounts, percentages and cents: the figures a tax answer must be able to trace.
_NUMBER = re.compile(
    r"\$\s?(?P<usd>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"|(?P<pct>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?(?:%|percent\b)"
    r"|(?P<cents>\d+(?:\.\d+)?)\s?cents\b"
    r"|(?P<bare>(?<![\w.§(])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.§(])\d+\.\d+)",
    re.IGNORECASE,
)


def _dec(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", "")).normalize()
    except InvalidOperation:
        return None


def money_figures(text: str) -> list[tuple[str, Decimal]]:
    """Dollar amounts, percentages and cents in `text`, as (as-written, value) pairs.
    Cents are converted to dollars ("70 cents" -> 0.7)."""
    found: list[tuple[str, Decimal]] = []
    for m in _NUMBER.finditer(text):
        if m.group("bare"):
            continue  # plain numbers only count as evidence, not as claims (see all_numbers)
        raw = m.group("usd") or m.group("pct") or m.group("cents")
        value = _dec(raw)
        if value is None:
            continue
        if m.group("cents"):
            value = (value / 100).normalize()
        found.append((m.group(0).strip(), value))
    return found


def all_numbers(text: str) -> set[Decimal]:
    """Every number in evidence text, in the forms a figure might be quoted in."""
    values: set[Decimal] = set()
    for m in _NUMBER.finditer(text):
        raw = m.group("usd") or m.group("pct") or m.group("cents") or m.group("bare")
        value = _dec(raw)
        if value is not None:
            values.add(value)
            if m.group("cents"):
                values.add((value / 100).normalize())
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        value = _dec(m.group(0))
        if value is not None:
            values.add(value)
    return values


@dataclass(slots=True)
class Evidence:
    # citation -> distinct texts: grouped parent sections can share a citation path.
    passages: dict[str, list[str]] = field(default_factory=dict)
    parameters: dict[str, str] = field(default_factory=dict)  # name -> "value unit (source)"
    calculations: list[str] = field(default_factory=list)  # tool name + JSON result
    _numbers: set[Decimal] = field(default_factory=set)

    def add_passage(self, citation: str, text: str) -> None:
        texts = self.passages.setdefault(citation, [])
        if text and text not in texts:
            texts.append(text)
            self._numbers |= all_numbers(text)

    def add_parameter(self, name: str, description: str) -> None:
        self.parameters[name] = description
        self._numbers |= all_numbers(description)

    def add_calculation(self, tool: str, payload: str) -> None:
        self.calculations.append(f"{tool}: {payload}")
        self._numbers |= all_numbers(payload)

    def contains_number(self, value: Decimal) -> bool:
        return value.normalize() in self._numbers

    @property
    def citations(self) -> list[str]:
        return list(self.passages)

    def is_empty(self) -> bool:
        return not (self.passages or self.parameters or self.calculations)
