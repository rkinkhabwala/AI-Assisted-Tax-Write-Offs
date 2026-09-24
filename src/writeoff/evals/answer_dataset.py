"""End-to-end answer eval cases (`evals/golden_set.jsonl`, `evals/safety_cases.jsonl`).

Golden cases (spec section 7b) label what a correct answer concludes and what it should
cite. Citations are labeled as citation paths, like the retrieval evals, and may be
coarse ("Pub 463, ch. 2" covers everything under it). A required citation is a group of
alternatives: citing any one of them satisfies the group.

Safety cases (spec section 7c) label the behavior expected: refuse an evasion request
and offer a legitimate route, answer a legitimate question without refusing, or treat
instructions embedded in user data as data.
"""

import json
import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from writeoff.models import EntityType, NonEmptyStr, TaxYear

Treatment = Literal[
    "deductible",
    "limited",
    "not_deductible",
    "capitalize",
    "informational",
    "clarify",
    "no_authority",
    "refuse",
]
SafetyExpectation = Literal["refuse_with_alternative", "answer", "resist_injection"]
SafetyCategory = Literal["evasion", "over_refusal", "injection"]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _compiles(patterns: tuple[str, ...]) -> tuple[str, ...]:
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid pattern {pattern!r}: {exc}") from exc
    return patterns


class GoldenCase(_Strict):
    id: NonEmptyStr
    category: NonEmptyStr
    question: NonEmptyStr
    entity_type: EntityType | None = None
    tax_year: TaxYear | None = None
    expected_treatment: tuple[Treatment, ...] = Field(min_length=1)
    required_citations: tuple[tuple[NonEmptyStr, ...], ...] = ()
    acceptable_citations: tuple[NonEmptyStr, ...] = ()
    # Case-insensitive regular expressions checked against the final answer.
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    key_points: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @field_validator("must_contain", "must_not_contain")
    @classmethod
    def _valid_patterns(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        return _compiles(patterns)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if any(not group for group in self.required_citations):
            raise ValueError("empty required-citation group")
        if self.expected_treatment == ("clarify",) and self.required_citations:
            raise ValueError("a clarify-only case can't require citations")
        return self

    @property
    def label_citations(self) -> tuple[str, ...]:
        """Every citation that counts as on-topic for precision."""
        required = tuple(c for group in self.required_citations for c in group)
        return tuple(dict.fromkeys(required + self.acceptable_citations))

    @property
    def expects_clarification(self) -> bool:
        return "clarify" in self.expected_treatment


class SafetyCase(_Strict):
    id: NonEmptyStr
    category: SafetyCategory
    question: NonEmptyStr
    entity_type: EntityType | None = None
    tax_year: TaxYear | None = None
    expected: SafetyExpectation
    must_not_contain: tuple[str, ...] = ()
    notes: str = ""

    @field_validator("must_not_contain")
    @classmethod
    def _valid_patterns(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        return _compiles(patterns)

    @model_validator(mode="after")
    def _category_matches(self) -> Self:
        expected = {
            "evasion": "refuse_with_alternative",
            "over_refusal": "answer",
            "injection": "resist_injection",
        }[self.category]
        if self.expected != expected:
            raise ValueError(f"{self.category} cases expect {expected!r}")
        return self


class AnswerDatasetError(ValueError):
    """An answer eval dataset file is missing or invalid."""


def _load[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise AnswerDatasetError(f"eval dataset not found: {path}") from exc
    cases: list[T] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            cases.append(model.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise AnswerDatasetError(f"{path}:{number}: {exc}") from exc
    ids = [getattr(c, "id", "") for c in cases]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise AnswerDatasetError(f"{path}: duplicate case ids {duplicates}")
    if not cases:
        raise AnswerDatasetError(f"{path}: no cases")
    return cases


def load_golden(path: Path) -> list[GoldenCase]:
    return _load(path, GoldenCase)


def load_safety(path: Path) -> list[SafetyCase]:
    return _load(path, SafetyCase)
