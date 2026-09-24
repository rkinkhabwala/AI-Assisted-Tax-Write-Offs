"""Retrieval eval cases (`evals/retrieval_queries.jsonl`).

Each case lists the provisions that answer it as citation paths, not chunk ids. Chunk
ids depend on how documents are chunked, so id labels would be invalidated by the very
chunk-size and merge experiments these evals exist to run. At evaluation time each
citation resolves to the chunks of the index under test that hold it (see `metrics`).

Grades: 2 = directly answers the question; 1 = relevant supporting context.
Out-of-scope cases have no targets. They calibrate the "weak results" threshold.
"""

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from writeoff.models import DocType, EntityType, NonEmptyStr, TaxYear


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Target(_Strict):
    citation: NonEmptyStr
    grade: int = Field(default=2, ge=1, le=2)
    # The provision is stored inside an enclosing chunk rather than as its own (e.g. a
    # paragraph folded into its section's opening chunk). `check` accepts that only when
    # the label says so, so a mistyped heading path can't silently widen a label.
    allow_enclosing: bool = False

    @property
    def doc_type(self) -> DocType:
        return doc_type_of(self.citation)


class RetrievalCase(_Strict):
    id: NonEmptyStr
    query: NonEmptyStr
    tax_year: TaxYear = 2025
    category: NonEmptyStr
    relevant: tuple[Target, ...] = ()
    entity_type: EntityType | None = None
    out_of_scope: bool = False

    @model_validator(mode="after")
    def _targets_iff_in_scope(self) -> Self:
        if self.out_of_scope == bool(self.relevant):
            raise ValueError("out-of-scope cases have no targets; in-scope cases need some")
        if len({t.citation for t in self.relevant}) != len(self.relevant):
            raise ValueError("duplicate target citations")
        return self


class DatasetError(ValueError):
    """The eval dataset file is missing or invalid."""


def doc_type_of(citation: str) -> DocType:
    if citation.startswith("IRC §"):
        return DocType.IRC
    if citation.startswith("Treas. Reg. §"):
        return DocType.TREASURY_REGULATION
    if citation.startswith("Pub "):
        return DocType.IRS_PUBLICATION
    if citation.startswith("Instructions for "):
        return DocType.FORM_INSTRUCTIONS
    raise ValueError(f"cannot tell the document type of citation {citation!r}")


def load_cases(path: Path) -> list[RetrievalCase]:
    cases: list[RetrievalCase] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise DatasetError(f"eval dataset not found: {path}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = RetrievalCase.model_validate(json.loads(line))
            for target in case.relevant:
                doc_type_of(target.citation)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise DatasetError(f"{path}:{number}: {exc}") from exc
        cases.append(case)
    ids = [c.id for c in cases]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise DatasetError(f"{path}: duplicate case ids {duplicates}")
    if not cases:
        raise DatasetError(f"{path}: no cases")
    return cases
