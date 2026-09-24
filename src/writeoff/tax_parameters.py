"""Loader for per-year tax parameters in `data/tax_parameters/{year}.yaml`.

These files are the only place dollar limits, percentages and rates may come from; the
agent's `get_tax_parameter` tool reads them and returns the value together with its
irs.gov source. A parameter without a `source_url` is rejected at load time, so no
unsourced number can reach a user. A `null` value means "not yet verified" and must be
reported as unavailable, never guessed.

Some parameters change mid-year (e.g. bonus depreciation depends on the acquisition date).
Those use `periods` instead of a single `value`.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, model_validator

from writeoff.models import NonEmptyStr, TaxYear


class Unit(StrEnum):
    USD = "usd"
    PERCENT = "percent"
    USD_PER_MILE = "usd_per_mile"
    USD_PER_SQ_FT = "usd_per_sq_ft"
    SQ_FT = "sq_ft"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ParameterPeriod(_Strict):
    start: date
    end: date
    value: Decimal | None

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError("period end is before start")
        return self


class TaxParameter(_Strict):
    description: NonEmptyStr
    unit: Unit
    source_url: HttpUrl
    value: Decimal | None = None
    periods: tuple[ParameterPeriod, ...] | None = None
    # Where the value was confirmed, e.g. "Pub 463 (2025), ch. 4, Standard Mileage Rate".
    # Required for any value that is filled in.
    evidence: str | None = None

    @model_validator(mode="after")
    def _value_or_periods(self) -> Self:
        if self.value is not None and self.periods is not None:
            raise ValueError("set either value or periods, not both")
        has_value = self.value is not None or any(p.value is not None for p in self.periods or ())
        if has_value and not self.evidence:
            raise ValueError("a filled-in value needs `evidence` naming where it was confirmed")
        if self.periods is not None:
            ordered = sorted(self.periods, key=lambda p: p.start)
            for prev, nxt in pairwise(ordered):
                if nxt.start <= prev.end:
                    raise ValueError("periods overlap")
        return self

    @property
    def is_verified(self) -> bool:
        if self.periods is not None:
            return bool(self.periods) and all(p.value is not None for p in self.periods)
        return self.value is not None


class TaxParameterSet(_Strict):
    tax_year: TaxYear
    parameters: dict[NonEmptyStr, TaxParameter] = Field(min_length=1)

    def get(self, name: str) -> TaxParameter:
        try:
            return self.parameters[name]
        except KeyError:
            known = ", ".join(sorted(self.parameters))
            raise KeyError(
                f"unknown tax parameter {name!r} for {self.tax_year}; known: {known}"
            ) from None


class TaxParameterError(ValueError):
    """A tax-parameter file is missing, malformed, or fails validation."""


def load_tax_parameters(tax_year: int, directory: Path) -> TaxParameterSet:
    path = directory / f"{tax_year}.yaml"
    if not path.is_file():
        raise TaxParameterError(f"no tax parameter file for {tax_year} at {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TaxParameterError(f"{path}: invalid YAML: {exc}") from exc
    try:
        params = TaxParameterSet.model_validate(raw)
    except ValidationError as exc:
        raise TaxParameterError(f"{path}: {exc}") from exc
    if params.tax_year != tax_year:
        raise TaxParameterError(f"{path} declares tax_year {params.tax_year}, expected {tax_year}")
    return params


class ParameterUnavailableError(LookupError):
    """A calculation needs a parameter that is unknown or not yet verified on irs.gov.

    Tools turn this into an explicit "unavailable" answer. A number is never guessed.
    """

    def __init__(self, name: str, tax_year: int, reason: str) -> None:
        super().__init__(f"{name} ({tax_year}): {reason}")
        self.name = name
        self.tax_year = tax_year
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ResolvedParameter:
    """A verified value, with the source a calculation must cite for it."""

    name: str
    tax_year: int
    value: Decimal
    unit: Unit
    source_url: str
    description: str


class TaxParameters:
    """Read-only access to the per-year parameter files, cached after first load."""

    def __init__(self, directory: Path, supported_years: tuple[int, ...]) -> None:
        self._directory = directory
        self._supported = supported_years
        self._sets: dict[int, TaxParameterSet] = {}

    def for_year(self, tax_year: int) -> TaxParameterSet:
        if tax_year not in self._supported:
            raise ParameterUnavailableError(
                "*", tax_year, f"tax year not supported {self._supported}"
            )
        if tax_year not in self._sets:
            self._sets[tax_year] = load_tax_parameters(tax_year, self._directory)
        return self._sets[tax_year]

    def describe(self, name: str, tax_year: int) -> TaxParameter:
        try:
            return self.for_year(tax_year).get(name)
        except KeyError as exc:
            raise ParameterUnavailableError(name, tax_year, str(exc.args[0])) from None

    def value(self, name: str, tax_year: int, *, on: date | None = None) -> ResolvedParameter:
        """The verified value of `name`. Parameters that vary within the year (`periods`)
        need `on`, the date that decides which period applies."""
        param = self.describe(name, tax_year)
        if param.periods is not None:
            if on is None:
                raise ParameterUnavailableError(
                    name, tax_year, "value depends on a date; none given"
                )
            period = next((p for p in param.periods if p.start <= on <= p.end), None)
            if period is None:
                raise ParameterUnavailableError(
                    name, tax_year, f"no period covers {on.isoformat()}"
                )
            raw = period.value
        else:
            raw = param.value
        if raw is None:
            raise ParameterUnavailableError(name, tax_year, "not yet verified on irs.gov")
        return ResolvedParameter(
            name, tax_year, raw, param.unit, str(param.source_url), param.description
        )
