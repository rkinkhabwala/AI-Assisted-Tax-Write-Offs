"""Shared result types. Every calculator shows its work: each step, and each tax
parameter used together with its irs.gov source, so any number in an answer can be traced
(spec section 6)."""

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from writeoff.tax_parameters import ParameterUnavailableError, ResolvedParameter, TaxParameters

CENT = Decimal("0.01")
HUNDRED = Decimal(100)


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def pct(value: Decimal) -> Decimal:
    """A percentage (0-100) as a fraction."""
    return value / HUNDRED


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ParameterUsed(FrozenModel):
    name: str
    value: Decimal
    unit: str
    source_url: str


class Step(FrozenModel):
    description: str
    amount: Decimal | None = None


class CalculationResult(FrozenModel):
    """Base for calculator outputs. `status="unavailable"` means a required parameter is
    not verified yet. The answer must then say so, not estimate."""

    status: Literal["ok", "unavailable"] = "ok"
    missing_parameters: list[str] = Field(default_factory=list)
    parameters_used: list[ParameterUsed] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    authorities: list[str] = Field(default_factory=list)


class Workbook:
    """Collects steps, parameters and notes while a calculation runs."""

    def __init__(self, params: TaxParameters, tax_year: int) -> None:
        self.params = params
        self.tax_year = tax_year
        self.steps: list[Step] = []
        self.used: dict[str, ParameterUsed] = {}
        self.notes: list[str] = []

    def param(self, name: str, *, on: date | None = None) -> Decimal:
        """A verified parameter value, recorded with its source for the result."""
        resolved: ResolvedParameter = self.params.value(name, self.tax_year, on=on)
        self.used[name] = ParameterUsed(
            name=name,
            value=resolved.value,
            unit=resolved.unit.value,
            source_url=resolved.source_url,
        )
        return resolved.value

    def step(self, description: str, amount: Decimal | None = None) -> Decimal | None:
        self.steps.append(
            Step(description=description, amount=money(amount) if amount is not None else None)
        )
        return amount

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def finish[R: CalculationResult](
        self, result_cls: type[R], authorities: list[str], **values: object
    ) -> R:
        return result_cls.model_validate(
            {
                "parameters_used": list(self.used.values()),
                "steps": self.steps,
                "notes": self.notes,
                "authorities": authorities,
                **values,
            }
        )


def unavailable[R: CalculationResult](
    result_cls: type[R], exc: ParameterUnavailableError, authorities: list[str]
) -> R:
    """An explicit 'unavailable' result naming the parameter that isn't verified yet."""
    return result_cls.model_validate(
        {
            "status": "unavailable",
            "missing_parameters": [exc.name],
            "notes": [
                f"Parameter {exc.name!r} for {exc.tax_year} is {exc.reason}. "
                "State that the figure is unavailable rather than estimating it."
            ],
            "authorities": authorities,
        }
    )
