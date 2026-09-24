"""Business use of the home (IRC § 280A(c); Pub 587; Form 8829).

Simplified method: the prescribed rate x business square footage, up to the maximum
footage. No depreciation, no carryover.

Regular method: business percentage = business sq ft / total sq ft. Direct expenses count
in full, indirect expenses at the business percentage. If `gross_income_limit` (gross
income from the business use, less business expenses not tied to the home) is given, the
Form 8829 ordering applies (§ 280A(c)(5)):
  1. mortgage interest, real estate taxes, casualty losses (deductible anyway),
  2. operating expenses, up to what the limit leaves,
  3. depreciation, up to what then remains.
Disallowed operating expenses and depreciation carry over to next year.
"""

from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from writeoff.calculators.common import (
    CalculationResult,
    FrozenModel,
    Workbook,
    money,
    unavailable,
)
from writeoff.tax_parameters import ParameterUnavailableError, TaxParameters

AUTHORITIES = ["IRC § 280A(c)(1)", "IRC § 280A(c)(5)", "Pub 587", "Instructions for Form 8829"]


class ExpenseCategory(StrEnum):
    MORTGAGE_INTEREST = "mortgage_interest"
    REAL_ESTATE_TAXES = "real_estate_taxes"
    CASUALTY_LOSSES = "casualty_losses"
    INSURANCE = "insurance"
    RENT = "rent"
    REPAIRS = "repairs"
    UTILITIES = "utilities"
    OTHER = "other"
    DEPRECIATION = "depreciation"


_TIER = {
    ExpenseCategory.MORTGAGE_INTEREST: 1,
    ExpenseCategory.REAL_ESTATE_TAXES: 1,
    ExpenseCategory.CASUALTY_LOSSES: 1,
    ExpenseCategory.DEPRECIATION: 3,
}


class HomeExpense(FrozenModel):
    category: ExpenseCategory
    amount: Decimal = Field(ge=0)
    direct: bool = False  # only for the business part of the home (e.g. painting the office)


class HomeOfficeInput(FrozenModel):
    tax_year: int
    method: Literal["simplified", "regular"]
    sq_ft: Decimal = Field(gt=0)
    total_sq_ft: Decimal = Field(gt=0)
    expenses: tuple[HomeExpense, ...] = ()
    gross_income_limit: Decimal | None = None

    @model_validator(mode="after")
    def _check(self) -> "HomeOfficeInput":
        if self.sq_ft > self.total_sq_ft:
            raise ValueError("business sq_ft cannot exceed total_sq_ft")
        return self


class HomeOfficeResult(CalculationResult):
    method: str | None = None
    business_pct: Decimal | None = None
    deduction: Decimal | None = None
    carryover_operating: Decimal | None = None
    carryover_depreciation: Decimal | None = None


def calc_home_office(inp: HomeOfficeInput, params: TaxParameters) -> HomeOfficeResult:
    wb = Workbook(params, inp.tax_year)
    try:
        if inp.method == "simplified":
            return _simplified(inp, wb)
        return _regular(inp, wb)
    except ParameterUnavailableError as exc:
        return unavailable(HomeOfficeResult, exc, AUTHORITIES)


def _simplified(inp: HomeOfficeInput, wb: Workbook) -> HomeOfficeResult:
    rate = wb.param("home_office_simplified_rate")
    max_sq_ft = wb.param("home_office_simplified_max_sq_ft")
    area = min(inp.sq_ft, max_sq_ft)
    if inp.sq_ft > max_sq_ft:
        wb.note(f"Only {max_sq_ft} sq ft count under the simplified method.")
    deduction = money(area * rate)
    wb.step(f"{area} sq ft x simplified rate", deduction)
    if inp.gross_income_limit is not None and deduction > max(inp.gross_income_limit, Decimal(0)):
        deduction = money(max(inp.gross_income_limit, Decimal(0)))
        wb.step("Limited to gross income from the business use; no carryover", deduction)
    if inp.expenses:
        wb.note(
            "Actual home expenses are not deducted under the simplified method; mortgage "
            "interest and real estate taxes may be itemized instead."
        )
    return wb.finish(
        HomeOfficeResult,
        AUTHORITIES,
        method="simplified",
        business_pct=money(_share(inp) * 100),
        deduction=deduction,
    )


def _share(inp: HomeOfficeInput) -> Decimal:
    return inp.sq_ft / inp.total_sq_ft


def _regular(inp: HomeOfficeInput, wb: Workbook) -> HomeOfficeResult:
    share = _share(inp)
    wb.step(f"Business percentage: {inp.sq_ft} / {inp.total_sq_ft} sq ft = {money(share * 100)}%")
    tiers = {1: Decimal(0), 2: Decimal(0), 3: Decimal(0)}
    for expense in inp.expenses:
        amount = expense.amount if expense.direct else expense.amount * share
        tiers[_TIER.get(expense.category, 2)] += amount
    for tier, label in (
        (1, "Mortgage interest, taxes, casualty losses"),
        (2, "Operating expenses"),
        (3, "Depreciation"),
    ):
        wb.step(f"{label} (business portion)", tiers[tier])
    if inp.gross_income_limit is None:
        wb.note(
            "The deduction cannot exceed gross income from the business use of the home; "
            "give gross_income_limit to apply that limit (IRC § 280A(c)(5))."
        )
        deduction = sum(tiers.values(), Decimal(0))
        return wb.finish(
            HomeOfficeResult,
            AUTHORITIES,
            method="regular",
            business_pct=money(share * 100),
            deduction=money(deduction),
            carryover_operating=Decimal(0),
            carryover_depreciation=Decimal(0),
        )
    room = max(inp.gross_income_limit - tiers[1], Decimal(0))
    operating = min(tiers[2], room)
    room -= operating
    depreciation = min(tiers[3], room)
    deduction = tiers[1] + operating + depreciation
    wb.step("Allowed after the gross income limit", deduction)
    carry_op, carry_dep = tiers[2] - operating, tiers[3] - depreciation
    if carry_op or carry_dep:
        wb.note("Expenses above the gross income limit carry over to next year (Form 8829).")
    return wb.finish(
        HomeOfficeResult,
        AUTHORITIES,
        method="regular",
        business_pct=money(share * 100),
        deduction=money(deduction),
        carryover_operating=money(carry_op),
        carryover_depreciation=money(carry_dep),
    )
