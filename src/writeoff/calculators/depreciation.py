"""Depreciation: § 179 expensing, special (bonus) allowance, MACRS, and § 280F caps.

Order of operations (Form 4562; Pub 946 ch. 2-5):
1. Business basis = cost x business-use %.
2. § 179 (if elected and eligible), limited by the dollar limit less the phaseout
   (§ 179(b)(1)-(2)), the SUV cap (§ 179(b)(5)), and the business income limit
   (§ 179(b)(3); the excess carries over).
3. Special depreciation allowance (§ 168(k)) on what remains. The percentage comes from
   the tax parameters for the acquisition date, unless the caller elects out (0).
4. MACRS on the rest, by the class's method, recovery period and convention.
5. Passenger automobiles: § 280F(a) caps each year's total, scaled by business use.
   Basis the caps leave unrecovered is deducted after the recovery period, up to the
   later-years cap.

Listed property used 50% or less for business (§ 280F(b)) gets no § 179 or bonus and must
use ADS straight line.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from writeoff.calculators.common import (
    CalculationResult,
    FrozenModel,
    Workbook,
    money,
    pct,
    unavailable,
)
from writeoff.calculators.macrs import Convention, Method, percentages
from writeoff.tax_parameters import ParameterUnavailableError, TaxParameters

MAX_EXTRA_YEARS = 60  # safety bound on post-recovery-period § 280F deductions


@dataclass(frozen=True, slots=True)
class AssetClass:
    description: str
    gds_years: Decimal
    gds_method: Method
    ads_years: Decimal
    real_property: bool = False
    section_179: bool = True
    bonus: bool = True
    listed: bool = False
    passenger_auto: bool = False
    suv_179_cap: bool = False


ASSET_CLASSES: dict[str, AssetClass] = {
    "computer_equipment": AssetClass(
        "Computers and peripheral equipment", Decimal(5), Method.DB200, Decimal(5)
    ),
    "office_furniture": AssetClass(
        "Office furniture and fixtures", Decimal(7), Method.DB200, Decimal(10)
    ),
    "machinery_equipment": AssetClass(
        "Machinery and equipment (7-year class)", Decimal(7), Method.DB200, Decimal(10)
    ),
    "passenger_auto": AssetClass(
        "Passenger automobile (§ 280F caps apply)",
        Decimal(5),
        Method.DB200,
        Decimal(5),
        listed=True,
        passenger_auto=True,
    ),
    "light_truck_van": AssetClass(
        "Truck or van not subject to the § 280F caps",
        Decimal(5),
        Method.DB200,
        Decimal(5),
        listed=True,
    ),
    "heavy_suv": AssetClass(
        "SUV rated 6,001-14,000 lbs GVWR (§ 179 SUV cap)",
        Decimal(5),
        Method.DB200,
        Decimal(5),
        listed=True,
        suv_179_cap=True,
    ),
    "land_improvements": AssetClass(
        "Land improvements (15-year)", Decimal(15), Method.DB150, Decimal(20), section_179=False
    ),
    "qualified_improvement_property": AssetClass(
        "Qualified improvement property", Decimal(15), Method.SL, Decimal(20)
    ),
    "residential_rental_property": AssetClass(
        "Residential rental property",
        Decimal("27.5"),
        Method.SL,
        Decimal(30),
        real_property=True,
        section_179=False,
        bonus=False,
    ),
    "nonresidential_real_property": AssetClass(
        "Nonresidential real property",
        Decimal(39),
        Method.SL,
        Decimal(40),
        real_property=True,
        section_179=False,
        bonus=False,
    ),
}

AssetClassName = StrEnum("AssetClassName", {k.upper(): k for k in ASSET_CLASSES})  # type: ignore[misc]
"""Asset class names as an enum, so tool schemas list the valid choices."""

AUTHORITIES = ["IRC § 168", "IRC § 179", "IRC § 168(k)", "Pub 946"]


class DepreciationInput(FrozenModel):
    cost: Decimal = Field(gt=0)
    placed_in_service: date
    asset_class: AssetClassName
    method: Literal["GDS", "GDS_SL", "ADS"] = "GDS"
    elect_179: bool = False
    section_179_amount: Decimal | None = Field(default=None, gt=0)
    bonus_pct: Decimal | None = Field(default=None, ge=0, le=100)
    business_use_pct: Decimal = Field(default=Decimal(100), gt=0, le=100)
    convention: Literal["half_year", "mid_quarter"] | None = None
    acquired: date | None = None
    total_section_179_property_cost: Decimal | None = Field(default=None, gt=0)
    business_income: Decimal | None = None

    @model_validator(mode="after")
    def _check(self) -> "DepreciationInput":
        if self.acquired is not None and self.acquired > self.placed_in_service:
            raise ValueError("acquired must not be after placed_in_service")
        return self


class ScheduleYear(FrozenModel):
    tax_year: int
    macrs: Decimal
    section_179: Decimal = Decimal(0)
    bonus: Decimal = Decimal(0)
    cap: Decimal | None = None
    deduction: Decimal


class DepreciationResult(CalculationResult):
    business_basis: Decimal | None = None
    section_179: Decimal | None = None
    section_179_carryover: Decimal | None = None
    bonus: Decimal | None = None
    bonus_pct: Decimal | None = None
    macrs_basis: Decimal | None = None
    method: str | None = None
    recovery_period: Decimal | None = None
    convention: str | None = None
    first_year_deduction: Decimal | None = None
    schedule: list[ScheduleYear] = Field(default_factory=list)
    total_deductions: Decimal | None = None


def calc_depreciation(inp: DepreciationInput, params: TaxParameters) -> DepreciationResult:
    cls = ASSET_CLASSES[inp.asset_class.value]
    authorities = [*AUTHORITIES, *(["IRC § 280F"] if cls.listed else [])]
    try:
        return _calculate(inp, cls, Workbook(params, inp.placed_in_service.year), authorities)
    except ParameterUnavailableError as exc:
        return unavailable(DepreciationResult, exc, authorities)


def _calculate(
    inp: DepreciationInput, cls: AssetClass, wb: Workbook, authorities: list[str]
) -> DepreciationResult:
    method_choice = inp.method
    low_business_use = cls.listed and inp.business_use_pct <= 50
    if low_business_use and method_choice != "ADS":
        method_choice = "ADS"
        wb.note(
            "Listed property used 50% or less for business must use ADS straight line; "
            "no section 179 or special allowance (IRC § 280F(b))."
        )

    basis = money(inp.cost * pct(inp.business_use_pct))
    wb.step(f"Business basis: cost x {inp.business_use_pct}% business use", basis)

    sec179, carryover = _section_179(inp, cls, wb, basis, low_business_use)
    remaining = basis - sec179

    bonus_pct = Decimal(0)
    bonus_ok = cls.bonus and not low_business_use and method_choice != "ADS"
    if not bonus_ok and inp.bonus_pct:
        wb.note("This property does not qualify for the special depreciation allowance.")
    elif bonus_ok:
        on = inp.acquired or inp.placed_in_service
        bonus_pct = (
            inp.bonus_pct
            if inp.bonus_pct is not None
            else wb.param("bonus_depreciation_pct", on=on)
        )
        if inp.bonus_pct is not None and inp.bonus_pct == 0:
            wb.note(
                "Elected out of the special depreciation allowance for this class "
                "(IRC § 168(k)(7))."
            )
    bonus = money(remaining * pct(bonus_pct))
    if bonus:
        wb.step(f"Special depreciation allowance: {bonus_pct}% of remaining basis", bonus)
    macrs_basis = remaining - bonus
    wb.step("Basis left for regular MACRS", macrs_basis)

    if method_choice == "ADS":
        method, years = Method.SL, cls.ads_years
    elif method_choice == "GDS_SL":
        method, years = Method.SL, cls.gds_years
    else:
        method, years = cls.gds_method, cls.gds_years
    if cls.real_property:
        convention, period = Convention.MID_MONTH, inp.placed_in_service.month
    elif inp.convention == "mid_quarter":
        convention, period = Convention.MID_QUARTER, (inp.placed_in_service.month - 1) // 3 + 1
    else:
        convention, period = Convention.HALF_YEAR, 1
        if inp.placed_in_service.month >= 10:
            wb.note(
                "Placed in service in the last quarter: the mid-quarter convention applies "
                "if more than 40% of the year's depreciable basis was placed in service then "
                "(IRC § 168(d)(3))."
            )
    rates = percentages(method, years, convention, period)
    wb.step(f"MACRS {method.value}, {years}-year recovery, {convention.value} convention")

    schedule = _schedule(
        inp, cls, wb, rates=rates, macrs_basis=macrs_basis, sec179=sec179, bonus=bonus
    )
    total = sum((y.deduction for y in schedule), Decimal(0))
    return wb.finish(
        DepreciationResult,
        authorities,
        business_basis=basis,
        section_179=sec179,
        section_179_carryover=carryover,
        bonus=bonus,
        bonus_pct=bonus_pct,
        macrs_basis=macrs_basis,
        method=method.value,
        recovery_period=years,
        convention=convention.value,
        first_year_deduction=schedule[0].deduction,
        schedule=schedule,
        total_deductions=total,
    )


def _section_179(
    inp: DepreciationInput, cls: AssetClass, wb: Workbook, basis: Decimal, low_business_use: bool
) -> tuple[Decimal, Decimal]:
    if not inp.elect_179:
        return Decimal(0), Decimal(0)
    if not cls.section_179 or low_business_use:
        wb.note("This property does not qualify for the section 179 deduction.")
        return Decimal(0), Decimal(0)
    limit = wb.param("section_179_dollar_limit")
    threshold = wb.param("section_179_phaseout_threshold")
    total_cost = inp.total_section_179_property_cost or inp.cost
    reduction = max(Decimal(0), total_cost - threshold)
    dollar_limit = max(Decimal(0), limit - reduction)
    wb.step("Section 179 dollar limit after phaseout", dollar_limit)
    if cls.suv_179_cap:
        dollar_limit = min(dollar_limit, wb.param("section_179_suv_limit"))
        wb.step("Section 179 limit for SUVs", dollar_limit)
    elected = min(inp.section_179_amount or basis, basis, dollar_limit)
    carryover = Decimal(0)
    if inp.business_income is not None:
        allowed = min(elected, max(Decimal(0), inp.business_income))
        carryover = elected - allowed
        if carryover:
            wb.note(
                f"Section 179 is limited to business income; {money(carryover)} carries "
                "over to next year (IRC § 179(b)(3))."
            )
        elected = allowed
    wb.step("Section 179 deduction", elected)
    return money(elected), money(carryover)


def _schedule(
    inp: DepreciationInput,
    cls: AssetClass,
    wb: Workbook,
    *,
    rates: list[Decimal],
    macrs_basis: Decimal,
    sec179: Decimal,
    bonus: Decimal,
) -> list[ScheduleYear]:
    start = inp.placed_in_service.year
    macrs = [money(macrs_basis * pct(r)) for r in rates]
    macrs[-1] = macrs_basis - sum(macrs[:-1], Decimal(0))  # absorb cent rounding
    if not cls.passenger_auto:
        return [
            ScheduleYear(
                tax_year=start + i,
                macrs=m,
                section_179=sec179 if i == 0 else Decimal(0),
                bonus=bonus if i == 0 else Decimal(0),
                deduction=m + (sec179 + bonus if i == 0 else Decimal(0)),
            )
            for i, m in enumerate(macrs)
        ]
    share = pct(inp.business_use_pct)
    caps = [
        wb.param(
            "passenger_auto_first_year_limit_with_bonus"
            if bonus
            else "passenger_auto_first_year_limit_without_bonus"
        )
        * share,
        wb.param("passenger_auto_second_year_limit") * share,
        wb.param("passenger_auto_third_year_limit") * share,
    ]
    later = wb.param("passenger_auto_succeeding_years_limit") * share
    rows: list[ScheduleYear] = []
    unrecovered = sec179 + bonus + macrs_basis
    i = 0
    while unrecovered > 0 and i < len(macrs) + MAX_EXTRA_YEARS:
        scheduled = macrs[i] if i < len(macrs) else unrecovered
        if i == 0:
            scheduled += sec179 + bonus
        cap = money(caps[i] if i < len(caps) else later)
        deduction = min(scheduled, cap, unrecovered)
        rows.append(
            ScheduleYear(
                tax_year=start + i,
                macrs=macrs[i] if i < len(macrs) else Decimal(0),
                section_179=sec179 if i == 0 else Decimal(0),
                bonus=bonus if i == 0 else Decimal(0),
                cap=cap,
                deduction=deduction,
            )
        )
        unrecovered -= deduction
        i += 1
    wb.note(
        "Passenger automobile depreciation is capped each year (IRC § 280F(a)); basis the "
        "caps leave unrecovered is deducted after the recovery period."
    )
    return rows
