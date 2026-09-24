"""Business use of a car or truck (Pub 463 ch. 4; Pub 334 ch. 8).

Standard mileage: business miles x the standard mileage rate, plus business parking and
tolls. Actual expenses: operating costs x business-use percentage, plus depreciation
(computed separately with `calc_depreciation`, already limited to business use), plus
business parking and tolls. Commuting miles are personal and never business miles.
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

AUTHORITIES = ["Pub 463, ch. 4", "IRC § 162(a)", "IRC § 280F"]


class CarCost(StrEnum):
    GAS_OIL = "gas_oil"
    REPAIRS = "repairs"
    TIRES = "tires"
    INSURANCE = "insurance"
    REGISTRATION_LICENSES = "registration_licenses"
    LEASE_PAYMENTS = "lease_payments"
    GARAGE_RENT = "garage_rent"
    OTHER = "other"


class VehicleInput(FrozenModel):
    tax_year: int
    method: Literal["standard_mileage", "actual"]
    total_miles: Decimal = Field(gt=0)
    business_miles: Decimal = Field(ge=0)
    actual_costs: dict[CarCost, Decimal] = Field(default_factory=dict)
    depreciation: Decimal = Field(default=Decimal(0), ge=0)
    parking_tolls: Decimal = Field(default=Decimal(0), ge=0)
    vehicles_used_simultaneously: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "VehicleInput":
        if self.business_miles > self.total_miles:
            raise ValueError("business_miles cannot exceed total_miles")
        if any(v < 0 for v in self.actual_costs.values()):
            raise ValueError("costs cannot be negative")
        return self


class VehicleResult(CalculationResult):
    method: str | None = None
    business_use_pct: Decimal | None = None
    deduction: Decimal | None = None


def calc_vehicle(inp: VehicleInput, params: TaxParameters) -> VehicleResult:
    wb = Workbook(params, inp.tax_year)
    share = inp.business_miles / inp.total_miles
    business_pct = money(share * 100)
    wb.step(f"Business use: {inp.business_miles} / {inp.total_miles} miles = {business_pct}%")
    if business_pct <= 50:
        wb.note(
            "Business use of 50% or less: no section 179 or special allowance, and "
            "depreciation must use ADS (IRC § 280F(b))."
        )
    try:
        if inp.method == "standard_mileage":
            if inp.vehicles_used_simultaneously >= 5:
                wb.note(
                    "The standard mileage rate is not allowed for five or more vehicles used "
                    "at the same time (Pub 463, ch. 4)."
                )
                return wb.finish(
                    VehicleResult,
                    AUTHORITIES,
                    method="standard_mileage",
                    business_use_pct=business_pct,
                    deduction=Decimal(0),
                )
            rate = wb.param("standard_mileage_rate_business")
            mileage = money(inp.business_miles * rate)
            wb.step(f"{inp.business_miles} business miles x standard mileage rate", mileage)
            if inp.actual_costs or inp.depreciation:
                wb.note(
                    "Actual operating costs and depreciation are replaced by the standard "
                    "mileage rate and are not deducted separately."
                )
            deduction = mileage + inp.parking_tolls
        else:
            operating = sum(inp.actual_costs.values(), Decimal(0))
            allocated = money(operating * share)
            wb.step(f"Operating costs {money(operating)} x {business_pct}% business use", allocated)
            if inp.depreciation:
                wb.step("Depreciation (business portion, from calc_depreciation)", inp.depreciation)
            deduction = allocated + inp.depreciation + inp.parking_tolls
            wb.note(
                "If you used actual expenses with MACRS depreciation for this car, you "
                "cannot switch to the standard mileage rate later (Pub 463, ch. 4)."
            )
    except ParameterUnavailableError as exc:
        return unavailable(VehicleResult, exc, AUTHORITIES)
    if inp.parking_tolls:
        wb.step("Business parking fees and tolls", inp.parking_tolls)
    wb.step("Vehicle deduction", deduction)
    return wb.finish(
        VehicleResult,
        AUTHORITIES,
        method=inp.method,
        business_use_pct=business_pct,
        deduction=money(deduction),
    )
