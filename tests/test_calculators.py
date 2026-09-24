"""Calculators against hand-computed results and the IRS MACRS tables (Pub 946, App. A).

Parameter values come from tests/fixtures/tax_parameters (illustrative, not verified);
the shipped data files are all null, which exercises the "unavailable" path.
"""

from datetime import date
from decimal import Decimal as D
from pathlib import Path

import pytest
from pydantic import ValidationError

from writeoff.calculators.depreciation import DepreciationInput, calc_depreciation
from writeoff.calculators.home_office import (
    ExpenseCategory,
    HomeExpense,
    HomeOfficeInput,
    calc_home_office,
)
from writeoff.calculators.macrs import Convention, Method, percentages
from writeoff.calculators.vehicle import CarCost, VehicleInput, calc_vehicle
from writeoff.tax_parameters import TaxParameters

FIXTURE_PARAMS = TaxParameters(Path(__file__).parent / "fixtures" / "tax_parameters", (2025,))
SHIPPED_PARAMS = TaxParameters(
    Path(__file__).resolve().parents[1] / "data" / "tax_parameters", (2025, 2026)
)


def _d(*values: str) -> list[D]:
    return [D(v) for v in values]


# --- MACRS tables --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "years", "convention", "period", "expected"),
    [
        (
            Method.DB200,
            D(5),
            Convention.HALF_YEAR,
            1,
            _d("20.00", "32.00", "19.20", "11.52", "11.52", "5.76"),
        ),
        (
            Method.DB200,
            D(7),
            Convention.HALF_YEAR,
            1,
            _d("14.29", "24.49", "17.49", "12.49", "8.93", "8.92", "8.93", "4.46"),
        ),
        (
            Method.DB150,
            D(15),
            Convention.HALF_YEAR,
            1,
            _d(
                "5.00",
                "9.50",
                "8.55",
                "7.70",
                "6.93",
                "6.23",
                "5.90",
                "5.90",
                "5.91",
                "5.90",
                "5.91",
                "5.90",
                "5.91",
                "5.90",
                "5.91",
                "2.95",
            ),
        ),
        (
            Method.DB200,
            D(5),
            Convention.MID_QUARTER,
            1,
            _d("35.00", "26.00", "15.60", "11.01", "11.01", "1.38"),
        ),
        (
            Method.DB200,
            D(5),
            Convention.MID_QUARTER,
            4,
            _d("5.00", "38.00", "22.80", "13.68", "10.94", "9.58"),
        ),
    ],
)
def test_macrs_matches_irs_tables(
    method: Method, years: D, convention: Convention, period: int, expected: list[D]
) -> None:
    assert percentages(method, years, convention, period) == expected


def test_real_property_mid_month() -> None:
    jan = percentages(Method.SL, D(39), Convention.MID_MONTH, 1)
    assert (jan[0], jan[1], jan[-1], len(jan)) == (D("2.461"), D("2.564"), D("0.107"), 40)
    # Pub 587 Table 2 (in our corpus): first-year percentage by month placed in service.
    by_month = [percentages(Method.SL, D(39), Convention.MID_MONTH, m)[0] for m in range(1, 13)]
    assert by_month == _d(
        "2.461",
        "2.247",
        "2.033",
        "1.819",
        "1.605",
        "1.391",
        "1.177",
        "0.963",
        "0.749",
        "0.535",
        "0.321",
        "0.107",
    )
    assert percentages(Method.SL, D("27.5"), Convention.MID_MONTH, 1)[0] == D("3.485")
    for m in (1, 6, 12):
        assert sum(percentages(Method.SL, D("27.5"), Convention.MID_MONTH, m)) == 100


def test_macrs_validation() -> None:
    with pytest.raises(ValueError, match="mid-month"):
        percentages(Method.DB200, D(5), Convention.MID_MONTH, 1)
    with pytest.raises(ValueError, match="quarter"):
        percentages(Method.DB200, D(5), Convention.MID_QUARTER, 5)


# --- depreciation --------------------------------------------------------------------


def _dep(**kwargs: object) -> DepreciationInput:
    base: dict[str, object] = {
        "cost": D(10000),
        "placed_in_service": date(2025, 3, 1),
        "asset_class": "computer_equipment",
        "bonus_pct": D(0),
    }
    base.update(kwargs)
    return DepreciationInput.model_validate(base)


def test_plain_macrs_schedule() -> None:
    r = calc_depreciation(_dep(), FIXTURE_PARAMS)
    assert r.status == "ok"
    assert [y.deduction for y in r.schedule] == _d(
        "2000.00", "3200.00", "1920.00", "1152.00", "1152.00", "576.00"
    )
    assert [y.tax_year for y in r.schedule] == list(range(2025, 2031))
    assert r.total_deductions == D("10000.00")


@pytest.mark.parametrize(
    ("acquired", "bonus"), [(date(2025, 1, 10), D(40)), (date(2025, 2, 1), D(100))]
)
def test_bonus_follows_acquisition_date(acquired: date, bonus: D) -> None:
    r = calc_depreciation(
        _dep(asset_class="office_furniture", bonus_pct=None, acquired=acquired), FIXTURE_PARAMS
    )
    assert r.bonus_pct == bonus
    assert r.bonus == D(10000) * bonus / 100
    expected_first = r.bonus + (D(10000) - r.bonus) * D("0.1429")
    assert r.first_year_deduction == expected_first.quantize(D("0.01"))
    assert r.total_deductions == D("10000.00")
    assert any(p.name == "bonus_depreciation_pct" for p in r.parameters_used)


def test_section_179_phaseout_then_bonus() -> None:
    r = calc_depreciation(
        _dep(
            cost=D(3_000_000),
            asset_class="machinery_equipment",
            elect_179=True,
            bonus_pct=None,
            placed_in_service=date(2025, 6, 1),
            total_section_179_property_cost=D(4_600_000),
        ),
        FIXTURE_PARAMS,
    )
    assert r.section_179 == D(1_900_000)  # 2.5M limit reduced by 600k over the 4M threshold
    assert r.bonus == D(1_100_000)  # 100% bonus on the rest
    assert r.first_year_deduction == D(3_000_000)
    used = {p.name for p in r.parameters_used}
    assert {"section_179_dollar_limit", "section_179_phaseout_threshold"} <= used


def test_section_179_business_income_limit_carries_over() -> None:
    r = calc_depreciation(_dep(elect_179=True, business_income=D(4000)), FIXTURE_PARAMS)
    assert r.section_179 == D(4000)
    assert r.section_179_carryover == D(6000)
    assert any("carries over" in n for n in r.notes)


def test_heavy_suv_section_179_cap() -> None:
    r = calc_depreciation(
        _dep(cost=D(80000), asset_class="heavy_suv", elect_179=True, bonus_pct=D(0)), FIXTURE_PARAMS
    )
    assert r.section_179 == D(31000)
    assert r.first_year_deduction == D(31000) + (D(49000) * D("0.20"))


def test_passenger_auto_caps_and_post_recovery_years() -> None:
    r = calc_depreciation(
        _dep(
            cost=D(60000), asset_class="passenger_auto", bonus_pct=None, acquired=date(2025, 2, 1)
        ),
        FIXTURE_PARAMS,
    )
    deductions = [y.deduction for y in r.schedule]
    assert deductions[0] == D(20000)  # first-year cap with bonus
    # All basis went to the bonus, so years 2-6 have no MACRS to deduct; the 40,000 the
    # first-year cap disallowed is recovered after the recovery period at the later cap.
    assert deductions[1:6] == [D(0)] * 5
    assert deductions[6] == D(7000)
    assert all(y.deduction <= (y.cap or 0) for y in r.schedule)
    assert sum(deductions) == D(60000)  # fully recovered, some of it after the recovery period
    assert len(r.schedule) > 6
    assert "IRC § 280F" in r.authorities


def test_passenger_auto_caps_scale_with_business_use() -> None:
    r = calc_depreciation(
        _dep(cost=D(50000), asset_class="passenger_auto", bonus_pct=D(0), business_use_pct=D(80)),
        FIXTURE_PARAMS,
    )
    assert r.business_basis == D(40000)
    assert r.schedule[0].cap == D("9600.00")  # 12,000 without-bonus cap x 80%
    assert r.first_year_deduction == D(8000)  # 40,000 x 20% is under the cap


def test_listed_property_at_or_below_50pct_uses_ads() -> None:
    r = calc_depreciation(
        _dep(
            cost=D(30000),
            asset_class="passenger_auto",
            elect_179=True,
            bonus_pct=None,
            business_use_pct=D(40),
        ),
        FIXTURE_PARAMS,
    )
    assert r.section_179 == 0
    assert r.bonus == 0
    assert r.method == "SL"
    assert r.first_year_deduction == D(1200)  # 12,000 basis x 10% (5-year SL, half-year)
    assert any("ADS" in n for n in r.notes)


def test_real_property_mid_month_and_no_179() -> None:
    r = calc_depreciation(
        _dep(
            cost=D(100000),
            asset_class="nonresidential_real_property",
            placed_in_service=date(2025, 5, 15),
            elect_179=True,
        ),
        FIXTURE_PARAMS,
    )
    assert r.first_year_deduction == D("1605.00")
    assert r.section_179 == 0
    assert any("does not qualify for the section 179" in n for n in r.notes)


def test_mid_quarter_convention() -> None:
    r = calc_depreciation(
        _dep(placed_in_service=date(2025, 11, 1), convention="mid_quarter"), FIXTURE_PARAMS
    )
    assert r.first_year_deduction == D(500)
    hy = calc_depreciation(_dep(placed_in_service=date(2025, 11, 1)), FIXTURE_PARAMS)
    assert any("mid-quarter" in n for n in hy.notes)


def test_unverified_parameters_make_the_result_unavailable() -> None:
    # 2026 values are not verified yet (2025 ones are filled in).
    r = calc_depreciation(
        _dep(bonus_pct=None, placed_in_service=date(2026, 3, 1), acquired=date(2026, 2, 1)),
        SHIPPED_PARAMS,
    )
    assert r.status == "unavailable"
    assert r.missing_parameters == ["bonus_depreciation_pct"]
    assert r.first_year_deduction is None
    assert r.schedule == []
    assert "rather than estimating" in r.notes[0]


def test_depreciation_input_validation() -> None:
    with pytest.raises(ValidationError, match="asset_class"):
        _dep(asset_class="spaceship")
    with pytest.raises(ValidationError, match="acquired"):
        _dep(acquired=date(2025, 4, 1))
    with pytest.raises(ValidationError):
        _dep(business_use_pct=D(120))


# --- home office ---------------------------------------------------------------------


def _home(**kwargs: object) -> HomeOfficeInput:
    base: dict[str, object] = {
        "tax_year": 2025,
        "method": "simplified",
        "sq_ft": D(250),
        "total_sq_ft": D(2000),
    }
    base.update(kwargs)
    return HomeOfficeInput.model_validate(base)


def test_home_office_simplified() -> None:
    assert calc_home_office(_home(), FIXTURE_PARAMS).deduction == D(1250)
    capped = calc_home_office(_home(sq_ft=D(400)), FIXTURE_PARAMS)
    assert capped.deduction == D(1500)
    assert any("300" in n for n in capped.notes)
    limited = calc_home_office(_home(gross_income_limit=D(1000)), FIXTURE_PARAMS)
    assert limited.deduction == D(1000)


def test_home_office_regular_with_gross_income_limit() -> None:
    expenses = [
        HomeExpense(category=ExpenseCategory.MORTGAGE_INTEREST, amount=D(10000)),
        HomeExpense(category=ExpenseCategory.REAL_ESTATE_TAXES, amount=D(5000)),
        HomeExpense(category=ExpenseCategory.UTILITIES, amount=D(3000)),
        HomeExpense(category=ExpenseCategory.INSURANCE, amount=D(1000)),
        HomeExpense(category=ExpenseCategory.REPAIRS, amount=D(400), direct=True),
        HomeExpense(category=ExpenseCategory.DEPRECIATION, amount=D(3000)),
    ]
    unlimited = calc_home_office(
        _home(method="regular", sq_ft=D(200), expenses=expenses), FIXTURE_PARAMS
    )
    assert unlimited.business_pct == D("10.00")
    assert unlimited.deduction == D(2600)  # 1,500 + 800 + 300
    limited = calc_home_office(
        _home(method="regular", sq_ft=D(200), expenses=expenses, gross_income_limit=D(2000)),
        FIXTURE_PARAMS,
    )
    assert limited.deduction == D(2000)
    assert limited.carryover_operating == D(300)
    assert limited.carryover_depreciation == D(300)


def test_home_office_validation_and_unavailable() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        _home(sq_ft=D(3000))
    r = calc_home_office(_home(tax_year=2026), SHIPPED_PARAMS)
    assert r.status == "unavailable"
    assert r.deduction is None


# --- vehicle -------------------------------------------------------------------------


def _car(**kwargs: object) -> VehicleInput:
    base: dict[str, object] = {
        "tax_year": 2025,
        "method": "standard_mileage",
        "total_miles": D(20000),
        "business_miles": D(10000),
    }
    base.update(kwargs)
    return VehicleInput.model_validate(base)


def test_vehicle_standard_mileage() -> None:
    r = calc_vehicle(_car(parking_tolls=D(150)), FIXTURE_PARAMS)
    assert r.deduction == D(7150)  # 10,000 x 0.70 + 150
    assert r.parameters_used[0].name == "standard_mileage_rate_business"


def test_vehicle_actual_expenses() -> None:
    costs = {CarCost.GAS_OIL: D(3000), CarCost.INSURANCE: D(1200), CarCost.REPAIRS: D(800)}
    r = calc_vehicle(
        _car(
            method="actual",
            business_miles=D(15000),
            actual_costs=costs,
            depreciation=D(4000),
            parking_tolls=D(100),
        ),
        FIXTURE_PARAMS,
    )
    assert r.business_use_pct == D("75.00")
    assert r.deduction == D(7850)  # 5,000 x 75% + 4,000 + 100


def test_vehicle_rules_and_validation() -> None:
    fleet = calc_vehicle(_car(vehicles_used_simultaneously=5), FIXTURE_PARAMS)
    assert fleet.deduction == 0
    low = calc_vehicle(_car(business_miles=D(5000)), FIXTURE_PARAMS)
    assert any("50% or less" in n for n in low.notes)
    with pytest.raises(ValidationError, match="cannot exceed"):
        _car(business_miles=D(30000))
    assert calc_vehicle(_car(tax_year=2026), SHIPPED_PARAMS).status == "unavailable"
    assert calc_vehicle(_car(), SHIPPED_PARAMS).deduction == D(7000)  # 2025 rate is filled in
