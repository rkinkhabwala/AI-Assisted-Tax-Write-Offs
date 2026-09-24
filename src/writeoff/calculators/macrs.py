"""MACRS depreciation percentages computed from IRC § 168(b)-(d).

These are statutory methods, not annually indexed amounts, so they are computed here
rather than read from `data/tax_parameters`:

- Method (§ 168(b)): 200% or 150% declining balance, switching to straight line in the
  first year straight line gives a larger deduction, or straight line throughout.
- Convention (§ 168(d)): half-year; mid-quarter (the quarter placed in service); or
  mid-month for residential rental and nonresidential real property.

Rounding reproduces the IRS percentage tables (Pub 946, Appendix A). Personal property:
each year's percentage is computed on the remaining (already rounded) percentage and
rounded to 2 decimals. This matches Tables A-1 and A-2 exactly (e.g. 7-year half-year:
14.29, 24.49, 17.49, 12.49, 8.93, 8.92, 8.93, 4.46). Real property uses the IRS's
rounded monthly rate (39-year: 0.214% per month), which matches Table A-7a exactly and
Table A-6 (27.5-year) to within 0.01 percentage point in the middle years.
"""

from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

HUNDRED = Decimal(100)
TWELVE = Decimal(12)
HALF = Decimal("0.5")


class Method(StrEnum):
    DB200 = "200DB"
    DB150 = "150DB"
    SL = "SL"


class Convention(StrEnum):
    HALF_YEAR = "half_year"
    MID_QUARTER = "mid_quarter"
    MID_MONTH = "mid_month"


_DB_FACTOR = {Method.DB200: Decimal(2), Method.DB150: Decimal("1.5"), Method.SL: Decimal(1)}


def _r(value: Decimal, places: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def first_year_fraction(convention: Convention, period: int) -> Decimal:
    """Share of a full year's depreciation taken in the year placed in service.
    `period` is the quarter (1-4) for mid-quarter, the month (1-12) for mid-month."""
    if convention is Convention.HALF_YEAR:
        return HALF
    if convention is Convention.MID_QUARTER:
        if not 1 <= period <= 4:
            raise ValueError("mid-quarter needs the quarter placed in service (1-4)")
        return (Decimal(4 - period) + HALF) / 4
    if not 1 <= period <= 12:
        raise ValueError("mid-month needs the month placed in service (1-12)")
    return (TWELVE - period + HALF) / TWELVE


def percentages(
    method: Method, recovery_years: Decimal, convention: Convention, period: int = 1
) -> list[Decimal]:
    """Annual depreciation as percentages of basis, summing to exactly 100."""
    if recovery_years <= 0:
        raise ValueError("recovery period must be positive")
    if convention is Convention.MID_MONTH:
        if method is not Method.SL:
            raise ValueError("mid-month convention applies only to straight-line real property")
        return _mid_month(recovery_years, period)
    rate = _DB_FACTOR[method] / recovery_years
    first = first_year_fraction(convention, period)
    out = [_r(HUNDRED * rate * first, 2)]
    remaining = HUNDRED - out[0]
    elapsed = first
    while remaining > 0:
        left = recovery_years - elapsed
        if left <= 1:
            out.append(remaining)
            break
        declining = remaining * rate if method is not Method.SL else Decimal(0)
        straight = remaining / left
        amount = min(_r(max(declining, straight), 2), remaining)
        out.append(amount)
        remaining -= amount
        elapsed += 1
    return out


def _mid_month(recovery_years: Decimal, month: int) -> list[Decimal]:
    monthly = _r(HUNDRED / recovery_years / TWELVE, 3)
    full = _r(HUNDRED / recovery_years, 3)
    out = [_r(monthly * (TWELVE - month + HALF), 3)]
    remaining = HUNDRED - out[0]
    while remaining > full:
        out.append(full)
        remaining -= full
    out.append(remaining)
    return out
