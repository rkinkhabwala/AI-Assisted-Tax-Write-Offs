"""Agent tools that need no database: classify_expense, get_tax_parameter, citations."""

from decimal import Decimal as D
from pathlib import Path

import pytest

from writeoff.models import EntityType
from writeoff.tax_parameters import TaxParameters
from writeoff.tools.classify import (
    RULES,
    ExpenseInput,
    Treatment,
    classify_expense,
    match_rules,
)
from writeoff.tools.law import get_tax_parameter, normalize_citation

FIXTURE_PARAMS = TaxParameters(Path(__file__).parent / "fixtures" / "tax_parameters", (2025,))
SHIPPED_PARAMS = TaxParameters(
    Path(__file__).resolve().parents[1] / "data" / "tax_parameters", (2025, 2026)
)


def _expense(description: str, amount: str = "200", **kwargs: object) -> ExpenseInput:
    base: dict[str, object] = {
        "description": description,
        "amount": D(amount),
        "entity_type": EntityType.SOLE_PROP,
        "tax_year": 2025,
    }
    base.update(kwargs)
    return ExpenseInput.model_validate(base)


@pytest.mark.parametrize(
    ("description", "amount", "category", "treatment", "deductible"),
    [
        (
            "Lunch with a client to discuss a contract",
            "200",
            "meals",
            Treatment.PARTIALLY_DEDUCTIBLE,
            "100.00",
        ),
        (
            "Lakers basketball game tickets for a client",
            "400",
            "entertainment",
            Treatment.NOT_DEDUCTIBLE,
            "0",
        ),
        (
            "Company holiday party for employees",
            "1500",
            "employee_party",
            Treatment.FULLY_DEDUCTIBLE,
            "1500.00",
        ),
        (
            "Speeding ticket on the way to a client",
            "150",
            "fines_penalties",
            Treatment.NOT_DEDUCTIBLE,
            "0",
        ),
        (
            "Monthly Adobe software subscription",
            "60",
            "software_subscription",
            Treatment.FULLY_DEDUCTIBLE,
            "60.00",
        ),
        ("Printer ink and paper", "80", "supplies", Treatment.FULLY_DEDUCTIBLE, "80.00"),
        # One keyword each for supplies and equipment: the consumable wins (answer-eval pilot).
        ("Printer paper", "45", "supplies", Treatment.FULLY_DEDUCTIBLE, "45.00"),
        ("Toner cartridge", "90", "supplies", Treatment.FULLY_DEDUCTIBLE, "90.00"),
        (
            "Donation to the campaign of a local candidate",
            "500",
            "political_lobbying",
            Treatment.NOT_DEDUCTIBLE,
            "0",
        ),
        ("Groceries for my family", "300", "personal", Treatment.NOT_DEDUCTIBLE, "0"),
    ],
)
def test_classification(
    description: str, amount: str, category: str, treatment: Treatment, deductible: str
) -> None:
    r = classify_expense(_expense(description, amount), FIXTURE_PARAMS)
    assert r.category == category
    assert r.treatment is treatment
    assert r.deductible_amount == D(deductible)
    assert r.authorities


def test_meals_limit_is_sourced() -> None:
    r = classify_expense(_expense("dinner with a customer"), FIXTURE_PARAMS)
    assert [p.name for p in r.parameters_used] == ["business_meals_deduction_pct"]
    assert r.parameters_used[0].source_url.startswith("https://www.irs.gov/")
    assert "IRC § 274(n)" in r.authorities


def test_gifts_limited_per_recipient() -> None:
    r = classify_expense(
        _expense("holiday gift baskets for clients", "120", recipients=3), FIXTURE_PARAMS
    )
    assert r.category == "gifts"
    assert r.deductible_amount == D(75)
    single = classify_expense(_expense("gift basket for a client", "120"), FIXTURE_PARAMS)
    assert single.deductible_amount == D(25)
    assert any("single recipient" in n for n in single.notes)


def test_equipment_is_capitalized_with_de_minimis_note() -> None:
    r = classify_expense(_expense("new laptop for the business", "1800"), FIXTURE_PARAMS)
    assert r.treatment is Treatment.CAPITALIZE_AND_DEPRECIATE
    assert r.deductible_amount is None
    assert r.calculator == "calc_depreciation"
    assert any("de minimis" in n for n in r.notes)


def test_mixed_use_and_startup() -> None:
    phone = classify_expense(
        _expense("cell phone bill", "100", business_use_pct=D(60)), FIXTURE_PARAMS
    )
    assert phone.deductible_amount == D(60)
    startup = classify_expense(_expense("start-up costs before we opened", "53000"), FIXTURE_PARAMS)
    assert startup.deductible_amount == D(2000)  # 5,000 reduced by 3,000 over the 50,000 threshold


def test_entity_specific_reasoning() -> None:
    sole = classify_expense(_expense("health insurance premiums", "6000"), FIXTURE_PARAMS)
    corp = classify_expense(
        _expense("health insurance premiums", "6000", entity_type=EntityType.C_CORP), FIXTURE_PARAMS
    )
    assert any("162(l)" in line for line in sole.reasoning)
    assert any("corporation" in line for line in corp.reasoning)
    assert sole.treatment is Treatment.DEPENDS_ON_FACTS


def test_category_override_unknown_and_ambiguity() -> None:
    forced = classify_expense(_expense("misc", "100", category="advertising"), FIXTURE_PARAMS)
    assert forced.category == "advertising"
    with pytest.raises(ValueError, match="unknown category"):
        classify_expense(_expense("misc", category="nonsense"), FIXTURE_PARAMS)
    unknown = classify_expense(_expense("zzzz qqqq"), FIXTURE_PARAMS)
    assert unknown.category == "unknown"
    assert unknown.treatment is Treatment.DEPENDS_ON_FACTS
    food_at_game = classify_expense(
        _expense("food and drinks at a basketball game with a client"), FIXTURE_PARAMS
    )
    assert {food_at_game.category, *food_at_game.other_matching_categories} >= {
        "meals",
        "entertainment",
    }


def test_unverified_limit_keeps_treatment_but_withholds_amount() -> None:
    r = classify_expense(_expense("client lunch", tax_year=2026), SHIPPED_PARAMS)
    assert r.treatment is Treatment.PARTIALLY_DEDUCTIBLE
    assert r.status == "unavailable"
    assert r.deductible_amount is None
    assert r.missing_parameters == ["business_meals_deduction_pct"]


def test_every_rule_keyword_is_a_valid_regex_and_categories_unique() -> None:
    assert len({r.category for r in RULES}) == len(RULES)
    assert match_rules("office rent for march")[0].category == "rent"


# --- get_tax_parameter ---------------------------------------------------------------


def test_get_tax_parameter_states() -> None:
    ok = get_tax_parameter(FIXTURE_PARAMS, "standard_mileage_rate_business", 2025)
    assert ok.status == "ok"
    assert ok.value == D("0.70")
    assert ok.source_url == "https://www.irs.gov/tax-professionals/standard-mileage-rates"
    periods = get_tax_parameter(FIXTURE_PARAMS, "bonus_depreciation_pct", 2025)
    assert periods.status == "ok"
    assert periods.periods is not None
    assert [p.value for p in periods.periods] == [D(40), D(100)]
    shipped = get_tax_parameter(SHIPPED_PARAMS, "section_179_dollar_limit", 2026)
    assert shipped.status == "unavailable"
    assert shipped.value is None
    assert shipped.guidance is not None
    assert "estimate" in shipped.guidance
    unknown = get_tax_parameter(FIXTURE_PARAMS, "moon_tax", 2025)
    assert unknown.status == "unknown"
    assert "standard_mileage_rate_business" in unknown.known_parameters
    wrong_year = get_tax_parameter(FIXTURE_PARAMS, "standard_mileage_rate_business", 1999)
    assert wrong_year.status == "unknown"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("§ 179(b)(1)", "IRC § 179(b)(1)"),
        ("section 280A", "IRC § 280A"),
        ("Reg. 1.162-5", "Treas. Reg. § 1.162-5"),
        ("IRC § 274(n)", "IRC § 274(n)"),
        ("Pub 463, ch. 2", "Pub 463, ch. 2"),
    ],
)
def test_normalize_citation(raw: str, expected: str) -> None:
    assert normalize_citation(raw) == expected


def test_specific_rule_beats_generic_keywords() -> None:
    assert match_rules("health insurance premiums")[0].category == "health_insurance"
    assert match_rules("general liability insurance premiums")[0].category == "insurance"
