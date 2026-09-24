"""`classify_expense`: a first-pass, rule-based treatment for a business expense.

Each rule maps an expense category to one of the spec's treatments, the parameter that
sets any limit, the authorities the agent should retrieve before relying on it, and the
facts that could change the answer. Callers may pass `category` directly (the agent
usually knows it); otherwise keywords in the description choose one, and the other
matching categories are reported so ambiguity is visible. Amounts are computed here, never by
the model, and any limit comes from the verified tax parameters.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from pydantic import Field

from writeoff.calculators.common import (
    CalculationResult,
    FrozenModel,
    Workbook,
    money,
    pct,
)
from writeoff.models import EntityType
from writeoff.tax_parameters import ParameterUnavailableError, TaxParameters


class Treatment(StrEnum):
    FULLY_DEDUCTIBLE = "fully_deductible"
    PARTIALLY_DEDUCTIBLE = "partially_deductible"
    CAPITALIZE_AND_DEPRECIATE = "capitalize_and_depreciate"
    NOT_DEDUCTIBLE = "not_deductible"
    DEPENDS_ON_FACTS = "depends_on_facts"


@dataclass(frozen=True, slots=True)
class Rule:
    category: str
    keywords: tuple[str, ...]
    treatment: Treatment
    reasoning: str
    authorities: tuple[str, ...]
    questions: tuple[str, ...] = ()
    calculator: str | None = None
    entity_notes: dict[EntityType, str] = field(default_factory=dict)


_SE_HEALTH_NOTE = (
    "Deducted as the self-employed health insurance adjustment (IRC § 162(l)), not as a "
    "business expense on the business return."
)

RULES: tuple[Rule, ...] = (
    Rule(
        "fines_penalties",
        (r"\bfines?\b", r"penalt", r"parking ticket", r"speeding ticket", r"citation fee"),
        Treatment.NOT_DEDUCTIBLE,
        "Fines and penalties paid to a government for violating a law are not deductible.",
        ("IRC § 162(f)", "Treas. Reg. § 1.162-21"),
    ),
    Rule(
        "political_lobbying",
        (r"politic", r"campaign", r"lobby"),
        Treatment.NOT_DEDUCTIBLE,
        "Political contributions and lobbying expenses are not deductible.",
        ("IRC § 162(e)", "Treas. Reg. § 1.162-20"),
    ),
    Rule(
        "employee_party",
        (
            r"(holiday|company|office|staff|employee|team)\s+(party|picnic|outing|celebration)",
            r"team[- ]building",
        ),
        Treatment.FULLY_DEDUCTIBLE,
        "Recreational and social activities primarily for employees (such as a holiday party) are "
        "excepted from the entertainment and 50% meal limits.",
        ("IRC § 274(e)(4)", "Pub 463, ch. 2, 50% Limit, Exception to the 50% Limit for Meals"),
        ("Was the event mainly for employees rather than highly compensated employees or owners?",),
    ),
    Rule(
        "entertainment",
        (
            r"tickets?\b",
            r"sporting event",
            r"\bgolf",
            r"country club",
            r"club dues",
            r"concert",
            r"theater|theatre",
            r"season tickets",
            r"skybox",
            r"hunting|fishing trip",
            r"yacht",
            r"basketball|football|baseball|hockey game",
        ),
        Treatment.NOT_DEDUCTIBLE,
        "Entertainment, amusement and recreation expenses are not deductible, even with a "
        "business purpose. Food and drinks bought separately at the event may be 50% deductible.",
        ("IRC § 274(a)", "Treas. Reg. § 1.274-11(a)", "Treas. Reg. § 1.274-11(b)(1)(ii)"),
        ("Were food or beverages purchased separately from the entertainment?",),
    ),
    Rule(
        "meals",
        (
            r"\bmeals?\b",
            r"\blunch",
            r"\bdinner",
            r"breakfast",
            r"restaurant",
            r"\bcoffee",
            r"\bfood\b",
            r"catering",
            r"\bdrinks?\b",
        ),
        Treatment.PARTIALLY_DEDUCTIBLE,
        "Business meals with a client, customer or employee, or while traveling away from home, "
        "are deductible at the business meals percentage if not lavish and the taxpayer or an "
        "employee is present.",
        ("IRC § 274(n)", "Treas. Reg. § 1.274-12(a)", "Pub 463, ch. 2, 50% Limit"),
        ("Was there a business purpose and was the taxpayer (or an employee) present?",),
    ),
    Rule(
        "employee_gifts",
        (r"gifts?.{0,20}(employee|staff)", r"(employee|staff).{0,20}gifts?"),
        Treatment.FULLY_DEDUCTIBLE,
        "Gifts to employees are deductible as compensation. Small non-cash items can be excluded "
        "from wages as de minimis fringe benefits, but cash and gift cards are always wages.",
        ("IRC § 132(e)", "Pub 15-B, ch. 2, De Minimis (Minimal) Benefits", "IRC § 274(j)"),
        ("Was the gift cash or a gift card (always taxable wages)?",),
    ),
    Rule(
        "gifts",
        (r"\bgifts?\b", r"gift basket", r"gift cards?"),
        Treatment.PARTIALLY_DEDUCTIBLE,
        "Business gifts are deductible up to the per-recipient annual limit; incidental costs "
        "such as engraving, packaging and shipping don't count toward it.",
        ("IRC § 274(b)", "Pub 463, ch. 3"),
        ("How many recipients received the gifts?",),
    ),
    Rule(
        "travel",
        (
            r"airfare",
            r"\bflights?\b",
            r"\bhotel",
            r"lodging",
            r"train ticket",
            r"rental car",
            r"business trip",
            r"\btravel",
        ),
        Treatment.FULLY_DEDUCTIBLE,
        "Ordinary and necessary travel away from your tax home overnight for business is "
        "deductible (meals during travel follow the meals rule).",
        ("IRC § 162(a)", "Pub 463, ch. 1, What Travel Expenses Are Deductible?"),
        ("Was the trip away from your tax home overnight and primarily for business?",),
    ),
    Rule(
        "commuting",
        (r"commut", r"drive (to|from) (the )?(office|work)"),
        Treatment.NOT_DEDUCTIBLE,
        "Commuting between home and a regular place of business is a personal expense.",
        ("IRC § 262", "Pub 463, ch. 4"),
    ),
    Rule(
        "vehicle",
        (
            r"\bmileage",
            r"\bgas\b|gasoline|\bfuel",
            r"car (repair|insurance|wash)",
            r"oil change",
            r"\btruck\b",
            r"\bvehicle",
            r"\bcar\b",
        ),
        Treatment.DEPENDS_ON_FACTS,
        "Car and truck expenses are deductible for business use only, using the standard mileage "
        "rate or actual expenses; use calc_vehicle for the amount.",
        ("Pub 463, ch. 4, Car Expenses", "IRC § 280F"),
        (
            "What were total and business miles for the year?",
            "Standard mileage or actual expenses?",
        ),
        calculator="calc_vehicle",
    ),
    Rule(
        "home_office",
        (r"home office", r"office in (my|the) home", r"business use of (my|the) home"),
        Treatment.DEPENDS_ON_FACTS,
        "The business part of a home is deductible only if it is used regularly and exclusively "
        "as a principal place of business (or another qualifying use); use calc_home_office.",
        ("IRC § 280A(c)(1)", "Pub 587, Qualifying for a Deduction, Exclusive Use"),
        (
            "Is the space used regularly and exclusively for business?",
            "What are the square footages?",
        ),
        calculator="calc_home_office",
        entity_notes={
            EntityType.S_CORP: "Owner-employees of an S or C corporation generally "
            "recover home office costs through an accountable-plan reimbursement.",
            EntityType.C_CORP: "Owner-employees of an S or C corporation generally "
            "recover home office costs through an accountable-plan reimbursement.",
        },
    ),
    Rule(
        "improvements",
        (
            r"new roof",
            r"renovat",
            r"remodel",
            r"\baddition\b",
            r"replace(ment)? (the )?(hvac|roof)",
            r"improvement",
        ),
        Treatment.CAPITALIZE_AND_DEPRECIATE,
        "Amounts that better, restore or adapt a unit of property must be capitalized and "
        "depreciated, unless a safe harbor applies.",
        ("IRC § 263(a)", "Treas. Reg. § 1.263(a)-3(d)", "Treas. Reg. § 1.263(a)-3(h)"),
        ("Does the work better, restore, or adapt the property to a new use?",),
        calculator="calc_depreciation",
    ),
    Rule(
        "repairs",
        (r"\brepairs?\b", r"\bfix", r"maintenance", r"\bpatch"),
        Treatment.DEPENDS_ON_FACTS,
        "Routine repairs and maintenance that keep property in ordinary operating condition are "
        "deductible; work that betters, restores or adapts it must be capitalized.",
        ("Treas. Reg. § 1.162-4", "Treas. Reg. § 1.263(a)-3(i)", "Treas. Reg. § 1.263(a)-3(d)"),
        ("Does the work merely keep the property in ordinary operating condition?",),
    ),
    # Before equipment: consumables for a device ("printer paper", "toner cartridge")
    # are supplies, and ties go to the earlier rule.
    Rule(
        "supplies",
        (r"supplies", r"\bpaper\b", r"\bink\b", r"toner", r"cartridge", r"stationery", r"postage"),
        Treatment.FULLY_DEDUCTIBLE,
        "Incidental materials and supplies are deductible when paid; others when used or consumed.",
        ("Treas. Reg. § 1.162-3(a)",),
    ),
    Rule(
        "equipment",
        (
            r"laptop",
            r"computer",
            r"printer",
            r"\bdesk\b",
            r"\bchairs?\b",
            r"furniture",
            r"machinery",
            r"equipment",
            r"\bcamera",
            r"tablet",
            r"monitor",
            r"\btools?\b",
            r"server",
        ),
        Treatment.CAPITALIZE_AND_DEPRECIATE,
        "Property with a useful life beyond the year is capitalized and depreciated. It can be "
        "expensed through the de minimis safe harbor election (if within the per-item limit), "
        "section 179, or the special depreciation allowance.",
        ("IRC § 263(a)", "Treas. Reg. § 1.263(a)-1(f)", "IRC § 179", "IRC § 168(k)"),
        ("Do you elect the de minimis safe harbor, section 179, or bonus depreciation?",),
        calculator="calc_depreciation",
    ),
    Rule(
        "software_subscription",
        (r"subscription", r"\bsaas\b", r"software", r"\bapp\b"),
        Treatment.FULLY_DEDUCTIBLE,
        "Software subscriptions used in the business are ordinary and necessary expenses.",
        ("IRC § 162(a)",),
    ),
    Rule(
        "rent",
        (r"\brent\b", r"coworking", r"office lease", r"lease (of|for) (office|space)"),
        Treatment.FULLY_DEDUCTIBLE,
        "Rent for property used in the business is deductible (but not rent paid toward "
        "acquiring equity in the property).",
        ("Treas. Reg. § 1.162-11", "Pub 334, ch. 8, Rent Expense"),
    ),
    Rule(
        "utilities",
        (
            r"electric",
            r"phone bill",
            r"cell ?phone (bill|plan|service)",
            r"internet",
            r"water bill",
            r"utilities",
        ),
        Treatment.FULLY_DEDUCTIBLE,
        "Utilities used in the business are deductible; mixed personal and business use is "
        "deductible only for the business share.",
        ("IRC § 162(a)", "IRC § 262"),
    ),
    Rule(
        "health_insurance",
        (r"health insurance", r"medical insurance", r"dental insurance"),
        Treatment.DEPENDS_ON_FACTS,
        "Treatment depends on the entity and who is covered.",
        (
            "IRC § 162(l)(1)",
            "Pub 334, ch. 8, Insurance",
            "Pub 15-B, ch. 2, Accident and Health Benefits",
        ),
        entity_notes={
            EntityType.SOLE_PROP: _SE_HEALTH_NOTE,
            EntityType.PARTNERSHIP: _SE_HEALTH_NOTE,
            EntityType.S_CORP: "Premiums for a 2% shareholder are deductible by the S "
            "corporation and included in the shareholder's wages; the shareholder may then "
            "take the IRC § 162(l) adjustment.",
            EntityType.C_CORP: "Premiums paid for employees, including owner-employees, "
            "are deductible by the corporation and generally excluded from their wages.",
        },
    ),
    Rule(
        "insurance",
        (r"(?<!health )(?<!medical )(?<!dental )insurance", r"premiums?"),
        Treatment.FULLY_DEDUCTIBLE,
        "Premiums for business insurance (liability, property, malpractice, workers' "
        "compensation) are deductible.",
        ("Pub 334, ch. 8, Insurance",),
    ),
    Rule(
        "advertising",
        (r"advertis", r"marketing", r"\bads\b", r"business cards", r"website", r"\bsignage\b"),
        Treatment.FULLY_DEDUCTIBLE,
        "Advertising and promotion related to the business are ordinary and necessary expenses.",
        ("IRC § 162(a)",),
    ),
    Rule(
        "professional_fees",
        (
            r"accountant",
            r"\bcpa\b",
            r"lawyer",
            r"attorney",
            r"legal fees",
            r"bookkeep",
            r"consultant",
        ),
        Treatment.FULLY_DEDUCTIBLE,
        "Legal and professional fees for the business are deductible, unless they are part of "
        "acquiring property (then capitalized).",
        ("Pub 334, ch. 8, Legal and Professional Fees",),
    ),
    Rule(
        "wages_contractors",
        (r"contractor", r"freelanc", r"payroll", r"\bwages?\b", r"salar"),
        Treatment.FULLY_DEDUCTIBLE,
        "Reasonable pay for services actually rendered is deductible.",
        ("IRC § 162(a)", "Treas. Reg. § 1.162-7"),
        ("Is the pay reasonable for the services actually performed?",),
    ),
    Rule(
        "education",
        (
            r"\bcourse",
            r"\bclass(es)?\b",
            r"training",
            r"seminar",
            r"workshop",
            r"certification",
            r"tuition",
        ),
        Treatment.DEPENDS_ON_FACTS,
        "Education that maintains or improves skills used in your current business is deductible; "
        "education that qualifies you for a new trade or business is not.",
        ("Treas. Reg. § 1.162-5(a)", "Treas. Reg. § 1.162-5(b)"),
        ("Does it qualify you for a new trade or business?",),
    ),
    Rule(
        "startup_costs",
        (r"start-?up", r"before (the business|we|I) (opened|launched|started)", r"pre-?opening"),
        Treatment.PARTIALLY_DEDUCTIBLE,
        "Start-up costs are deductible up to the first-year limit (reduced when total start-up "
        "costs exceed the phaseout threshold); the rest is amortized over 180 months.",
        ("IRC § 195(b)", "Treas. Reg. § 1.195-1(a)"),
        ("What is the total of all start-up costs?",),
    ),
    Rule(
        "interest",
        (r"loan interest", r"credit card interest", r"interest on"),
        Treatment.FULLY_DEDUCTIBLE,
        "Interest on debt used for the business is deductible, allocated by how the loan "
        "proceeds were used.",
        ("Pub 334, ch. 8, Interest",),
    ),
    Rule(
        "bank_fees",
        (r"bank fee", r"merchant fee", r"processing fee", r"stripe|paypal|square fee"),
        Treatment.FULLY_DEDUCTIBLE,
        "Bank and payment-processing fees of the business are ordinary and necessary expenses.",
        ("IRC § 162(a)",),
    ),
    Rule(
        "dues",
        (r"membership dues", r"association dues", r"chamber of commerce", r"professional dues"),
        Treatment.FULLY_DEDUCTIBLE,
        "Dues to professional and trade associations are deductible, except the part used for "
        "lobbying; club dues are not.",
        ("Treas. Reg. § 1.162-15", "IRC § 162(e)"),
    ),
    Rule(
        "clothing",
        (r"clothing", r"clothes", r"\bsuits?\b", r"\bshoes\b", r"uniforms?"),
        Treatment.DEPENDS_ON_FACTS,
        "Clothing suitable for everyday wear is personal even if bought for work; uniforms or "
        "protective gear not suitable for street wear can be deductible.",
        ("IRC § 262", "IRC § 162(a)"),
        ("Is the clothing suitable for everyday wear outside work?",),
    ),
    Rule(
        "charitable",
        (r"donation", r"charit", r"contribution to"),
        Treatment.DEPENDS_ON_FACTS,
        "Charitable gifts are not business expenses for sole proprietors and pass-through "
        "entities; C corporations deduct them subject to the taxable income limit.",
        (
            "Instructions for Form 1120, Specific Instructions, Deductions, "
            "Line 19. Charitable Contributions",
        ),
    ),
    Rule(
        "personal",
        (r"groceries", r"\bpersonal\b", r"vacation", r"family", r"gym membership", r"haircut"),
        Treatment.NOT_DEDUCTIBLE,
        "Personal, living and family expenses are not deductible.",
        ("IRC § 262",),
    ),
)

RULES_BY_CATEGORY = {r.category: r for r in RULES}
_MIXED_USE = {"utilities", "vehicle", "software_subscription", "interest", "equipment"}


class ExpenseInput(FrozenModel):
    description: str = Field(min_length=1)
    amount: Decimal = Field(ge=0)
    business_use_pct: Decimal = Field(default=Decimal(100), ge=0, le=100)
    entity_type: EntityType
    tax_year: int
    category: str | None = None
    recipients: int | None = Field(default=None, ge=1)


class ExpenseClassification(CalculationResult):
    category: str
    treatment: Treatment
    deductible_amount: Decimal | None = None
    reasoning: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    calculator: str | None = None
    other_matching_categories: list[str] = Field(default_factory=list)


def match_rules(description: str) -> list[Rule]:
    """Rules whose keywords appear, most matches first (ties keep table order)."""
    text = description.lower()
    scored = [
        (sum(bool(re.search(k, text)) for k in rule.keywords), i, rule)
        for i, rule in enumerate(RULES)
    ]
    return [rule for hits, _, rule in sorted(scored, key=lambda t: (-t[0], t[1])) if hits]


def classify_expense(inp: ExpenseInput, params: TaxParameters) -> ExpenseClassification:
    if inp.category is not None:
        if inp.category not in RULES_BY_CATEGORY:
            raise ValueError(f"unknown category; choose from {sorted(RULES_BY_CATEGORY)}")
        rule, others = RULES_BY_CATEGORY[inp.category], []
    else:
        matches = match_rules(inp.description)
        if not matches:
            return ExpenseClassification(
                category="unknown",
                treatment=Treatment.DEPENDS_ON_FACTS,
                reasoning=["No rule matched; search the law for this expense before answering."],
                questions=["What exactly was purchased and how is it used in the business?"],
                authorities=["IRC § 162(a)", "IRC § 262"],
            )
        rule, others = matches[0], [r.category for r in matches[1:]]

    wb = Workbook(params, inp.tax_year)
    reasoning = [rule.reasoning]
    if note := rule.entity_notes.get(inp.entity_type):
        reasoning.append(note)
    share = pct(inp.business_use_pct)
    business_amount = money(inp.amount * share)
    if inp.business_use_pct < 100:
        wb.step(f"Business share: {inp.business_use_pct}% of {money(inp.amount)}", business_amount)
        if rule.category not in _MIXED_USE:
            wb.note("Only the business-use share of a mixed-use expense is deductible.")

    status, missing, deductible = "ok", [], None
    try:
        deductible = _amount(rule, inp, wb, business_amount)
    except ParameterUnavailableError as exc:
        status, missing = "unavailable", [exc.name]
        wb.note(
            f"{exc.name} for {exc.tax_year} is {exc.reason}; the deductible amount is unavailable."
        )
    return wb.finish(
        ExpenseClassification,
        list(rule.authorities),
        status=status,
        missing_parameters=missing,
        category=rule.category,
        treatment=rule.treatment,
        deductible_amount=deductible,
        reasoning=reasoning,
        questions=list(rule.questions),
        calculator=rule.calculator,
        other_matching_categories=others,
    )


def _amount(  # noqa: PLR0911 - one return per treatment kind
    rule: Rule, inp: ExpenseInput, wb: Workbook, business_amount: Decimal
) -> Decimal | None:
    if rule.treatment is Treatment.NOT_DEDUCTIBLE:
        return Decimal(0)
    if rule.treatment is Treatment.FULLY_DEDUCTIBLE:
        return business_amount
    if rule.category == "meals":
        rate = wb.param("business_meals_deduction_pct")
        amount = money(business_amount * pct(rate))
        wb.step(f"Meals at {rate}% deductible", amount)
        return amount
    if rule.category == "gifts":
        limit = wb.param("business_gift_limit_per_recipient") * (inp.recipients or 1)
        amount = money(min(business_amount, limit))
        wb.step(
            f"Gifts limited to the per-recipient limit x {inp.recipients or 1} recipient(s)", amount
        )
        if inp.recipients is None:
            wb.note("Assumed a single recipient; the limit applies per recipient per year.")
        return amount
    if rule.category == "startup_costs":
        limit = wb.param("startup_cost_immediate_deduction_limit")
        threshold = wb.param("startup_cost_phaseout_threshold")
        allowed = max(Decimal(0), limit - max(Decimal(0), inp.amount - threshold))
        amount = money(min(business_amount, allowed))
        wb.step("Start-up costs deductible in the first year", amount)
        wb.step("Remainder amortized ratably over 180 months", business_amount - amount)
        return amount
    if rule.category == "equipment":
        try:
            threshold = wb.param("de_minimis_safe_harbor_without_afs")
        except ParameterUnavailableError:
            wb.note(
                "The de minimis safe harbor limit is not verified yet; "
                "cannot say whether it applies."
            )
            return None
        if business_amount <= threshold:
            wb.note(
                "Within the de minimis safe harbor per-item limit (no applicable financial "
                "statement): deductible in full if the annual election is made."
            )
        return None
    return None
