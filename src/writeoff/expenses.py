"""CSV expense import (spec section 9).

Cells are data, never instructions. An upload is screened before anything else happens:
if any cell (or header) reads like an instruction to an AI system, the whole upload is
refused and nothing about it is logged or stored. Clean rows are classified by the
deterministic rules in `tools.classify` (no model sees the cell text), which gives a
first-pass treatment the user can then ask the agent about.
"""

import csv
import io
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, Field

from writeoff.models import EntityType
from writeoff.tax_parameters import TaxParameters
from writeoff.tools.classify import ExpenseInput, Treatment, classify_expense

MAX_BYTES = 1_000_000
MAX_ROWS = 500

# Column names accepted for each field, in order of preference (lower-cased, trimmed).
DESCRIPTION_COLUMNS = ("description", "memo", "item", "details", "merchant", "vendor", "payee")
AMOUNT_COLUMNS = ("amount", "cost", "total", "price", "debit", "value")
BUSINESS_USE_COLUMNS = ("business_use_pct", "business_use", "business %", "business_pct")

_INJECTION = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore\b.{0,40}\b(instructions?|prompts?|rules|above)\b",
        r"\bdisregard\b.{0,40}\b(instructions?|prompts?|rules|above)\b",
        r"\b(system|developer)\s+(prompt|message|note|instructions?)\b",
        r"^\s*system\s*:",
        r"\byou are now\b",
        r"\bnew (task|instructions?)\b",
        r"</?\s*(data|system|instructions?)\s*>",
        r"\b(reveal|print|show|repeat)\b.{0,30}\b(system prompt|instructions|your prompt)\b",
        r"\bnote to (the )?(ai|assistant|model|llm)\b",
        # Addressed to the assistant with an instruction verb later in the same sentence.
        r"\b(assistant|ai|model)\s*[,:][^.\n]{0,160}\b(ignore|confirm|say|reply|respond|"
        r"treat|classify|print|reveal|mark|approve|output)\b",
        r"\bjailbreak\b|\bdeveloper mode\b",
    )
]


class CSVImportError(ValueError):
    """The file isn't a usable expense CSV (the message is safe to show the user)."""


class CSVInjectionError(ValueError):
    """Cells contain instruction-like text; the upload is refused."""

    def __init__(self, cells: list[str]) -> None:
        self.cells = cells  # positions such as "row 3, column memo"; never the content
        super().__init__(
            "This file was not processed: some cells contain text addressed to an AI "
            f"system ({', '.join(cells[:5])}). Spreadsheet cells are treated strictly as "
            "data. Remove that text and upload again."
        )


@dataclass(frozen=True, slots=True)
class ExpenseRow:
    line: int  # 1-based line in the file, header = 1
    description: str
    amount: Decimal
    business_use_pct: Decimal


class RowResult(BaseModel):
    line: int
    description: str
    amount: Decimal
    business_use_pct: Decimal
    category: str
    treatment: Treatment
    deductible_amount: Decimal | None = None
    questions: list[str] = Field(default_factory=list)
    authorities: list[str] = Field(default_factory=list)
    note: str | None = None


def _pick(columns: dict[str, int], names: tuple[str, ...]) -> int | None:
    for name in names:
        if name in columns:
            return columns[name]
    return None


def _decimal(raw: str) -> Decimal:
    text = raw.strip().replace("$", "").replace(",", "")
    negative = text.startswith("(") and text.endswith(")")
    try:
        value = Decimal(text.strip("()"))
    except InvalidOperation as exc:
        raise ValueError(raw) from exc
    return -value if negative else value


def screen_for_injection(table: list[list[str]], header: list[str]) -> list[str]:
    """Positions of cells that read like instructions to an AI system."""
    flagged = [
        f"header column {i + 1}"
        for i, name in enumerate(header)
        if any(p.search(name) for p in _INJECTION)
    ]
    for r, row in enumerate(table, start=2):
        for c, cell in enumerate(row):
            if any(p.search(cell) for p in _INJECTION):
                column = header[c] if c < len(header) and header[c] else f"#{c + 1}"
                flagged.append(f"row {r}, column {column}")
    return flagged


def _parse_row(
    line: int, cells: list[str], desc_col: int, amount_col: int, use_col: int | None
) -> ExpenseRow | str:
    """The row, or a problem description."""
    description = cells[desc_col].strip()
    raw_use = cells[use_col].strip() if use_col is not None else ""
    try:
        amount = _decimal(cells[amount_col])
        use = _decimal(raw_use) if raw_use else Decimal(100)
    except ValueError:
        return f"row {line}: amount or business-use % isn't a number"
    if not description:
        return f"row {line}: no description"
    if amount < 0:
        return f"row {line}: negative amount (refunds aren't expenses)"
    if not Decimal(0) <= use <= Decimal(100):
        return f"row {line}: business-use % must be 0-100"
    return ExpenseRow(line, description, amount, use)


def parse_expense_csv(data: bytes) -> list[ExpenseRow]:
    """Screen, then parse. Raises CSVInjectionError before any other processing."""
    if len(data) > MAX_BYTES:
        raise CSVImportError(f"The file is larger than {MAX_BYTES // 1_000_000} MB.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CSVImportError("The file isn't UTF-8 text. Export it as CSV (UTF-8).") from exc
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        raise CSVImportError("The file is empty.")
    header, body = rows[0], rows[1:]
    if flagged := screen_for_injection(body, header):
        raise CSVInjectionError(flagged)
    if len(body) > MAX_ROWS:
        raise CSVImportError(f"The file has more than {MAX_ROWS} rows.")
    columns = {name.strip().lower(): i for i, name in enumerate(header)}
    desc_col, amount_col = _pick(columns, DESCRIPTION_COLUMNS), _pick(columns, AMOUNT_COLUMNS)
    use_col = _pick(columns, BUSINESS_USE_COLUMNS)
    if desc_col is None or amount_col is None:
        raise CSVImportError(
            "The header needs a description column (description, memo, item, merchant, "
            "vendor or payee) and an amount column (amount, cost, total, price or debit)."
        )
    parsed: list[ExpenseRow] = []
    problems: list[str] = []
    for line, row in enumerate(body, start=2):
        cells = row + [""] * (len(header) - len(row))
        result = _parse_row(line, cells, desc_col, amount_col, use_col)
        if isinstance(result, str):
            problems.append(result)
        else:
            parsed.append(result)
    if problems:
        raise CSVImportError("; ".join(problems[:10]))
    return parsed


def classify_rows(
    rows: list[ExpenseRow], params: TaxParameters, entity_type: EntityType, tax_year: int
) -> list[RowResult]:
    results = []
    for row in rows:
        r = classify_expense(
            ExpenseInput(
                description=row.description,
                amount=row.amount,
                business_use_pct=row.business_use_pct,
                entity_type=entity_type,
                tax_year=tax_year,
            ),
            params,
        )
        note = None
        if r.missing_parameters:
            note = "A tax parameter for this year isn't available yet: " + ", ".join(
                r.missing_parameters
            )
        results.append(
            RowResult(
                line=row.line,
                description=row.description,
                amount=row.amount,
                business_use_pct=row.business_use_pct,
                category=r.category,
                treatment=r.treatment,
                deductible_amount=r.deductible_amount,
                questions=r.questions,
                authorities=r.authorities,
                note=note,
            )
        )
    return results
