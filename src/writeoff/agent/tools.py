"""Expose the phase 5 tools to the Agent SDK as an in-process MCP server.

Input schemas are generated from the same pydantic models the tools validate with, so
the schema Claude sees cannot drift from the code. Every handler validates its
arguments, runs under a timeout, and returns errors as readable `is_error` results rather
than raising, so the agent can recover (spec section 5 guardrails).

`ToolRuntime` also carries per-question state: it enforces the weak-retrieval retry
limit (spec section 5: "rewrite the query and retry, max 2 retries").
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from claude_agent_sdk import SdkMcpTool, ToolAnnotations, create_sdk_mcp_server, tool
from claude_agent_sdk.types import McpSdkServerConfig
from pydantic import BaseModel, Field, ValidationError

from writeoff.agent.evidence import Evidence, SourceDocument
from writeoff.calculators.common import CalculationResult, FrozenModel
from writeoff.calculators.depreciation import DepreciationInput, calc_depreciation
from writeoff.calculators.home_office import HomeOfficeInput, calc_home_office
from writeoff.calculators.vehicle import VehicleInput, calc_vehicle
from writeoff.models import DocType, EntityType
from writeoff.retrieval.hybrid import HybridRetriever
from writeoff.tax_parameters import TaxParameters
from writeoff.tools.classify import ExpenseInput, classify_expense
from writeoff.tools.law import (
    CitationToolResult,
    ParameterToolResult,
    SearchToolResult,
    get_citation,
    get_tax_parameter,
    search_tax_law,
)

SERVER_NAME = "writeoff"
RETRY_LIMIT_GUIDANCE = (
    "Retry limit reached for weak searches. Answer with only what the retrieved passages "
    'support, and say "I couldn\'t find authority for this in my sources" for the rest.'
)


class SearchArgs(FrozenModel):
    query: str = Field(min_length=3, description="What to look up, in tax terms where possible")
    tax_year: int
    entity_type: EntityType | None = None
    doc_types: list[DocType] = Field(default_factory=list)


class CitationArgs(FrozenModel):
    citation: str = Field(
        min_length=2, description='e.g. "§ 274(n)(1)", "Reg. 1.162-5", or a stored citation'
    )
    tax_year: int


class ParameterArgs(FrozenModel):
    name: str = Field(description="Parameter name, e.g. standard_mileage_rate_business")
    tax_year: int


@dataclass(slots=True)
class RequestState:
    request_id: UUID
    weak_searches: int = 0
    tool_calls: int = 0
    started: dict[str, float] = field(default_factory=dict)
    evidence: Evidence = field(default_factory=Evidence)


def tool_names() -> list[str]:
    return [f"mcp__{SERVER_NAME}__{name}" for name in _DESCRIPTIONS]


_DESCRIPTIONS: dict[str, str] = {
    "search_tax_law": (
        "Hybrid search over the IRC, Treasury Regulations, IRS publications and form "
        "instructions for one tax year. Returns ranked passages (each with a `citation` to cite "
        "verbatim) and their full parent `sections` text. `weak: true` means no strong "
        "authority was found."
    ),
    "get_citation": (
        "Exact text of a provision by citation, e.g. '§ 280A(c)(1)', 'Reg. 1.263(a)-1(f)', or a "
        "`citation` value returned by search. Returns the enclosing section when the provision "
        "is stored inside a larger passage."
    ),
    "get_tax_parameter": (
        "A dollar limit, rate or percentage for a tax year (e.g. section_179_dollar_limit, "
        "standard_mileage_rate_business, business_meals_deduction_pct) with its irs.gov source. "
        "status 'unavailable' means it is not verified yet: say so, never estimate. Unknown "
        "names return the list of known parameters."
    ),
    "classify_expense": (
        "First-pass treatment of an expense (fully/partially deductible, capitalize and "
        "depreciate, not deductible, or depends on facts), with the deductible amount when a "
        "sourced limit applies, the authorities to check, and follow-up questions. Confirm "
        "with retrieved law."
    ),
    "calc_depreciation": (
        "Depreciation for one asset: section 179, special allowance (by acquisition date), "
        "MACRS (GDS/ADS, half-year/mid-quarter/mid-month) and passenger-auto caps, with a full "
        "schedule and every parameter's source."
    ),
    "calc_home_office": (
        "Home office deduction by the simplified or regular method, applying the gross income "
        "limit and carryovers when given."
    ),
    "calc_vehicle": (
        "Car or truck deduction by the standard mileage rate or actual expenses and "
        "business-use percentage."
    ),
}

_LARGE = ToolAnnotations(readOnlyHint=True, maxResultSizeChars=400_000)
_SMALL = ToolAnnotations(readOnlyHint=True)


def _text(result: BaseModel) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": result.model_dump_json(exclude_none=True)}]}


def _error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Error: {message}"}], "is_error": True}


class ToolRuntime:
    def __init__(
        self,
        params: TaxParameters,
        retriever: HybridRetriever,
        *,
        timeout_seconds: float = 30.0,
        max_weak_retries: int = 2,
    ) -> None:
        self._params = params
        self._retriever = retriever
        self._timeout = timeout_seconds
        self._max_weak = max_weak_retries
        self.state = RequestState(request_id=UUID(int=0))

    def begin(self, request_id: UUID) -> RequestState:
        self.state = RequestState(request_id=request_id)
        return self.state

    def server(self) -> McpSdkServerConfig:
        return create_sdk_mcp_server(name=SERVER_NAME, version="1.0.0", tools=self.tools())

    def tools(self) -> list[SdkMcpTool[Any]]:
        specs: list[
            tuple[str, type[FrozenModel], Callable[[Any], Awaitable[BaseModel]], ToolAnnotations]
        ] = [
            ("search_tax_law", SearchArgs, self._search, _LARGE),
            ("get_citation", CitationArgs, self._citation, _LARGE),
            ("get_tax_parameter", ParameterArgs, self._parameter, _SMALL),
            ("classify_expense", ExpenseInput, self._classify, _SMALL),
            ("calc_depreciation", DepreciationInput, self._depreciation, _SMALL),
            ("calc_home_office", HomeOfficeInput, self._home_office, _SMALL),
            ("calc_vehicle", VehicleInput, self._vehicle, _SMALL),
        ]
        return [
            tool(name, _DESCRIPTIONS[name], model.model_json_schema(), annotations=notes)(
                self._handler(model, fn)
            )
            for name, model, fn, notes in specs
        ]

    def _handler(
        self, model: type[FrozenModel], fn: Callable[[Any], Awaitable[BaseModel]]
    ) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
        async def handle(args: dict[str, Any]) -> dict[str, Any]:
            try:
                parsed = model.model_validate(args)
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                )
                return _error(f"invalid arguments ({problems})")
            try:
                result = await asyncio.wait_for(fn(parsed), timeout=self._timeout)
            except TimeoutError:
                return _error(
                    "the tool timed out; answer with what you have and say what is missing"
                )
            except ValueError as exc:
                return _error(str(exc))
            return _text(result)

        return handle

    async def _search(self, args: SearchArgs) -> BaseModel:
        result = await search_tax_law(
            self._retriever, args.query, args.tax_year, args.entity_type, frozenset(args.doc_types)
        )
        if result.weak:
            self.state.weak_searches += 1
            if self.state.weak_searches > self._max_weak:
                result = result.model_copy(update={"guidance": RETRY_LIMIT_GUIDANCE})
        self._record_search(result)
        return result

    async def _citation(self, args: CitationArgs) -> BaseModel:
        result = await get_citation(self._retriever, args.citation, args.tax_year)
        self._record_citation(result)
        return result

    async def _parameter(self, args: ParameterArgs) -> BaseModel:
        result = get_tax_parameter(self._params, args.name, args.tax_year)
        self._record_parameter(result)
        return result

    async def _classify(self, args: ExpenseInput) -> BaseModel:
        return self._record_calculation("classify_expense", classify_expense(args, self._params))

    async def _depreciation(self, args: DepreciationInput) -> BaseModel:
        return self._record_calculation("calc_depreciation", calc_depreciation(args, self._params))

    async def _home_office(self, args: HomeOfficeInput) -> BaseModel:
        return self._record_calculation("calc_home_office", calc_home_office(args, self._params))

    async def _vehicle(self, args: VehicleInput) -> BaseModel:
        return self._record_calculation("calc_vehicle", calc_vehicle(args, self._params))

    # --- evidence capture (what the verifier checks against) --------------------------

    def _record_search(self, result: SearchToolResult) -> None:
        evidence = self.state.evidence
        by_section: dict[str, SourceDocument] = {}
        for passage in result.passages:
            source = SourceDocument(passage.title, passage.source_url)
            evidence.add_passage(passage.citation, passage.text, source)
            if passage.section_id:
                by_section.setdefault(passage.section_id, source)
        for section_id, section in result.sections.items():
            evidence.add_passage(section.citation, section.text, by_section.get(section_id))

    def _record_citation(self, result: CitationToolResult) -> None:
        for passage in result.passages:
            self.state.evidence.add_passage(
                passage.citation, passage.text, SourceDocument(passage.title, passage.source_url)
            )

    def _record_parameter(self, result: ParameterToolResult) -> None:
        if result.status == "ok":
            value = result.value if result.value is not None else result.periods
            self.state.evidence.add_parameter(
                result.name,
                f"{value} {result.unit} for {result.tax_year} (source: {result.source_url})",
            )

    def _record_calculation[R: CalculationResult](self, tool_name: str, result: R) -> R:
        self.state.evidence.add_calculation(tool_name, result.model_dump_json(exclude_none=True))
        for used in result.parameters_used:
            self.state.evidence.add_parameter(
                used.name, f"{used.value} {used.unit} (source: {used.source_url})"
            )
        return result
