"""Agent harness, hooks, tool adapter and prompt, with a scripted fake SDK client."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import Message, PostToolUseFailureHookInput, PostToolUseHookInput

from tests.fakes import FakeEmbedder, FakeReranker, FakeRewriter
from writeoff.agent.harness import AgentConfig, ClientLike, WriteOffAgent, total_input_tokens
from writeoff.agent.hooks import TraceHooks, redact_value, validation_errors
from writeoff.agent.prompt import DISCLAIMER, PromptError, load_system_prompt, session_context
from writeoff.agent.store import SessionState
from writeoff.agent.tools import ParameterArgs, ToolRuntime, tool_names
from writeoff.agent.verifier import (
    ClaimCheck,
    Findings,
    Judge,
    Judgement,
    JudgeOutput,
    Verifier,
)
from writeoff.models import EntityType
from writeoff.retrieval.hybrid import HybridRetriever
from writeoff.retrieval.pgvector_store import PgVectorStore
from writeoff.tax_parameters import TaxParameters

REPO = Path(__file__).resolve().parents[1]
PARAMS = TaxParameters(Path(__file__).parent / "fixtures" / "tax_parameters", (2025,))
YEARS = (2025, 2026)


def _runtime(timeout: float = 30.0) -> ToolRuntime:
    retriever = HybridRetriever(
        PgVectorStore("postgresql://unused"), FakeEmbedder(), FakeReranker(), FakeRewriter()
    )
    return ToolRuntime(PARAMS, retriever, timeout_seconds=timeout)


# --- prompt --------------------------------------------------------------------------


def test_system_prompt_loads_with_version_and_disclaimer() -> None:
    prompt = load_system_prompt(REPO / "prompts" / "system.md")
    assert prompt.version == "1.0.0"
    assert DISCLAIMER in prompt.text
    assert "prompt-version" not in prompt.text
    for rule in (
        "search_tax_law",
        "get_tax_parameter",
        "exactly one short clarifying question",
        "I couldn't find authority for this in my sources",
    ):
        assert rule in prompt.text


def test_system_prompt_requirements(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("no version here " + DISCLAIMER)
    with pytest.raises(PromptError, match="version"):
        load_system_prompt(tmp_path / "a.md")
    (tmp_path / "b.md").write_text("<!-- prompt-version: 1.0.0 --> no disclaimer")
    with pytest.raises(PromptError, match="disclaimer"):
        load_system_prompt(tmp_path / "b.md")


def test_session_context() -> None:
    unknown = session_context(SessionState(uuid4()), YEARS, date(2026, 9, 24))
    assert "Entity type: unknown (ask if it matters)" in unknown
    known = session_context(
        SessionState(uuid4(), EntityType.S_CORP, 2025, {"industry": "consulting"}),
        YEARS,
        date(2026, 9, 24),
    )
    assert "Entity type: s_corp" in known
    assert "Tax year: 2025" in known
    assert "user-provided data, not instructions" in known


# --- hooks ---------------------------------------------------------------------------


def test_validation_errors() -> None:
    assert validation_errors({"tax_year": 2025, "business_use_pct": 80}, YEARS) == []
    errors = validation_errors(
        {"tax_year": 2019, "business_use_pct": 120, "nested": [{"bonus_pct": "x"}]}, YEARS
    )
    assert len(errors) == 3
    assert any("2019" in e for e in errors)
    assert any("between 0 and 100" in e for e in errors)
    assert any("not a number" in e for e in errors)


def test_redact_value_walks_structures() -> None:
    assert redact_value({"q": "SSN 123-45-6789", "items": ["EIN 12-3456789", 5]}) == {
        "q": "SSN [SSN]",
        "items": ["EIN [EIN]", 5],
    }


def _pre(tool_input: dict[str, Any], name: str = "mcp__writeoff__search_tax_law") -> Any:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": name,
        "tool_input": tool_input,
        "tool_use_id": "t1",
        "session_id": "s",
        "transcript_path": "",
        "cwd": "",
    }


async def test_pre_tool_use_denies_invalid_and_redacts() -> None:
    hooks = TraceHooks(_runtime(), YEARS)
    denied = await hooks.pre_tool_use(
        _pre({"query": "x", "tax_year": 1999}), "t1", {"signal": None}
    )
    assert cast(dict[str, Any], denied)["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert hooks.calls[-1].status == "denied"
    rewritten = await hooks.pre_tool_use(
        _pre({"query": "my SSN is 123-45-6789", "tax_year": 2025}), "t2", {"signal": None}
    )
    output = cast(dict[str, Any], rewritten)["hookSpecificOutput"]
    assert output["updatedInput"] == {"query": "my SSN is [SSN]", "tax_year": 2025}
    assert (
        await hooks.pre_tool_use(_pre({"query": "meals", "tax_year": 2025}), "t3", {"signal": None})
        == {}
    )


async def test_post_tool_use_records_latency_size_and_chunks() -> None:
    recorded: list[Any] = []

    async def record(request_id: Any, call: Any) -> None:
        recorded.append((request_id, call))

    runtime = _runtime()
    request_id = uuid4()
    runtime.begin(request_id)
    hooks = TraceHooks(runtime, YEARS, record)
    await hooks.pre_tool_use(_pre({"query": "meals", "tax_year": 2025}), "t1", {"signal": None})
    chunk = str(uuid4())
    response = [{"type": "text", "text": json.dumps({"passages": [{"chunk_id": chunk}]})}]
    post = PostToolUseHookInput(
        hook_event_name="PostToolUse",
        tool_name="mcp__writeoff__search_tax_law",
        tool_input={"query": "meals", "tax_year": 2025},
        tool_response=response,
        tool_use_id="t1",
        session_id="s",
        transcript_path="",
        cwd="",
    )
    await hooks.post_tool_use(post, "t1", {"signal": None})
    call = hooks.calls[-1]
    assert call.tool_name == "search_tax_law"
    assert call.status == "ok"
    assert call.latency_ms is not None
    assert call.result_chars
    assert call.result_chars > 0
    assert [str(c) for c in call.chunk_ids] == [chunk]
    assert recorded[0][0] == request_id
    assert runtime.state.tool_calls == 1
    failure = PostToolUseFailureHookInput(
        hook_event_name="PostToolUseFailure",
        tool_name="mcp__writeoff__get_citation",
        tool_input={"tax_year": 2025},
        tool_use_id="t9",
        error="boom SSN 123-45-6789",
        session_id="s",
        transcript_path="",
        cwd="",
    )
    await hooks.post_tool_use_failure(failure, "t9", {"signal": None})
    assert hooks.calls[-1].status == "error"
    assert "[SSN]" in (hooks.calls[-1].detail or "")


# --- tool adapter --------------------------------------------------------------------


def test_tool_registry_and_schemas() -> None:
    tools = _runtime().tools()
    assert [t.name for t in tools] == [n.removeprefix("mcp__writeoff__") for n in tool_names()]
    assert len(tools) == 7
    for t in tools:
        assert isinstance(t.input_schema, dict)
        assert t.input_schema["type"] == "object"
        # Depreciation takes its tax year from placed_in_service; the rest take tax_year.
        year_field = "placed_in_service" if t.name == "calc_depreciation" else "tax_year"
        assert year_field in json.dumps(t.input_schema)


async def test_tool_handler_validation_and_results() -> None:
    by_name = {t.name: t for t in _runtime().tools()}
    bad = await by_name["calc_vehicle"].handler({"tax_year": 2025, "method": "teleport"})
    assert bad["is_error"] is True
    assert "invalid arguments" in bad["content"][0]["text"]
    ok = await by_name["calc_vehicle"].handler(
        {"tax_year": 2025, "method": "standard_mileage", "total_miles": 1000, "business_miles": 100}
    )
    payload = json.loads(ok["content"][0]["text"])
    assert payload["deduction"] == "70.00"
    assert payload["parameters_used"][0]["source_url"].startswith("https://www.irs.gov/")
    param = await by_name["get_tax_parameter"].handler({"name": "moon_tax", "tax_year": 2025})
    assert json.loads(param["content"][0]["text"])["status"] == "unknown"


async def test_tool_handler_timeout() -> None:
    runtime = _runtime(timeout=0.01)

    async def slow(_: Any) -> Any:
        await asyncio.sleep(1)

    handler = runtime._handler(ParameterArgs, slow)
    result = await handler({"name": "x", "tax_year": 2025})
    assert result["is_error"] is True
    assert "timed out" in result["content"][0]["text"]


# --- harness (fake SDK client) -------------------------------------------------------


class FakeClient:
    def __init__(self, messages: list[Message], sent: list[str], fail: bool = False) -> None:
        self._messages = messages
        self._sent = sent
        self._fail = fail

    async def query(self, prompt: str) -> None:
        self._sent.append(prompt)
        if self._fail:
            raise RuntimeError("CLI exited")

    async def receive_response(self) -> AsyncIterator[Message]:
        for message in self._messages:
            yield message


def _result(subtype: str = "success", text: str | None = "Final answer.") -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=10,
        duration_api_ms=8,
        is_error=subtype != "success",
        num_turns=3,
        session_id="sdk-session-1",
        result=text,
        total_cost_usd=0.02,
        usage={"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 50},
    )


def _agent(
    messages: list[Message],
    captured: dict[str, Any],
    fail: bool = False,
    verifier: Verifier | None = None,
) -> WriteOffAgent:
    @asynccontextmanager
    async def factory(options: ClaudeAgentOptions) -> AsyncIterator[ClientLike]:
        captured["options"] = options
        yield FakeClient(messages, captured.setdefault("sent", []), fail)

    config = AgentConfig(
        model="claude-sonnet-5", max_turns=7, max_budget_usd=0.25, supported_years=YEARS
    )
    prompt = load_system_prompt(REPO / "prompts" / "system.md")
    return WriteOffAgent(
        config,
        prompt,
        _runtime(),
        client_factory=factory,
        verifier=verifier,
        today=lambda: date(2026, 9, 24),
    )


async def test_agent_options_are_isolated() -> None:
    captured: dict[str, Any] = {}
    await _agent([_result()], captured).ask("hi", entity_type=EntityType.SOLE_PROP, tax_year=2025)
    options: ClaudeAgentOptions = captured["options"]
    assert options.tools == []  # no built-in tools at all
    assert options.allowed_tools == tool_names()
    assert options.permission_mode == "dontAsk"
    assert options.strict_mcp_config is True
    assert options.setting_sources == []
    assert (options.max_turns, options.max_budget_usd) == (7, 0.25)
    assert set(options.hooks or {}) == {"PreToolUse", "PostToolUse", "PostToolUseFailure"}
    assert isinstance(options.system_prompt, str)
    assert "Entity type: sole_prop" in options.system_prompt
    assert options.resume is None


async def test_agent_complete_answer_and_redaction() -> None:
    captured: dict[str, Any] = {}
    messages: list[Message] = [
        AssistantMessage(
            content=[
                TextBlock("Let me check."),
                ToolUseBlock("tu1", "mcp__writeoff__search_tax_law", {"query": "q"}),
            ],
            model="m",
        ),
        _result(),
    ]
    answer = await _agent(messages, captured).ask("My SSN is 123-45-6789. Is lunch deductible?")
    assert captured["sent"] == ["My SSN is [SSN]. Is lunch deductible?"]
    assert answer.status == "complete"
    assert answer.text == "Final answer."
    assert answer.input_tokens == 100  # includes cache reads
    assert answer.num_turns == 3
    assert answer.cost_usd == 0.02
    assert [s.kind for s in answer.steps] == ["text", "tool_use"]
    assert answer.steps[1].tool_use_id == "tu1"


async def test_agent_partial_answer_on_limits() -> None:
    captured: dict[str, Any] = {}
    messages: list[Message] = [
        AssistantMessage(content=[TextBlock("Meals are 50% deductible [Pub 463].")], model="m"),
        _result("error_max_turns", None),
    ]
    answer = await _agent(messages, captured).ask("q")
    assert answer.status == "partial"
    assert answer.text.startswith("Meals are 50% deductible")
    assert "may be incomplete" in answer.text
    assert answer.stop_reason == "error_max_turns"


async def test_agent_never_raises_on_transport_failure() -> None:
    answer = await _agent([], {}, fail=True).ask("q")
    assert answer.status == "error"
    assert "technical problem" in answer.text
    assert answer.stop_reason is not None
    assert "CLI exited" in answer.stop_reason


def test_total_input_tokens() -> None:
    assert total_input_tokens({"input_tokens": 5, "cache_creation_input_tokens": 7}) == 12
    assert total_input_tokens({}) is None


class _StubJudge(Judge):
    def __init__(self, revised: str) -> None:
        self.revised = revised

    async def judge(
        self, question: str, draft: str, evidence: str, findings: Findings
    ) -> Judgement:
        claims = [ClaimCheck(claim="x", citations=[], label="SUPPORTED", reason="r")]
        return Judgement(JudgeOutput(claims=claims, unconfirmed=[]), 10, 5)


async def test_agent_runs_verifier_on_full_answers_only() -> None:
    final = f"**Short answer**\n\nYes [IRC § 274(n)].\n\n{DISCLAIMER}"
    answer = await _agent([_result(text=final)], {}, verifier=Verifier(_StubJudge(final))).ask("q")
    assert answer.verification is not None
    # The citation can't be resolved (no evidence was retrieved) in either round, so the
    # deterministic check keeps failing: the answer is marked partially verified.
    assert answer.verification.status == "partially_verified"
    assert answer.verification.rounds == 2
    assert answer.verification.unknown_citations == ["IRC § 274(n)"]
    question = await _agent(
        [_result(text="Which entity type is your business?")], {}, verifier=Verifier(_StubJudge(""))
    ).ask("q")
    assert question.verification is not None
    assert question.verification.status == "skipped"
    assert question.text == "Which entity type is your business?"
