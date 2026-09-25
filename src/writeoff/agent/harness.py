"""The WriteOff agent: Claude Agent SDK harness, guardrails and trace (spec sections 4-5).

Isolation: `tools=[]` removes every built-in tool (no Bash, files or web); only our
in-process MCP tools are available and pre-approved; `permission_mode="dontAsk"` denies
anything else; `strict_mcp_config` and `setting_sources=[]` keep local Claude Code
configuration out.

Authentication: the API key is passed to the Claude Code subprocess explicitly (`env`).
Otherwise the CLI falls back to whatever login exists on the machine, such as a personal
claude.ai subscription, which Anthropic doesn't allow for products built on the SDK.

Guardrails: `max_turns` and `max_budget_usd` bound the loop, and every tool call has a
timeout (ToolRuntime). Hitting a limit returns a partial answer that says so. It never
raises to the caller.
"""

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from datetime import date
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import Message
from pydantic import BaseModel, Field, SecretStr

from writeoff.agent.evidence import Evidence
from writeoff.agent.hooks import TraceHooks
from writeoff.agent.prompt import SystemPrompt, session_context
from writeoff.agent.store import AgentStore, SessionState, ToolCallRecord
from writeoff.agent.tools import SERVER_NAME, ToolRuntime, tool_names
from writeoff.agent.verifier import VerificationReport, Verifier, cited_citations
from writeoff.models import EntityType
from writeoff.privacy import redact

logger = logging.getLogger(__name__)
AnswerStatus = Literal["complete", "partial", "error"]
_LIMIT_NOTES = {
    "error_max_turns": "the maximum number of reasoning steps",
    "error_max_budget_usd": "the cost budget for one question",
}


class ClientLike(Protocol):
    async def query(self, prompt: str) -> None: ...
    def receive_response(self) -> AsyncIterator[Message]: ...


ClientFactory = Callable[[ClaudeAgentOptions], AbstractAsyncContextManager[ClientLike]]


def default_client(options: ClaudeAgentOptions) -> AbstractAsyncContextManager[ClientLike]:
    return ClaudeSDKClient(options=options)


class TraceStep(BaseModel):
    kind: Literal["text", "tool_use"]
    detail: str
    tool_input: dict[str, Any] | None = None
    tool_use_id: str | None = None


class CitedSource(BaseModel):
    """A passage the final answer cites, for the UI's Sources panel."""

    citation: str
    title: str | None = None
    url: str | None = None
    text: str


class AgentEvent(BaseModel):
    """Progress while an answer is prepared. Only activity is streamed, never draft text:
    the draft may still change when the verifier checks it."""

    kind: Literal["tool", "verifying"]
    message: str
    tool: str | None = None


EventHandler = Callable[[AgentEvent], Awaitable[None]]


def describe_tool(name: str, tool_input: dict[str, Any]) -> str:
    short = name.removeprefix(f"mcp__{SERVER_NAME}__")
    if short == "search_tax_law":
        return f"Searching the tax law: {tool_input.get('query', '')}".rstrip(": ")
    if short == "get_citation":
        return f"Reading {tool_input.get('citation', 'a provision')}"
    if short == "get_tax_parameter":
        return f"Looking up {tool_input.get('name', 'a tax parameter')}"
    if short == "classify_expense":
        return "Classifying the expense"
    if short.startswith("calc_"):
        return f"Running the {short.removeprefix('calc_').replace('_', ' ')} calculator"
    return f"Using {short}"


def cited_sources(text: str, evidence: Evidence) -> list[CitedSource]:
    """The retrieved passages the answer cites. A passage whose text is already inside a
    cited enclosing section is left out, so the panel doesn't repeat itself."""
    wanted = cited_citations(text, evidence)
    joined = {c: "\n\n".join(evidence.passages[c]) for c in wanted}
    sources = []
    for citation in wanted:
        body = joined[citation]
        if any(
            other != citation
            and citation.startswith((other + "(", other + ", "))
            and all(t in joined[other] for t in evidence.passages[citation])
            for other in wanted
        ):
            continue
        doc = evidence.documents.get(citation)
        sources.append(
            CitedSource(
                citation=citation,
                title=doc.title if doc else None,
                url=doc.url if doc else None,
                text=body,
            )
        )
    return sources


class AgentAnswer(BaseModel):
    request_id: UUID
    session_id: UUID
    status: AnswerStatus
    text: str
    stop_reason: str | None = None
    num_turns: int | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    steps: list[TraceStep] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int = 0
    prompt_version: str = ""
    draft: str | None = None  # the agent's text before verification edits
    retrieved_citations: list[str] = Field(default_factory=list)
    sources: list[CitedSource] = Field(default_factory=list)
    verification: VerificationReport | None = None


class AgentConfig(BaseModel):
    model: str
    max_turns: int = Field(default=12, ge=1)
    max_budget_usd: float = Field(default=0.5, gt=0)
    supported_years: tuple[int, ...]
    # Required in production (see factory); optional so tests can use a fake client.
    api_key: SecretStr | None = None


class WriteOffAgent:
    def __init__(
        self,
        config: AgentConfig,
        prompt: SystemPrompt,
        runtime: ToolRuntime,
        *,
        store: AgentStore | None = None,
        verifier: Verifier | None = None,
        client_factory: ClientFactory = default_client,
        today: Callable[[], date] = date.today,
        record_traces: bool = True,
    ) -> None:
        self._config = config
        self._prompt = prompt
        self._runtime = runtime
        self._store = store
        self._verifier = verifier
        self._client_factory = client_factory
        self._today = today
        # Sessions are always kept (they carry the conversation); traces only if enabled.
        self._traces = store if record_traces else None

    def options(self, state: SessionState, hooks: TraceHooks) -> ClaudeAgentOptions:
        context = session_context(state, self._config.supported_years, self._today())
        system = f"{self._prompt.text}\n\n{context}"
        return ClaudeAgentOptions(
            system_prompt=system,
            model=self._config.model,
            tools=[],
            allowed_tools=tool_names(),
            mcp_servers={SERVER_NAME: self._runtime.server()},
            strict_mcp_config=True,
            setting_sources=[],
            permission_mode="dontAsk",
            max_turns=self._config.max_turns,
            max_budget_usd=self._config.max_budget_usd,
            hooks=hooks.matchers(),
            resume=state.sdk_session_id,
            env=(
                {"ANTHROPIC_API_KEY": self._config.api_key.get_secret_value()}
                if self._config.api_key
                else {}
            ),
        )

    async def ask(
        self,
        question: str,
        *,
        session_id: UUID | None = None,
        entity_type: EntityType | None = None,
        tax_year: int | None = None,
        business_profile: dict[str, Any] | None = None,
        on_event: EventHandler | None = None,
    ) -> AgentAnswer:
        started = time.monotonic()
        session_id = session_id or uuid4()
        state = (
            await self._store.load_session(session_id) if self._store else SessionState(session_id)
        )
        state.entity_type = entity_type or state.entity_type
        state.tax_year = tax_year or state.tax_year
        if business_profile:
            state.business_profile = {**state.business_profile, **redact_profile(business_profile)}
        request_id = uuid4()
        self._runtime.begin(request_id)
        question_redacted = redact(question)
        if self._store:
            await self._store.save_session(state)
        if self._traces:
            await self._traces.start_request(
                request_id, session_id, question_redacted, self._prompt.version, self._config.model
            )
        hooks = TraceHooks(
            self._runtime,
            self._config.supported_years,
            self._traces.record_tool_call if self._traces else None,
        )
        run = _Run()
        try:
            async with self._client_factory(self.options(state, hooks)) as client:
                await client.query(question_redacted)
                async for message in client.receive_response():
                    run.add(message)
                    if on_event is not None and isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                await _emit(
                                    on_event,
                                    AgentEvent(
                                        kind="tool",
                                        message=describe_tool(block.name, block.input),
                                        tool=block.name,
                                    ),
                                )
        except Exception as exc:  # the SDK surfaces CLI/transport failures as varied types
            run.failure = f"{type(exc).__name__}: {exc}"
        steps, texts, result, failure = run.steps, run.texts, run.result, run.failure

        status, text, stop = _outcome(result, texts, failure)
        draft, report = text, None
        if self._verifier is not None and status in {"complete", "partial"}:
            if on_event is not None:
                await _emit(
                    on_event,
                    AgentEvent(kind="verifying", message="Checking the answer against the sources"),
                )
            text, report = await self._verifier.verify(
                draft, question_redacted, self._runtime.state.evidence
            )
        if result is not None and result.session_id:
            state.sdk_session_id = result.session_id
        usage = (result.usage or {}) if result else {}
        answer = AgentAnswer(
            request_id=request_id,
            session_id=session_id,
            status=status,
            text=text,
            stop_reason=stop,
            num_turns=result.num_turns if result else None,
            tool_calls=hooks.calls,
            steps=steps,
            input_tokens=total_input_tokens(usage),
            output_tokens=usage.get("output_tokens"),
            cost_usd=result.total_cost_usd if result else None,
            latency_ms=int((time.monotonic() - started) * 1000),
            prompt_version=self._prompt.version,
            draft=draft if report and draft != text else None,
            verification=report,
            retrieved_citations=self._runtime.state.evidence.citations,
            sources=cited_sources(text, self._runtime.state.evidence),
        )
        await self._finish(state, answer, report)
        return answer

    async def _finish(
        self, state: SessionState, answer: AgentAnswer, report: VerificationReport | None
    ) -> None:
        if self._store:
            await self._store.save_session(state)
        if self._traces:
            await self._traces.finish_request(
                answer.request_id,
                status=answer.status,
                stop_reason=answer.stop_reason,
                num_turns=answer.num_turns,
                tool_calls=len(answer.tool_calls),
                input_tokens=answer.input_tokens,
                output_tokens=answer.output_tokens,
                cost_usd=answer.cost_usd,
                latency_ms=answer.latency_ms,
            )
            if report is not None:
                await self._traces.record_verification(answer.request_id, report)


class _Run:
    """Messages collected from one agent run."""

    def __init__(self) -> None:
        self.steps: list[TraceStep] = []
        self.texts: list[str] = []
        self.result: ResultMessage | None = None
        self.failure: str | None = None

    def add(self, message: Message) -> None:
        if isinstance(message, ResultMessage):
            self.result = message
            return
        if not isinstance(message, AssistantMessage):
            return
        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                self.texts.append(block.text)
                self.steps.append(TraceStep(kind="text", detail=block.text))
            elif isinstance(block, ToolUseBlock):
                self.steps.append(
                    TraceStep(
                        kind="tool_use",
                        detail=block.name,
                        tool_input=block.input,
                        tool_use_id=block.id,
                    )
                )


async def _emit(handler: EventHandler, event: AgentEvent) -> None:
    """Progress is best effort: a failing listener (e.g. a closed stream) never breaks
    the answer."""
    try:
        await handler(event)
    except Exception:  # the listener is caller code of any kind
        logger.warning("progress listener failed", exc_info=True)


def total_input_tokens(usage: dict[str, Any]) -> int | None:
    """Input tokens including prompt-cache reads and writes, which the API reports apart."""
    keys = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    values = [usage[k] for k in keys if isinstance(usage.get(k), int)]
    return sum(values) if values else None


def redact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {k: redact(v) if isinstance(v, str) else v for k, v in profile.items()}


def _outcome(
    result: ResultMessage | None, texts: list[str], failure: str | None
) -> tuple[AnswerStatus, str, str | None]:
    final = texts[-1].strip() if texts else ""
    if failure is not None or result is None:
        reason = failure or "the agent ended without a result"
        message = (
            "I couldn't complete this answer because of a technical problem. Please try again."
        )
        return "error", (f"{final}\n\n{message}" if final else message), reason
    if result.subtype == "success" and not result.is_error:
        return "complete", (result.result or final).strip(), result.stop_reason or "end_turn"
    limit = _LIMIT_NOTES.get(result.subtype)
    if limit is not None:
        note = (
            f"_I reached {limit} before finishing, so this answer may be incomplete. "
            "Anything not supported by a citation above could not be confirmed._"
        )
        body = final or "I couldn't finish researching this question."
        return "partial", f"{body}\n\n{note}", result.subtype
    return "error", final or "I couldn't complete this answer. Please try again.", result.subtype
