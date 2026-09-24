"""Agent SDK hooks (spec section 4).

PreToolUse: denies calls whose tax_year is outside the supported years or whose
percentages fall outside 0-100, and rewrites SSN/EIN patterns out of any string argument
(`updatedInput`), so identifiers never reach tools, traces, or later model turns.

PostToolUse / PostToolUseFailure: record the tool name, latency, result size, retrieved
chunk ids and status for the request's trace.
"""

import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from claude_agent_sdk import HookMatcher
from claude_agent_sdk.types import HookContext, HookEvent, HookInput, HookJSONOutput

from writeoff.agent.store import ToolCallRecord
from writeoff.agent.tools import SERVER_NAME, ToolRuntime
from writeoff.privacy import redact

_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_TOOL_PREFIX = f"mcp__{SERVER_NAME}__"

Recorder = Callable[[UUID, ToolCallRecord], Awaitable[None]]


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value


def validation_errors(tool_input: dict[str, Any], supported_years: tuple[int, ...]) -> list[str]:
    errors: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                where = f"{path}.{key}" if path else key
                if key == "tax_year" and item not in supported_years:
                    errors.append(
                        f"{where}={item!r} is not a supported tax year {list(supported_years)}"
                    )
                if key.endswith("_pct") and item is not None:
                    try:
                        number = float(item)
                    except (TypeError, ValueError):
                        errors.append(f"{where}={item!r} is not a number")
                    else:
                        if not 0 <= number <= 100:
                            errors.append(f"{where}={item!r} must be between 0 and 100")
                walk(item, where)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, f"{path}[{i}]")

    walk(tool_input, "")
    return errors


def _response_text(response: Any) -> tuple[str, bool]:
    """Flatten an MCP tool response to text; report whether it was an error result."""
    if isinstance(response, str):
        return response, False
    is_error = isinstance(response, dict) and bool(
        response.get("is_error") or response.get("isError")
    )
    parts: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("text"), str):
                parts.append(value["text"])
            for item in value.values():
                if isinstance(item, (dict, list)):
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(response)
    return "\n".join(parts), is_error


def _chunk_ids(text: str) -> tuple[UUID, ...]:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return ()
    found: list[UUID] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            chunk_id = value.get("chunk_id")
            if isinstance(chunk_id, str) and _UUID.fullmatch(chunk_id):
                found.append(UUID(chunk_id))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    return tuple(dict.fromkeys(found))


class TraceHooks:
    def __init__(
        self, runtime: ToolRuntime, supported_years: tuple[int, ...], record: Recorder | None = None
    ) -> None:
        self._runtime = runtime
        self._years = supported_years
        self._record = record
        self.calls: list[ToolCallRecord] = []

    def matchers(self) -> dict[HookEvent, list[HookMatcher]]:
        match = f"{_TOOL_PREFIX}.*"
        return {
            "PreToolUse": [HookMatcher(matcher=match, hooks=[self.pre_tool_use])],
            "PostToolUse": [HookMatcher(matcher=match, hooks=[self.post_tool_use])],
            "PostToolUseFailure": [HookMatcher(matcher=match, hooks=[self.post_tool_use_failure])],
        }

    async def _save(self, call: ToolCallRecord) -> None:
        self.calls.append(call)
        self._runtime.state.tool_calls += 1
        if self._record is not None:
            await self._record(self._runtime.state.request_id, call)

    async def pre_tool_use(
        self, data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        tool_input: dict[str, Any] = dict(data.get("tool_input", {}))  # type: ignore[call-overload]
        tool_name = str(data.get("tool_name", ""))
        use_id = tool_use_id or str(data.get("tool_use_id", ""))
        clean = redact_value(tool_input)
        errors = validation_errors(clean, self._years)
        if errors:
            await self._save(
                ToolCallRecord(
                    use_id,
                    tool_name.removeprefix(_TOOL_PREFIX),
                    clean,
                    "denied",
                    None,
                    None,
                    (),
                    "; ".join(errors),
                )
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Invalid arguments: " + "; ".join(errors),
                }
            }
        self._runtime.state.started[use_id] = time.monotonic()
        if clean != tool_input:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "updatedInput": clean,
                }
            }
        return {}

    def _latency(self, use_id: str) -> int | None:
        started = self._runtime.state.started.pop(use_id, None)
        return None if started is None else int((time.monotonic() - started) * 1000)

    async def post_tool_use(
        self, data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        use_id = tool_use_id or str(data.get("tool_use_id", ""))
        text, is_error = _response_text(data.get("tool_response"))
        await self._save(
            ToolCallRecord(
                use_id,
                str(data.get("tool_name", "")).removeprefix(_TOOL_PREFIX),
                redact_value(dict(data.get("tool_input", {}))),  # type: ignore[call-overload]
                "error" if is_error else "ok",
                self._latency(use_id),
                len(text),
                _chunk_ids(text),
                redact(text[:300]) if is_error else None,
            )
        )
        return {}

    async def post_tool_use_failure(
        self, data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        use_id = tool_use_id or str(data.get("tool_use_id", ""))
        await self._save(
            ToolCallRecord(
                use_id,
                str(data.get("tool_name", "")).removeprefix(_TOOL_PREFIX),
                redact_value(dict(data.get("tool_input", {}))),  # type: ignore[call-overload]
                "error",
                self._latency(use_id),
                None,
                (),
                redact(str(data.get("error", ""))[:300]),
            )
        )
        return {}
