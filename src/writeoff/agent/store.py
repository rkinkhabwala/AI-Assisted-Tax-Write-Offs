"""Postgres persistence for agent sessions and traces (migration 0002).

Sessions keep the facts that persist across questions (entity type, tax year, business
profile) and the Agent SDK session id used to resume the conversation. Traces record one
row per question and one per tool call. They hold redacted text only (spec section 9).
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from writeoff.models import EntityType

if TYPE_CHECKING:  # the verifier imports the prompt module, which imports this one
    from writeoff.agent.verifier import VerificationReport


@dataclass(slots=True)
class SessionState:
    session_id: UUID
    entity_type: EntityType | None = None
    tax_year: int | None = None
    business_profile: dict[str, Any] = field(default_factory=dict)
    sdk_session_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    tool_use_id: str
    tool_name: str
    input_redacted: dict[str, Any]
    status: str  # ok | error | denied
    latency_ms: int | None
    result_chars: int | None
    chunk_ids: tuple[UUID, ...] = ()
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RequestRecord:
    request_id: UUID
    session_id: UUID
    question_redacted: str
    prompt_version: str
    model: str
    status: str
    stop_reason: str | None
    num_turns: int | None
    tool_calls: int
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: Decimal | None
    latency_ms: int | None
    created_at: datetime


class AgentStore:
    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo

    async def _connect(self) -> psycopg.AsyncConnection[dict[str, Any]]:
        return await psycopg.AsyncConnection.connect(self._conninfo, row_factory=dict_row)

    # --- sessions --------------------------------------------------------------------

    async def load_session(self, session_id: UUID) -> SessionState:
        async with await self._connect() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM agent_sessions WHERE session_id = %s", (session_id,)
                )
            ).fetchone()
        if row is None:
            return SessionState(session_id=session_id)
        return SessionState(
            session_id=session_id,
            entity_type=EntityType(row["entity_type"]) if row["entity_type"] else None,
            tax_year=row["tax_year"],
            business_profile=row["business_profile"] or {},
            sdk_session_id=row["sdk_session_id"],
        )

    async def save_session(self, state: SessionState) -> None:
        async with await self._connect() as conn:
            await conn.execute(
                """INSERT INTO agent_sessions (session_id, entity_type, tax_year, business_profile,
                                              sdk_session_id)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (session_id) DO UPDATE SET
                       entity_type = EXCLUDED.entity_type, tax_year = EXCLUDED.tax_year,
                       business_profile = EXCLUDED.business_profile,
                       sdk_session_id = EXCLUDED.sdk_session_id, updated_at = now()""",
                (
                    state.session_id,
                    state.entity_type.value if state.entity_type else None,
                    state.tax_year,
                    Jsonb(state.business_profile),
                    state.sdk_session_id,
                ),
            )

    # --- traces ----------------------------------------------------------------------

    async def start_request(
        self,
        request_id: UUID,
        session_id: UUID,
        question_redacted: str,
        prompt_version: str,
        model: str,
    ) -> None:
        async with await self._connect() as conn:
            await conn.execute(
                """INSERT INTO agent_requests (request_id, session_id, question_redacted,
                                              prompt_version, model, status)
                   VALUES (%s, %s, %s, %s, %s, 'running')""",
                (request_id, session_id, question_redacted, prompt_version, model),
            )

    async def record_tool_call(self, request_id: UUID, call: ToolCallRecord) -> None:
        async with await self._connect() as conn:
            await conn.execute(
                """INSERT INTO agent_tool_calls (request_id, tool_use_id, tool_name, input_redacted,
                                                status, latency_ms, result_chars, chunk_ids, detail)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    request_id,
                    call.tool_use_id,
                    call.tool_name,
                    Jsonb(json.loads(json.dumps(call.input_redacted, default=str))),
                    call.status,
                    call.latency_ms,
                    call.result_chars,
                    list(call.chunk_ids),
                    call.detail,
                ),
            )

    async def finish_request(
        self,
        request_id: UUID,
        *,
        status: str,
        stop_reason: str | None,
        num_turns: int | None,
        tool_calls: int,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float | None,
        latency_ms: int,
    ) -> None:
        async with await self._connect() as conn:
            await conn.execute(
                """UPDATE agent_requests SET status = %s, stop_reason = %s, num_turns = %s,
                          tool_calls = %s, input_tokens = %s, output_tokens = %s, cost_usd = %s,
                          latency_ms = %s, finished_at = now()
                   WHERE request_id = %s""",
                (
                    status,
                    stop_reason,
                    num_turns,
                    tool_calls,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    latency_ms,
                    request_id,
                ),
            )

    async def record_verification(self, request_id: UUID, report: "VerificationReport") -> None:
        async with await self._connect() as conn:
            await conn.execute(
                """UPDATE agent_requests SET verification_status = %s, verification_rounds = %s,
                          claims_supported = %s, claims_partial = %s, claims_unsupported = %s,
                          unknown_citations = %s, untraced_numbers = %s,
                          verifier_input_tokens = %s, verifier_output_tokens = %s
                   WHERE request_id = %s""",
                (
                    report.status,
                    report.rounds,
                    report.count("SUPPORTED"),
                    report.count("PARTIALLY_SUPPORTED"),
                    report.count("UNSUPPORTED"),
                    report.unknown_citations,
                    report.untraced_numbers,
                    report.input_tokens,
                    report.output_tokens,
                    request_id,
                ),
            )

    async def tool_calls(self, request_id: UUID) -> list[ToolCallRecord]:
        async with await self._connect() as conn:
            rows = await (
                await conn.execute(
                    "SELECT * FROM agent_tool_calls WHERE request_id = %s ORDER BY id",
                    (request_id,),
                )
            ).fetchall()
        return [
            ToolCallRecord(
                r["tool_use_id"],
                r["tool_name"],
                r["input_redacted"],
                r["status"],
                r["latency_ms"],
                r["result_chars"],
                tuple(r["chunk_ids"]),
                r["detail"],
            )
            for r in rows
        ]

    async def request(self, request_id: UUID) -> RequestRecord | None:
        async with await self._connect() as conn:
            r = await (
                await conn.execute(
                    "SELECT * FROM agent_requests WHERE request_id = %s", (request_id,)
                )
            ).fetchone()
        if r is None:
            return None
        return RequestRecord(
            r["request_id"],
            r["session_id"],
            r["question_redacted"],
            r["prompt_version"],
            r["model"],
            r["status"],
            r["stop_reason"],
            r["num_turns"],
            r["tool_calls"],
            r["input_tokens"],
            r["output_tokens"],
            r["cost_usd"],
            r["latency_ms"],
            r["created_at"],
        )

    async def purge_traces(self, older_than_days: int) -> int:
        """Retention (spec section 9): delete requests (and their tool calls) past the window."""
        async with await self._connect() as conn:
            rows = await (
                await conn.execute(
                    "DELETE FROM agent_requests "
                    "WHERE created_at < now() - make_interval(days => %s) "
                    "RETURNING request_id",
                    (older_than_days,),
                )
            ).fetchall()
        return len(rows)
