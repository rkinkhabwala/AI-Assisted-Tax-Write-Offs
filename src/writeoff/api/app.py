"""FastAPI application (spec sections 4, 8, 9).

Endpoints (all under /v1 need `Authorization: Bearer <API_TOKEN>` when API_TOKEN is set):

    POST  /v1/sessions               create a session (entity type, tax year, profile)
    GET   /v1/sessions/{id}          read it
    PATCH /v1/sessions/{id}          update it
    POST  /v1/chat                   ask; returns the verified answer with its sources
    POST  /v1/chat/stream            same, as server-sent events: progress, then answer
    POST  /v1/expenses/csv           classify an uploaded expense CSV (cells are data only)
    GET   /healthz, /readyz          liveness and database readiness

Streaming sends activity ("Searching the tax law…"), not draft text: the draft can still
change when the verifier checks it, and unverified claims should never reach the user.
"""

import asyncio
import contextlib
import hmac
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Annotated, Any, Literal, Protocol
from uuid import UUID, uuid4

import httpx
import psycopg
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi import status as http
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from writeoff import __doc__ as package_doc
from writeoff.agent.factory import build_agent
from writeoff.agent.harness import AgentAnswer, AgentEvent, CitedSource, redact_profile
from writeoff.agent.store import AgentStore, SessionState
from writeoff.config import Settings, get_settings
from writeoff.expenses import (
    CSVImportError,
    CSVInjectionError,
    RowResult,
    classify_rows,
    parse_expense_csv,
)
from writeoff.models import EntityType
from writeoff.retention import purge
from writeoff.tax_parameters import TaxParameters

logger = logging.getLogger(__name__)
KEEPALIVE_SECONDS = 15.0
CSV_PATH = "/v1/expenses/csv"


class _SkipCSVUploads(logging.Filter):
    """Spec section 9: uploads that try prompt injection are refused and not logged. The
    access log would still record that one happened, so upload requests are left out of
    it altogether (success or refusal; the lines never held content)."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        return not (isinstance(args, tuple) and len(args) > 2 and str(args[2]) == CSV_PATH)


_ACCESS_FILTER = _SkipCSVUploads()


class Asker(Protocol):
    async def ask(
        self,
        question: str,
        *,
        session_id: UUID | None = None,
        entity_type: EntityType | None = None,
        tax_year: int | None = None,
        business_profile: dict[str, Any] | None = None,
        on_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> AgentAnswer: ...


AgentFactory = Callable[[], Asker]


class SessionStore(Protocol):
    async def find_session(self, session_id: UUID) -> SessionState | None: ...
    async def save_session(self, state: SessionState) -> None: ...


# --- request and response models ---------------------------------------------------------


class Health(BaseModel):
    status: Literal["ok", "unavailable"]


class SessionIn(BaseModel):
    entity_type: EntityType | None = None
    tax_year: int | None = None
    business_profile: dict[str, str | int | float | bool] = Field(default_factory=dict)


class SessionOut(BaseModel):
    session_id: UUID
    entity_type: EntityType | None
    tax_year: int | None
    business_profile: dict[str, Any]


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    session_id: UUID | None = None
    entity_type: EntityType | None = None
    tax_year: int | None = None


class VerificationOut(BaseModel):
    status: str
    supported: int
    partially_supported: int
    unsupported: int
    unconfirmed: list[str]


class UsageOut(BaseModel):
    turns: int | None
    tool_calls: int
    latency_ms: int
    cost_usd: float | None


class AnswerOut(BaseModel):
    session_id: UUID
    request_id: UUID
    status: str
    answer: str
    sources: list[CitedSource]
    verification: VerificationOut | None
    usage: UsageOut
    prompt_version: str

    @classmethod
    def of(cls, a: AgentAnswer) -> "AnswerOut":
        v = a.verification
        return cls(
            session_id=a.session_id,
            request_id=a.request_id,
            status=a.status,
            answer=a.text,
            sources=a.sources,
            verification=None
            if v is None
            else VerificationOut(
                status=v.status,
                supported=v.count("SUPPORTED"),
                partially_supported=v.count("PARTIALLY_SUPPORTED"),
                unsupported=v.count("UNSUPPORTED"),
                unconfirmed=v.unconfirmed,
            ),
            usage=UsageOut(
                turns=a.num_turns,
                tool_calls=len(a.tool_calls),
                latency_ms=a.latency_ms,
                cost_usd=a.cost_usd,
            ),
            prompt_version=a.prompt_version,
        )


class ExpensesOut(BaseModel):
    entity_type: EntityType
    tax_year: int
    rows: list[RowResult]
    total_amount: Decimal
    total_deductible: Decimal
    rows_needing_review: int


# --- app ---------------------------------------------------------------------------------


class AppState:
    def __init__(
        self,
        settings: Settings,
        agent_factory: AgentFactory,
        sessions: SessionStore,
        params: TaxParameters,
    ) -> None:
        self.settings = settings
        self.agent_factory = agent_factory
        self.sessions = sessions
        self.params = params
        self.answer_slots = asyncio.Semaphore(settings.max_concurrent_answers)


def _state(request: Request) -> AppState:
    state: AppState = request.app.state.writeoff
    return state


StateDep = Annotated[AppState, Depends(_state)]


def require_token(request: Request, state: StateDep) -> None:
    token = state.settings.api_token
    if token is None:
        return
    header = request.headers.get("authorization", "")
    supplied = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
    if not hmac.compare_digest(supplied.encode(), token.get_secret_value().encode()):
        raise HTTPException(
            http.HTTP_401_UNAUTHORIZED,
            "missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _check_year(state: AppState, year: int | None) -> None:
    supported = state.settings.supported_tax_years
    if year is not None and year not in supported:
        raise HTTPException(
            http.HTTP_422_UNPROCESSABLE_CONTENT,
            f"tax year {year} isn't supported; choose one of {list(supported)}",
        )


async def _retention_loop(store: AgentStore, settings: Settings) -> None:
    while True:
        try:
            await purge(store, settings.trace_retention_days, settings.session_retention_days)
        except psycopg.Error:
            logger.warning("retention run failed; will retry next interval", exc_info=True)
        await asyncio.sleep(settings.retention_interval_hours * 3600)


def create_app(
    settings: Settings | None = None,
    *,
    agent_factory: AgentFactory | None = None,
    sessions: SessionStore | None = None,
    run_retention: bool = True,
) -> FastAPI:
    """The application. Tests pass fakes for the agent and the session store."""
    settings = settings or get_settings()
    access_log = logging.getLogger("uvicorn.access")
    if _ACCESS_FILTER not in access_log.filters:
        access_log.addFilter(_ACCESS_FILTER)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = AgentStore(str(settings.database_url))
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            app.state.writeoff = AppState(
                settings,
                agent_factory or (lambda: build_agent(settings, client)),
                sessions or store,
                TaxParameters(settings.tax_parameters_dir, settings.supported_tax_years),
            )
            if settings.api_token is None:
                logger.warning("API_TOKEN is not set: the API is open to anyone who can reach it")
            retention = None
            if run_retention:
                retention = asyncio.create_task(_retention_loop(store, settings))
            try:
                yield
            finally:
                if retention is not None:
                    retention.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await retention

    app = FastAPI(title="WriteOff Assistant", description=package_doc or "", lifespan=lifespan)
    _routes(app, settings)
    return app


def _routes(app: FastAPI, settings: Settings) -> None:  # noqa: PLR0915 - route table
    auth = [Depends(require_token)]

    @app.get("/healthz")
    async def healthz() -> Health:
        """Liveness: the process is up."""
        return Health(status="ok")

    @app.get("/readyz")
    async def readyz(response: Response) -> Health:
        """Readiness: the database is reachable."""
        try:
            async with await psycopg.AsyncConnection.connect(
                str(settings.database_url), connect_timeout=3
            ) as conn:
                await conn.execute("SELECT 1")
        except psycopg.Error:
            response.status_code = http.HTTP_503_SERVICE_UNAVAILABLE
            return Health(status="unavailable")
        return Health(status="ok")

    def session_out(s: SessionState) -> SessionOut:
        return SessionOut(
            session_id=s.session_id,
            entity_type=s.entity_type,
            tax_year=s.tax_year,
            business_profile=s.business_profile,
        )

    async def existing(state: AppState, session_id: UUID) -> SessionState:
        found = await state.sessions.find_session(session_id)
        if found is None:
            raise HTTPException(http.HTTP_404_NOT_FOUND, "session not found")
        return found

    @app.post("/v1/sessions", dependencies=auth, status_code=http.HTTP_201_CREATED)
    async def create_session(body: SessionIn, state: StateDep) -> SessionOut:
        _check_year(state, body.tax_year)
        session = SessionState(
            session_id=uuid4(),
            entity_type=body.entity_type,
            tax_year=body.tax_year,
            business_profile=redact_profile(dict(body.business_profile)),
        )
        await state.sessions.save_session(session)
        return session_out(session)

    @app.get("/v1/sessions/{session_id}", dependencies=auth)
    async def get_session(session_id: UUID, state: StateDep) -> SessionOut:
        return session_out(await existing(state, session_id))

    @app.patch("/v1/sessions/{session_id}", dependencies=auth)
    async def update_session(session_id: UUID, body: SessionIn, state: StateDep) -> SessionOut:
        _check_year(state, body.tax_year)
        session = await existing(state, session_id)
        fields = body.model_fields_set
        if "entity_type" in fields:
            session.entity_type = body.entity_type
        if "tax_year" in fields:
            session.tax_year = body.tax_year
        if "business_profile" in fields:
            session.business_profile = {
                **session.business_profile,
                **redact_profile(dict(body.business_profile)),
            }
        await state.sessions.save_session(session)
        return session_out(session)

    async def prepare(state: AppState, body: ChatIn) -> Asker:
        _check_year(state, body.tax_year)
        if body.session_id is not None:
            await existing(state, body.session_id)
        try:
            return state.agent_factory()
        except RuntimeError as exc:  # missing API keys
            raise HTTPException(http.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    @app.post("/v1/chat", dependencies=auth)
    async def chat(body: ChatIn, state: StateDep) -> AnswerOut:
        agent = await prepare(state, body)
        async with state.answer_slots:
            answer = await agent.ask(
                body.question,
                session_id=body.session_id,
                entity_type=body.entity_type,
                tax_year=body.tax_year,
            )
        return AnswerOut.of(answer)

    @app.post("/v1/chat/stream", dependencies=auth)
    async def chat_stream(body: ChatIn, state: StateDep) -> StreamingResponse:
        agent = await prepare(state, body)
        return StreamingResponse(
            _answer_events(state, agent, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post(CSV_PATH, dependencies=auth)
    async def expenses_csv(
        state: StateDep,
        file: Annotated[UploadFile, File(description="CSV with description and amount columns")],
        entity_type: Annotated[EntityType, Form()],
        tax_year: Annotated[int, Form()],
    ) -> ExpensesOut:
        # Nothing about the upload is logged or stored: not on success, not on refusal.
        _check_year(state, tax_year)
        data = await file.read()
        try:
            rows = parse_expense_csv(data)
        except CSVInjectionError as exc:
            raise HTTPException(
                http.HTTP_422_UNPROCESSABLE_CONTENT, {"refused": True, "message": str(exc)}
            ) from None
        except CSVImportError as exc:
            raise HTTPException(
                http.HTTP_400_BAD_REQUEST, {"refused": False, "message": str(exc)}
            ) from None
        results = classify_rows(rows, state.params, entity_type, tax_year)
        return ExpensesOut(
            entity_type=entity_type,
            tax_year=tax_year,
            rows=results,
            total_amount=sum((r.amount for r in results), Decimal(0)),
            total_deductible=sum(
                (r.deductible_amount for r in results if r.deductible_amount is not None),
                Decimal(0),
            ),
            rows_needing_review=sum(r.deductible_amount is None for r in results),
        )


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _answer_events(state: AppState, agent: Asker, body: ChatIn) -> AsyncGenerator[str, None]:
    """Progress events while the agent works, keep-alives during quiet stretches (the
    verifier can take 20 seconds), then the answer. If the client goes away the run is
    cancelled so it stops spending."""
    queue: asyncio.Queue[tuple[str, object] | None] = asyncio.Queue()

    async def on_event(event: AgentEvent) -> None:
        await queue.put(("progress", event.model_dump()))

    async def run() -> None:
        try:
            if state.answer_slots.locked():
                await queue.put(("progress", {"kind": "queued", "message": "Waiting for a slot"}))
            async with state.answer_slots:
                answer = await agent.ask(
                    body.question,
                    session_id=body.session_id,
                    entity_type=body.entity_type,
                    tax_year=body.tax_year,
                    on_event=on_event,
                )
            await queue.put(("answer", AnswerOut.of(answer).model_dump(mode="json")))
        except Exception:  # the harness shouldn't raise; if it does, tell the client
            logger.exception("streamed answer failed")
            await queue.put(("error", {"message": "The answer failed. Please try again."}))
        finally:
            await queue.put(None)

    task = asyncio.create_task(run())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), KEEPALIVE_SECONDS)
            except TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                break
            yield _sse(*item)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


app = create_app()
