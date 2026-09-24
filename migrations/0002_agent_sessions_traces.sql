-- 0002: per-session agent state and per-request traces (spec sections 4 and 9).
--
-- Traces hold no raw user text: questions are stored after SSN/EIN redaction, tool inputs
-- likewise. Rows older than TRACE_RETENTION_DAYS are purged by the retention job.

CREATE TABLE agent_sessions (
    session_id        uuid PRIMARY KEY,
    entity_type       text CHECK (entity_type IN ('sole_prop', 'partnership', 's_corp', 'c_corp')),
    tax_year          smallint CHECK (tax_year BETWEEN 2000 AND 2100),
    business_profile  jsonb       NOT NULL DEFAULT '{}'::jsonb,
    sdk_session_id    text,                     -- Claude Agent SDK session, for resuming
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE agent_requests (
    request_id          uuid PRIMARY KEY,
    session_id          uuid        NOT NULL REFERENCES agent_sessions (session_id) ON DELETE CASCADE,
    question_redacted   text        NOT NULL,
    prompt_version      text        NOT NULL,
    model               text        NOT NULL,
    status              text        NOT NULL
        CHECK (status IN ('running', 'complete', 'partial', 'error')),
    stop_reason         text,
    num_turns           integer,
    tool_calls          integer     NOT NULL DEFAULT 0,
    input_tokens        integer,
    output_tokens       integer,
    cost_usd            numeric(10, 6),
    latency_ms          integer,
    created_at          timestamptz NOT NULL DEFAULT now(),
    finished_at         timestamptz
);

CREATE TABLE agent_tool_calls (
    id              bigserial PRIMARY KEY,
    request_id      uuid        NOT NULL REFERENCES agent_requests (request_id) ON DELETE CASCADE,
    tool_use_id     text        NOT NULL,
    tool_name       text        NOT NULL,
    input_redacted  jsonb       NOT NULL,
    status          text        NOT NULL CHECK (status IN ('ok', 'error', 'denied')),
    latency_ms      integer,
    result_chars    integer,
    chunk_ids       uuid[]      NOT NULL DEFAULT '{}',
    detail          text,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX agent_requests_session_idx ON agent_requests (session_id, created_at);
CREATE INDEX agent_requests_created_idx ON agent_requests (created_at);
CREATE INDEX agent_tool_calls_request_idx ON agent_tool_calls (request_id);
