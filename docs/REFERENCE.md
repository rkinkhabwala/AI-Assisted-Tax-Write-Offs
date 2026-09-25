# WriteOff Assistant: interface reference

Every interface in the system, in one place: the HTTP API, the tools the agent can call
(and the MCP server that exposes them), the hooks around those tools, the external
services and models, configuration, and the command-line entry points.

For how a request moves through these pieces, see the README's "Web app and API" section
and `src/writeoff/agent/harness.py`.

| Layer | Who calls it | Where |
|---|---|---|
| [HTTP API](#1-http-api) | the Streamlit UI, or any HTTP client | `src/writeoff/api/app.py` |
| [MCP tools](#3-mcp-server-and-tools) | the model, inside the Claude Code subprocess | `src/writeoff/agent/tools.py` |
| [Hooks](#4-hooks) | the Agent SDK, around every tool call | `src/writeoff/agent/hooks.py` |
| [External services](#5-external-services-and-models) | our code | Anthropic, Voyage AI, irs.gov / uscode / eCFR |
| [Configuration](#7-configuration) | everything | `src/writeoff/config.py`, `.env` |
| [Command line](#8-command-line) | you | `Makefile`, `python -m writeoff.…` |

---

## 1. HTTP API

Base URL: `http://localhost:8000` (interactive docs at `/docs`, OpenAPI JSON at `/openapi.json`).

**Authentication.** When `API_TOKEN` is set, every `/v1/*` endpoint requires
`Authorization: Bearer <API_TOKEN>` (constant-time comparison). When it's unset the API is
open and logs a warning at startup; use that only on your own machine. `/healthz` and
`/readyz` never need a token.

**Errors.** FastAPI's standard `{"detail": ...}` body.

| Status | When |
|---|---|
| 400 | CSV file unusable (`detail: {"refused": false, "message": ...}`) |
| 401 | missing or wrong bearer token |
| 404 | unknown `session_id` |
| 422 | invalid body; unsupported `tax_year`; CSV refused for injected instructions (`detail: {"refused": true, "message": ...}`) |
| 503 | `ANTHROPIC_API_KEY` or `VOYAGE_API_KEY` not configured |

### Enumerations used in requests

- `entity_type`: `sole_prop` (also single-member LLCs) · `partnership` · `s_corp` · `c_corp`
- `tax_year`: one of `SUPPORTED_TAX_YEARS` (default `2025`, `2026`)

---

### `GET /healthz`
Liveness. Always `200 {"status": "ok"}` while the process is up.

### `GET /readyz`
Readiness: can the API reach Postgres? `200 {"status": "ok"}`, or `503 {"status": "unavailable"}`.

---

### `POST /v1/sessions` → 201
Create a session. A session holds the facts that persist across questions, plus the
Agent SDK conversation (so follow-up questions keep context).

Request (all fields optional):
```json
{
  "entity_type": "s_corp",
  "tax_year": 2025,
  "business_profile": {"industry": "consulting", "employees": 3}
}
```
`business_profile` values may be strings, numbers or booleans. SSN and EIN patterns in
string values are redacted before storage.

Response (`SessionOut`):
```json
{"session_id": "uuid", "entity_type": "s_corp", "tax_year": 2025,
 "business_profile": {"industry": "consulting", "employees": 3}}
```

### `GET /v1/sessions/{session_id}` → 200
Returns `SessionOut`, or 404.

### `PATCH /v1/sessions/{session_id}` → 200
Same body as create. Only the fields you send change. `business_profile` is merged into
the existing profile, not replaced. Returns `SessionOut`.

---

### `POST /v1/chat` → 200
Ask one question and wait for the verified answer (typically 30–60 s).

Request (`ChatIn`):

| Field | Type | Notes |
|---|---|---|
| `question` | string, 1–8000 chars | required |
| `session_id` | uuid | optional; omit to start a new session (its id comes back in the response) |
| `entity_type` | enum | optional; overrides and updates the session's value |
| `tax_year` | int | optional; overrides and updates the session's value |

Response (`AnswerOut`):

| Field | Type | Meaning |
|---|---|---|
| `session_id` | uuid | pass it back to continue the conversation |
| `request_id` | uuid | key for tracing (`agent_requests`, `agent_tool_calls`) |
| `status` | `complete` \| `partial` \| `error` | `partial`: hit the turn or cost cap; `error`: the agent couldn't run |
| `answer` | markdown | the verified answer with inline `[citation]` tags and the disclaimer |
| `sources` | list | every passage the answer cites: `{citation, title, url, text}` |
| `verification` | object \| null | `{status, supported, partially_supported, unsupported, unconfirmed[]}`; status is `verified`, `revised`, `partially_verified`, `skipped` (clarifying question) or `error` |
| `usage` | object | `{turns, tool_calls, latency_ms, cost_usd}`. `cost_usd` covers the agent loop only, not the verifier (about +50%) |
| `prompt_version` | string | system prompt version that produced the answer |

```bash
curl -s localhost:8000/v1/chat -H "Authorization: Bearer $API_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"question": "How much of a $180 client dinner can I deduct?", "entity_type": "sole_prop", "tax_year": 2025}'
```

### `POST /v1/chat/stream` → 200 `text/event-stream`
Same request as `/v1/chat`. The response is a stream of server-sent events:

```
event: progress
data: {"kind": "tool", "message": "Searching the tax law: business meal 50% limit", "tool": "mcp__writeoff__search_tax_law"}

event: progress
data: {"kind": "verifying", "message": "Checking the answer against the sources", "tool": null}

: keep-alive

event: answer
data: { ...AnswerOut... }
```

| Event | When |
|---|---|
| `progress` | `kind` is `queued` (waiting for a free slot), `tool` (each tool call, described in plain words) or `verifying` |
| `answer` | once, at the end: the full `AnswerOut` |
| `error` | once, if the run failed unexpectedly: `{"message": ...}` |
| `: keep-alive` | an SSE comment every 15 s of silence (the verifier can take ~20 s) |

Draft text is never streamed: the verifier may still change it. Closing the connection
cancels the run, so it stops spending.

---

### `POST /v1/expenses/csv` → 200
Classify an expense spreadsheet with the fixed rules. No model reads the cells.

Request: `multipart/form-data` with

| Field | Notes |
|---|---|
| `file` | CSV, UTF-8, ≤ 1 MB, ≤ 500 rows |
| `entity_type` | enum, required |
| `tax_year` | int, required |

Recognized columns (case-insensitive, first match wins):
- description: `description`, `memo`, `item`, `details`, `merchant`, `vendor`, `payee`
- amount: `amount`, `cost`, `total`, `price`, `debit`, `value` (`$1,045.50` and `(20.00)` accepted; negatives rejected)
- business use (optional, default 100): `business_use_pct`, `business_use`, `business %`, `business_pct`

**Screening.** Every cell and header is checked first. If any reads like an instruction
to an AI system ("ignore previous instructions", "system prompt", "note to the assistant:
…", role tags, …), the whole file is refused with **422** and
`{"refused": true, "message": "... row 3, column memo ..."}`. The message gives cell
positions only, never the text. Nothing about the upload is stored or logged, including
the access log line.

Response (`ExpensesOut`):
```json
{
  "entity_type": "sole_prop", "tax_year": 2025,
  "rows": [
    {"line": 2, "description": "Printer paper", "amount": "45.00", "business_use_pct": "100",
     "category": "supplies", "treatment": "fully_deductible", "deductible_amount": "45.00",
     "questions": [], "authorities": ["Treas. Reg. § 1.162-3(a)"], "note": null}
  ],
  "total_amount": "45.00", "total_deductible": "45.00", "rows_needing_review": 0
}
```
`treatment` is one of `fully_deductible`, `partially_deductible`,
`capitalize_and_depreciate`, `not_deductible`, `depends_on_facts`. Rows with
`deductible_amount: null` need facts or research, so ask about them in the chat.

---

## 2. The agent (Claude Agent SDK)

`WriteOffAgent` (`agent/harness.py`) drives `ClaudeSDKClient`, which runs the Claude Code
CLI bundled with `claude-agent-sdk` as a subprocess. Options set on every run:

| Option | Value | Why |
|---|---|---|
| `model` | `AGENT_MODEL` (`claude-sonnet-5`) | |
| `system_prompt` | `prompts/system.md` (versioned, currently 1.1.1) + a session-context block | |
| `tools` | `[]` | no built-in tools at all: no shell, files or web |
| `allowed_tools` | the 7 `mcp__writeoff__*` tools | pre-approved; nothing else can run |
| `mcp_servers` | `{"writeoff": <in-process server>}` | see section 3 |
| `permission_mode` | `dontAsk` | anything not allowed is denied, never prompted |
| `strict_mcp_config`, `setting_sources=[]` | on | ignore local Claude Code settings and MCP config |
| `max_turns` | `AGENT_MAX_TURNS` (12) | stop → `partial` answer |
| `max_budget_usd` | `AGENT_MAX_BUDGET_USD` (0.50) | stop → `partial` answer |
| `hooks` | PreToolUse, PostToolUse, PostToolUseFailure | section 4 |
| `resume` | the session's SDK session id | multi-turn conversations |
| `env` | `{"ANTHROPIC_API_KEY": ...}` | the CLI must bill the API key, never a local claude.ai login |

---

## 3. MCP server and tools

There are **no external MCP servers.** The only MCP server is **in-process**: created with
`create_sdk_mcp_server(name="writeoff", version="1.0.0")` inside the API (or CLI) process.
The model runs in the Claude Code subprocess and calls these tools over MCP; the tool code
runs in our Python process, with direct access to Postgres and the corpus.

- **Names:** `mcp__writeoff__<tool>`, e.g. `mcp__writeoff__search_tax_law`.
- **Schemas:** generated from the same pydantic models the handlers validate with, so the
  schema Claude sees can't drift from the code.
- **Annotations:** all tools are `readOnlyHint: true`. The two retrieval tools allow results
  up to 400,000 characters (`maxResultSizeChars`).
- **Results:** JSON text. **Errors** come back as a normal result with `is_error: true` and
  `"Error: <reason>"` (invalid arguments, timeout after `TOOL_TIMEOUT_SECONDS` = 30 s,
  unknown category, …). They're never raised, so the model can recover.
- **Evidence:** every result is recorded in the request's `Evidence` (passages with their
  source documents, parameters, calculations). The verifier checks the answer against it.

### 3.1 `search_tax_law`
Hybrid search over the IRC, Treasury Regulations, IRS publications and form instructions
for one tax year.

| Argument | Type | Notes |
|---|---|---|
| `query` | string ≥ 3 chars | ideally in tax terms |
| `tax_year` | int | must be supported |
| `entity_type` | enum, optional | keeps passages for all entities plus this one |
| `doc_types` | list, optional | `irc`, `treasury_regulation`, `irs_publication`, `form_instructions` |

Result: `query`, `rewritten_terms[]`, `weak` (best rerank score < 0.6: no strong
authority), `guidance` (e.g. "retry limit reached"), `passages[]` (`rank`, `chunk_id`,
`citation`, `doc_type`, `title`, `source_url`, `tax_year`, `text`, `section_id`,
`relevance`, `via_citation`), `sections{section_id: {citation, text}}` (the full parent
section of each passage).

Pipeline: explicit-citation lookup · Claude Haiku query rewrite · dense (Voyage +
pgvector HNSW) and keyword (Postgres full-text) search in parallel · reciprocal rank fusion
· top 40 reranked by Voyage `rerank-2.5` · top 8, at least 2 of them statute or regulation ·
parent sections attached. After 2 weak searches the tool adds guidance telling the model
to stop retrying.

### 3.2 `get_citation`
Exact text of a provision.

| Argument | Type | Notes |
|---|---|---|
| `citation` | string | `§ 280A(c)(1)`, `Reg. 1.263(a)-1(f)`, or a `citation` returned by search |
| `tax_year` | int | |

Result: `requested`, `resolved` (the normalized citation), `found`, `passages[]`,
`guidance`. When a provision is stored inside a larger passage, the enclosing section is
returned.

### 3.3 `get_tax_parameter`
A dollar limit, rate or percentage for a year, from `data/tax_parameters/{year}.yaml`.

| Argument | Type |
|---|---|
| `name` | string (see table) |
| `tax_year` | int |

Result: `name`, `tax_year`, `status` (`ok`, `unavailable`: not yet verified, the model must
say so; `unknown`: bad name, with `known_parameters[]`), `value` or `periods[]`
(`{start, end, value}` when the value depends on a date), `unit`, `source_url`,
`description`, `guidance`.

Parameters (values shown for 2025; **all 2026 values are still unverified (null)**):

| Name | Unit | 2025 |
|---|---|---|
| `section_179_dollar_limit` | usd | 2,500,000 |
| `section_179_phaseout_threshold` | usd | 4,000,000 |
| `section_179_suv_limit` | usd | 31,300 |
| `bonus_depreciation_pct` | percent | by acquisition date: 40 (2017-09-28 → 2025-01-19), 100 (2025-01-20 → 2025-12-31) |
| `passenger_auto_first_year_limit_with_bonus` | usd | 20,200 |
| `passenger_auto_first_year_limit_without_bonus` | usd | 12,200 |
| `passenger_auto_second_year_limit` | usd | 19,600 |
| `passenger_auto_third_year_limit` | usd | 11,800 |
| `passenger_auto_succeeding_years_limit` | usd | 7,060 |
| `standard_mileage_rate_business` | usd_per_mile | 0.70 |
| `home_office_simplified_rate` | usd_per_sq_ft | 5 |
| `home_office_simplified_max_sq_ft` | sq_ft | 300 |
| `business_meals_deduction_pct` | percent | 50 |
| `business_gift_limit_per_recipient` | usd | 25 |
| `de_minimis_safe_harbor_with_afs` | usd | 5,000 |
| `de_minimis_safe_harbor_without_afs` | usd | 2,500 |
| `startup_cost_immediate_deduction_limit` | usd | 5,000 |
| `startup_cost_phaseout_threshold` | usd | 50,000 |
| `qbi_threshold_single` | usd | *unverified* |
| `qbi_threshold_joint` | usd | *unverified* |

Every filled value carries an `evidence` quote and an irs.gov `source_url`.

### 3.4 `classify_expense`
First-pass treatment of one expense by fixed rules. The model must confirm it against
retrieved law.

| Argument | Type | Notes |
|---|---|---|
| `description` | string | required |
| `amount` | decimal ≥ 0 | required |
| `business_use_pct` | 0–100 | default 100 |
| `entity_type` | enum | required |
| `tax_year` | int | required |
| `category` | string, optional | force a category (see list) |
| `recipients` | int ≥ 1, optional | for gifts |

Result (`ExpenseClassification`): `category`, `treatment`, `deductible_amount` (when a
sourced limit applies), `reasoning[]`, `questions[]` (facts that would change the answer),
`calculator` (which calc tool to use next), `other_matching_categories[]`, plus the
common calculation fields (section 3.8). `authorities[]` are leads to look up, not
citations.

Categories: `fines_penalties`, `political_lobbying`, `employee_party`, `entertainment`,
`meals`, `employee_gifts`, `gifts`, `travel`, `commuting`, `vehicle`, `home_office`,
`improvements`, `repairs`, `supplies`, `equipment`, `software_subscription`, `rent`,
`utilities`, `health_insurance`, `insurance`, `advertising`, `professional_fees`,
`wages_contractors`, `education`, `startup_costs`, `interest`, `bank_fees`, `dues`,
`clothing`, `charitable`, `personal`.

### 3.5 `calc_depreciation`
Depreciation for one asset: § 179, special (bonus) depreciation by acquisition date, MACRS
and the § 280F passenger-auto caps, with a full schedule.

| Argument | Type | Notes |
|---|---|---|
| `cost` | decimal > 0 | required |
| `placed_in_service` | date | required |
| `asset_class` | enum | see below |
| `method` | `GDS` \| `GDS_SL` \| `ADS` | default `GDS` |
| `elect_179` | bool | default false |
| `section_179_amount` | decimal > 0, optional | default: as much as allowed |
| `bonus_pct` | 0–100, optional | default: the parameter for the acquisition date; 0 = elect out |
| `business_use_pct` | (0, 100] | default 100; ≤ 50% for listed property forces ADS with no § 179 or bonus |
| `convention` | `half_year` \| `mid_quarter`, optional | real property always uses mid-month |
| `acquired` | date, optional | decides the bonus percentage; defaults to `placed_in_service` |
| `total_section_179_property_cost` | decimal, optional | for the phase-out |
| `business_income` | decimal, optional | § 179 income limit, with carryover |

Asset classes (GDS recovery period / method):

| `asset_class` | GDS | Notes |
|---|---|---|
| `computer_equipment` | 5 yr, 200% DB | |
| `office_furniture` | 7 yr, 200% DB | |
| `machinery_equipment` | 7 yr, 200% DB | |
| `passenger_auto` | 5 yr, 200% DB | listed property; § 280F caps |
| `light_truck_van` | 5 yr, 200% DB | listed property; no § 280F caps |
| `heavy_suv` | 5 yr, 200% DB | listed; § 179 limited to the SUV cap |
| `land_improvements` | 15 yr, 150% DB | no § 179 |
| `qualified_improvement_property` | 15 yr, SL | |
| `residential_rental_property` | 27.5 yr, SL | no § 179, no bonus |
| `nonresidential_real_property` | 39 yr, SL | no § 179, no bonus |

Result: `business_basis`, `section_179`, `section_179_carryover`, `bonus`, `bonus_pct`,
`macrs_basis`, `method`, `recovery_period`, `convention`, `first_year_deduction`,
`schedule[]` (`{tax_year, macrs, section_179, bonus, cap, deduction}`),
`total_deductions`, plus the common fields. MACRS percentages match the IRS tables
(Pub 946, Appendix A).

### 3.6 `calc_home_office`

| Argument | Type | Notes |
|---|---|---|
| `tax_year` | int | |
| `method` | `simplified` \| `regular` | |
| `sq_ft`, `total_sq_ft` | decimal > 0 | office area and home area |
| `expenses` | list of `{category, amount}`, regular method | categories: `mortgage_interest`, `real_estate_taxes`, `casualty_losses`, `insurance`, `rent`, `repairs`, `utilities`, `other`, `depreciation` |
| `gross_income_limit` | decimal, optional | applies the gross income limit and carryovers |

Result: `method`, `business_pct`, `deduction`, `carryover_operating`,
`carryover_depreciation`, plus the common fields. Simplified: $5 × up to 300 sq ft.

### 3.7 `calc_vehicle`

| Argument | Type | Notes |
|---|---|---|
| `tax_year` | int | |
| `method` | `standard_mileage` \| `actual` | |
| `total_miles` | decimal > 0 | |
| `business_miles` | decimal ≥ 0 | |
| `actual_costs` | map, actual method | keys: `gas_oil`, `repairs`, `tires`, `insurance`, `registration_licenses`, `lease_payments`, `garage_rent`, `other` |
| `depreciation` | decimal ≥ 0 | actual method |
| `parking_tolls` | decimal ≥ 0 | business parking and tolls, added under either method |
| `vehicles_used_simultaneously` | int ≥ 1 | 5 or more: the result notes the standard rate isn't allowed (Pub 463, ch. 4) |

Result: `method`, `business_use_pct`, `deduction`, plus the common fields.

### 3.8 Fields common to calculator results
`status` (`ok` or `unavailable` when a needed parameter isn't verified for that year),
`missing_parameters[]`, `parameters_used[]` (`{name, value, unit, source_url}`), `steps[]`
(`{description, amount}`: the arithmetic, step by step), `notes[]`, `authorities[]`.

---

## 4. Hooks

Registered on the three tool-lifecycle events for the `mcp__writeoff__*` tools
(`agent/hooks.py`, class `TraceHooks`):

| Hook | What it does |
|---|---|
| **PreToolUse** | Redacts SSN, ITIN and EIN patterns (`123-45-6789`, 9-digit runs, `12-3456789`) in the arguments. Validates every `tax_year` (must be supported) and every `*_pct` (0–100) anywhere in the input. Invalid → `permissionDecision: "deny"` with the reason, which the model sees and can correct. Records the start time. |
| **PostToolUse** | Writes one `agent_tool_calls` row: tool name, redacted input, `ok` or `error`, latency, result size, retrieved `chunk_ids`, and the first 300 characters of any error. Tool results themselves are not stored. |
| **PostToolUseFailure** | Same row with status `error` when the call failed outright. |

---

## 5. External services and models

| Service | Used for | Called from | Model / endpoint |
|---|---|---|---|
| **Claude Agent SDK → Claude Code CLI** | the agent loop | `agent/harness.py` | `AGENT_MODEL` = `claude-sonnet-5` |
| **Anthropic Messages API** (`messages.parse`, structured output) | grounding verifier | `agent/verifier.py` | `VERIFIER_MODEL` = `claude-sonnet-5`, thinking off |
| Anthropic Messages API | query rewrite into tax terms | `retrieval/rewrite.py` | `REWRITE_MODEL` = `claude-haiku-4-5-20251001` |
| Anthropic Messages API | per-chunk context summaries (ingestion only) | `chunking/context.py` | `CONTEXT_MODEL` = `claude-haiku-4-5-20251001` |
| Anthropic Messages API | answer-eval judge (evals only) | `evals/judge.py` | `EVAL_JUDGE_MODEL` = `claude-opus-5-5`, adaptive thinking |
| **Voyage AI** `POST https://api.voyageai.com/v1/embeddings` | query and chunk embeddings (1024-d) | `retrieval/voyage.py` | `voyage-4` |
| Voyage AI `POST https://api.voyageai.com/v1/rerank` | reranking search candidates | `retrieval/voyage.py` | `rerank-2.5` |
| **uscode.house.gov**, **ecfr.gov** (versioner API), **irs.gov** | source documents (ingestion only) | `ingestion/fetcher.py` | HTML, XML, PDF; cached in `data/raw/` |
| **PostgreSQL 16 + pgvector** | corpus (HNSW + full-text), sessions, traces | `retrieval/pgvector_store.py`, `agent/store.py` | |

List prices used for cost reporting (`PRICE_PER_MTOK`, USD per million input/output
tokens, checked 2026-09-24): Sonnet 5 2/10, Opus 5.5 4/20, Haiku 4.5 1/5.

---

## 6. Database tables

| Table | Holds | Written by |
|---|---|---|
| `documents` | one row per source document and tax year | ingestion |
| `chunks` | parent sections and child passages: text, `citation_path`, embedding, full-text vector, entity types | ingestion |
| `agent_sessions` | entity type, tax year, business profile, SDK session id | API / harness |
| `agent_requests` | one row per question: redacted question, prompt version, status, turns, tokens, cost, latency, verification results | harness |
| `agent_tool_calls` | one row per tool call (see PostToolUse) | hooks |

Answer text and tool results are not stored. Traces are deleted after
`TRACE_RETENTION_DAYS`; idle sessions after `SESSION_RETENTION_DAYS`.

---

## 7. Configuration

Set in `.env` or the environment; names are case-insensitive. Full list in
`src/writeoff/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | required for the agent, verifier, rewrite and summaries |
| `VOYAGE_API_KEY` | — | required for search and ingestion |
| `DATABASE_URL` | `postgresql://writeoff:writeoff@localhost:5432/writeoff` | Docker sets `@db:5432` |
| `API_TOKEN` | unset (open) | bearer token for `/v1/*` |
| `WRITEOFF_API_URL` | `http://localhost:8000` | where the UI finds the API |
| `MAX_CONCURRENT_ANSWERS` | 4 | answers in flight per API process |
| `AGENT_MODEL`, `VERIFIER_MODEL`, `REWRITE_MODEL`, `CONTEXT_MODEL`, `EVAL_JUDGE_MODEL` | see section 5 | |
| `AGENT_MAX_TURNS` | 12 | per question |
| `AGENT_MAX_BUDGET_USD` | 0.50 | per question (agent loop) |
| `TOOL_TIMEOUT_SECONDS` | 30 | per tool call |
| `MAX_WEAK_SEARCH_RETRIES` | 2 | rephrased searches after weak results |
| `SUPPORTED_TAX_YEARS` | `[2025, 2026]` | |
| `TRACE_RETENTION_DAYS` | 30 | 0 = traces never written |
| `SESSION_RETENTION_DAYS` | 30 | 0 = sessions kept |
| `RETENTION_INTERVAL_HOURS` | 24 | how often the API purges |
| `EMBEDDING_MODEL`, `RERANKER_MODEL` | `voyage-4`, `rerank-2.5` | the embedding dimension is fixed at 1024 by the schema |
| `SYSTEM_PROMPT_PATH` | `prompts/system.md` | |

---

## 8. Command line

`make help` lists every target. The main ones:

| Command | Does |
|---|---|
| `make up` / `make db-down` | start (rebuild) / stop db + API + UI in Docker |
| `make api` / `make ui` | run the API / UI locally |
| `make ask Q="..." ENTITY=sole_prop YEAR=2025` | one question with a full trace (`python -m writeoff.agent.cli`, flags `--entity --year --session --no-verify --no-persist`) |
| `make search Q="..." YEAR=2025` | hybrid search only (`python -m writeoff.retrieval.cli`, flags `--entity -k`) |
| `make ingest YEAR=…` / `make ingest-dry YEAR=…` | load the corpus (`python -m writeoff.ingestion.cli`, flags `--source --no-context --refresh --dry-run`) |
| `make migrate` | apply SQL migrations |
| `make purge` | apply retention now (`python -m writeoff.retention`) |
| `make eval` / `make eval-answers` / `make eval-full` | evaluations (`python -m writeoff.evals.cli …`, see `evals/README.md`) |
| `make smoke` | end-to-end check of the Docker stack |
| `make test` / `make lint` | tests / ruff + mypy |
