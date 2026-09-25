# WriteOff Assistant

A grounded AI assistant that helps U.S. small-business owners understand federal tax
deductions. Every substantive claim is backed by a retrieved passage from the IRC,
Treasury Regulations, or IRS publications, and every number comes from deterministic
Python rather than from the model. See [spec.md](spec.md) for the full specification and
[PLAN.md](PLAN.md) for build progress.

> General information only, not tax or legal advice. Consult a CPA or enrolled agent.

## Stack

Python 3.12 · Claude Agent SDK · FastAPI · Streamlit · PostgreSQL 16 + pgvector (HNSW + full-text
search) · Voyage AI embeddings and reranker (behind swappable interfaces) · pytest · Docker

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 for you if needed)
- Docker with the Compose plugin

## Running WriteOff Assistant

### 1. One-time setup

```bash
cp .env.example .env          # then edit .env (see below)
make install                  # creates .venv (Python deps + the bundled Claude Code CLI)
make db-up                    # starts Postgres 16 + pgvector in Docker, waits until healthy
make migrate                  # creates the tables
```

In `.env`, set at least:

| Variable | Why |
|---|---|
| `ANTHROPIC_API_KEY` | the agent, the verifier and context summaries. Required: the agent refuses to start without it rather than fall back to a personal claude.ai login |
| `VOYAGE_API_KEY` | embeddings and reranking (search) |
| `API_TOKEN` | any long random string, e.g. `openssl rand -hex 32`. Protects the API; leave empty only on your own machine |

### 2. Load the tax-law corpus (once per tax year)

The assistant can only answer from documents in the database, so ingest before first
use:

```bash
make ingest-dry YEAR=2025     # optional: fetch + parse + chunk only, no API cost
make ingest YEAR=2025         # fetch, summarize (Claude), embed (Voyage), store
make ingest YEAR=2026
```

This spends API credit and takes a while the first time. Downloads and summaries are
cached, and re-running is idempotent: only new or changed passages are processed again.
Run it again after adding sources to `scripts/generate_sources.py`, or when the IRS updates
a publication.

### 3a. Run the app (Docker: recommended)

```bash
make up                       # builds and starts db + API + UI
```

- Web app: **http://localhost:8501**
- API docs: http://localhost:8000/docs

Check the whole stack end to end:

```bash
make smoke                    # asks one live question (a few cents); NO_LLM=1 skips it
```

After changing code, run `make up` again to rebuild. `make db-down` stops everything; the
database volume, and with it the ingested corpus, is kept. (`docker compose down -v`
would delete the database and the corpus with it.)

### 3b. Run for development (no containers for the app)

```bash
make db-up                    # database only
make api                      # terminal 1: API with auto-reload on :8000
make ui                       # terminal 2: Streamlit UI on :8501, talks to WRITEOFF_API_URL
make ask Q="Can I deduct a client lunch?" ENTITY=sole_prop YEAR=2025   # CLI, with a full trace
```

### Using the app

1. In the sidebar, choose your **entity type** and **tax year**.
2. **Ask** a question in plain English, e.g. "I drove 8,000 business miles in 2025. What
   can I deduct?" You'll see what the assistant is doing while it researches, then the
   answer with inline citations. A note says how many claims were checked against the
   sources, and a collapsible **Sources** panel quotes each cited passage with a link.
   If a missing fact changes the answer, the assistant asks one question first.
3. **Upload expenses**: a CSV with a `description` (or memo, merchant, vendor) column
   and an `amount` column, optionally `business_use_pct`. Each row gets a first-pass
   treatment. Ask about any row in the chat for a sourced answer. Files with text
   addressed to an AI system are refused, and uploads are never stored.
4. **New conversation** in the sidebar starts over.

### Everyday commands

| Command | What it does |
|---|---|
| `make help` | list all targets |
| `make up` / `make db-down` | start (rebuilding) / stop the Docker stack |
| `docker compose logs -f app` | follow API logs (`ui` or `db` for the others) |
| `make test` / `make lint` | test suite (starts the database) / ruff + mypy |
| `make eval` | retrieval eval with its regression gate (cheap) |
| `make eval-answers` | answer and safety evals (~$17, capped at `MAX_COST`, default $20) |
| `make purge` | delete traces and idle sessions past the retention windows now |

### Troubleshooting

| Symptom | Fix |
|---|---|
| UI says it "can't reach the WriteOff API" | `docker compose ps` and `docker compose logs app`; the API must be healthy |
| `401 missing or invalid bearer token` | the UI and your client must send the same `API_TOKEN` as the API; re-run `make up` after changing `.env` |
| `503 … ANTHROPIC_API_KEY is required` | set the key in `.env`, then `make up` (or restart `make api`) |
| Every answer says "I couldn't find authority" | that tax year isn't ingested: `make ingest YEAR=<year>` |
| `CLINotFoundError: Claude Code not found` | the `.venv` (which contains the CLI) is missing or was moved: `make install` |
| `You've hit your session limit` in an answer | the agent is using a claude.ai login instead of the API key; make sure `ANTHROPIC_API_KEY` is set |
| Voyage `429` during ingest | rate limit or billing not yet active; wait and re-run `make ingest` (it resumes) |
| Port 5432, 8000 or 8501 already in use | stop the other service, or set `POSTGRES_PORT` in `.env` (and update `DATABASE_URL`) |

## Web app and API

Full reference for every endpoint, agent tool, hook, external service and setting:
[docs/REFERENCE.md](docs/REFERENCE.md).

The Streamlit UI (`ui/app.py`) is a thin client of the API. Pick the entity type and tax
year in the sidebar, then ask. While the agent works you see what it's doing
("Searching the tax law…", "Checking the answer against the sources"), then the verified
answer with a collapsible **Sources** panel quoting every cited passage with a link to
irs.gov, uscode.house.gov or eCFR. The **Upload expenses** tab classifies a CSV.

| Endpoint | |
|---|---|
| `POST /v1/sessions`, `GET`/`PATCH /v1/sessions/{id}` | session facts: entity type, tax year, business profile |
| `POST /v1/chat` | ask; returns the answer, sources, verification counts and usage |
| `POST /v1/chat/stream` | the same as server-sent events: `progress` events, then `answer` |
| `POST /v1/expenses/csv` | classify an expense CSV (multipart: `file`, `entity_type`, `tax_year`) |
| `GET /healthz`, `GET /readyz` | liveness, and readiness (database reachable) |

- **Auth:** set `API_TOKEN` and send `Authorization: Bearer <token>`. Without it the API
  is open, which is fine only on your own machine; the app logs a warning at startup.
- **Streaming shows activity, not draft text:** the verifier may still change the
  draft, so unverified claims never reach the screen.
- **CSV uploads are data only:** every cell is screened first. A file with text
  addressed to an AI system is refused with the cell positions (never the text); nothing
  about it is stored or logged, and upload requests are kept out of the access log.
  Clean rows get a first-pass treatment from the fixed classification rules; no model
  reads the cells.
- **Cost control:** `MAX_CONCURRENT_ANSWERS` (default 4) bounds parallel answers, and each
  answer is capped by `AGENT_MAX_TURNS` and `AGENT_MAX_BUDGET_USD`.
- **Retention:** traces are kept `TRACE_RETENTION_DAYS` (0 = never written); sessions idle
  longer than `SESSION_RETENTION_DAYS` are deleted (0 = kept). The API purges daily;
  `make purge` does it on demand.
- **Authentication with Anthropic:** the agent's Claude Code subprocess is given
  `ANTHROPIC_API_KEY` explicitly and won't start without it, so it never falls back to a
  personal claude.ai login.

## Layout

```
src/writeoff/
  config.py          Settings (pydantic-settings, reads .env)
  models.py          Document, Chunk, RetrievalResult and friends
  tax_parameters.py  Loader for data/tax_parameters/{year}.yaml
  db/migrate.py      Minimal forward-only SQL migration runner
  ingestion/         registry, fetcher + cache, parsers (uscode, eCFR, irs.gov, PDF), pipeline, CLI
  chunking/          structure-aware parent/child chunker, token estimate, context summaries
  retrieval/         interfaces, pgvector store, Voyage embedder/reranker, citations, RRF, hybrid search
  agent/             Claude Agent SDK harness, hooks and loop
  tools/             in-process MCP tools exposed to the agent
  calculators/       deterministic tax math (depreciation, home office, vehicle)
  evals/             (src) eval harness: retrieval + answer/safety evals, judge, reports
  api/               FastAPI app: sessions, chat (JSON + SSE), CSV upload, retention task
  ui/                Streamlit UI and its API client
  expenses.py        CSV parsing, injection screening, rule-based row classification
  retention.py       purge old traces and idle sessions
migrations/          plain SQL, applied in filename order
data/tax_parameters/ per-year limits and rates, each with an irs.gov source_url
data/sources.yaml    the corpus: sources and their editions per tax year
scripts/             generate_sources.py (writes data/sources.yaml), smoke_test.py
prompts/system.md    versioned system prompt (earlier versions in prompts/archive/)
docs/REFERENCE.md    interface reference: API, tools, hooks, services, settings
evals/               retrieval, answer and safety evaluations
```

## Ingestion

The corpus is defined in [data/sources.yaml](data/sources.yaml): 18 IRC sections
(uscode.house.gov), 54 Treasury Regulation sections (eCFR, pinned to a date per tax year),
7 IRS publications and 7 sets of form instructions (irs.gov HTML, or irs-prior PDFs for
editions the live page has moved past). Regenerate it with
`python scripts/generate_sources.py` rather than editing it by hand.

```bash
make ingest-dry YEAR=2025   # fetch + parse + chunk, no keys needed -> data/staging/chunks-2025.jsonl
make ingest YEAR=2025       # + Claude context summaries + Voyage embeddings -> Postgres
PYTHONPATH=src uv run --no-sync python -m writeoff.ingestion.cli --year 2025 --source irc-280a
```

Pipeline: fetch (polite, retried, cached in `data/raw/`) -> parse into a document tree
(`ingestion/parsers/`) -> structure-aware chunking into parent sections and child chunks
(`chunking/chunker.py`) -> per-chunk context summary (Claude) -> embedding (Voyage) ->
sync into Postgres. Re-running is idempotent: unchanged chunks cost no API calls and no
writes, changed ones are re-embedded, and removed ones are deleted. `--no-context` skips
the summaries, and `--refresh` re-downloads the sources.

## Search

```bash
make search Q="home office exclusive use requirement" YEAR=2025
PYTHONPATH=src uv run --no-sync python -m writeoff.retrieval.cli "Section 179 limit" --entity s_corp -k 5
```

Hybrid retrieval (`retrieval/hybrid.py`): explicit IRC and regulation citations are
fetched directly; the question is rewritten into tax terms (Claude); semantic
(pgvector HNSW) and keyword (Postgres full-text) results are fused with Reciprocal Rank
Fusion; the Voyage reranker picks the top 8 against the original question; each hit
comes with its parent section.

## Ask the agent

```bash
make ask Q="Can I deduct a lunch with a client?" ENTITY=sole_prop YEAR=2025
```

The agent (`agent/harness.py`) runs on the Claude Agent SDK with only our seven tools:
search_tax_law, get_citation, get_tax_parameter, classify_expense, calc_depreciation,
calc_home_office and calc_vehicle. It has no built-in tools and no file or web access,
and it stops at turn and budget limits. Hooks validate and redact tool inputs and record
each call to `agent_tool_calls` for tracing.

## Evaluation

```bash
make eval         # retrieval eval on the production index + regression gate (Recall@10)
make eval-sweep   # compare retrieval configurations; results in evals/retrieval_runs.csv
make eval-answers # end-to-end answer + safety evals (~$17, capped at MAX_COST=20)
make eval-full    # both layers with both regression gates
make eval-agreement FILE=evals/reports/spot_check_<run>.md   # judge vs your hand check
```

See [evals/README.md](evals/README.md). Current retrieval baseline: Recall@10 0.842,
MRR 0.815, nDCG@10 0.737 on 75 labeled queries.

## Tax parameters

`data/tax_parameters/{year}.yaml` holds every dollar limit, percentage and rate the agent
may quote. Values ship as `null` and must be filled in by hand from the cited irs.gov
page. The loader refuses any entry without a `source_url`, and the agent is never
allowed to hard-code these numbers.

## Database migrations

Migrations are plain SQL files in `migrations/`, applied in order and recorded in
`schema_migrations`. They are forward-only: to change the schema, add a new file rather
than editing an applied one. The embedding column is `vector(1024)`, which matches
`EMBEDDING_DIMENSION`. Switching to an embedder with a different dimension requires a new
migration.
