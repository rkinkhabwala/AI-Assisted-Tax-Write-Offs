# WriteOff Assistant

A grounded AI assistant that helps U.S. small-business owners understand federal tax
deductions. Every substantive claim is backed by a retrieved passage from the IRC,
Treasury Regulations, or IRS publications, and every number comes from deterministic
Python rather than from the model. See [spec.md](spec.md) for the full specification and
[PLAN.md](PLAN.md) for build progress.

> General information only, not tax or legal advice. Consult a CPA or enrolled agent.

## Stack

Python 3.12 · Claude Agent SDK · FastAPI · PostgreSQL 16 + pgvector (HNSW + full-text
search) · Voyage AI embeddings and reranker (behind swappable interfaces) · pytest · Docker

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 for you if needed)
- Docker with the Compose plugin

## Setup

```bash
cp .env.example .env          # then fill in ANTHROPIC_API_KEY and VOYAGE_API_KEY
make install                  # uv sync: creates .venv with runtime + dev deps
make db-up                    # starts Postgres 16 + pgvector and waits until healthy
make migrate                  # applies migrations/*.sql to DATABASE_URL
make lint                     # ruff check, ruff format --check, mypy --strict
make test                     # full test suite, including the DB migration test
```

Run the API (currently only `/healthz`):

```bash
uv run uvicorn writeoff.api.app:app --reload
# or, fully containerised:
docker compose up --build
```

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
  evals/             (src) eval harness: dataset, metrics, variants, reports
  api/               FastAPI app
migrations/          plain SQL, applied in filename order
data/tax_parameters/ per-year limits and rates, each with an irs.gov source_url
data/sources.yaml    the corpus: sources and their editions per tax year
scripts/             generate_sources.py (writes data/sources.yaml)
prompts/system.md    versioned system prompt
evals/               retrieval, answer and safety evaluations
```

## Ingestion

The corpus is defined in [data/sources.yaml](data/sources.yaml): 18 IRC sections
(uscode.house.gov), 54 Treasury Regulation sections (eCFR, pinned to a date per tax year),
6 IRS publications and 6 sets of form instructions (irs.gov HTML, or irs-prior PDFs for
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
