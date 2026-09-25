.DEFAULT_GOAL := help
.PHONY: help install lint format test test-unit db-up db-down migrate ingest ingest-dry search eval eval-answers eval-full eval-agreement eval-sweep eval-index ask api ui up purge smoke

UV := uv run --no-sync
YEAR ?= 2025

# macOS may flag files under .venv as UF_HIDDEN, and Python 3.12.8+ silently skips hidden
# .pth files, which breaks the editable install of `writeoff`. The flag can be re-applied
# at any time, so don't rely on the .pth: put src/ on the path explicitly.
export PYTHONPATH := $(CURDIR)/src

help:  ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Create .venv with runtime and dev dependencies
	uv sync --python 3.12

lint:  ## Ruff lint + format check, mypy --strict
	$(UV) ruff check src tests
	$(UV) ruff format --check src tests
	$(UV) mypy

format:  ## Auto-fix lint and formatting
	$(UV) ruff check --fix --exit-zero src tests
	$(UV) ruff format src tests

test: db-up  ## Full test suite; DB tests fail (not skip) if Postgres is unreachable
	WRITEOFF_REQUIRE_DB=1 $(UV) pytest

test-unit:  ## Tests that need no database
	$(UV) pytest -m "not db"

db-up:  ## Start Postgres 16 + pgvector and wait until healthy
	docker compose up -d --wait db

db-down:  ## Stop all containers (db, API, UI); the data volume is kept
	docker compose down

migrate:  ## Apply pending SQL migrations to DATABASE_URL
	$(UV) python -m writeoff.db.migrate

ingest:  ## Ingest YEAR (default 2025) into Postgres; needs VOYAGE_API_KEY and ANTHROPIC_API_KEY
	$(UV) python -m writeoff.ingestion.cli --year $(YEAR)

ingest-dry:  ## Fetch, parse and chunk YEAR with no keys; writes data/staging/chunks-YEAR.jsonl
	$(UV) python -m writeoff.ingestion.cli --year $(YEAR) --dry-run

search:  ## Hybrid search, e.g. make search Q="Section 179 limit" YEAR=2025
	$(UV) python -m writeoff.retrieval.cli "$(Q)" --year $(YEAR) --text

eval:  ## Retrieval eval on the prod index; fails if Recall@10 drops >3 points vs baseline
	$(UV) python -m writeoff.evals.cli check
	$(UV) python -m writeoff.evals.cli retrieval --gate

eval-answers:  ## End-to-end answer + safety evals (costs API credit; capped by MAX_COST, default $20). ARGS="--ids a,b" etc.
	$(UV) python -m writeoff.evals.cli check-answers
	$(UV) python -m writeoff.evals.cli answers --max-cost $(or $(MAX_COST),20) $(ARGS)

eval-full:  ## Retrieval + answer evals with both regression gates (Recall@10 and treatment accuracy)
	$(UV) python -m writeoff.evals.cli check
	$(UV) python -m writeoff.evals.cli retrieval --gate
	$(UV) python -m writeoff.evals.cli check-answers
	$(UV) python -m writeoff.evals.cli answers --gate --max-cost $(or $(MAX_COST),20)

eval-agreement:  ## Judge vs hand spot-check agreement, e.g. make eval-agreement FILE=evals/reports/spot_check_X.md
	$(UV) python -m writeoff.evals.cli agreement $(FILE)

eval-sweep:  ## All query-time retrieval variants (INDEX=prod by default); logs to evals/retrieval_runs.csv
	$(UV) python -m writeoff.evals.cli sweep --index $(or $(INDEX),prod)

eval-index:  ## Build an experiment index, e.g. make eval-index VARIANT=large_noctx
	$(UV) python -m writeoff.evals.cli build-index $(VARIANT)

api:  ## Run the API locally with reload (http://localhost:8000/docs)
	$(UV) uvicorn writeoff.api.app:app --reload --port 8000

ui:  ## Run the Streamlit UI locally against WRITEOFF_API_URL (default http://localhost:8000)
	$(UV) streamlit run src/writeoff/ui/app.py --browser.gatherUsageStats=false

up:  ## Build and start db + API + UI in docker compose (UI on http://localhost:8501)
	docker compose up -d --build

purge:  ## Delete traces and idle sessions past TRACE_/SESSION_RETENTION_DAYS
	$(UV) python -m writeoff.retention

smoke:  ## End-to-end smoke test against docker compose (one cheap live question; NO_LLM=1 skips it)
	docker compose up -d --build --wait
	$(UV) python scripts/smoke_test.py $(if $(NO_LLM),--no-llm)

ask:  ## Ask the agent and print the trace, e.g. make ask Q="Can I deduct a client lunch?" ENTITY=sole_prop
	$(UV) python -m writeoff.agent.cli "$(Q)" --year $(YEAR) $(if $(ENTITY),--entity $(ENTITY))
