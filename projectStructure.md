The full eval run is still going in the background; I'll report on it when it finishes. Here's how the project fits together.

## 1. Project structure

```
data/
  sources.yaml              the 84 source documents (generated; don't hand-edit)
  tax_parameters/2025.yaml  dollar limits and rates, each with an irs.gov source
  tax_parameters/2026.yaml
scripts/generate_sources.py builds sources.yaml
migrations/                 Postgres schema (documents, chunks, agent traces, verification)
prompts/system.md           the agent's versioned instructions

src/writeoff/
  ingestion/     fetch → parse → sync.  Turns IRS/eCFR/US Code pages into a document tree
  chunking/      splits the tree into chunks; Haiku writes a context summary for each
  retrieval/     Voyage embeddings + reranker, pgvector store, hybrid search
  calculators/   depreciation, home office and vehicle calculations (no LLM)
  tools/         law lookup, parameter lookup, expense classifier
  agent/         Claude Agent SDK harness, MCP tools, hooks, grounding verifier
  evals/         retrieval, answer and safety evals
  tax_parameters.py, config.py, models.py, privacy.py
tests/           325 tests
evals/           datasets, rubric, baselines, reports
```

It works as two separate pipelines that share one Postgres database.

## 2. How the code flows

### A. Ingestion (offline, run once per tax year): `make ingest YEAR=2025`

```
sources.yaml ──► fetcher (HTTP + disk cache)
             ──► parser (uscode / ecfr / irs_html / pdf) ──► document tree
             ──► chunker ──► parent chunks (≤2,000 tokens) + child chunks (300–800)
             ──► context.py: Haiku writes a 1–2 sentence summary per child (cached on disk)
             ──► voyage.py: embed "summary + text" (voyage-4, 1024 dims)
             ──► pipeline.py: sync to Postgres (pgvector index + full-text index)
```

`pipeline.py` compares content hashes. An unchanged chunk costs no API calls, a changed one is re-embedded, and a chunk that disappeared from the source is deleted.

### B. Answering a question: `make ask Q="..."` (the API arrives in Phase 9)

```
question
  └► WriteOffAgent.ask (agent/harness.py)
       ├─ loads the session (entity type, year) and redacts SSNs and EINs
       ├─ Claude Sonnet 5 runs the loop: understand → retrieve → compute → draft
       │     tools it can call (agent/tools.py → tools/, calculators/):
       │       search_tax_law ─► HybridRetriever:
       │           exact-citation match, else dense + keyword search
       │           → RRF fusion → Voyage rerank → top 8 (flagged "weak" if poor)
       │       get_citation, get_tax_parameter (reads the YAML)
       │       calc_depreciation / calc_home_office / calc_vehicle, classify_expense
       │     hooks.py records each tool call; evidence.py stores everything retrieved
       ├─ Verifier (agent/verifier.py):
       │     deterministic check: every [citation] was retrieved; every $ or % is in the evidence
       │     a separate Claude call labels each claim against the cited passages
       │     edits → round 2 → unsupported claims stripped, with a note saying so
       └─ AgentAnswer (text + trace + verification) → saved to agent_requests
```

The model never states a number from memory. Limits come from the parameter YAML, amounts come from the calculators, and the verifier checks both.

## 3. Adding tax year 2027

Each year is a separate slice of the database. Documents are unique per `(source_url, tax_year)` and every search filters by `tax_year`. Adding 2027 creates new rows and new embeddings, and leaves the 2025 and 2026 data untouched.

**Step 1: Register the 2027 sources.** In [scripts/generate_sources.py](scripts/generate_sources.py):
- Add a 2027 date to `ECFR_DATES`, e.g. `2027: "2027-12-31"`. That covers every regulation and statute for 2027.
- For each publication and form instruction, add a `2027:` URL once the IRS posts the 2027 edition. Until then, leave it out: the IRS reuses the same URL, and early in the year it still serves the old edition.

Then:
```bash
uv run --no-sync python scripts/generate_sources.py
```

**Step 2: Add tax parameters.** Copy `data/tax_parameters/2026.yaml` to `2027.yaml` and set `tax_year: 2027`. Fill in each value with its irs.gov `evidence` once the IRS announces it (mileage rate, § 179 limit, § 280F caps, …). Anything left `null` makes the agent say "not available yet" rather than guess.

**Step 3: Allow the year.** In [config.py](src/writeoff/config.py#L76), change `supported_tax_years = (2025, 2026, 2027)`, or set it in `.env`.

**Step 4: Ingest (embeds the new year).**
```bash
make ingest-dry YEAR=2027   # fetch + parse + chunk, no API cost: check it parses
make ingest YEAR=2027       # Haiku summaries + Voyage embeddings → Postgres
```
The cost should be close to the 2025 ingest, since it covers roughly the same number of chunks.

**Step 5: Re-sync mid-year when the IRS updates a publication.** Run `make ingest YEAR=2027` again. Only the chunks whose text changed are re-summarized and re-embedded; everything else is skipped.

**Step 6: Evals.** Add 2027 cases to the golden set, or copy existing ones with `tax_year: 2027`. Then run `make eval-full` to check that retrieval and answers held up.

Watch out for two things:
- **Statutes aren't frozen per year.** The US Code source is the live "current law" page for every year. A 2027 ingest will capture 2027 law, but re-ingesting 2025 later would also pull current law, not the 2025 text. Regulations don't have this problem because the eCFR URLs are pinned to a date.
- **A year change breaks the pipeline's diffing** if you ever change the embedding model or the chunking settings. Those changes need a full re-embed of every year, which is what the eval index variants (`make eval-index`) are for.