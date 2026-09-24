"""Application settings, loaded from environment variables and `.env`.

Every secret and tunable lives here so nothing is hard-coded in the pipeline. Fields that
later phases will need (agent limits, retrieval depths) are added in those phases rather
than stubbed now.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Must match the `vector(N)` column in migrations/0001_init.sql. Both default embedder
# candidates (voyage-4, voyage-law-2) and the open-source fallback (bge-large-en-v1.5)
# emit 1024-dim vectors, so the choice between them needs no schema change.
EMBEDDING_DIMENSION = 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Infrastructure -------------------------------------------------------------
    database_url: PostgresDsn = PostgresDsn(
        "postgresql://writeoff:writeoff@localhost:5432/writeoff"
    )
    migrations_dir: Path = Path("migrations")

    # --- Model providers ------------------------------------------------------------
    anthropic_api_key: SecretStr | None = None
    voyage_api_key: SecretStr | None = None

    agent_model: str = "claude-sonnet-5"
    # Contextual-retrieval summaries run once per chunk, so a small fast model is used.
    context_model: str = "claude-haiku-4-5-20251001"
    # Query rewriting runs on every search, so it uses the same small, fast model.
    rewrite_model: str = "claude-haiku-4-5-20251001"
    # The grounding verifier judges legal support claim by claim; it runs on the agent model.
    verifier_model: str = "claude-sonnet-5"
    # Answer-eval judge: a different, stronger model than the agent to limit self-preference.
    eval_judge_model: str = "claude-opus-5-5"
    # USD per million (input, output) tokens, from platform.claude.com/docs/en/about-claude/pricing
    # (checked 2026-09-24). Used only to report eval and verifier cost.
    price_per_mtok: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "claude-sonnet-5": (2.0, 10.0),
            "claude-opus-5-5": (4.0, 20.0),
            "claude-haiku-4-5-20251001": (1.0, 5.0),
        }
    )

    # Only Voyage is implemented; bge (D1 fallback) is added if and when it is needed.
    embedding_provider: Literal["voyage"] = "voyage"
    embedding_model: str = "voyage-4"
    embedding_dimension: int = EMBEDDING_DIMENSION
    reranker_provider: Literal["voyage"] = "voyage"
    reranker_model: str = "rerank-2.5"

    # --- Ingestion ------------------------------------------------------------------
    sources_path: Path = Path("data/sources.yaml")
    raw_cache_dir: Path = Path("data/raw")
    fetch_min_interval_seconds: float = Field(default=1.0, ge=0)
    # Sent to irs.gov, ecfr.gov and uscode.house.gov. Operators may add contact details.
    http_user_agent: str = "WriteOffAssistant/0.1 (tax-law research corpus)"
    context_concurrency: int = Field(default=8, ge=1)
    context_cache_dir: Path = Path("data/cache/context")

    # --- Agent (spec section 5 guardrails) -------------------------------------------
    agent_max_turns: int = Field(default=12, ge=1)
    agent_max_budget_usd: float = Field(default=0.50, gt=0)
    tool_timeout_seconds: float = Field(default=30.0, gt=0)
    max_weak_search_retries: int = Field(default=2, ge=0)

    # --- Domain ---------------------------------------------------------------------
    supported_tax_years: tuple[int, ...] = (2025, 2026)
    tax_parameters_dir: Path = Path("data/tax_parameters")
    system_prompt_path: Path = Path("prompts/system.md")

    # --- Privacy (spec section 9) ---------------------------------------------------
    trace_retention_days: int = Field(default=30, ge=0)

    @field_validator("embedding_dimension")
    @classmethod
    def _dimension_matches_schema(cls, value: int) -> int:
        if value != EMBEDDING_DIMENSION:
            raise ValueError(
                f"embedding_dimension={value} but the chunks.embedding column is "
                f"vector({EMBEDDING_DIMENSION}); add a migration before changing it"
            )
        return value

    @field_validator("supported_tax_years")
    @classmethod
    def _years_sorted_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("supported_tax_years must not be empty")
        return tuple(sorted(set(value)))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
