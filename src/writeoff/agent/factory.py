"""Assemble a ready-to-use WriteOffAgent from settings."""

import anthropic
import httpx

from writeoff.agent.harness import AgentConfig, WriteOffAgent
from writeoff.agent.prompt import load_system_prompt
from writeoff.agent.store import AgentStore
from writeoff.agent.tools import ToolRuntime
from writeoff.agent.verifier import ClaudeJudge, Verifier
from writeoff.config import Settings
from writeoff.retrieval.factory import build_retriever
from writeoff.tax_parameters import TaxParameters


def build_agent(
    settings: Settings, http: httpx.AsyncClient, *, persist: bool = True, verify: bool = True
) -> WriteOffAgent:
    runtime = ToolRuntime(
        TaxParameters(settings.tax_parameters_dir, settings.supported_tax_years),
        build_retriever(settings, http),
        timeout_seconds=settings.tool_timeout_seconds,
        max_weak_retries=settings.max_weak_search_retries,
    )
    config = AgentConfig(
        model=settings.agent_model,
        max_turns=settings.agent_max_turns,
        max_budget_usd=settings.agent_max_budget_usd,
        supported_years=settings.supported_tax_years,
    )
    store = AgentStore(str(settings.database_url)) if persist else None
    verifier = None
    if verify:
        if settings.anthropic_api_key is None:
            raise RuntimeError("ANTHROPIC_API_KEY is required for the grounding verifier")
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        verifier = Verifier(ClaudeJudge(client, settings.verifier_model))
    prompt = load_system_prompt(settings.system_prompt_path)
    return WriteOffAgent(config, prompt, runtime, store=store, verifier=verifier)
