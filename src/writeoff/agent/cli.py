"""Ask the agent one question and print the full trace.

python -m writeoff.agent.cli "Can I deduct a lunch with a client?" --entity sole_prop --year 2025
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from uuid import UUID

import httpx

from writeoff.agent.factory import build_agent
from writeoff.agent.harness import AgentAnswer
from writeoff.config import get_settings
from writeoff.models import EntityType


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask WriteOff Assistant a question.")
    parser.add_argument("question")
    parser.add_argument("--entity", choices=[e.value for e in EntityType])
    parser.add_argument("--year", type=int)
    parser.add_argument("--session", type=UUID, help="continue an existing session")
    parser.add_argument("--no-verify", action="store_true", help="skip the grounding verifier")
    parser.add_argument(
        "--no-persist", action="store_true", help="don't write the trace to Postgres"
    )
    return parser.parse_args(argv)


def print_trace(answer: AgentAnswer) -> None:
    calls = {c.tool_use_id: c for c in answer.tool_calls}
    print(
        f"request {answer.request_id}  session {answer.session_id}  "
        f"prompt v{answer.prompt_version}\n"
    )
    tool_steps = [s for s in answer.steps if s.kind == "tool_use"]
    for number, step in enumerate(tool_steps, start=1):
        name = step.detail.removeprefix("mcp__writeoff__")
        args = json.dumps(step.tool_input, ensure_ascii=False)
        print(f"[{number}] {name}({args[:300]}{'...' if len(args) > 300 else ''})")
        call = calls.get(step.tool_use_id or "")
        if call is not None:
            chunks = f", {len(call.chunk_ids)} chunks" if call.chunk_ids else ""
            detail = f" ({call.detail})" if call.detail else ""
            print(
                f"     -> {call.status}, {call.latency_ms} ms, "
                f"{call.result_chars} chars{chunks}{detail}"
            )
    report = answer.verification
    if report is not None:
        print(
            f"\nverification: {report.status} after {report.rounds} round(s) | claims "
            f"supported {report.count('SUPPORTED')} / partial {report.count('PARTIALLY_SUPPORTED')}"
            f" / unsupported {report.count('UNSUPPORTED')} | unknown citations "
            f"{report.unknown_citations or '-'} | untraced figures {report.untraced_numbers or '-'}"
        )
        for claim in report.claims:
            if claim.label != "SUPPORTED":
                print(f"   {claim.label}: {claim.claim[:140]}")
                print(f"      why: {claim.reason[:200]}")
        if report.unconfirmed:
            print(f"   unconfirmed: {report.unconfirmed}")
    print(f"\n{'=' * 78}\n{answer.text}\n{'=' * 78}")
    print(
        f"status {answer.status} ({answer.stop_reason}) | turns {answer.num_turns} | "
        f"tool calls {len(answer.tool_calls)} | tokens in/out {answer.input_tokens}/"
        f"{answer.output_tokens} | cost ${answer.cost_usd or 0:.4f} | "
        f"{answer.latency_ms / 1000:.1f}s"
    )


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as http:
        agent = build_agent(settings, http, persist=not args.no_persist, verify=not args.no_verify)
        answer = await agent.ask(
            args.question,
            session_id=args.session,
            entity_type=EntityType(args.entity) if args.entity else None,
            tax_year=args.year,
        )
    print_trace(answer)
    return 0 if answer.status != "error" else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(run(_parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
