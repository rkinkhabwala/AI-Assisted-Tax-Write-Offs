"""Data retention (spec section 9): delete old traces and idle sessions.

    python -m writeoff.retention     # TRACE_RETENTION_DAYS and SESSION_RETENTION_DAYS

`TRACE_RETENTION_DAYS=0` means traces are never written (so there is nothing to delete);
`SESSION_RETENTION_DAYS=0` keeps sessions. The API also runs this once a day (see
`writeoff.api.app`).
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from writeoff.agent.store import AgentStore
from writeoff.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PurgeResult:
    requests: int
    sessions: int


async def purge(store: AgentStore, trace_days: int, session_days: int) -> PurgeResult:
    """Traces older than `trace_days`, then sessions idle for `session_days` (their
    traces cascade). A window of 0 skips that step."""
    requests = await store.purge_traces(trace_days) if trace_days > 0 else 0
    sessions = await store.purge_sessions(session_days) if session_days > 0 else 0
    logger.info("retention: deleted %d requests and %d sessions", requests, sessions)
    return PurgeResult(requests, sessions)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delete traces and sessions past retention")
    parser.parse_args(argv)
    settings = get_settings()
    trace_days, session_days = settings.trace_retention_days, settings.session_retention_days
    store = AgentStore(str(settings.database_url))
    result = asyncio.run(purge(store, trace_days, session_days))
    print(
        f"deleted {result.requests} requests (window {trace_days} days) and "
        f"{result.sessions} sessions (window {session_days} days)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
