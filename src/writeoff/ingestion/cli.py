"""Command-line ingestion.

    python -m writeoff.ingestion.cli --year 2025                 # full run (needs API keys)
    python -m writeoff.ingestion.cli --year 2025 --dry-run       # fetch, parse, chunk only
    python -m writeoff.ingestion.cli --year 2025 --source pub-463 --source irc-280a

A dry run writes the chunks as JSON Lines for inspection and needs no keys or database.
A full run embeds with Voyage, summarizes with Claude (skip with --no-context), and syncs
into Postgres. A failing source is reported and the run continues. The exit code is
non-zero if any source failed.
"""

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import anthropic
import httpx

from writeoff.chunking.context import (
    CachingSummarizer,
    ClaudeContextSummarizer,
    ContextSummarizer,
)
from writeoff.config import Settings, get_settings
from writeoff.ingestion.fetcher import CachingFetcher, HttpDocumentFetcher
from writeoff.ingestion.pipeline import IngestionPipeline, IngestResult
from writeoff.ingestion.registry import load_registry
from writeoff.models import ChunkLevel
from writeoff.retrieval.pgvector_store import PgVectorStore
from writeoff.retrieval.voyage import VoyageEmbedder

logger = logging.getLogger("writeoff.ingest")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest tax sources into the vector store.")
    parser.add_argument("--year", type=int, required=True, help="tax year to ingest")
    parser.add_argument("--source", action="append", default=[], help="source id (repeatable)")
    parser.add_argument(
        "--dry-run", action="store_true", help="fetch/parse/chunk only; write JSONL"
    )
    parser.add_argument(
        "--out", type=Path, help="dry-run output (default data/staging/chunks-YEAR.jsonl)"
    )
    parser.add_argument("--no-context", action="store_true", help="skip Claude context summaries")
    parser.add_argument(
        "--refresh", action="store_true", help="re-download instead of using the cache"
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace, settings: Settings) -> int:
    if args.year not in settings.supported_tax_years:
        logger.error(
            "tax year %d is not in SUPPORTED_TAX_YEARS %s", args.year, settings.supported_tax_years
        )
        return 2
    try:
        entries = load_registry(settings.sources_path).select(
            args.year, frozenset(args.source) or None
        )
    except KeyError as exc:  # unknown --source ids
        logger.error("%s", exc.args[0])
        return 2
    if not args.dry_run and settings.voyage_api_key is None:
        logger.error("VOYAGE_API_KEY is not set (use --dry-run to chunk without embedding)")
        return 2
    if not args.dry_run and not args.no_context and settings.anthropic_api_key is None:
        logger.error("ANTHROPIC_API_KEY is not set (use --no-context to skip context summaries)")
        return 2

    timeout = httpx.Timeout(60.0, connect=10.0)
    async with (
        httpx.AsyncClient(
            headers={"User-Agent": settings.http_user_agent}, timeout=timeout, follow_redirects=True
        ) as web,
        httpx.AsyncClient(timeout=timeout) as api,
    ):
        fetcher = CachingFetcher(
            HttpDocumentFetcher(web, min_interval=settings.fetch_min_interval_seconds),
            settings.raw_cache_dir,
            refresh=args.refresh,
        )
        pipeline = _pipeline(args, settings, fetcher, api)
        results: list[IngestResult] = []
        failures = 0
        for entry in entries:
            try:
                results.append(await pipeline.ingest(entry, args.year))
            except Exception:  # report and continue with the remaining sources
                failures += 1
                logger.exception("failed: %s (%d)", entry.id, args.year)
                continue
            _report(results[-1])
    if args.dry_run:
        out = args.out or Path(f"data/staging/chunks-{args.year}.jsonl")
        _write_jsonl(results, out)
        logger.info("wrote %d chunks to %s", sum(len(r.chunks) for r in results), out)
    logger.info("%d sources ingested, %d failed", len(results), failures)
    return 1 if failures else 0


def _pipeline(
    args: argparse.Namespace, settings: Settings, fetcher: CachingFetcher, api: httpx.AsyncClient
) -> IngestionPipeline:
    if args.dry_run:
        return IngestionPipeline(fetcher)
    assert settings.voyage_api_key is not None  # noqa: S101 - checked in run()
    summarizer: ContextSummarizer | None = None
    if not args.no_context and settings.anthropic_api_key is not None:
        client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(), max_retries=5
        )
        summarizer = CachingSummarizer(
            ClaudeContextSummarizer(client, settings.context_model),
            settings.context_cache_dir,
            model=settings.context_model,
        )
    return IngestionPipeline(
        fetcher,
        store=PgVectorStore(str(settings.database_url)),
        embedder=VoyageEmbedder(
            api,
            settings.voyage_api_key.get_secret_value(),
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
        ),
        summarizer=summarizer,
        summary_concurrency=settings.context_concurrency,
    )


def _report(result: IngestResult) -> None:
    parents = sum(c.level is ChunkLevel.PARENT for c in result.chunks)
    children = len(result.chunks) - parents
    stats = f" | {result.stats}" if result.stats else ""
    logger.info(
        "%-16s %d  parents=%-4d children=%-4d%s",
        result.source_id,
        result.tax_year,
        parents,
        children,
        stats,
    )


def _write_jsonl(results: Sequence[IngestResult], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for result in results:
            for chunk in result.chunks:
                record = {"source_id": result.source_id, **json.loads(chunk.model_dump_json())}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return asyncio.run(run(_parse_args(argv), get_settings()))


if __name__ == "__main__":
    sys.exit(main())
