"""Search the corpus from the command line and show how each result was found.

python -m writeoff.retrieval.cli "home office exclusive use requirement" --year 2025
python -m writeoff.retrieval.cli "Section 179 limit" --entity s_corp -k 5 --no-rewrite
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence

import httpx

from writeoff.config import get_settings
from writeoff.models import DocType, EntityType, SearchFilters
from writeoff.retrieval.factory import RetrieverConfigError, build_retriever
from writeoff.retrieval.hybrid import SearchResponse


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid search over the tax-law corpus.")
    parser.add_argument("query")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--entity", choices=[e.value for e in EntityType])
    parser.add_argument(
        "--doc-type", action="append", choices=[d.value for d in DocType], default=[]
    )
    parser.add_argument("-k", type=int, default=5, help="results to show")
    parser.add_argument("--no-rewrite", action="store_true", help="skip Claude query rewriting")
    parser.add_argument("--text", action="store_true", help="print each result's passage")
    return parser.parse_args(argv)


def _print(response: SearchResponse, k: int, show_text: bool) -> None:
    print(f"query:      {response.query}")
    print(f"rewritten:  {'; '.join(response.rewritten_terms) or '(none)'}")
    print(f"citations:  {', '.join(r.citation for r in response.citations) or '(none)'}")
    print(f"weak:       {response.weak}\n")
    print(f"{'#':>2}  {'rerank':>6}  {'rrf':>6}  {'dense':>5}  {'lex':>4}  via       citation")
    for r in response.results[:k]:
        via = "citation" if r.citation_lookup else "search"
        rrf = f"{r.fused_score:.4f}" if not r.citation_lookup else "-"
        print(
            f"{r.rank:>2}  {r.rerank_score or 0:6.3f}  {rrf:>6}  {r.dense_rank or '-':>5}  "
            f"{r.lexical_rank or '-':>4}  {via:<8}  {r.chunk.citation_path}"
        )
        if show_text:
            print("      " + r.chunk.text[:600].replace("\n", "\n      ") + "\n")


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    filters = SearchFilters(
        tax_year=args.year,
        doc_types=frozenset(DocType(d) for d in args.doc_type),
        entity_type=EntityType(args.entity) if args.entity else None,
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as http:
        try:
            retriever = build_retriever(settings, http, rewrite=not args.no_rewrite)
        except RetrieverConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _print(await retriever.search(args.query, filters), args.k, args.text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    return asyncio.run(run(_parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
