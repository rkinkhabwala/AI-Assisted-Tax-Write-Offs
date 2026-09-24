"""Tools that read the law: `search_tax_law`, `get_citation`, `get_tax_parameter`.

Search results reference their parent sections by id, and each section's text appears
once: several hits often share a parent, and repeating up to 2k tokens per hit would
waste the agent's context.
"""

import contextlib
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import Field

from writeoff.calculators.common import FrozenModel
from writeoff.models import ChunkLevel, DocType, EntityType, SearchFilters
from writeoff.retrieval.citations import extract_citations
from writeoff.retrieval.hybrid import HybridRetriever
from writeoff.tax_parameters import ParameterUnavailableError, TaxParameters

NO_AUTHORITY = (
    "No strong authority was found. Say that the sources don't cover this rather than "
    "answering from general knowledge, or retry with more specific tax terms."
)


class Passage(FrozenModel):
    rank: int
    chunk_id: str
    citation: str
    doc_type: DocType
    title: str
    source_url: str
    tax_year: int
    text: str
    section_id: str | None = None
    relevance: float | None = None
    via_citation: bool = False


class Section(FrozenModel):
    citation: str
    text: str


class SearchToolResult(FrozenModel):
    query: str
    rewritten_terms: list[str]
    weak: bool
    guidance: str | None = None
    passages: list[Passage]
    sections: dict[str, Section] = Field(default_factory=dict)


class CitationToolResult(FrozenModel):
    requested: str
    resolved: str | None
    found: bool
    passages: list[Passage] = Field(default_factory=list)
    guidance: str | None = None


class ParameterPeriodValue(FrozenModel):
    start: date
    end: date
    value: Decimal | None


class ParameterToolResult(FrozenModel):
    name: str
    tax_year: int
    status: Literal["ok", "unavailable", "unknown"]
    value: Decimal | None = None
    periods: list[ParameterPeriodValue] | None = None
    unit: str | None = None
    source_url: str | None = None
    description: str | None = None
    guidance: str | None = None
    known_parameters: list[str] = Field(default_factory=list)


async def search_tax_law(
    retriever: HybridRetriever,
    query: str,
    tax_year: int,
    entity_type: EntityType | None = None,
    doc_types: frozenset[DocType] = frozenset(),
) -> SearchToolResult:
    response = await retriever.search(
        query, SearchFilters(tax_year=tax_year, doc_types=doc_types, entity_type=entity_type)
    )
    passages: list[Passage] = []
    sections: dict[str, Section] = {}
    for r in response.results:
        section_id = None
        if r.parent is not None:
            section_id = str(r.parent.id)
            sections.setdefault(
                section_id, Section(citation=r.parent.citation_path, text=r.parent.text)
            )
        passages.append(
            Passage(
                rank=r.rank,
                chunk_id=str(r.chunk.id),
                citation=r.chunk.citation_path,
                doc_type=r.chunk.doc_type,
                title=r.chunk.title,
                source_url=str(r.chunk.source_url),
                tax_year=r.chunk.tax_year,
                text=r.chunk.text,
                section_id=section_id,
                relevance=r.rerank_score,
                via_citation=r.citation_lookup,
            )
        )
    return SearchToolResult(
        query=query,
        rewritten_terms=response.rewritten_terms,
        weak=response.weak,
        guidance=NO_AUTHORITY if response.weak else None,
        passages=passages,
        sections=sections,
    )


def normalize_citation(citation: str) -> str:
    """Accept '§ 179(b)', 'section 179(b)', 'Reg. 1.162-5' or a stored path as-is."""
    stripped = citation.strip()
    if stripped.startswith(("IRC §", "Treas. Reg. §", "Pub ", "Instructions for ")):
        return stripped
    refs = extract_citations(stripped)
    return refs[0].citation if refs else stripped


async def get_citation(
    retriever: HybridRetriever, citation: str, tax_year: int
) -> CitationToolResult:
    resolved = normalize_citation(citation)
    chunks = await retriever.lookup_citation(resolved, tax_year)
    if not chunks:
        return CitationToolResult(
            requested=citation,
            resolved=resolved,
            found=False,
            guidance=f"{resolved} is not in the {tax_year} sources. Don't quote it from memory.",
        )
    passages = [
        Passage(
            rank=i,
            chunk_id=str(c.id),
            citation=c.citation_path,
            doc_type=c.doc_type,
            title=c.title,
            source_url=str(c.source_url),
            tax_year=c.tax_year,
            text=c.text,
            section_id=str(c.parent_id) if c.parent_id else None,
        )
        for i, c in enumerate((c for c in chunks if c.level is ChunkLevel.PARENT), start=1)
    ]
    if not passages:  # no parent sections: return the matching passages themselves
        passages = [
            Passage(
                rank=i,
                chunk_id=str(c.id),
                citation=c.citation_path,
                doc_type=c.doc_type,
                title=c.title,
                source_url=str(c.source_url),
                tax_year=c.tax_year,
                text=c.text,
            )
            for i, c in enumerate(chunks, start=1)
        ]
    return CitationToolResult(requested=citation, resolved=resolved, found=True, passages=passages)


def get_tax_parameter(params: TaxParameters, name: str, tax_year: int) -> ParameterToolResult:
    try:
        param = params.describe(name, tax_year)
    except ParameterUnavailableError as exc:
        known: list[str] = []
        with contextlib.suppress(ParameterUnavailableError):
            known = sorted(params.for_year(tax_year).parameters)
        return ParameterToolResult(
            name=name,
            tax_year=tax_year,
            status="unknown",
            guidance=exc.reason,
            known_parameters=known,
        )
    common = {
        "name": name,
        "tax_year": tax_year,
        "unit": param.unit.value,
        "source_url": str(param.source_url),
        "description": param.description,
    }
    if not param.is_verified:
        return ParameterToolResult.model_validate(
            {
                **common,
                "status": "unavailable",
                "guidance": "Not yet verified on irs.gov. Say the figure is unavailable; "
                "don't estimate it.",
            }
        )
    periods = None
    if param.periods is not None:
        periods = [
            ParameterPeriodValue(start=p.start, end=p.end, value=p.value) for p in param.periods
        ]
    return ParameterToolResult.model_validate(
        {**common, "status": "ok", "value": param.value, "periods": periods}
    )
