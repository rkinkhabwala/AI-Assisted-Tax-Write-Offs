"""Core data models shared by ingestion, chunking, retrieval and the agent.

Design notes:
- IDs are deterministic (UUIDv5) so re-ingesting the same source for the same tax year
  targets the same rows. Combined with `content_hash`, that makes upserts idempotent:
  unchanged chunks are no-ops and a new tax year produces new rows, not overwrites.
- Parent/child: small CHILD chunks are embedded and searched; the PARENT section they
  belong to is what the agent reads, so nearby exceptions ("except as provided in...")
  are never cut off. Parents have no embedding and no size cap.
- `text` is the raw passage shown to users and cited. The breadcrumb and the generated
  context summary are prepended only in `embedding_text`, never stored in `text`.
"""

import hashlib
from datetime import date
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID, uuid5

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    model_validator,
)

CHUNK_HARD_MAX_TOKENS = 1200
MIN_TAX_YEAR = 2000
MAX_TAX_YEAR = 2100

_NAMESPACE = UUID("6f1c0d2e-5b8a-4f7e-9c3d-2a1b0e9f8d7c")

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
TaxYear = Annotated[int, Field(ge=MIN_TAX_YEAR, le=MAX_TAX_YEAR)]


class DocType(StrEnum):
    IRC = "irc"
    TREASURY_REGULATION = "treasury_regulation"
    IRS_PUBLICATION = "irs_publication"
    FORM_INSTRUCTIONS = "form_instructions"


class EntityType(StrEnum):
    # A single-member LLC is disregarded by default and files like a sole proprietor.
    SOLE_PROP = "sole_prop"
    PARTNERSHIP = "partnership"
    S_CORP = "s_corp"
    C_CORP = "c_corp"


class ChunkLevel(StrEnum):
    PARENT = "parent"
    CHILD = "child"


class SourceFormat(StrEnum):
    HTML = "html"
    PDF = "pdf"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_document_id(source_url: str, tax_year: int) -> UUID:
    return uuid5(_NAMESPACE, f"doc|{source_url}|{tax_year}")


def make_chunk_id(document_id: UUID, level: ChunkLevel, citation_path: str, ordinal: int) -> UUID:
    return uuid5(_NAMESPACE, f"chunk|{document_id}|{level}|{citation_path}|{ordinal}")


class SourceSpec(_Frozen):
    """A document to ingest: the input to a DocumentFetcher."""

    source_url: HttpUrl
    title: NonEmptyStr
    doc_type: DocType
    tax_year: TaxYear
    format: SourceFormat
    effective_date: date | None = None


class FetchedDocument(_Frozen):
    """Raw bytes returned by a DocumentFetcher, before parsing."""

    spec: SourceSpec
    content: bytes = Field(min_length=1)
    content_type: NonEmptyStr
    retrieved_at: AwareDatetime


class Document(_Frozen):
    """A parsed, normalized source document for one tax year."""

    id: UUID
    source_url: HttpUrl
    title: NonEmptyStr
    doc_type: DocType
    tax_year: TaxYear
    effective_date: date | None = None
    retrieved_at: AwareDatetime
    text: NonEmptyStr
    content_hash: Sha256Hex

    @model_validator(mode="after")
    def _check_integrity(self) -> Self:
        if self.content_hash != sha256_hex(self.text):
            raise ValueError("content_hash does not match sha256(text)")
        if self.id != make_document_id(str(self.source_url), self.tax_year):
            raise ValueError("id must be make_document_id(source_url, tax_year)")
        return self


class Chunk(_Frozen):
    """A retrievable, citable passage. See the module docstring for parent/child rules."""

    id: UUID
    document_id: UUID
    parent_id: UUID | None = None
    level: ChunkLevel
    ordinal: int = Field(
        ge=0,
        description="Occurrence index among this document's chunks with the same level "
        "and citation_path; keeps ids stable when unrelated sections change",
    )

    citation_path: NonEmptyStr = Field(description='e.g. "IRC § 280A(c)(1)(A)"')
    breadcrumb: NonEmptyStr = Field(
        description="Human-readable hierarchy, e.g. "
        '"IRC § 280A — Disallowance of ... > (c) Exceptions > (1) Certain business use"'
    )
    text: NonEmptyStr
    context_summary: str | None = Field(
        default=None, description="1-2 sentence LLM-generated context (contextual retrieval)"
    )
    token_count: int = Field(gt=0)
    content_hash: Sha256Hex

    source_url: HttpUrl
    title: NonEmptyStr
    doc_type: DocType
    tax_year: TaxYear
    effective_date: date | None = None
    retrieved_at: AwareDatetime
    # Empty means the passage applies to every entity type.
    entity_types: frozenset[EntityType] = frozenset()

    @model_validator(mode="after")
    def _check_integrity(self) -> Self:
        if self.content_hash != sha256_hex(self.text):
            raise ValueError("content_hash does not match sha256(text)")
        if self.level is ChunkLevel.CHILD:
            if self.parent_id is None:
                raise ValueError("child chunks must have a parent_id")
            if self.token_count > CHUNK_HARD_MAX_TOKENS:
                raise ValueError(
                    f"child chunk has {self.token_count} tokens; hard max is "
                    f"{CHUNK_HARD_MAX_TOKENS}"
                )
        elif self.parent_id is not None:
            raise ValueError("parent chunks must not have a parent_id")
        if self.parent_id == self.id:
            raise ValueError("a chunk cannot be its own parent")
        return self

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedder: context summary + breadcrumb + raw passage."""
        parts = [self.context_summary, self.breadcrumb, self.text]
        return "\n\n".join(p for p in parts if p)


class EmbeddedChunk(_Frozen):
    """A chunk paired with its vector for upsert. Parents carry no embedding."""

    chunk: Chunk
    embedding: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _check_embedding(self) -> Self:
        if self.chunk.level is ChunkLevel.CHILD and not self.embedding:
            raise ValueError("child chunks must be embedded")
        if self.chunk.level is ChunkLevel.PARENT and self.embedding is not None:
            raise ValueError("parent chunks are not embedded")
        return self


class SearchFilters(_Frozen):
    """Metadata filters applied to both dense and lexical search."""

    tax_year: TaxYear
    doc_types: frozenset[DocType] = frozenset()
    entity_type: EntityType | None = None


class ScoredChunk(_Frozen):
    """One hit from a single retriever (dense or lexical), before fusion."""

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)


class UpsertStats(_Frozen):
    inserted: int = Field(ge=0)
    updated: int = Field(ge=0)
    unchanged: int = Field(ge=0)
    deleted: int = Field(ge=0)


class StoredChunkState(_Frozen):
    """What the store already holds for a chunk id; decides whether re-ingest is a no-op."""

    content_hash: Sha256Hex
    parent_id: UUID | None
    has_context_summary: bool


class RerankScore(_Frozen):
    index: int = Field(ge=0, description="Index into the documents passed to the reranker")
    score: float


class RetrievalResult(_Frozen):
    """A final, fused and reranked hit handed to the agent."""

    chunk: Chunk = Field(description="The child chunk that matched")
    parent: Chunk | None = Field(default=None, description="Its parent section, for context")
    rank: int = Field(ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    lexical_rank: int | None = Field(default=None, ge=1)
    citation_lookup: bool = Field(
        default=False, description="Found via the explicit-citation fast path, not search"
    )
    fused_score: float = Field(default=0.0, ge=0, description="Reciprocal Rank Fusion score")
    rerank_score: float | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.dense_rank is None and self.lexical_rank is None and not self.citation_lookup:
            raise ValueError("a result must come from a retriever or the citation fast path")
        if self.parent is not None:
            if self.parent.level is not ChunkLevel.PARENT:
                raise ValueError("parent must be a PARENT-level chunk")
            if self.chunk.parent_id != self.parent.id:
                raise ValueError("parent does not match chunk.parent_id")
        return self

    @property
    def citation(self) -> str:
        return self.chunk.citation_path
