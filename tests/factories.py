"""Builders for valid model instances; tests override single fields to probe validation."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from writeoff.models import (
    Chunk,
    ChunkLevel,
    DocType,
    make_chunk_id,
    make_document_id,
    sha256_hex,
)

SOURCE_URL = "https://www.law.cornell.edu/uscode/text/26/280A"
RETRIEVED_AT = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
DOC_ID = make_document_id(SOURCE_URL, 2025)


def chunk_kwargs(
    *,
    level: ChunkLevel = ChunkLevel.PARENT,
    citation_path: str = "IRC § 280A(c)",
    text: str = "(c) Exceptions for certain business or rental use ...",
    parent_id: UUID | None = None,
    ordinal: int = 0,
    **overrides: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "id": make_chunk_id(DOC_ID, level, citation_path, ordinal),
        "document_id": DOC_ID,
        "parent_id": parent_id,
        "level": level,
        "ordinal": ordinal,
        "citation_path": citation_path,
        "breadcrumb": "IRC § 280A — Disallowance of certain expenses in connection with "
        "business use of home > (c) Exceptions",
        "text": text,
        "token_count": 250,
        "content_hash": sha256_hex(text),
        "source_url": SOURCE_URL,
        "title": "26 U.S. Code § 280A",
        "doc_type": DocType.IRC,
        "tax_year": 2025,
        "retrieved_at": RETRIEVED_AT,
    }
    kwargs.update(overrides)
    return kwargs


def make_parent(**overrides: Any) -> Chunk:
    return Chunk(**chunk_kwargs(**overrides))


def make_child(parent: Chunk, **overrides: Any) -> Chunk:
    defaults: dict[str, Any] = {
        "level": ChunkLevel.CHILD,
        "citation_path": "IRC § 280A(c)(1)(A)",
        "text": "(A) as the principal place of business for any trade or business ...",
        "parent_id": parent.id,
    }
    defaults.update(overrides)
    return Chunk(**chunk_kwargs(**defaults))
