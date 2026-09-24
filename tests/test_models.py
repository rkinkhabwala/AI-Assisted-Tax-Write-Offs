from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from tests.factories import DOC_ID, RETRIEVED_AT, SOURCE_URL, chunk_kwargs, make_child, make_parent
from writeoff.models import (
    CHUNK_HARD_MAX_TOKENS,
    Chunk,
    ChunkLevel,
    DocType,
    Document,
    EmbeddedChunk,
    EntityType,
    RetrievalResult,
    SearchFilters,
    make_chunk_id,
    make_document_id,
    sha256_hex,
)

# --- Chunk --------------------------------------------------------------------------


def test_valid_parent_and_child() -> None:
    parent = make_parent()
    child = make_child(parent, entity_types=frozenset({EntityType.SOLE_PROP}))
    assert parent.level is ChunkLevel.PARENT
    assert child.parent_id == parent.id
    assert child.citation_path == "IRC § 280A(c)(1)(A)"
    assert child.entity_types == {EntityType.SOLE_PROP}


def test_chunk_round_trips_through_json() -> None:
    parent = make_parent()
    assert Chunk.model_validate_json(parent.model_dump_json()) == parent


def test_chunk_is_immutable() -> None:
    parent = make_parent()
    with pytest.raises(ValidationError):
        parent.text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"citation_path": "   "}, "citation_path"),
        ({"content_hash": "0" * 64}, "content_hash does not match"),
        ({"content_hash": "not-a-hash"}, "content_hash"),
        ({"tax_year": 1999}, "tax_year"),
        ({"token_count": 0}, "token_count"),
        ({"retrieved_at": datetime(2026, 9, 1)}, "timezone"),
        ({"source_url": "not a url"}, "source_url"),
        ({"doc_type": "blog_post"}, "doc_type"),
        ({"unexpected": "field"}, "Extra inputs"),
    ],
)
def test_chunk_rejects_invalid_fields(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Chunk(**chunk_kwargs(**overrides))


def test_child_requires_parent() -> None:
    with pytest.raises(ValidationError, match="must have a parent_id"):
        Chunk(**chunk_kwargs(level=ChunkLevel.CHILD))


def test_parent_must_not_have_parent() -> None:
    other = make_parent(citation_path="IRC § 280A")
    with pytest.raises(ValidationError, match="must not have a parent_id"):
        make_parent(parent_id=other.id)


def test_child_hard_max_tokens() -> None:
    parent = make_parent()
    make_child(parent, token_count=CHUNK_HARD_MAX_TOKENS)
    with pytest.raises(ValidationError, match="hard max"):
        make_child(parent, token_count=CHUNK_HARD_MAX_TOKENS + 1)


def test_parent_may_exceed_hard_max() -> None:
    assert make_parent(token_count=5000).token_count == 5000


def test_chunk_cannot_be_its_own_parent() -> None:
    kwargs = chunk_kwargs(level=ChunkLevel.CHILD)
    kwargs["parent_id"] = kwargs["id"]
    with pytest.raises(ValidationError, match="own parent"):
        Chunk(**kwargs)


def test_embedding_text_prepends_context_and_breadcrumb() -> None:
    chunk = make_parent(context_summary="From IRC § 280A on home offices.")
    assert chunk.embedding_text.split("\n\n") == [
        "From IRC § 280A on home offices.",
        chunk.breadcrumb,
        chunk.text,
    ]
    assert "From IRC" not in chunk.text


# --- Deterministic ids ---------------------------------------------------------------


def test_ids_are_deterministic_and_year_scoped() -> None:
    assert make_document_id(SOURCE_URL, 2025) == make_document_id(SOURCE_URL, 2025)
    assert make_document_id(SOURCE_URL, 2025) != make_document_id(SOURCE_URL, 2026)
    a = make_chunk_id(DOC_ID, ChunkLevel.CHILD, "IRC § 179(b)", 0)
    assert a == make_chunk_id(DOC_ID, ChunkLevel.CHILD, "IRC § 179(b)", 0)
    assert a != make_chunk_id(DOC_ID, ChunkLevel.CHILD, "IRC § 179(b)", 1)


# --- Document ------------------------------------------------------------------------


def _document(**overrides: Any) -> Document:
    text = "Publication 946 — How To Depreciate Property ..."
    url = "https://www.irs.gov/publications/p946"
    kwargs: dict[str, Any] = {
        "id": make_document_id(url, 2025),
        "source_url": url,
        "title": "Publication 946",
        "doc_type": DocType.IRS_PUBLICATION,
        "tax_year": 2025,
        "retrieved_at": RETRIEVED_AT,
        "text": text,
        "content_hash": sha256_hex(text),
    }
    kwargs.update(overrides)
    return Document(**kwargs)


def test_valid_document() -> None:
    assert _document().doc_type is DocType.IRS_PUBLICATION


def test_document_id_must_match_url_and_year() -> None:
    with pytest.raises(ValidationError, match="make_document_id"):
        _document(tax_year=2026)


def test_document_hash_must_match_text() -> None:
    with pytest.raises(ValidationError, match="content_hash"):
        _document(text="different text")


# --- EmbeddedChunk -------------------------------------------------------------------


def test_embedded_chunk_rules() -> None:
    parent = make_parent()
    child = make_child(parent)
    EmbeddedChunk(chunk=child, embedding=(0.1, 0.2))
    EmbeddedChunk(chunk=parent)
    with pytest.raises(ValidationError, match="must be embedded"):
        EmbeddedChunk(chunk=child)
    with pytest.raises(ValidationError, match="not embedded"):
        EmbeddedChunk(chunk=parent, embedding=(0.1,))


# --- RetrievalResult -----------------------------------------------------------------


def test_valid_retrieval_result() -> None:
    parent = make_parent()
    child = make_child(parent)
    result = RetrievalResult(
        chunk=child, parent=parent, rank=1, dense_rank=3, lexical_rank=1, fused_score=0.032
    )
    assert result.citation == "IRC § 280A(c)(1)(A)"


def test_citation_lookup_result_needs_no_retriever_rank() -> None:
    parent = make_parent()
    assert RetrievalResult(chunk=parent, rank=1, citation_lookup=True).fused_score == 0.0


def test_retrieval_result_requires_a_source() -> None:
    parent = make_parent()
    with pytest.raises(ValidationError, match="retriever or the citation fast path"):
        RetrievalResult(chunk=make_child(parent), rank=1)


def test_retrieval_result_parent_must_match() -> None:
    parent = make_parent()
    unrelated = make_parent(citation_path="IRC § 280A(d)")
    with pytest.raises(ValidationError, match="does not match"):
        RetrievalResult(chunk=make_child(parent), parent=unrelated, rank=1, dense_rank=1)


def test_retrieval_result_rejects_child_as_parent() -> None:
    parent = make_parent()
    child = make_child(parent)
    with pytest.raises(ValidationError, match="PARENT-level"):
        RetrievalResult(chunk=child, parent=child, rank=1, dense_rank=1)


# --- SearchFilters -------------------------------------------------------------------


def test_search_filters() -> None:
    filters = SearchFilters(tax_year=2025, doc_types=frozenset({"irc"}))  # type: ignore[arg-type]
    assert filters.doc_types == {DocType.IRC}
    with pytest.raises(ValidationError):
        SearchFilters(tax_year=1990)
