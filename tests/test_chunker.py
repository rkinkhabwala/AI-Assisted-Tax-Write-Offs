"""Chunker invariants (spec section 2) on real fixtures and synthetic edge cases."""

from collections.abc import Mapping
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import pytest
from pydantic import HttpUrl

from writeoff.chunking.chunker import (
    ChunkingConfig,
    CitationStyle,
    SourceContext,
    chunk_document,
)
from writeoff.chunking.tokens import estimate_tokens
from writeoff.ingestion.parsers import ParserName, parse
from writeoff.ingestion.tree import Block, BlockKind, Node, ParsedDocument, make_table, render
from writeoff.models import (
    CHUNK_HARD_MAX_TOKENS,
    Chunk,
    ChunkLevel,
    DocType,
    Document,
    EntityType,
    make_document_id,
    sha256_hex,
)

FIXTURES = Path(__file__).parent / "fixtures"
PASS_THROUGH = frozenset({EntityType.SOLE_PROP, EntityType.PARTNERSHIP, EntityType.S_CORP})

REAL = [
    ("uscode_280A.html", ParserName.USCODE_HTML, "IRC § 280A", DocType.IRC),
    ("uscode_168_a_to_c.html", ParserName.USCODE_HTML, "IRC § 168", DocType.IRC),
    ("ecfr_1.162-5.xml", ParserName.ECFR_XML, "Treas. Reg. § 1.162-5", DocType.TREASURY_REGULATION),
    (
        "ecfr_1.263a-1.xml",
        ParserName.ECFR_XML,
        "Treas. Reg. § 1.263(a)-1",
        DocType.TREASURY_REGULATION,
    ),
    ("irs_p463_excerpt.html", ParserName.IRS_HTML, "Pub 463", DocType.IRS_PUBLICATION),
    ("irs_i8829_2025.pdf", ParserName.PDF, "Instructions for Form 8829", DocType.FORM_INSTRUCTIONS),
]


def _document(parsed: ParsedDocument, doc_type: DocType, name: str = "doc") -> Document:
    url = f"https://www.irs.gov/test/{name}"
    text = render(parsed.root)
    return Document(
        id=make_document_id(url, 2025),
        source_url=HttpUrl(url),
        title=parsed.title,
        doc_type=doc_type,
        tax_year=2025,
        retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
        text=text,
        content_hash=sha256_hex(text),
    )


def _chunk(
    parsed: ParsedDocument,
    root: str,
    doc_type: DocType,
    config: ChunkingConfig | None = None,
    entity_overrides: Mapping[str, frozenset[EntityType]] | None = None,
) -> list[Chunk]:
    style = (
        CitationStyle.STATUTORY
        if doc_type in {DocType.IRC, DocType.TREASURY_REGULATION}
        else CitationStyle.HEADINGS
    )
    context = SourceContext(
        document=_document(parsed, doc_type),
        citation_root=root,
        style=style,
        entity_overrides=entity_overrides,
    )
    return chunk_document(parsed, context, config)


def _real(
    name: str, parser: ParserName, root: str, doc_type: DocType
) -> tuple[ParsedDocument, list[Chunk]]:
    parsed = parse(parser, (FIXTURES / name).read_bytes())
    return parsed, _chunk(parsed, root, doc_type)


def _children(chunks: list[Chunk]) -> list[Chunk]:
    return [c for c in chunks if c.level is ChunkLevel.CHILD]


def _tables(node: Node) -> list[Block]:
    out: list[Block] = []
    for item in node.content:
        if isinstance(item, Node):
            out.extend(_tables(item))
        elif item.kind is BlockKind.TABLE:
            out.append(item)
    return out


# --- spec-required invariants on every real fixture ----------------------------------


@pytest.mark.parametrize(("name", "parser", "root", "doc_type"), REAL)
def test_every_chunk_has_a_citation_path(
    name: str, parser: ParserName, root: str, doc_type: DocType
) -> None:
    _, chunks = _real(name, parser, root, doc_type)
    assert chunks
    for chunk in chunks:
        assert chunk.citation_path.startswith(root)
        assert chunk.breadcrumb


@pytest.mark.parametrize(("name", "parser", "root", "doc_type"), REAL)
def test_no_child_exceeds_hard_max(
    name: str, parser: ParserName, root: str, doc_type: DocType
) -> None:
    _, chunks = _real(name, parser, root, doc_type)
    for child in _children(chunks):
        assert child.token_count <= CHUNK_HARD_MAX_TOKENS
        assert estimate_tokens(child.text) <= CHUNK_HARD_MAX_TOKENS


@pytest.mark.parametrize(("name", "parser", "root", "doc_type"), REAL)
def test_never_splits_a_table(name: str, parser: ParserName, root: str, doc_type: DocType) -> None:
    parsed, chunks = _real(name, parser, root, doc_type)
    for table in _tables(parsed.root):
        if estimate_tokens(table.text) <= CHUNK_HARD_MAX_TOKENS:
            # A table that fits is always inside a single child chunk, intact.
            assert any(table.text in c.text for c in _children(chunks)), table.title


@pytest.mark.parametrize(("name", "parser", "root", "doc_type"), REAL)
def test_structure_ids_and_parents(
    name: str, parser: ParserName, root: str, doc_type: DocType
) -> None:
    _, chunks = _real(name, parser, root, doc_type)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    parents = {c.id: c for c in chunks if c.level is ChunkLevel.PARENT}
    assert parents
    for child in _children(chunks):
        assert child.parent_id in parents
        # A child never cites a provision outside its parent section.
        assert child.citation_path.startswith(parents[child.parent_id].citation_path)


@pytest.mark.parametrize(("name", "parser", "root", "doc_type"), REAL)
def test_chunking_is_deterministic(
    name: str, parser: ParserName, root: str, doc_type: DocType
) -> None:
    _, first = _real(name, parser, root, doc_type)
    _, second = _real(name, parser, root, doc_type)
    assert [(c.id, c.content_hash) for c in first] == [(c.id, c.content_hash) for c in second]


def test_parents_respect_parent_max() -> None:
    for name, parser, root, doc_type in REAL:
        _, chunks = _real(name, parser, root, doc_type)
        for parent in (c for c in chunks if c.level is ChunkLevel.PARENT):
            assert parent.token_count <= ChunkingConfig().parent_max, (name, parent.citation_path)


# --- citation precision --------------------------------------------------------------


def test_statutory_citation_is_the_exact_provision() -> None:
    _, chunks = _real(*REAL[0])
    home_office = next(
        c for c in _children(chunks) if "principal place of business for any" in c.text
    )
    assert home_office.citation_path == "IRC § 280A(c)(1)"
    assert home_office.breadcrumb == (
        "IRC § 280A — Disallowance of certain expenses in connection with business use of home, "
        "rental of vacation homes, etc. > (c) Exceptions for certain business or rental use; "
        "limitation on deductions for such use > (1) Certain business use"
    )
    # The chapeau travels with the subparagraphs it introduces, and the flush text too.
    assert "exclusively used on a regular basis" in home_office.text
    assert "In the case of an employee" in home_office.text


def test_table_chunk_cites_its_subsection() -> None:
    _, chunks = _real(*REAL[1])
    table_chunk = next(c for c in _children(chunks) if "| Nonresidential real property |" in c.text)
    assert table_chunk.citation_path == "IRC § 168(c)"
    assert "(c) Applicable recovery period" in table_chunk.text


def test_publication_citations_use_chapters_and_headings() -> None:
    _, chunks = _real(*REAL[4])
    tax_home = next(c for c in _children(chunks) if "Main place of business or work" in c.text)
    assert tax_home.citation_path.startswith("Pub 463, ch. 1, Traveling Away From Home")


def test_statutory_siblings_are_not_merged() -> None:
    _, chunks = _real(*REAL[0])
    for child in _children(chunks):
        if (
            child.citation_path == "IRC § 280A"
        ):  # only a chunk with root-level text may cite the root
            pytest.fail(f"imprecise root citation for: {child.text[:80]!r}")


# --- synthetic edge cases ------------------------------------------------------------


def _sentence(i: int) -> str:
    return f"Sentence {i} states a rule about deductible business expenses in some detail."


def _synthetic(*content: Block | Node) -> ParsedDocument:
    return ParsedDocument(
        title="IRC § 999 — Test section",
        root=Node(label="999", heading="Test", content=list(content)),
    )


def test_oversized_table_is_split_by_rows_with_header_repeated() -> None:
    rows = [[f"row {i}", "some descriptive text " * 12] for i in range(60)]
    table = make_table("Table 9. Big", [["Item", "Rule"]], rows)
    assert estimate_tokens(table.text) > CHUNK_HARD_MAX_TOKENS
    chunks = _chunk(
        _synthetic(Node(label="(a)", heading="Rates", content=[table])), "IRC § 999", DocType.IRC
    )
    pieces = [c for c in _children(chunks) if "| Item | Rule |" in c.text]
    assert len(pieces) > 1
    seen: list[str] = []
    for piece in pieces:
        assert piece.token_count <= CHUNK_HARD_MAX_TOKENS
        lines = piece.text.split("\n")
        assert "| Item | Rule |" in lines  # header repeated in every piece
        seen.extend(line for line in lines if line.startswith("| row "))
    assert seen == list(table.rows)  # every row exactly once, never cut
    # The parent still holds the whole table.
    parent = next(c for c in chunks if c.level is ChunkLevel.PARENT and table.rows[-1] in c.text)
    assert table.rows[0] in parent.text


def test_oversized_example_repeats_its_title() -> None:
    body = "\n".join(_sentence(i) for i in range(400))
    example = Block(BlockKind.EXAMPLE, f"Example 1. Big facts\n{body}")
    chunks = _chunk(
        _synthetic(Node(label="(a)", heading="Examples", content=[example])),
        "IRC § 999",
        DocType.IRC,
    )
    pieces = [c for c in _children(chunks) if "Example 1. Big facts" in c.text]
    assert len(pieces) > 1
    assert all("(continued)" in p.text for p in pieces[1:])


def test_long_paragraph_overlaps_only_within_its_section() -> None:
    long_text = " ".join(_sentence(i) for i in range(300))
    doc = _synthetic(
        Node(label="(a)", heading="Long", content=[Block(BlockKind.TEXT, long_text)]),
        Node(
            label="(b)", heading="Short", content=[Block(BlockKind.TEXT, "Short provision text.")]
        ),
    )
    children = _children(_chunk(doc, "IRC § 999", DocType.IRC))
    long_pieces = [c for c in children if c.citation_path == "IRC § 999(a)"]
    assert len(long_pieces) > 2
    for first, second in pairwise(long_pieces):
        last_sentence = first.text.rsplit(". ", 1)[-1]
        assert last_sentence in second.text  # sentence overlap between consecutive pieces
    short = [c for c in children if "Short provision text." in c.text]
    assert len(short) == 1
    assert short[0].citation_path == "IRC § 999(b)"
    assert "Sentence" not in short[0].text  # nothing from (a) leaks into (b)


def test_entity_overrides_use_longest_matching_citation_prefix() -> None:
    doc = _synthetic(
        Node(label="(a)", heading="General", content=[Block(BlockKind.TEXT, "General rule.")]),
        Node(
            label="(l)", heading="Health insurance", content=[Block(BlockKind.TEXT, "SE health.")]
        ),
    )
    chunks = _chunk(
        doc,
        "IRC § 999",
        DocType.IRC,
        config=ChunkingConfig(target_min=1, target_max=10, overlap=0),
        entity_overrides={"IRC § 999(l)": PASS_THROUGH},
    )
    by_citation = {c.citation_path: c.entity_types for c in _children(chunks)}
    assert by_citation["IRC § 999(l)"] == PASS_THROUGH
    assert by_citation["IRC § 999(a)"] == frozenset()


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="target_min"):
        ChunkingConfig(target_min=900, target_max=800)
    with pytest.raises(ValueError, match="overlap"):
        ChunkingConfig(overlap=400)


def test_publication_citations_drop_the_article_wrapper() -> None:
    body = [Block(BlockKind.TEXT, "Deduct the business part of your home expenses.")]
    doc = ParsedDocument(
        title="Publication 587 (2025), Business Use of Your Home",
        root=Node(
            heading="Publication 587 (2025), Business Use of Your Home",
            content=[
                Node(
                    heading="Publication 587 - Main Contents",
                    content=[
                        Node(heading="Figuring the Deduction", content=list(body)),
                    ],
                ),
                Node(
                    heading="Publication 587 - Introductory Material",
                    content=[
                        Node(heading="1. Overview", content=list(body)),
                    ],
                ),
            ],
        ),
    )
    tiny = ChunkingConfig(target_min=1, target_max=10, parent_max=10, overlap=0)
    chunks = _chunk(doc, "Pub 587", DocType.IRS_PUBLICATION, config=tiny)
    citations = {c.citation_path for c in chunks}
    assert "Pub 587, Figuring the Deduction" in citations
    assert "Pub 587, Introductory Material, 1. Overview" in citations  # chapters only lead


def test_merge_statutory_siblings_trades_precision_for_size() -> None:
    parsed = parse(ParserName.USCODE_HTML, (FIXTURES / "uscode_280A.html").read_bytes())
    strict = _children(_chunk(parsed, "IRC § 280A", DocType.IRC))
    merged = _children(
        _chunk(
            parsed, "IRC § 280A", DocType.IRC, config=ChunkingConfig(merge_statutory_siblings=True)
        )
    )
    assert len(merged) < len(strict)
    assert max(c.token_count for c in merged) <= CHUNK_HARD_MAX_TOKENS
