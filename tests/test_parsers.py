"""Parsers against trimmed real documents (tests/fixtures) plus synthetic edge cases."""

from pathlib import Path

import pytest

from writeoff.ingestion.normalize import normalize_text
from writeoff.ingestion.parsers import ParseError, ParserName, parse
from writeoff.ingestion.parsers.ecfr import parse_ecfr
from writeoff.ingestion.tree import Block, BlockKind, Node, ParsedDocument, render

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str, parser: ParserName) -> ParsedDocument:
    return parse(parser, (FIXTURES / name).read_bytes())


def _paths(node: Node, prefix: str = "") -> list[str]:
    out = []
    for child in node.children():
        out.append(prefix + child.label)
        out.extend(_paths(child, prefix + child.label))
    return out


def _find(node: Node, path: str) -> Node:
    for child in node.children():
        if path == child.label:
            return child
        if path.startswith(child.label):
            return _find(child, path.removeprefix(child.label))
    raise KeyError(path)


def _blocks(node: Node) -> list[Block]:
    out: list[Block] = []
    for item in node.content:
        out.extend(_blocks(item) if isinstance(item, Node) else [item])
    return out


def _headings(node: Node) -> list[str]:
    return [c.heading for c in node.children()]


# --- normalize -----------------------------------------------------------------------


def test_normalize_text() -> None:
    raw = (
        "  An\u00a0ordinary  expense\u00adis \u2018common\u2019 and "
        "\u201caccepted\u201d\u200b.\n\n\n\nNext  "
    )
    assert normalize_text(raw) == "An ordinary expenseis 'common' and \"accepted\".\n\nNext"
    assert normalize_text("§ 280A(c)(1)") == "§ 280A(c)(1)"


# --- uscode --------------------------------------------------------------------------


def test_uscode_title_and_exact_hierarchy() -> None:
    doc = _load("uscode_280A.html", ParserName.USCODE_HTML)
    assert doc.title.startswith("IRC § 280A — Disallowance of certain expenses")
    assert doc.root.label == "280A"
    paths = _paths(doc.root)
    assert {"(a)", "(b)", "(c)", "(c)(1)", "(c)(1)(A)", "(c)(4)(B)(iii)"} <= set(paths)
    assert _find(doc.root, "(c)(1)").heading == "Certain business use"


def test_uscode_flush_text_belongs_to_parent_after_children() -> None:
    doc = _load("uscode_280A.html", ParserName.USCODE_HTML)
    c1 = _find(doc.root, "(c)(1)")
    kinds = [type(item).__name__ for item in c1.content]
    assert kinds == ["Block", "Node", "Node", "Node", "Block"]  # chapeau, (A)-(C), flush text
    last = c1.content[-1]
    assert isinstance(last, Block)
    assert last.text.startswith("In the case of an employee")


def test_uscode_excludes_editorial_notes() -> None:
    doc = _load("uscode_280A.html", ParserName.USCODE_HTML)
    assert "editorial note" not in render(doc.root)


def test_uscode_table() -> None:
    doc = _load("uscode_168_a_to_c.html", ParserName.USCODE_HTML)
    tables = [b for b in _blocks(_find(doc.root, "(c)")) if b.kind is BlockKind.TABLE]
    assert len(tables) == 1
    table = tables[0]
    assert table.header[0] == "| In the case of: | The applicable recovery period is: |"
    assert "| Nonresidential real property | 39 years. |" in table.rows
    assert len(table.rows) == 10


def test_uscode_rejects_non_statute_page() -> None:
    with pytest.raises(ParseError, match="field"):
        parse(ParserName.USCODE_HTML, b"<html><body>Not found</body></html>")


# --- eCFR ----------------------------------------------------------------------------


def test_ecfr_hierarchy_with_chained_labels_and_italic_letters() -> None:
    doc = _load("ecfr_1.162-5.xml", ParserName.ECFR_XML)
    assert doc.title == "Treas. Reg. § 1.162-5 — Expenses for education."
    paths = _paths(doc.root)
    assert paths[:5] == ["(a)", "(a)(1)", "(a)(2)", "(b)", "(b)(1)"]
    # "(b) Heading—(1) In general." opens (b) and (b)(1) from one paragraph.
    assert _find(doc.root, "(b)").heading == "Nondeductible educational expenditures"
    assert _find(doc.root, "(b)(1)").heading == "In general"
    # Old-style italic letters sit below the roman numeral, not at the top level.
    assert "(b)(3)(i)(a)" in paths
    assert [p for p in paths if p.count("(") == 1] == ["(a)", "(b)", "(c)", "(d)", "(e)"]


def test_ecfr_keeps_label_inline_when_paragraph_has_no_heading() -> None:
    doc = _load("ecfr_1.162-5.xml", ParserName.ECFR_XML)
    first = _find(doc.root, "(b)(2)(i)").content[0]
    assert isinstance(first, Block)
    assert first.text.startswith("(i) The first category")


def test_ecfr_examples_are_atomic_blocks() -> None:
    doc = _load("ecfr_1.263a-1.xml", ParserName.ECFR_XML)
    examples = [b for b in _blocks(doc.root) if b.kind is BlockKind.EXAMPLE]
    assert examples
    assert examples[0].text.startswith("Example 1.")
    assert "\n\n" in examples[0].text  # title and body kept together in one block


def _ecfr(*paragraphs: str) -> str:
    body = "".join(f"<P>{p}</P>" for p in paragraphs)
    return f'<DIV8 N="1.1-1" TYPE="SECTION"><HEAD>§ 1.1-1 Test.</HEAD>{body}</DIV8>'


@pytest.mark.parametrize(
    ("paragraphs", "expected"),
    [
        # "(i)" right after "(h)" is the letter i ...
        (["(h) H.", "(i) I."], ["(h)", "(i)"]),
        # ... but inside a numbered paragraph it is roman one.
        (
            ["(a) A.", "(1) One.", "(i) Roman one.", "(ii) Roman two."],
            ["(a)", "(a)(1)", "(a)(1)(i)", "(a)(1)(ii)"],
        ),
        # Both readings possible: a chained digit child means it was a letter.
        (
            ["(h) H.", "(1) One.", "(i) <I>Heading</I>—(1) Child."],
            ["(h)", "(h)(1)", "(i)", "(i)(1)"],
        ),
        (["(h) H.", "(1) One.", "(i) Roman."], ["(h)", "(h)(1)", "(h)(1)(i)"]),
        # Italic digits nest below capital letters.
        (
            ["(a) A.", "(1) One.", "(i) R.", "(A) Cap.", "(<I>1</I>) Italic one."],
            ["(a)", "(a)(1)", "(a)(1)(i)", "(a)(1)(i)(A)", "(a)(1)(i)(A)(1)"],
        ),
    ],
)
def test_ecfr_label_disambiguation(paragraphs: list[str], expected: list[str]) -> None:
    doc = parse_ecfr(_ecfr(*paragraphs).encode())
    assert _paths(doc.root) == expected


def test_ecfr_rejects_invalid_xml() -> None:
    with pytest.raises(ParseError):
        parse(ParserName.ECFR_XML, b"<DIV8><HEAD>unclosed")


# --- irs.gov HTML --------------------------------------------------------------------


def test_irs_html_outline() -> None:
    doc = _load("irs_p463_excerpt.html", ParserName.IRS_HTML)
    assert doc.title == "Publication 463 (2025), Travel, Gift, and Car Expenses"
    chapters = _headings(doc.root)
    assert chapters == [
        "Publication 463 - Introductory Material",
        "1. Travel",
        "2. Meals and Entertainment",
    ]  # "How To Get Tax Help" is skipped
    travel = next(c for c in doc.root.children() if c.heading == "1. Travel")
    assert "Traveling Away From Home" in _headings(travel)
    away = next(c for c in travel.children() if c.heading == "Traveling Away From Home")
    assert "Tax Home" in _headings(away)


def test_irs_html_atomic_blocks() -> None:
    doc = _load("irs_p463_excerpt.html", ParserName.IRS_HTML)
    blocks = _blocks(doc.root)
    kinds = {b.kind for b in blocks}
    assert {BlockKind.TABLE, BlockKind.EXAMPLE, BlockKind.NOTE, BlockKind.TEXT} <= kinds
    table = next(b for b in blocks if b.kind is BlockKind.TABLE and "Table 1-1" in b.title)
    assert table.header[0].startswith("| IF you have expenses for...")
    assert any(r.startswith("| transportation |") for r in table.rows)
    example = next(b for b in blocks if b.kind is BlockKind.EXAMPLE)
    assert example.text.startswith("Example")


def test_irs_html_rejects_non_docbook_page() -> None:
    with pytest.raises(ParseError, match=r"div\.book"):
        parse(ParserName.IRS_HTML, b"<html><body><h1>Search</h1></body></html>")


# --- PDF -----------------------------------------------------------------------------


def test_pdf_title_sections_and_columns() -> None:
    doc = _load("irs_i8829_2025.pdf", ParserName.PDF)
    assert doc.title == "2025 Instructions for Form 8829"
    headings = [h for h in _headings(doc.root)]
    assert {"General Instructions", "Specific Instructions"} <= set(headings)
    text = render(doc.root)
    assert "Fileid" not in text  # proof marks outside the page box are dropped
    general = next(c for c in doc.root.children() if c.heading == "General Instructions")
    assert "Purpose of Form" in render(general)


def test_pdf_rejects_garbage() -> None:
    with pytest.raises(ParseError):
        parse(ParserName.PDF, b"not a pdf")


def test_pdf_drops_page_numbers() -> None:
    doc = _load("irs_i8829_2025.pdf", ParserName.PDF)
    for paragraph in render(doc.root).split("\n\n"):
        first = paragraph.split(" ", 1)[0]
        assert not (first.isdigit() and len(first) <= 2 and " " in paragraph), paragraph[:60]


def test_reserved_placeholders_are_pruned_but_pointers_kept() -> None:
    xml = (
        '<DIV8 N="1.1-1" TYPE="SECTION"><HEAD>§ 1.1-1 Test.</HEAD>'
        "<P>(a) Rule text.</P><P>(b) [Reserved]</P>"
        "<P>(c) [Reserved]. For further guidance, see § 1.274-5(c).</P></DIV8>"
    )
    doc = parse(ParserName.ECFR_XML, xml.encode())
    assert _paths(doc.root) == ["(a)", "(c)"]
