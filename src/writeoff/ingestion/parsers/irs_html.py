"""Parser for IRS publications and form instructions on irs.gov (DocBook-generated HTML).

The page body is a `div.book` whose `div.article` / `div.chapter` / `div.section` elements
nest exactly like the document outline; each carries its heading in a `div.titlepage`,
a direct h1-h6 child, or (for run-in heads) a leading `p.inlinehd`. We walk that tree
directly instead of flattening headings, so H1/H2/H3 levels never need guessing.

Atomic blocks, never split by the chunker: `div.table` / `div.informaltable` (tables),
`div.example` (worked examples), `div.note` (Tip/Caution callouts), and figure sections
(the `role-figure` heading plus the flowchart's accessible text alternative).
"""

from selectolax.lexbor import LexborHTMLParser, LexborNode

from writeoff.ingestion.normalize import normalize_text
from writeoff.ingestion.parsers.common import ParseError, cell_texts, element_text
from writeoff.ingestion.tree import Block, BlockKind, Node, ParsedDocument, make_table

_CONTAINERS = frozenset(
    {"article", "chapter", "section", "appendix", "part", "preface", "glossary"}
)
_SKIPPED_HEADINGS = frozenset({"How To Get Tax Help", "Index"})
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_SKIPPED_CLASSES = frozenset({"index", "indexdiv", "sectioninfo", "caption", "toc"})


def parse_irs_html(content: bytes) -> ParsedDocument:
    tree = LexborHTMLParser(content.decode("utf-8", errors="replace"))
    book = tree.css_first("div.book")
    if book is None:
        raise ParseError("irs.gov page has no div.book (not a DocBook publication page)")
    h1 = book.css_first("div.titlepage h1")
    title = element_text(h1) if h1 else ""
    if not title:
        raise ParseError("irs.gov publication has no title")
    root = Node(heading=title)
    _walk(book, root, skip_first_titlepage=True)
    if root.is_empty():
        raise ParseError(f"no content parsed from {title!r}")
    return ParsedDocument(title=title, root=root)


def _classes(el: LexborNode) -> set[str]:
    return set((el.attributes.get("class") or "").split())


def _heading_of(container: LexborNode) -> tuple[str, str, LexborNode | None]:
    """(heading text, heading role class, element holding the heading)."""
    for child in container.iter():
        classes = _classes(child)
        if "titlepage" in classes:
            heading = child.css_first("h1, h2, h3, h4, h5, h6")
            if heading is not None:
                role = next((c for c in _classes(heading) if c.startswith("role-")), "")
                return element_text(heading), role, child
            return "", "", child
        if child.tag in _HEADING_TAGS:
            role = next((c for c in classes if c.startswith("role-")), "")
            return element_text(child), role, child
        if child.tag == "p" and "inlinehd" in classes:
            return element_text(child).rstrip(".:"), "role-inline", child
        if child.tag not in {"a", "-text", "-comment", "span"} and "sectioninfo" not in classes:
            break
    return "", "", None


def _walk(el: LexborNode, node: Node, *, skip_first_titlepage: bool = False) -> None:
    skipped_titlepage = not skip_first_titlepage
    for child in el.iter():
        classes = _classes(child)
        if child.tag == "div" and "titlepage" in classes and not skipped_titlepage:
            skipped_titlepage = True
            continue
        _handle(child, classes, node)


def _handle(child: LexborNode, classes: set[str], node: Node) -> None:  # noqa: PLR0912
    # One branch per DocBook element kind; a flat dispatch reads better than a table here.
    tag = child.tag
    if tag == "div" and classes & _CONTAINERS:
        heading, role, heading_el = _heading_of(child)
        if heading in _SKIPPED_HEADINGS:
            return
        if role == "role-figure":
            node.add_text(BlockKind.FIGURE, _figure_text(child, heading))
            return
        section = Node(heading=heading) if heading else node
        for grandchild in child.iter():
            if heading_el is not None and grandchild.mem_id == heading_el.mem_id:
                continue
            _handle(grandchild, _classes(grandchild), section)
        if section is not node and not section.is_empty():
            node.content.append(section)
    elif tag == "div" and classes & _SKIPPED_CLASSES:
        return
    elif tag == "div" and classes & {"table", "informaltable"}:
        node.content.append(_table(child))
    elif tag == "div" and "example" in classes:
        node.add_text(BlockKind.EXAMPLE, _block_text(child))
    elif tag == "div" and "note" in classes:
        node.add_text(BlockKind.NOTE, _block_text(child))
    elif tag == "div" and "textobject" in classes:
        node.add_text(BlockKind.FIGURE, _block_text(child))
    elif (tag == "div" and classes & {"itemizedlist", "orderedlist"}) or tag in {"ul", "ol"}:
        node.add_text(BlockKind.TEXT, _list_text(child))
    elif tag == "div":
        for grandchild in child.iter():
            _handle(grandchild, _classes(grandchild), node)
    elif tag in {"p", "blockquote", "pre"} or tag in _HEADING_TAGS:
        node.add_text(BlockKind.TEXT, element_text(child))
    elif tag == "table":
        node.content.append(_table(child))


def _block_text(el: LexborNode) -> str:
    """Paragraph-preserving text of a composite block (example, note, figure)."""
    parts = []
    for p in el.css("p, li, td, h1, h2, h3, h4, h5, h6"):
        if p.css_first("p, li, td") is not None:  # keep only the innermost text holders
            continue
        text = element_text(p)
        if text:
            parts.append(f"- {text}" if p.tag == "li" else text)
    return "\n".join(parts) if parts else element_text(el)


def _list_text(el: LexborNode) -> str:
    items = [normalize_text(li.text(deep=True, separator=" ")) for li in el.css("li")]
    return "\n".join(f"- {item}" for item in items if item)


def _figure_text(section: LexborNode, heading: str) -> str:
    alt = section.css_first("div.textobject")
    body = _block_text(alt) if alt is not None else ""
    return f"{heading}\n{body}" if body else heading


def _table(el: LexborNode) -> Block:
    title_el = el.css_first("p.title")
    title = element_text(title_el) if title_el is not None else ""
    header: list[list[str]] = []
    rows: list[list[str]] = []
    for tr in el.css("tr"):
        cells = cell_texts(tr)
        in_thead = tr.parent is not None and tr.parent.tag == "thead"
        if in_thead or (tr.css_first("th") is not None and tr.css_first("td") is None):
            header.append(cells)
        else:
            rows.append(cells)
    return make_table(title, header, rows)
