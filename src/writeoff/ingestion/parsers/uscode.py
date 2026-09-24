"""Parser for IRC sections from uscode.house.gov (`view.xhtml?req=granuleid:USC-prelim-...`).

The page marks every statutory unit with an anchor `substructure-location_c_1_A`, so the
section > subsection > paragraph > subparagraph hierarchy is read exactly rather than
inferred from labels. Paragraph classes encode indentation: `statutory-body-Nem` is text
of the current unit, while `statutory-body-block-Nem` is flush language after a list of
children, belonging to the ancestor at depth N+1. Editorial notes (amendments, effective
dates) fall outside the `field-start:statute` markers and are excluded.
"""

import re

from selectolax.lexbor import LexborHTMLParser, LexborNode

from writeoff.ingestion.parsers.common import ParseError, cell_texts, element_text
from writeoff.ingestion.tree import BlockKind, Node, ParsedDocument, make_table

_SECTION_HEAD = re.compile(r"^§\s*(?P<num>[0-9A-Za-z-]+)\.\s*(?P<title>.+)$", re.S)
_ANCHOR = "substructure-location_"
_BLOCK_DEPTH = re.compile(r"statutory-body-block-(\d+)em")
_HEAD_CLASSES = (
    "subsection-head",
    "paragraph-head",
    "subparagraph-head",
    "clause-head",
    "subclause-head",
    "item-head",
    "subitem-head",
)


def _field(html: str, name: str) -> str:
    start, end = f"<!-- field-start:{name} -->", f"<!-- field-end:{name} -->"
    i, j = html.find(start), html.find(end)
    if i < 0 or j < i:
        raise ParseError(f"uscode page has no {name!r} field")
    return html[i + len(start) : j]


class _Builder:
    def __init__(self, root: Node) -> None:
        self.root = root
        self.index: dict[tuple[str, ...], Node] = {(): root}
        self.path: tuple[str, ...] = ()

    def enter(self, path: tuple[str, ...]) -> None:
        for depth in range(1, len(path) + 1):
            key = path[:depth]
            if key not in self.index:
                node = Node(label=f"({key[-1]})")
                self.index[key[:-1]].content.append(node)
                self.index[key] = node
        self.path = path

    def node_at(self, depth: int | None = None) -> Node:
        path = self.path if depth is None else self.path[:depth]
        return self.index[path]


def parse_uscode(content: bytes) -> ParsedDocument:
    html = content.decode("utf-8", errors="replace")
    head_html = _field(html, "head")
    head = LexborHTMLParser(head_html).css_first("h3.section-head")
    match = _SECTION_HEAD.match(element_text(head)) if head else None
    if match is None:
        raise ParseError("uscode page has no parsable section heading")
    number, title = match["num"], match["title"].strip()
    root = Node(label=number, heading=title)
    builder = _Builder(root)

    body = LexborHTMLParser(_field(html, "statute")).body
    if body is None:
        raise ParseError("uscode statute field is empty")
    for el in body.iter():
        _handle(el, builder)
    if root.is_empty():
        raise ParseError(f"no statutory text found for § {number}")
    return ParsedDocument(title=f"IRC § {number} — {title}", root=root)


def _handle(el: LexborNode, builder: _Builder) -> None:
    classes = (el.attributes.get("class") or "").split()
    if el.tag == "a" and (name := el.attributes.get("name") or "").startswith(_ANCHOR):
        builder.enter(tuple(name.removeprefix(_ANCHOR).split("_")))
    elif el.tag == "h4" and any(c in _HEAD_CLASSES for c in classes):
        node = builder.node_at()
        heading = element_text(el)
        if node.label and heading.startswith(node.label):
            heading = heading.removeprefix(node.label).strip()
        node.heading = heading
    elif el.tag == "p":
        _drop_footnote_refs(el)
        depth_match = next((m for c in classes if (m := _BLOCK_DEPTH.fullmatch(c))), None)
        depth = int(depth_match.group(1)) + 1 if depth_match else None
        builder.node_at(depth).add_text(BlockKind.TEXT, element_text(el))
    elif el.tag == "table":
        header = [cell_texts(tr) for tr in el.css("tr") if tr.css_first("th")]
        rows = [cell_texts(tr) for tr in el.css("tr") if not tr.css_first("th")]
        caption = el.css_first("caption")
        builder.node_at().content.append(
            make_table(element_text(caption) if caption else "", header, rows)
        )


def _drop_footnote_refs(el: LexborNode) -> None:
    for sup in el.css("sup"):
        sup.decompose()
