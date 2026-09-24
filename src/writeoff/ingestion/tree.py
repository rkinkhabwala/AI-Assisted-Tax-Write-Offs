"""Source-independent document tree produced by every parser and consumed by the chunker.

A `Node` is one structural unit: an IRC subsection "(c)", a regulation paragraph "(b)(2)",
or a publication heading. Its `content` keeps blocks and child nodes in document order,
because statutes interleave them: a chapeau ("...exclusively used on a regular basis-")
precedes subparagraphs (A)-(C), and flush language after them still belongs to the
paragraph. Blocks are the smallest units the chunker places: text paragraphs may be split
at sentence boundaries, but tables, examples, notes and figures are atomic.
"""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from writeoff.ingestion.normalize import normalize_text


class BlockKind(StrEnum):
    TEXT = "text"
    TABLE = "table"
    EXAMPLE = "example"
    NOTE = "note"
    FIGURE = "figure"


@dataclass(frozen=True, slots=True)
class Block:
    kind: BlockKind
    text: str
    # Tables only: pre-rendered title, header and body lines, so an oversized table can be
    # split into row groups that each repeat the title and header.
    title: str = ""
    header: tuple[str, ...] = ()
    rows: tuple[str, ...] = ()

    @property
    def is_atomic(self) -> bool:
        return self.kind is not BlockKind.TEXT


@dataclass(slots=True)
class Node:
    label: str = ""  # "(c)", "(1)"; empty for heading-based documents
    heading: str = ""
    content: list["Block | Node"] = field(default_factory=list)

    @property
    def heading_line(self) -> str:
        return " ".join(part for part in (self.label, self.heading) if part)

    def children(self) -> Iterator["Node"]:
        return (item for item in self.content if isinstance(item, Node))

    def add_text(self, kind: BlockKind, text: str) -> None:
        text = normalize_text(text)
        if text:
            self.content.append(Block(kind, text))

    def is_empty(self) -> bool:
        return not self.content


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Parser output. `title` heads every breadcrumb; `root.content` is the body."""

    title: str
    root: Node


def _cell(text: str) -> str:
    return normalize_text(text).replace("\n", " ").replace("|", "/")


def make_table(title: str, header: Sequence[Sequence[str]], rows: Sequence[Sequence[str]]) -> Block:
    """Render a table as pipe-delimited lines. Empty rows are dropped."""
    header_lines = [
        f"| {' | '.join(_cell(c) for c in r)} |" for r in header if any(map(str.strip, r))
    ]
    if header_lines:
        width = max(len(r) for r in header)
        header_lines.append("|" + " --- |" * width)
    row_lines = [f"| {' | '.join(_cell(c) for c in r)} |" for r in rows if any(map(str.strip, r))]
    title = _cell(title)
    text = "\n".join([title, *header_lines, *row_lines] if title else [*header_lines, *row_lines])
    return Block(BlockKind.TABLE, text, title, tuple(header_lines), tuple(row_lines))


_RESERVED = re.compile(r"^(?:\([0-9A-Za-z]+\)\s*)*\[Reserved\]\.?$")


def prune_reserved(node: Node) -> Node:
    """Drop provisions whose only content is "(x) [Reserved]": placeholders with nothing to
    retrieve. Reserved paragraphs that point elsewhere ("see § 1.274-5(g)") are kept."""
    kept: list[Block | Node] = []
    for item in node.content:
        if isinstance(item, Node):
            prune_reserved(item)
            if item.is_empty():
                continue
        elif _RESERVED.match(item.text):
            continue
        kept.append(item)
    node.content = kept
    return node


def render(node: Node, *, include_heading: bool = True) -> str:
    """Full text of a subtree, headings included, paragraphs separated by blank lines."""
    parts: list[str] = []
    if include_heading and node.heading:  # label-only nodes carry the label in their text
        parts.append(node.heading_line)
    for item in node.content:
        text = render(item) if isinstance(item, Node) else item.text
        if text:
            parts.append(text)
    return "\n\n".join(parts)
