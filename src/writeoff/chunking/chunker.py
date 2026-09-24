"""Structure-aware chunking (spec section 2).

Two passes over a `ParsedDocument` tree:

1. Parents. Walking down from the document root, a node becomes a PARENT chunk if its
   whole subtree fits in `parent_max` tokens. Otherwise its content (own text and
   child sections) is packed in order into parents of up to `parent_max`, and only
   child sections that are themselves too big are descended into. Parents are what
   the agent reads when one of their children matches, so `parent_max` (2,000) is sized
   to hold a typical IRC subsection or publication H2 section, where "except as
   provided in..." qualifications live, while keeping the reranked top-8 context near
   16k tokens. The one exception is a single table larger than `parent_max`. Tables are
   never split, so it becomes a parent on its own.

2. Children. The same fit-or-descend rule at `target_max` (800) tokens: a CHILD chunk is
   the largest subtree that fits, and when a node is too big its content is packed in
   order. For statutes, two sibling provisions are never merged, so a chunk holding
   (c)(1) cites exactly "IRC § 280A(c)(1)" rather than "IRC § 280A". A chapeau still
   joins the first provision it introduces. Publications merge small sibling sections
   up to `target_min` (300), because their heading-based citations lose little by it.
   No chunk exceeds `hard_max` (1,200). Heading lines are glued to the content they
   introduce. Tables, examples, notes and figures are atomic. One larger than
   the hard max is split into pieces that each repeat its title (and, for tables, its
   header row), with the parent still holding it whole. A long text paragraph is split
   at sentence boundaries with `overlap` tokens of overlap. That is the only overlap we
   create, and it never crosses a section boundary, which would mislead citations.

A chunk's citation path is the deepest node containing all of its content.
"""

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from writeoff.chunking.tokens import estimate_tokens
from writeoff.ingestion.tree import Block, BlockKind, Node, ParsedDocument, render
from writeoff.models import (
    CHUNK_HARD_MAX_TOKENS,
    Chunk,
    ChunkLevel,
    Document,
    EntityType,
    make_chunk_id,
    sha256_hex,
)

Path = tuple[Node, ...]
_SEP = "\n\n"
_SENTENCE_END = re.compile(r"(?<=[.;:?!])\s+(?=[\"'(A-Z0-9§])|\n+")
_CHAPTER = re.compile(r"^(\d+)\.\s+\S")
_ARTICLE = re.compile(r"^(?:Publication|Instructions for)\b.* - (?P<part>.+)$")
_DROPPED_ARTICLES = frozenset({"Main Contents"})


class ChunkingError(RuntimeError):
    """The chunker produced output violating its own invariants (a bug, not bad input)."""


class CitationStyle(StrEnum):
    STATUTORY = "statutory"  # IRC § 280A(c)(1)(A): labels concatenated
    HEADINGS = "headings"  # Pub 463, ch. 1, Tax Home: headings comma-joined


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_min: int = 300
    target_max: int = 800
    hard_max: int = CHUNK_HARD_MAX_TOKENS
    parent_max: int = 2000
    overlap: int = 60
    # Pack small sibling provisions together like publication sections, at the cost of
    # coarser citations. Off by default; phase 4 measures the trade-off.
    merge_statutory_siblings: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.target_min <= self.target_max <= self.hard_max <= CHUNK_HARD_MAX_TOKENS:
            raise ValueError("require 0 < target_min <= target_max <= hard_max <= 1200")
        if not 0 <= self.overlap < self.target_min:
            raise ValueError("overlap must be smaller than target_min")


@dataclass(frozen=True, slots=True)
class SourceContext:
    """Everything the chunker needs besides the tree."""

    document: Document
    citation_root: str  # "IRC § 280A", "Treas. Reg. § 1.162-5", "Pub 463"
    style: CitationStyle
    entity_types: frozenset[EntityType] = frozenset()
    # Citation prefix -> entity types, e.g. {"IRC § 162(l)": {sole_prop, partnership, s_corp}}
    entity_overrides: Mapping[str, frozenset[EntityType]] | None = None


@dataclass(frozen=True, slots=True)
class _Group:
    """A run of content under one node, with the heading line(s) that introduce it."""

    path: Path
    heading: str
    content: tuple[Block | Node, ...]

    @property
    def text(self) -> str:
        return _join([self.heading, *(_text(item) for item in self.content)])


class _ChildBuffer:
    """Accumulates one child chunk; tracks the paths of its content for the citation."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.paths: list[Path] = []
        self.has_node = False

    @property
    def tokens(self) -> int:
        return estimate_tokens(_join(self.parts)) if self.parts else 0

    def tokens_with(self, text: str) -> int:
        return estimate_tokens(_join([*self.parts, text]))


def chunk_document(
    parsed: ParsedDocument, ctx: SourceContext, config: ChunkingConfig | None = None
) -> list[Chunk]:
    return _Chunker(parsed, ctx, config or ChunkingConfig()).run()


class _Chunker:
    def __init__(self, parsed: ParsedDocument, ctx: SourceContext, config: ChunkingConfig) -> None:
        self.parsed = parsed
        self.ctx = ctx
        self.config = config
        self.strict = ctx.style is CitationStyle.STATUTORY and not config.merge_statutory_siblings
        self.ordinals: Counter[tuple[ChunkLevel, str]] = Counter()

    def run(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        for group in self._parents(self.parsed.root, (), ""):
            parent = self._chunk(ChunkLevel.PARENT, group.path, group.text, parent=None)
            chunks.append(parent)
            for path, text in self._children(group.content, group.path, group.heading):
                child = self._chunk(ChunkLevel.CHILD, path, text, parent=parent)
                if child.token_count > self.config.hard_max:
                    raise ChunkingError(f"child {child.citation_path} exceeds hard max")
                chunks.append(child)
        return chunks

    # --- pass 1: parents -------------------------------------------------------------

    def _parents(self, node: Node, path: Path, heading: str) -> list[_Group]:
        whole = _Group(path, heading, tuple(node.content))
        if not node.content:
            return []
        if estimate_tokens(whole.text) <= self.config.parent_max:
            return [whole]
        # Too big: pack consecutive blocks and small child sections into parents of up to
        # parent_max, descending only into child sections that are too big themselves.
        groups: list[_Group] = []
        buffer: list[Block | Node] = []
        for item in node.content:
            if isinstance(item, Node) and estimate_tokens(_text(item)) > self.config.parent_max:
                groups.extend(self._group(path, heading, buffer))
                buffer = []
                groups.extend(self._parents(item, (*path, item), _heading(item)))
                continue
            if (
                buffer
                and estimate_tokens(_Group(path, heading, (*buffer, item)).text)
                > self.config.parent_max
            ):
                groups.extend(self._group(path, heading, buffer))
                buffer = []
            buffer.append(item)
        groups.extend(self._group(path, heading, buffer))
        return groups

    @staticmethod
    def _group(path: Path, heading: str, items: Sequence[Block | Node]) -> list[_Group]:
        if not items:
            return []
        if len(items) == 1 and isinstance(items[0], Node):
            node = items[0]
            return [_Group((*path, node), _heading(node), tuple(node.content))]
        return [_Group(path, heading, tuple(items))]

    # --- pass 2: children ------------------------------------------------------------

    def _children(
        self, content: Sequence[Block | Node], path: Path, pending: str
    ) -> list[tuple[Path, str]]:
        """Child chunks for `content` under `path`; `pending` heading text leads the first."""
        cfg = self.config
        whole = _join([pending, *(_text(item) for item in content)])
        if content and estimate_tokens(whole) <= cfg.target_max:
            return [(path, whole)]
        out: list[tuple[Path, str]] = []
        buf = _ChildBuffer()

        def flush() -> None:
            nonlocal buf
            if buf.parts:
                out.append((_common_path(buf.paths), _join(buf.parts)))
            buf = _ChildBuffer()

        def add(text: str, item_path: Path, *, is_node: bool) -> None:
            nonlocal pending
            if pending:
                text, pending = _join([pending, text]), ""
            buf.parts.append(text)
            buf.paths.append(item_path)
            buf.has_node = buf.has_node or is_node

        for item in content:
            if isinstance(item, Node):
                sub = (*path, item)
                text = _text(item)
                if estimate_tokens(text) <= cfg.target_max:
                    joined = buf.tokens_with(text)
                    if buf.parts and (
                        (self.strict and buf.has_node)
                        or joined > cfg.target_max
                        or (not self.strict and buf.tokens >= cfg.target_min)
                    ):
                        flush()
                    add(text, sub, is_node=True)
                else:
                    heading = _join([pending, _heading(item)])
                    pending = ""
                    if buf.parts:
                        flush()
                    out.extend(self._children(item.content, sub, heading))
                continue
            budget = (cfg.hard_max if item.is_atomic else cfg.target_max) - (
                estimate_tokens(pending) + 1 if pending else 0
            )
            pieces = self._split(item, max(budget, cfg.target_min))
            for i, piece in enumerate(pieces):
                limit = cfg.hard_max if item.is_atomic else cfg.target_max
                if buf.parts and (i > 0 or buf.tokens_with(piece) > limit):
                    flush()
                add(piece, path, is_node=False)
        flush()
        return out

    def _split(self, block: Block, budget: int) -> list[str]:
        if estimate_tokens(block.text) <= budget:
            return [block.text]
        if block.kind is BlockKind.TABLE and block.rows:
            return _split_table(block, budget)
        if block.is_atomic:
            return _split_composite(block.text, budget)
        return _split_text(block.text, budget, self.config.overlap)

    # --- chunk construction ----------------------------------------------------------

    def _chunk(self, level: ChunkLevel, path: Path, text: str, parent: Chunk | None) -> Chunk:
        doc = self.ctx.document
        text = text.strip()
        citation = self._citation(path)
        ordinal = self.ordinals[(level, citation)]
        self.ordinals[(level, citation)] += 1
        return Chunk(
            id=make_chunk_id(doc.id, level, citation, ordinal),
            document_id=doc.id,
            parent_id=parent.id if parent else None,
            level=level,
            ordinal=ordinal,
            citation_path=citation,
            breadcrumb=" > ".join(
                [self.parsed.title, *(n.heading_line for n in path if n.heading_line)]
            ),
            text=text,
            token_count=estimate_tokens(text),
            content_hash=sha256_hex(text),
            source_url=doc.source_url,
            title=doc.title,
            doc_type=doc.doc_type,
            tax_year=doc.tax_year,
            effective_date=doc.effective_date,
            retrieved_at=doc.retrieved_at,
            entity_types=self._entity_types(citation),
        )

    def _citation(self, path: Path) -> str:
        root = self.ctx.citation_root
        if self.ctx.style is CitationStyle.STATUTORY:
            return root + "".join(node.label for node in path)
        return ", ".join([root, *_heading_segments(path)])

    def _entity_types(self, citation: str) -> frozenset[EntityType]:
        overrides = self.ctx.entity_overrides or {}
        matches = [key for key in overrides if _is_within(citation, key)]
        return overrides[max(matches, key=len)] if matches else self.ctx.entity_types


def _join(parts: Iterable[str]) -> str:
    return _SEP.join(p for p in parts if p)


def _heading(node: Node) -> str:
    # Label-only nodes ("(A)") carry their label inline in their text already.
    return node.heading_line if node.heading else ""


def _text(item: Block | Node) -> str:
    return render(item) if isinstance(item, Node) else item.text


def _common_path(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    common: list[Node] = []
    for nodes in zip(*paths, strict=False):
        if all(n is nodes[0] for n in nodes):
            common.append(nodes[0])
        else:
            break
    return tuple(common)


def _heading_segments(path: Path) -> list[str]:
    """Citation segments for a heading path: "Pub 463, ch. 1, Tax Home" style.

    irs.gov wraps content in articles titled "Publication 587 - Main Contents" or
    "Instructions for Form 1120 - Notices"; the document name is already the citation
    root, so only the part after the dash is kept, and "Main Contents" is dropped.
    """
    segments: list[str] = []
    for node in path:
        heading = node.heading_line
        if article := _ARTICLE.match(heading):
            heading = "" if article["part"] in _DROPPED_ARTICLES else article["part"]
        if not heading:
            continue
        if not segments and (chapter := _CHAPTER.match(heading)):
            heading = f"ch. {chapter.group(1)}"
        segments.append(heading)
    return segments


def _is_within(citation: str, prefix: str) -> bool:
    return citation == prefix or (citation.startswith(prefix) and citation[len(prefix)] in "(,")


def _pack_lines(
    lines: Sequence[str], budget: int, measure: Callable[[list[str]], int]
) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if current and measure([*current, line]) > budget:
            groups.append(current)
            current = []
        current.append(line)
    if current:
        groups.append(current)
    return groups


def _split_table(block: Block, budget: int) -> list[str]:
    """Row groups that each repeat the title and header; rows themselves are never cut."""
    head = [block.title] if block.title else []
    head_cont = [f"{block.title} (continued)"] if block.title else []
    header = list(block.header)
    room = budget - estimate_tokens("\n".join([*head_cont, *header]))
    pieces: list[str] = []
    for i, rows in enumerate(
        _pack_lines(block.rows, room, lambda ls: estimate_tokens("\n".join(ls)))
    ):
        lead = head if i == 0 else head_cont
        if len(rows) == 1 and estimate_tokens(rows[0]) > room:
            # A single row too large for any chunk: split its text but keep the header.
            pieces.extend(
                "\n".join([*lead, *header, part]) for part in _split_text(rows[0], room, 0)
            )
        else:
            pieces.append("\n".join([*lead, *header, *rows]))
    return pieces


def _split_composite(text: str, budget: int) -> list[str]:
    """Split an oversized example/note/figure at paragraph lines, repeating its first line."""
    first, *rest = text.split("\n")
    lead = f"{first} (continued)"
    room = budget - estimate_tokens(lead) - 1
    pieces: list[str] = []
    for i, lines in enumerate(
        _pack_lines(rest or [first], room, lambda ls: estimate_tokens("\n".join(ls)))
    ):
        body = "\n".join(lines)
        parts = _split_text(body, room, 0) if estimate_tokens(body) > room else [body]
        for j, part in enumerate(parts):
            pieces.append(f"{first}\n{part}" if i == 0 and j == 0 and rest else f"{lead}\n{part}")
    return pieces


def _split_text(text: str, budget: int, overlap: int) -> list[str]:
    """Sentence-window split with `overlap` tokens carried into the next window."""
    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    units: list[str] = []
    for sentence in sentences:
        if estimate_tokens(sentence) <= budget:
            units.append(sentence)
        else:  # a single run-on "sentence" longer than a chunk: fall back to word windows
            units.extend(
                " ".join(w)
                for w in _pack_lines(
                    sentence.split(), budget, lambda ws: estimate_tokens(" ".join(ws))
                )
            )
    pieces: list[str] = []
    window: list[str] = []
    for unit in units:
        if window and estimate_tokens(" ".join([*window, unit])) > budget:
            pieces.append(" ".join(window))
            carry: list[str] = []
            for previous in reversed(window):
                if (
                    estimate_tokens(" ".join([previous, *carry, unit])) > budget
                    or estimate_tokens(" ".join([previous, *carry])) > overlap
                ):
                    break
                carry.insert(0, previous)
            window = carry
        window.append(unit)
    if window:
        pieces.append(" ".join(window))
    return pieces
