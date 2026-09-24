"""Parser for IRS publication and instruction PDFs (pdfplumber).

Used for editions that exist only as PDFs, e.g. a prior-year publication once irs.gov's
HTML page moves on to the next year. IRS PDFs are typeset in two columns, so each page is
split into horizontal bands at full-width tables, and each band into columns at the
gutter, before reading lines top to bottom. Headings are lines set in bold at a size
larger than the body text; distinct heading sizes map to outline levels (largest first).
Bold run-in heads at body size ("Performance of services.") stay part of their
paragraph, matching the HTML rendering. Text outside the page box (printer's proof
marks), running footers and page numbers (the bottom 42pt margin, where the column split
would otherwise glue a page number onto the next paragraph) are dropped.
"""

import io
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pdfplumber
from pdfplumber.page import Page

from writeoff.ingestion.parsers.common import ParseError
from writeoff.ingestion.tree import BlockKind, Node, ParsedDocument, make_table

_FOOTER = re.compile(r"^(Page \d+|.*\bPage \d+)$|^(Publication|Instructions for) .+\(\d{4}\)\s*$")
_EXAMPLE = re.compile(r"^Examples?\b[.:]?", re.I)
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_FOOTER_MARGIN = 42.0  # points above the bottom edge holding page numbers and running footers
_GUTTER_BAND = (0.35, 0.65)  # search the middle 30% of the page width for a column gutter
_MAX_HEADING_LEVELS = 4
_WORDY = re.compile(r"[A-Za-z]{3}")  # rejects icon glyphs such as the bold "!" caution mark
_SKIPPED_HEADINGS = frozenset({"Contents", "Index", "How To Get Tax Help"})


@dataclass(slots=True)
class _Line:
    top: float
    bottom: float
    text: str
    size: float
    bold: bool
    starts_bold: bool


@dataclass(slots=True)
class _Table:
    top: float
    rows: list[list[str]]


def parse_pdf(content: bytes) -> ParsedDocument:
    try:
        pdf = pdfplumber.open(io.BytesIO(content))
    except Exception as exc:  # pdfminer raises a variety of exception types
        raise ParseError(f"unreadable PDF: {exc}") from exc
    with pdf:
        metadata_title = str(pdf.metadata.get("Title") or "").strip()
        events: list[_Line | _Table] = []
        for page in pdf.pages:
            events.extend(_page_events(page))
    lines = [e for e in events if isinstance(e, _Line)]
    if not lines:
        raise ParseError("PDF has no extractable text (scanned image?)")
    body_size = Counter(round(line.size, 1) for line in lines).most_common(1)[0][0]
    heading_sizes = sorted(
        {round(line.size, 1) for line in lines if _is_heading_candidate(line, body_size)},
        reverse=True,
    )[:_MAX_HEADING_LEVELS]
    title = metadata_title or _title(lines, heading_sizes)
    root = _build_tree(events, heading_sizes, title)
    _drop_skipped(root)
    return ParsedDocument(title=title, root=root)


def _is_heading_candidate(line: _Line, body_size: float) -> bool:
    return line.bold and line.size >= body_size + 1.0 and bool(_WORDY.search(line.text))


def _drop_skipped(node: Node) -> None:
    node.content = [
        item
        for item in node.content
        if not (isinstance(item, Node) and item.heading in _SKIPPED_HEADINGS)
    ]
    for child in node.children():
        _drop_skipped(child)


def _page_events(page: Page) -> list[_Line | _Table]:
    page = page.crop((0, 0, page.width, page.height), strict=False)  # drops proof marks
    wide = [t for t in page.find_tables() if (t.bbox[2] - t.bbox[0]) > page.width * 0.6]
    events: list[_Line | _Table] = []
    cursor = 0.0
    for table in sorted(wide, key=lambda t: t.bbox[1]):
        events.extend(_band_events(page, cursor, table.bbox[1]))
        events.append(_Table(table.bbox[1], _clean_rows(table.extract())))
        cursor = table.bbox[3]
    events.extend(_band_events(page, cursor, page.height))
    footer_top = page.height - _FOOTER_MARGIN
    return [e for e in events if isinstance(e, _Table) or e.top < footer_top]


def _band_events(page: Page, top: float, bottom: float) -> list[_Line | _Table]:
    if bottom - top < 1:
        return []
    band = page.crop((0, top, page.width, bottom), strict=False)
    gutter = _find_gutter(band)
    columns = (
        [band]
        if gutter is None
        else [
            band.crop((0, top, gutter, bottom), strict=False),
            band.crop((gutter, top, page.width, bottom), strict=False),
        ]
    )
    events: list[_Line | _Table] = []
    for column in columns:
        events.extend(_column_events(column))
    return events


def _find_gutter(band: Page) -> float | None:
    """x of an empty vertical strip in the middle of the band, if the band has two columns."""
    chars = band.chars
    if len(chars) < 50:
        return None
    lo, hi = (int(band.width * f) for f in _GUTTER_BAND)
    covered = [0] * (hi - lo)
    for c in chars:
        for x in range(max(lo, int(c["x0"])), min(hi, int(c["x1"]) + 1)):
            covered[x - lo] += 1
    best = min(range(len(covered)), key=lambda i: covered[i])
    return float(lo + best) if covered[best] <= len(chars) * 0.002 else None


def _column_events(column: Page) -> list[_Line | _Table]:
    tables = column.find_tables()
    boxes = [t.bbox for t in tables]
    events: list[_Line | _Table] = [_Table(t.bbox[1], _clean_rows(t.extract())) for t in tables]
    for raw in column.extract_text_lines(return_chars=True):
        mid = (raw["top"] + raw["bottom"]) / 2
        if any(b[1] <= mid <= b[3] and b[0] <= raw["x0"] <= b[2] for b in boxes):
            continue
        text = raw["text"].strip()
        if not text or _FOOTER.match(text) or _PAGE_NUMBER.match(text):
            continue
        chars: list[dict[str, Any]] = raw["chars"]
        sizes = Counter(round(c["size"], 1) for c in chars)
        bold = sum("Bold" in c["fontname"] for c in chars) > len(chars) / 2
        events.append(
            _Line(
                raw["top"],
                raw["bottom"],
                text,
                sizes.most_common(1)[0][0],
                bold,
                "Bold" in chars[0]["fontname"],
            )
        )
    return sorted(events, key=lambda e: e.top)


def _clean_rows(rows: list[list[str | None]]) -> list[list[str]]:
    return [[(cell or "").replace("\n", " ") for cell in row] for row in rows]


def _title(lines: list[_Line], heading_sizes: list[float]) -> str:
    top_size = heading_sizes[0] if heading_sizes else None
    first = next(
        (line for line in lines if line.bold and round(line.size, 1) == top_size), lines[0]
    )
    return first.text


def _build_tree(events: list[_Line | _Table], heading_sizes: list[float], title: str) -> Node:
    root = Node(heading=title)
    stack: list[tuple[int, Node]] = []
    paragraph: list[str] = []
    kind = BlockKind.TEXT
    last: _Line | None = None

    def current() -> Node:
        return stack[-1][1] if stack else root

    def flush() -> None:
        nonlocal paragraph, kind
        if paragraph:
            current().add_text(kind, _join(paragraph))
        paragraph, kind = [], BlockKind.TEXT

    for event in events:
        if isinstance(event, _Table):
            flush()
            if event.rows:
                current().content.append(make_table("", event.rows[:1], event.rows[1:]))
            last = None
            continue
        size = round(event.size, 1)
        if size in heading_sizes and event.bold and _WORDY.search(event.text):
            flush()
            level = heading_sizes.index(size)
            if (
                stack
                and stack[-1][0] == level
                and last is not None
                and last.bold
                and round(last.size, 1) == size
                and event.top - last.bottom < event.size
            ):
                stack[-1][1].heading += " " + event.text  # heading wrapped onto a second line
            elif event.text not in title:  # cover-page fragments of the title aren't sections
                while stack and stack[-1][0] >= level:
                    stack.pop()
                node = Node(heading=event.text)
                current().content.append(node)
                stack.append((level, node))
            last = event
            continue
        new_paragraph = (
            last is None
            or (event.top - last.bottom) > event.size * 0.8
            or (event.starts_bold and last.text.endswith("."))
        )
        if new_paragraph:
            flush()
            if _EXAMPLE.match(event.text):
                kind = BlockKind.EXAMPLE
        paragraph.append(event.text)
        last = event
    flush()
    return root


def _join(lines: list[str]) -> str:
    text = ""
    for line in lines:
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line  # re-join a word hyphenated across lines
        else:
            text = f"{text} {line}" if text else line
    return text
