"""Parser for Treasury Regulation sections from the eCFR versioner API (XML).

eCFR delivers a section as a flat run of <P> paragraphs, so the (a) > (1) > (i) > (A) >
(1-italic) > (i-italic) hierarchy is reconstructed from paragraph labels. Two quirks:

- Chained labels: "(e) *Heading*—(1) *In general.* Text" opens (e) and its child (1) in one
  paragraph.
- Older regulations use italic letters "(*a*)" below roman numerals; modern ones use "(A)".
  Both are placed at the same depth.
- "(i)", "(v)", "(x)" may be letters (after "(h)") or roman numerals. We pick whichever is
  the expected next sibling given the open paragraphs; if both are, a chained digit
  label (only letters have digit children) decides, otherwise roman wins.
"""

import re
from dataclasses import dataclass
from xml.etree.ElementTree import Element

from defusedxml.ElementTree import ParseError as XmlParseError
from defusedxml.ElementTree import fromstring

from writeoff.ingestion.normalize import normalize_text
from writeoff.ingestion.parsers.common import ParseError
from writeoff.ingestion.tree import Block, BlockKind, Node, ParsedDocument, make_table

_ITALIC_TAGS = frozenset({"I", "E"})
_SKIP_TAGS = frozenset({"CITA", "SECAUTH", "AUTH", "SOURCE", "FTNT", "EDNOTE", "HEAD"})
_ITALIC_OPEN, _ITALIC_CLOSE = "\x01", "\x02"
_LABEL = re.compile(r"^\(\x01?([a-zA-Z]{1,4}|\d{1,3})\x02?\)\s*")
_HEADING = re.compile(r"^\x01([^\x01\x02]{1,300})\x02\s*[.—-]*\s*")
_ROMAN = re.compile(r"^(?=[ivx])(x{0,3})(ix|iv|v?i{0,3})$")
_SECTION_HEAD = re.compile(r"^§\s*(?P<num>\S+)\s+(?P<title>.+)$", re.S)

LETTER, DIGIT, ROMAN, UPPER, ITALIC_DIGIT, ITALIC_ROMAN = range(1, 7)


@dataclass(frozen=True, slots=True)
class _Label:
    text: str
    italic: bool
    heading: str


@dataclass(slots=True)
class _Open:
    level: int
    label: str
    node: Node


def parse_ecfr(content: bytes) -> ParsedDocument:
    try:
        section = fromstring(content)
    except XmlParseError as exc:
        raise ParseError(f"invalid eCFR XML: {exc}") from exc
    head = section.find("HEAD")
    match = (
        _SECTION_HEAD.match(normalize_text("".join(head.itertext()))) if head is not None else None
    )
    if match is None:
        raise ParseError("eCFR section has no parsable HEAD")
    number, title = match["num"], match["title"].strip()
    root = Node(label=number, heading=title)
    stack: list[_Open] = []

    for el in section:
        _handle(el, root, stack)
    if root.is_empty():
        raise ParseError(f"no regulation text found for § {number}")
    return ParsedDocument(title=f"Treas. Reg. § {number} — {title}", root=root)


def _handle(el: Element, root: Node, stack: list[_Open]) -> None:
    current = stack[-1].node if stack else root
    if el.tag in _SKIP_TAGS:
        return
    if el.tag in {"P", "FP"}:
        labels, body = _split_labels(_marked_text(el))
        for i, label in enumerate(labels):
            following = labels[i + 1].text if i + 1 < len(labels) else None
            current = _open(label, following, root, stack)
        # Labels without a heading aren't rendered as heading lines, so keep them inline:
        # "(i) The first category..." rather than a bare "The first category...".
        inline: list[str] = []
        for label in reversed(labels):
            if label.heading:
                break
            inline.insert(0, f"({label.text})")
        current.add_text(BlockKind.TEXT, f"{''.join(inline)} {body}" if inline else body)
    elif el.tag == "EXAMPLE":
        current.add_text(BlockKind.EXAMPLE, _paragraphs(el))
    elif el.tag == "NOTE":
        current.add_text(BlockKind.NOTE, _paragraphs(el))
    elif el.tag == "GPOTABLE":
        current.content.append(_table(el))
    else:
        current.add_text(BlockKind.TEXT, _paragraphs(el))


def _marked_text(el: Element) -> str:
    """Element text with italic runs wrapped in sentinel characters."""
    parts = [el.text or ""]
    for child in el:
        inner = "".join(child.itertext())
        parts.append(
            f"{_ITALIC_OPEN}{inner}{_ITALIC_CLOSE}" if child.tag in _ITALIC_TAGS else inner
        )
        parts.append(child.tail or "")
    return "".join(parts)


def _split_labels(marked: str) -> tuple[list[_Label], str]:
    labels: list[_Label] = []
    rest = marked.lstrip()
    while match := _LABEL.match(rest):
        italic = _ITALIC_OPEN in match.group(0)
        rest = rest[match.end() :]
        heading = ""
        if heading_match := _HEADING.match(rest):
            heading = heading_match.group(1).strip().rstrip(".")
            rest = rest[heading_match.end() :]
        labels.append(_Label(match.group(1), italic, normalize_text(heading)))
    body = rest.replace(_ITALIC_OPEN, "").replace(_ITALIC_CLOSE, "")
    return labels, body


def _open(label: _Label, following: str | None, root: Node, stack: list[_Open]) -> Node:
    level = _level(label, following, stack)
    while stack and stack[-1].level >= level:
        stack.pop()
    parent = stack[-1].node if stack else root
    node = Node(label=f"({label.text})", heading=label.heading)
    parent.content.append(node)
    stack.append(_Open(level, label.text, node))
    return node


def _level(label: _Label, following: str | None, stack: list[_Open]) -> int:  # noqa: PLR0911
    # A decision table over label forms; each return is one documented case.
    text = label.text
    if text.isdigit():
        return ITALIC_DIGIT if label.italic else DIGIT
    if text.isupper():
        return UPPER
    is_roman = bool(_ROMAN.match(text))
    by_level = {o.level: o.label for o in stack}
    if label.italic:
        # Older regulations nest italic letters under roman numerals, the depth that
        # modern ones give (A). An italic roman numeral is the level below italic digits,
        # unless it continues an italic letter run ("h" then "i").
        if is_roman and by_level.get(UPPER) != _prev_letter(text):
            return ITALIC_ROMAN
        return UPPER
    if not is_roman:
        return LETTER
    if len(text) > 1:
        return ROMAN
    # Single-character "i", "v" or "x": letter or roman numeral?
    letter = by_level.get(LETTER)
    expects_letter = letter is not None and len(letter) == 1 and chr(ord(letter) + 1) == text
    roman = by_level.get(ROMAN)
    if roman is not None:
        expects_roman = _next_roman(roman) == text
    else:
        expects_roman = text == "i" and DIGIT in by_level
    if expects_letter and expects_roman:
        return LETTER if following is not None and following.isdigit() else ROMAN
    if expects_letter:
        return LETTER
    if expects_roman:
        return ROMAN
    return ROMAN if DIGIT in by_level else LETTER


_ROMANS = [
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "xi",
    "xii",
    "xiii",
    "xiv",
    "xv",
    "xvi",
    "xvii",
    "xviii",
    "xix",
    "xx",
    "xxi",
    "xxii",
    "xxiii",
    "xxiv",
    "xxv",
]


def _prev_letter(letter: str) -> str:
    return chr(ord(letter[0]) - 1)


def _next_roman(numeral: str) -> str | None:
    try:
        return _ROMANS[_ROMANS.index(numeral) + 1]
    except (ValueError, IndexError):
        return None


def _paragraphs(el: Element) -> str:
    lines = []
    for child in el.iter():
        if child is el:
            continue
        text = (
            normalize_text("".join(child.itertext()))
            if child.tag in {"HED", "P", "PSPACE", "FP"}
            else ""
        )
        if text:
            lines.append(text)
    return "\n\n".join(lines) if lines else normalize_text("".join(el.itertext()))


def _table(el: Element) -> Block:
    title_el = el.find("TTITLE")
    title = normalize_text("".join(title_el.itertext())) if title_el is not None else ""
    header = [[normalize_text("".join(c.itertext())) for c in el.iter("CHED")]]
    rows = [
        [normalize_text("".join(e.itertext())) for e in row.iter("ENT")] for row in el.iter("ROW")
    ]
    return make_table(title, header, rows)
