"""Helpers shared by the HTML parsers."""

from selectolax.lexbor import LexborNode

from writeoff.ingestion.normalize import normalize_text


class ParseError(ValueError):
    """A fetched document does not have the structure its parser expects."""


def element_text(el: LexborNode) -> str:
    return normalize_text(el.text(deep=True, separator=""))


def cell_texts(tr: LexborNode) -> list[str]:
    return [normalize_text(c.text(deep=True, separator=" ")) for c in tr.css("th, td")]
