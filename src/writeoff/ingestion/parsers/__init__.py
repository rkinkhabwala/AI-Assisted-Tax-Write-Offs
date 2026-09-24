"""Source parsers: raw bytes -> `ParsedDocument` tree."""

from collections.abc import Callable
from enum import StrEnum

from writeoff.ingestion.parsers.common import ParseError
from writeoff.ingestion.parsers.ecfr import parse_ecfr
from writeoff.ingestion.parsers.irs_html import parse_irs_html
from writeoff.ingestion.parsers.pdf import parse_pdf
from writeoff.ingestion.parsers.uscode import parse_uscode
from writeoff.ingestion.tree import ParsedDocument, prune_reserved
from writeoff.models import SourceFormat


class ParserName(StrEnum):
    USCODE_HTML = "uscode_html"
    ECFR_XML = "ecfr_xml"
    IRS_HTML = "irs_html"
    PDF = "pdf"

    @property
    def source_format(self) -> SourceFormat:
        return SourceFormat.PDF if self is ParserName.PDF else SourceFormat.HTML


_PARSERS: dict[ParserName, Callable[[bytes], ParsedDocument]] = {
    ParserName.USCODE_HTML: parse_uscode,
    ParserName.ECFR_XML: parse_ecfr,
    ParserName.IRS_HTML: parse_irs_html,
    ParserName.PDF: parse_pdf,
}


def parse(parser: ParserName, content: bytes) -> ParsedDocument:
    parsed = _PARSERS[parser](content)
    prune_reserved(parsed.root)
    return parsed


__all__ = ["ParseError", "ParserName", "parse"]
