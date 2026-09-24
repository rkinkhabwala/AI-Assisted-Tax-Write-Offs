"""The corpus: which documents to ingest, from where, per tax year (`data/sources.yaml`).

Each source lists its editions by tax year. Statutes and regulations have no annual
editions, so the same text is snapshotted for every supported year (decision D7).
Publications and instructions point at the edition for that year: the live HTML page
for the current edition, or the irs-prior PDF once the page has moved on.
"""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, model_validator

from writeoff.chunking.chunker import CitationStyle
from writeoff.ingestion.parsers import ParserName
from writeoff.models import DocType, EntityType, NonEmptyStr, SourceSpec, TaxYear

ALLOWED_HOSTS = frozenset({"uscode.house.gov", "www.ecfr.gov", "www.irs.gov"})


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Edition(_Strict):
    url: HttpUrl
    parser: ParserName

    @model_validator(mode="after")
    def _allowed_host(self) -> Self:
        if self.url.scheme != "https" or self.url.host not in ALLOWED_HOSTS:
            raise ValueError(f"{self.url} is not an https URL on {sorted(ALLOWED_HOSTS)}")
        return self


class SourceEntry(_Strict):
    id: NonEmptyStr
    title: NonEmptyStr
    citation_root: NonEmptyStr
    doc_type: DocType
    entity_types: frozenset[EntityType] = frozenset()
    entity_overrides: dict[NonEmptyStr, frozenset[EntityType]] = Field(default_factory=dict)
    editions: dict[TaxYear, Edition] = Field(min_length=1)

    @property
    def citation_style(self) -> CitationStyle:
        statutory = self.doc_type in {DocType.IRC, DocType.TREASURY_REGULATION}
        return CitationStyle.STATUTORY if statutory else CitationStyle.HEADINGS

    def spec_for(self, tax_year: int) -> SourceSpec:
        edition = self.editions[tax_year]
        return SourceSpec(
            source_url=edition.url,
            title=self.title,
            doc_type=self.doc_type,
            tax_year=tax_year,
            format=edition.parser.source_format,
        )


class Registry(_Strict):
    sources: tuple[SourceEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [s.id for s in self.sources]
        if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
            raise ValueError(f"duplicate source ids: {duplicates}")
        return self

    def select(self, tax_year: int, ids: frozenset[str] | None = None) -> list[SourceEntry]:
        """Sources with an edition for `tax_year`, optionally restricted to `ids`."""
        if ids and (unknown := ids - {s.id for s in self.sources}):
            raise KeyError(f"unknown source ids: {sorted(unknown)}")
        return [s for s in self.sources if tax_year in s.editions and (not ids or s.id in ids)]


class RegistryError(ValueError):
    """The source registry file is missing or invalid."""


def load_registry(path: Path) -> Registry:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistryError(f"source registry not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise RegistryError(f"{path}: invalid YAML: {exc}") from exc
    try:
        return Registry.model_validate(raw)
    except ValidationError as exc:
        raise RegistryError(f"{path}: {exc}") from exc
