"""Abstract interface for retrieving raw source documents (irs.gov, ecfr.gov, uscode)."""

from abc import ABC, abstractmethod

from writeoff.models import FetchedDocument, SourceSpec


class DocumentFetcher(ABC):
    """Downloads a source document. Parsing happens downstream, not here."""

    @abstractmethod
    def supports(self, spec: SourceSpec) -> bool:
        """Whether this fetcher can retrieve `spec` (e.g. by host or format)."""

    @abstractmethod
    async def fetch(self, spec: SourceSpec) -> FetchedDocument:
        """Fetch the document's raw bytes, stamped with a timezone-aware retrieved_at."""
