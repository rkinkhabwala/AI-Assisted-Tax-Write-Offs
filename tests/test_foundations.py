"""Config, interfaces, the migration discovery and the health endpoint."""

import inspect
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from writeoff.api.app import app
from writeoff.config import EMBEDDING_DIMENSION, Settings
from writeoff.db.migrate import MigrationError, discover
from writeoff.ingestion.interfaces import DocumentFetcher
from writeoff.retrieval.interfaces import Embedder, Reranker, VectorStore


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.embedding_dimension == EMBEDDING_DIMENSION
    assert settings.supported_tax_years == (2025, 2026)


def test_settings_reject_dimension_that_does_not_match_schema() -> None:
    with pytest.raises(ValidationError, match="add a migration"):
        Settings(_env_file=None, embedding_dimension=768)


def test_settings_normalize_tax_years() -> None:
    settings = Settings(_env_file=None, supported_tax_years=(2026, 2025, 2026))
    assert settings.supported_tax_years == (2025, 2026)


@pytest.mark.parametrize("interface", [VectorStore, Embedder, Reranker, DocumentFetcher])
def test_interfaces_are_abstract(interface: type) -> None:
    assert inspect.isabstract(interface)
    with pytest.raises(TypeError, match="abstract"):
        interface()


def test_healthz() -> None:
    response = TestClient(app).get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rejects_bad_filename(tmp_path: Path) -> None:
    (tmp_path / "init.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(MigrationError, match="bad migration filename"):
        discover(tmp_path)
