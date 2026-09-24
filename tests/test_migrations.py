"""Applies migrations/*.sql to a throwaway database on the docker-compose Postgres."""

import shutil
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from writeoff.config import EMBEDDING_DIMENSION
from writeoff.db.migrate import MigrationError, apply_migrations, discover

MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"

pytestmark = pytest.mark.db


def _vector(value: float) -> str:
    return "[" + ",".join([str(value)] * EMBEDDING_DIMENSION) + "]"


def _insert_document(conn: psycopg.Connection) -> str:
    doc_id = str(uuid4())
    conn.execute(
        """INSERT INTO documents (id, source_url, title, doc_type, tax_year, retrieved_at,
                                  content_hash)
           VALUES (%s, 'https://www.law.cornell.edu/uscode/text/26/280A', '§ 280A', 'irc',
                   2025, now(), repeat('a', 64))""",
        (doc_id,),
    )
    return doc_id


def _insert_chunk(
    conn: psycopg.Connection,
    doc_id: str,
    *,
    level: str,
    parent_id: str | None = None,
    citation_path: str = "IRC § 280A(c)",
    text: str = "Exceptions for certain business use of a home",
    embedding: str | None = None,
    token_count: int = 100,
) -> str:
    chunk_id = str(uuid4())
    conn.execute(
        """INSERT INTO chunks (id, document_id, parent_id, level, ordinal, citation_path,
                               breadcrumb, text, token_count, content_hash, source_url, title,
                               doc_type, tax_year, retrieved_at, embedding)
           VALUES (%s, %s, %s, %s, 0, %s, 'IRC § 280A > (c) Exceptions', %s, %s,
                   repeat('b', 64), 'https://www.law.cornell.edu/uscode/text/26/280A',
                   '§ 280A', 'irc', 2025, now(), %s::vector)""",
        (chunk_id, doc_id, parent_id, level, citation_path, text, token_count, embedding),
    )
    return chunk_id


def test_migrations_apply_cleanly_and_are_idempotent(fresh_db: str) -> None:
    expected = [m.filename for m in discover(MIGRATIONS)]
    assert expected, "no migrations found"
    assert apply_migrations(fresh_db, MIGRATIONS) == expected
    assert apply_migrations(fresh_db, MIGRATIONS) == []


def test_schema_shape(fresh_db: str) -> None:
    apply_migrations(fresh_db, MIGRATIONS)
    with psycopg.connect(fresh_db) as conn:
        embedding_type = conn.execute(
            """SELECT format_type(atttypid, atttypmod) FROM pg_attribute
               WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"""
        ).fetchone()
        assert embedding_type == (f"vector({EMBEDDING_DIMENSION})",)

        tsv = conn.execute(
            """SELECT data_type, is_generated FROM information_schema.columns
               WHERE table_name = 'chunks' AND column_name = 'search_tsv'"""
        ).fetchone()
        assert tsv == ("tsvector", "ALWAYS")

        indexes: dict[str, str] = dict(
            conn.execute(
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'chunks'"
            ).fetchall()
        )
        assert "USING hnsw (embedding vector_cosine_ops)" in indexes["chunks_embedding_hnsw_idx"]
        assert "USING gin (search_tsv)" in indexes["chunks_search_tsv_gin_idx"]


def test_dense_and_lexical_queries_work(fresh_db: str) -> None:
    apply_migrations(fresh_db, MIGRATIONS)
    with psycopg.connect(fresh_db) as conn:
        doc_id = _insert_document(conn)
        parent = _insert_chunk(conn, doc_id, level="parent")
        near = _insert_chunk(
            conn,
            doc_id,
            level="child",
            parent_id=parent,
            citation_path="IRC § 280A(c)(1)(A)",
            text="the principal place of business, used exclusively and regularly",
            embedding=_vector(0.5),
        )
        far_embedding = "[" + ",".join(["1"] + ["0"] * (EMBEDDING_DIMENSION - 1)) + "]"
        _insert_chunk(
            conn,
            doc_id,
            level="child",
            parent_id=parent,
            citation_path="IRC § 280A(g)",
            text="rental of a dwelling unit for fewer than 15 days",
            embedding=far_embedding,
        )

        conn.execute("SET enable_seqscan = off")  # make the planner use the HNSW index
        dense = conn.execute(
            """SELECT id FROM chunks WHERE level = 'child'
               ORDER BY embedding <=> %s::vector LIMIT 1""",
            (_vector(0.4),),
        ).fetchone()
        assert dense is not None
        assert str(dense[0]) == near

        # Exact statutory token from the citation path ('simple' config keeps "280a").
        cited = conn.execute(
            "SELECT count(*) FROM chunks WHERE search_tsv @@ to_tsquery('simple', '280a')"
        ).fetchone()
        assert cited == (3,)
        # Stemmed prose match ('english' config: "exclusively" ~ "exclusive").
        prose = conn.execute(
            """SELECT id FROM chunks
               WHERE search_tsv @@ websearch_to_tsquery('english', 'exclusive use')"""
        ).fetchall()
        assert [str(r[0]) for r in prose] == [near]


@pytest.mark.parametrize(
    ("kwargs", "constraint"),
    [
        ({"level": "child"}, "chunks_parent_iff_child"),
        ({"level": "parent", "embedding": "EMBED"}, "chunks_embedding_level"),
        ({"level": "child", "parent_id": "PARENT", "token_count": 1201}, "chunks_child_size"),
        ({"level": "parent", "citation_path": ""}, "citation_path"),
    ],
)
def test_constraints(fresh_db: str, kwargs: dict[str, object], constraint: str) -> None:
    apply_migrations(fresh_db, MIGRATIONS)
    with psycopg.connect(fresh_db) as conn:
        doc_id = _insert_document(conn)
        parent = _insert_chunk(conn, doc_id, level="parent")
        if kwargs.get("parent_id") == "PARENT":
            kwargs["parent_id"] = parent
        if kwargs.get("embedding") == "EMBED":
            kwargs["embedding"] = _vector(0.1)
        with pytest.raises(psycopg.errors.CheckViolation, match=constraint):
            _insert_chunk(conn, doc_id, **kwargs)  # type: ignore[arg-type]


def test_rejects_edited_migration(fresh_db: str, tmp_path: Path) -> None:
    for path in MIGRATIONS.glob("*.sql"):
        shutil.copy(path, tmp_path / path.name)
    apply_migrations(fresh_db, tmp_path)
    first = sorted(tmp_path.glob("*.sql"))[0]
    first.write_text(first.read_text(encoding="utf-8") + "\n-- edited\n", encoding="utf-8")
    with pytest.raises(MigrationError, match="modified after being applied"):
        apply_migrations(fresh_db, tmp_path)
