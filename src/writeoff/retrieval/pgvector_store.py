"""PostgreSQL + pgvector implementation of `VectorStore`.

Dense search uses the HNSW cosine index; lexical search uses the generated `search_tsv`
column, querying it with both the 'english' config (stemmed prose) and the 'simple' config
(verbatim citation tokens like "280a"). Fusion, reranking and query rewriting belong to
the retrieval layer above this store (phase 3). Connections are opened per call, which
suits batch ingestion. The API server will add a pool in phase 9.
"""

from collections.abc import Collection, Sequence
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from writeoff.models import (
    Chunk,
    ChunkLevel,
    Document,
    EmbeddedChunk,
    ScoredChunk,
    SearchFilters,
    StoredChunkState,
    UpsertStats,
)
from writeoff.retrieval.interfaces import VectorStore

_CHUNK_COLUMNS = (
    "id",
    "document_id",
    "parent_id",
    "level",
    "ordinal",
    "citation_path",
    "breadcrumb",
    "text",
    "context_summary",
    "token_count",
    "content_hash",
    "source_url",
    "title",
    "doc_type",
    "tax_year",
    "effective_date",
    "retrieved_at",
    "entity_types",
)
_SELECT = ", ".join(_CHUNK_COLUMNS)
_UPSERT_CHUNK = f"""
INSERT INTO chunks ({_SELECT}, embedding)
VALUES ({", ".join(["%s"] * len(_CHUNK_COLUMNS))}, %s::vector)
ON CONFLICT (id) DO UPDATE SET
    {", ".join(f"{c} = EXCLUDED.{c}" for c in _CHUNK_COLUMNS if c != "id")},
    embedding = EXCLUDED.embedding,
    updated_at = now()
"""  # noqa: S608 - column names are module constants, values are parameters
_UPSERT_DOCUMENT = """
INSERT INTO documents (id, source_url, title, doc_type, tax_year, effective_date, retrieved_at,
                       content_hash)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    title = EXCLUDED.title, doc_type = EXCLUDED.doc_type, effective_date = EXCLUDED.effective_date,
    retrieved_at = EXCLUDED.retrieved_at, content_hash = EXCLUDED.content_hash, updated_at = now()
"""
# A stored path P encloses the requested path Q when Q continues P at a boundary: "(" for
# statutory paths, ", " for heading paths. Only the deepest enclosing level is returned.
_SELECT_ENCLOSING = f"""
WITH enclosing AS (
    SELECT {_SELECT}, length(citation_path) AS depth FROM chunks
    WHERE tax_year = %(year)s
      AND (starts_with(%(path)s, citation_path || '(')
           OR starts_with(%(path)s, citation_path || ', '))
)
SELECT {_SELECT} FROM enclosing WHERE depth = (SELECT max(depth) FROM enclosing)
ORDER BY level = 'child', ordinal
""".encode()  # noqa: S608 - column names are module constants
_SELECT_BY_IDS = f"SELECT {_SELECT} FROM chunks WHERE id = ANY(%s)".encode()  # noqa: S608
# ts_rank_cd normalization 4 divides by the mean distance between matched extents, which
# favours passages where the query's terms occur together. Without it, ORed queries were
# dominated by long chunks that repeat a common word ("Form 8829" ranked Form 1120's
# "Assembling the Return" first). Postgres FTS has no IDF; phase 4 evals decide whether
# a BM25 ranking is worth adding.
_LEXICAL_QUERY = "(websearch_to_tsquery('english', %(q)s) || websearch_to_tsquery('simple', %(q)s))"


def _vector(values: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def _filter_sql(filters: SearchFilters) -> tuple[str, dict[str, Any]]:
    clauses = ["level = 'child'", "tax_year = %(tax_year)s"]
    params: dict[str, Any] = {"tax_year": filters.tax_year}
    if filters.doc_types:
        clauses.append("doc_type = ANY(%(doc_types)s)")
        params["doc_types"] = sorted(str(d) for d in filters.doc_types)
    if filters.entity_type is not None:
        clauses.append("(entity_types = '{}' OR %(entity)s = ANY(entity_types))")
        params["entity"] = str(filters.entity_type)
    return " AND ".join(clauses), params


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    data = {c: row[c] for c in _CHUNK_COLUMNS}
    data["entity_types"] = frozenset(row["entity_types"])
    return Chunk.model_validate(data)


def _chunk_params(item: EmbeddedChunk) -> tuple[Any, ...]:
    c = item.chunk
    values = [getattr(c, col) for col in _CHUNK_COLUMNS]
    values[_CHUNK_COLUMNS.index("source_url")] = str(c.source_url)
    values[_CHUNK_COLUMNS.index("level")] = str(c.level)
    values[_CHUNK_COLUMNS.index("doc_type")] = str(c.doc_type)
    values[_CHUNK_COLUMNS.index("entity_types")] = sorted(str(e) for e in c.entity_types)
    return (*values, _vector(item.embedding) if item.embedding else None)


class PgVectorStore(VectorStore):
    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo

    async def _connect(self) -> psycopg.AsyncConnection[dict[str, Any]]:
        return await psycopg.AsyncConnection.connect(self._conninfo, row_factory=dict_row)

    async def existing_chunks(self, document_id: UUID) -> dict[UUID, StoredChunkState]:
        async with await self._connect() as conn:
            rows = await (
                await conn.execute(
                    "SELECT id, content_hash, parent_id, "
                    "context_summary IS NOT NULL AS has_summary "
                    "FROM chunks WHERE document_id = %s",
                    (document_id,),
                )
            ).fetchall()
        return {
            r["id"]: StoredChunkState(
                content_hash=r["content_hash"],
                parent_id=r["parent_id"],
                has_context_summary=r["has_summary"],
            )
            for r in rows
        }

    async def sync_document(
        self, document: Document, changed: Sequence[EmbeddedChunk], keep: Collection[UUID]
    ) -> UpsertStats:
        # Parents first: children reference them through a foreign key.
        ordered = sorted(changed, key=lambda e: e.chunk.level is not ChunkLevel.PARENT)
        changed_ids = [e.chunk.id for e in ordered]
        async with await self._connect() as conn, conn.transaction():
            await conn.execute(
                _UPSERT_DOCUMENT,
                (
                    document.id,
                    str(document.source_url),
                    document.title,
                    str(document.doc_type),
                    document.tax_year,
                    document.effective_date,
                    document.retrieved_at,
                    document.content_hash,
                ),
            )
            present = await (
                await conn.execute("SELECT id FROM chunks WHERE id = ANY(%s)", (changed_ids,))
            ).fetchall()
            deleted = await (
                await conn.execute(
                    "DELETE FROM chunks WHERE document_id = %s AND NOT (id = ANY(%s)) RETURNING id",
                    (document.id, [*changed_ids, *keep]),
                )
            ).fetchall()
            async with conn.cursor() as cur:
                await cur.executemany(_UPSERT_CHUNK, [_chunk_params(e) for e in ordered])
        updated = len(present)
        return UpsertStats(
            inserted=len(ordered) - updated,
            updated=updated,
            unchanged=len(keep),
            deleted=len(deleted),
        )

    async def dense_search(
        self, embedding: Sequence[float], k: int, filters: SearchFilters
    ) -> list[ScoredChunk]:
        where, params = _filter_sql(filters)
        params |= {"embedding": _vector(embedding), "k": k}
        sql = (
            f"SELECT {_SELECT}, 1 - (embedding <=> %(embedding)s::vector) AS score FROM chunks "  # noqa: S608
            f"WHERE {where} ORDER BY embedding <=> %(embedding)s::vector LIMIT %(k)s"
        )
        return await self._scored(sql, params)

    async def lexical_search(self, query: str, k: int, filters: SearchFilters) -> list[ScoredChunk]:
        where, params = _filter_sql(filters)
        params |= {"q": query, "k": k}
        sql = (
            f"SELECT {_SELECT}, ts_rank_cd(search_tsv, {_LEXICAL_QUERY}, 4) AS score FROM chunks "  # noqa: S608
            f"WHERE {where} AND search_tsv @@ {_LEXICAL_QUERY} ORDER BY score DESC LIMIT %(k)s"
        )
        return await self._scored(sql, params)

    async def _scored(self, sql: str, params: dict[str, Any]) -> list[ScoredChunk]:
        async with await self._connect() as conn:
            rows = await (await conn.execute(sql.encode(), params)).fetchall()
        return [
            ScoredChunk(chunk=_row_to_chunk(r), score=float(r["score"]), rank=i)
            for i, r in enumerate(rows, start=1)
        ]

    async def get_by_citation(self, citation_path: str, tax_year: int) -> list[Chunk]:
        escaped = citation_path.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
        sql = (
            f"SELECT {_SELECT} FROM chunks WHERE tax_year = %(year)s AND (citation_path = %(path)s "  # noqa: S608
            "OR citation_path LIKE %(sub)s OR citation_path LIKE %(heading)s) "
            "ORDER BY level = 'child', citation_path, ordinal"
        )
        params = {
            "year": tax_year,
            "path": citation_path,
            "sub": f"{escaped}(%",
            "heading": f"{escaped},%",
        }
        async with await self._connect() as conn:
            rows = await (await conn.execute(sql.encode(), params)).fetchall()
        return [_row_to_chunk(r) for r in rows]

    async def get_enclosing(self, citation_path: str, tax_year: int) -> list[Chunk]:
        async with await self._connect() as conn:
            rows = await (
                await conn.execute(_SELECT_ENCLOSING, {"year": tax_year, "path": citation_path})
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    async def get_chunks(self, ids: Sequence[UUID]) -> list[Chunk]:
        async with await self._connect() as conn:
            rows = await (await conn.execute(_SELECT_BY_IDS, (list(ids),))).fetchall()
        by_id = {r["id"]: _row_to_chunk(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]
