import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from writeoff.config import Settings


@pytest.fixture(scope="session")
def admin_conninfo() -> str:
    """Conninfo for the docker-compose Postgres. Skips (or fails, under `make test`) if down."""
    conninfo = str(Settings().database_url)
    try:
        with psycopg.connect(conninfo, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
    except psycopg.OperationalError as exc:
        reason = f"Postgres unreachable at DATABASE_URL (run `make db-up`): {exc}"
        if os.environ.get("WRITEOFF_REQUIRE_DB") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    return conninfo


@pytest.fixture
def fresh_db(admin_conninfo: str) -> Iterator[str]:
    """A brand-new, empty database for one test, dropped afterwards."""
    name = f"writeoff_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin_conninfo, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        yield make_conninfo(admin_conninfo, dbname=name)
    finally:
        with psycopg.connect(admin_conninfo, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )
