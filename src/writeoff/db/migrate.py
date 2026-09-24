"""Minimal forward-only migration runner.

Applies `migrations/NNNN_*.sql` in filename order, each in its own transaction, and
records applied filenames with a checksum in `schema_migrations`. Editing an already
applied file is an error: add a new migration instead.

Usage: python -m writeoff.db.migrate
"""

import hashlib
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg

from writeoff.config import get_settings

logger = logging.getLogger(__name__)

_FILENAME = re.compile(r"^\d{4}_[a-z0-9_]+\.sql$")

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   text PRIMARY KEY,
    checksum   char(64)    NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


class MigrationError(RuntimeError):
    """A migration file is invalid or conflicts with what was already applied."""


@dataclass(frozen=True, slots=True)
class Migration:
    filename: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


def discover(directory: Path) -> list[Migration]:
    if not directory.is_dir():
        raise MigrationError(f"migrations directory not found: {directory}")
    migrations = []
    for path in sorted(directory.glob("*.sql")):
        if not _FILENAME.match(path.name):
            raise MigrationError(f"bad migration filename {path.name!r}; expected NNNN_name.sql")
        migrations.append(Migration(path.name, path.read_text(encoding="utf-8")))
    return migrations


def apply_migrations(conninfo: str, directory: Path) -> list[str]:
    """Apply pending migrations; return the filenames applied in this run."""
    migrations = discover(directory)
    applied_now: list[str] = []
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute(_BOOTSTRAP)
        applied: dict[str, str] = dict(
            conn.execute("SELECT filename, checksum FROM schema_migrations").fetchall()
        )
        for migration in migrations:
            previous = applied.get(migration.filename)
            if previous is not None:
                if previous != migration.checksum:
                    raise MigrationError(
                        f"{migration.filename} was modified after being applied; "
                        "add a new migration instead"
                    )
                continue
            with conn.transaction():
                # Migration files are trusted repository content, not user input.
                conn.execute(migration.sql.encode("utf-8"))
                conn.execute(
                    "INSERT INTO schema_migrations (filename, checksum) VALUES (%s, %s)",
                    (migration.filename, migration.checksum),
                )
            logger.info("applied %s", migration.filename)
            applied_now.append(migration.filename)
    return applied_now


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    try:
        applied = apply_migrations(str(settings.database_url), settings.migrations_dir)
    except (MigrationError, psycopg.Error) as exc:
        logger.error("migration failed: %s", exc)
        return 1
    if applied:
        logger.info("%d migration(s) applied", len(applied))
    else:
        logger.info("up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
