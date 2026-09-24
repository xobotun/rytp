"""The one SQLite connection every stage is handed.

Design §3: "Every stage reads and writes SQLite; nothing is handed
between stages in memory." Contracts §8: "Every stage takes an open
``Database``; nothing opens its own connection." Only the surfaces
(``rytp/cli.py``, ``rytp/tui/app.py``) and the worker construct one.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType

from rytp import constants as C
from rytp.db.schema import LATEST_VERSION, MIGRATIONS

__all__ = ["LATEST_VERSION", "MIGRATIONS", "Database"]


class Database:
    """An open SQLite connection plus the migration runner.

    ``isolation_level=None`` turns off :mod:`sqlite3`'s implicit
    transaction handling, so a bare ``db.conn.execute(...)`` autocommits
    and :meth:`transaction` controls its own BEGIN/COMMIT. Multi-statement
    writes must use :meth:`transaction`.

    ``check_same_thread=False`` lets the worker hand a connection between
    threads; callers are responsible for not sharing one concurrently.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(
            self.path, isolation_level=None, check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row
        # PRAGMA does not accept bound parameters, hence the interpolation.
        # Both values come from rytp.constants, never from user input.
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute(f"PRAGMA journal_mode = {C.SQLITE_JOURNAL_MODE}")
        self.conn.execute(f"PRAGMA busy_timeout = {int(C.SQLITE_BUSY_TIMEOUT_MS)}")

    # -- migrations --------------------------------------------------

    def schema_version(self) -> int:
        """The highest migration applied, or 0 on an untouched file."""
        try:
            row = self.conn.execute(
                "SELECT value FROM schema_version WHERE key = 'version'"
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row["value"]) if row else 0

    def migrate(self) -> int:
        """Apply every pending migration. Returns the version reached."""
        return self.migrate_to(LATEST_VERSION)

    def migrate_to(self, target: int) -> int:
        """Apply pending migrations up to and including ``target``.

        Only the tests stop short of :data:`LATEST_VERSION`; the runner
        needs the parameter so a partly-migrated database can be built
        deliberately.
        """
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            " key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_version (key, value) VALUES ('version', '0')"
        )
        current = self.schema_version()
        for version, sql in MIGRATIONS:
            if version <= current or version > target:
                continue
            # executescript() commits any open transaction before it runs,
            # so BEGIN/COMMIT has to live inside the script itself for the
            # DDL and the version bump to land together. `version` is an
            # int from MIGRATIONS, never user input.
            self.conn.executescript(
                f"BEGIN;\n{sql}\n"
                f"UPDATE schema_version SET value = '{int(version)}'"
                " WHERE key = 'version';\nCOMMIT;"
            )
        return self.schema_version()

    # -- transactions ------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """BEGIN / COMMIT, rolling back and re-raising on any exception.

        Re-entrant. SQLite has no nested transactions, and helpers that
        each wrap their own write are routinely called inside a larger
        one (``channel.sync`` upserts hundreds of videos in a single
        transaction). An inner block therefore joins the outer one; the
        outermost block owns the COMMIT and the ROLLBACK.
        """
        if self.conn.in_transaction:
            yield
            return
        self.conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    # -- lifecycle ---------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
