"""Watching the queue (design §10, "watching job progress").

An adapter, not an implementation. Part 2's `jobs list`, `jobs stats`,
`jobs retry`, `jobs cancel` and `queue pause`/`resume` already do all of this
and are already registered, so this view calls them through the registry.
design §10's claim is that both surfaces come from one definition; a screen
that wrote its own SELECT would quietly make that false.

What it adds is what a command line cannot: filters you cycle with one key,
a refresh that keeps up with a worker in another process, and the highlighted
job's `note` shown in full rather than in a column.
"""

from __future__ import annotations

from typing import Final

from rytp import constants as C
from rytp.commands import CommandResult, resolve
from rytp.db import Database
from rytp.models import RytpError

__all__ = ["JobsView"]

#: Column positions in `jobs.list`'s result. Named rather than counted so a
#: reader can see what the view depends on; the test above pins the header.
_ID: Final = 0
_STATE: Final = 3
_NOTE: Final = 8


def _choices(command: str, param: str) -> tuple[str, ...]:
    """A parameter's declared choices, used as this view's filter cycle.

    Derived rather than restated: a new job state or a fourth pool shows up
    here with no edit. A hardcoded list of states would be the same defect as
    the hardcoded list of job kinds that went stale before any code existed.
    """
    for declared in resolve(command).params:
        if declared.name == param:
            return declared.choices or ("",)
    raise KeyError(f"{command} has no parameter {param!r}")


class JobsView:
    """What is queued, running, done and failed — and why."""

    def __init__(self, db: Database, *, limit: int = C.TUI_QUEUE_ROW_LIMIT) -> None:
        self._db = db
        self._limit = limit
        self.state: str = ""
        self.pool: str = ""
        self.state_choices: tuple[str, ...] = _choices("jobs.list", "state")
        self.pool_choices: tuple[str, ...] = _choices("jobs.list", "pool")
        self.columns: tuple[str, ...] = ()
        self.rows: tuple[tuple[str, ...], ...] = ()
        self.status: str = ""
        self.refresh()

    # -- reading -----------------------------------------------------

    def refresh(self) -> None:
        """Re-read the queue. Called on a timer and after every action."""
        listing: CommandResult = resolve("jobs.list").handler(
            self._db,
            state=self.state,
            pool=self.pool,
            kind="",
            limit=self._limit,
        )
        self.columns = listing.columns
        self.rows = listing.rows
        self.status = self._status_line()

    def _status_line(self) -> str:
        stats = resolve("jobs.stats").handler(self._db)
        flat = {name: value for name, value in stats.rows}
        parts = [
            f"{name} {value}"
            for name, value in flat.items()
            if value not in ("", "0") or name in ("paused", "throttled")
        ]
        where = []
        if self.state:
            where.append(f"state={self.state}")
        if self.pool:
            where.append(f"pool={self.pool}")
        shown = f"{len(self.rows)} shown"
        if where:
            shown += " (" + ", ".join(where) + ")"
        if not self.rows and not self.state and not self.pool:
            shown = "nothing queued"
        # `with notes` is spelled `notes` here so one short line carries it.
        return shown + " · " + " · ".join(parts).replace("with notes", "notes")

    # -- the highlighted row -----------------------------------------

    def _row(self, index: int) -> tuple[str, ...] | None:
        return self.rows[index] if 0 <= index < len(self.rows) else None

    def job_id_at(self, index: int) -> int | None:
        row = self._row(index)
        return int(row[_ID]) if row else None

    def note_at(self, index: int) -> str:
        """The full note, which the column can only show the start of."""
        job_id = self.job_id_at(index)
        if job_id is None:
            return ""
        found = self._db.conn.execute(
            "SELECT note FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return str(found["note"]) if found and found["note"] else ""

    # -- filters -----------------------------------------------------

    def cycle_state(self) -> str:
        self.state = self._next(self.state_choices, self.state)
        self.refresh()
        return self.state

    def cycle_pool(self) -> str:
        self.pool = self._next(self.pool_choices, self.pool)
        self.refresh()
        return self.pool

    @staticmethod
    def _next(choices: tuple[str, ...], current: str) -> str:
        position = choices.index(current) if current in choices else -1
        return choices[(position + 1) % len(choices)]

    # -- acting ------------------------------------------------------

    def retry_at(self, index: int) -> str:
        job_id = self.job_id_at(index)
        if job_id is None:
            return "no job selected"
        # jobs.retry's `state` always filters (Q.retry has no "any state"
        # wildcard: it is one `state = ?` clause, defaulting to "failed"), so
        # a retry by id must still name the one state a person retries from
        # a table. Finding, not fixed here: `jobs.retry --job-id` cannot
        # revive a `blocked` or `cancelled` row without also passing
        # `--state`, which the CLI's own choices already require anyway.
        return self._run("jobs.retry", job_id=job_id, kind="", state="failed")

    def retry_all_failed(self) -> str:
        return self._run("jobs.retry", job_id=0, kind="", state="failed")

    def cancel_at(self, index: int) -> str:
        job_id = self.job_id_at(index)
        if job_id is None:
            return "no job selected"
        return self._run(
            "jobs.cancel", job_id=job_id, kind="", state="", target_id=0, dry_run=False
        )

    def toggle_pause(self) -> str:
        from rytp.jobs.queue import is_paused

        name = "queue.resume" if is_paused(self._db) else "queue.pause"
        return self._run(name)

    def _run(self, command: str, **values: object) -> str:
        """Run a registered command and refresh. Errors become a status line."""
        try:
            result = resolve(command).handler(self._db, **values)
        except RytpError as exc:
            return str(exc)
        self.refresh()
        return result.message or f"{command} done"
