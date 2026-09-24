"""The queue view: one adapter over the commands both surfaces already share."""

from __future__ import annotations

import pytest

from rytp.db import Database
from rytp.jobs import queue as Q
from rytp.tui.jobs_view import JobsView
from tests.fakes import make_video


@pytest.fixture()
def queued(db: Database) -> Database:
    """One job in each interesting state, plus one carrying a note."""
    first = make_video(db, external_id="VIDEO_A", title="Первое видео")
    second = make_video(db, external_id="VIDEO_B", title="Второе видео")
    Q.enqueue(db, "download", first)
    failed = Q.enqueue(db, "captions", first)
    Q.fail(db, failed, error="HTTP Error 403")
    done = Q.enqueue(db, "download", second)
    Q.finish(db, done, note="взяты субтитры другого языка")
    return db


def test_the_columns_are_the_ones_the_cli_prints(queued: Database) -> None:
    """design §10: one definition, two surfaces. The view calls `jobs.list`
    rather than writing its own SELECT, so they cannot drift."""
    view = JobsView(queued)
    assert view.columns == (
        "id", "kind", "target", "state", "pool", "attempts", "not_before",
        "error", "note",
    )
    assert len(view.rows) == 3


def test_every_cell_is_a_string(queued: Database) -> None:
    view = JobsView(queued)
    assert all(isinstance(cell, str) for row in view.rows for cell in row)


def test_the_status_line_carries_the_stats_the_design_asks_for(queued: Database) -> None:
    """design §5: "Throttling shows up in `rytp jobs stats` rather than as a
    mystery stall"."""
    status = JobsView(queued).status
    for token in ("pending", "failed", "throttled", "notes", "paused"):
        assert token in status.lower(), status


def test_cycling_the_state_filter_walks_the_command_s_own_choices(
    queued: Database,
) -> None:
    """Derived, not listed: a new job state appears here without an edit, which
    is the fix for the same defect that made a hardcoded list of job kinds go
    stale before there was any code."""
    view = JobsView(queued)
    assert view.state == ""
    seen = [view.cycle_state() for _ in range(3)]
    assert seen[0] != ""
    assert all(value in view.state_choices for value in seen)


def test_filtering_by_state_narrows_the_rows(queued: Database) -> None:
    view = JobsView(queued)
    for _ in range(len(view.state_choices)):
        if view.state == "failed":
            break
        view.cycle_state()
    assert view.state == "failed", view.state_choices
    assert [row[1] for row in view.rows] == ["captions"]
    assert "failed" in view.status


def test_filtering_by_pool_narrows_the_rows(queued: Database) -> None:
    view = JobsView(queued)
    for _ in range(len(view.pool_choices)):
        if view.pool == "network":
            break
        view.cycle_pool()
    assert view.pool == "network", view.pool_choices
    assert {row[4] for row in view.rows} == {"network"}


def test_the_note_of_the_highlighted_job_is_readable_in_full(queued: Database) -> None:
    """The reason `jobs.note` exists: on the bulk path nobody reads stdout, so
    a warning that only fits in a truncated column is a warning nobody sees."""
    view = JobsView(queued)
    index = next(i for i, row in enumerate(view.rows) if row[3] == "done")
    assert view.note_at(index) == "взяты субтитры другого языка"


def test_a_job_with_no_note_reads_as_empty(queued: Database) -> None:
    view = JobsView(queued)
    index = next(i for i, row in enumerate(view.rows) if row[3] == "pending")
    assert view.note_at(index) == ""


def test_retrying_the_highlighted_job_revives_it(queued: Database) -> None:
    view = JobsView(queued)
    index = next(i for i, row in enumerate(view.rows) if row[3] == "failed")
    job_id = view.job_id_at(index)
    assert job_id is not None
    message = view.retry_at(index)
    assert "retried" in message
    assert Q.get_job(queued, job_id).state == "pending"


def test_cancelling_the_highlighted_job_drops_it(queued: Database) -> None:
    view = JobsView(queued)
    index = next(i for i, row in enumerate(view.rows) if row[3] == "pending")
    job_id = view.job_id_at(index)
    assert job_id is not None
    view.cancel_at(index)
    assert Q.get_job(queued, job_id).state == "cancelled"


def test_retrying_everything_failed_takes_one_key(queued: Database) -> None:
    view = JobsView(queued)
    assert "retried" in view.retry_all_failed()
    assert not [row for row in view.rows if row[3] == "failed"]


def test_pausing_and_resuming_round_trips(queued: Database) -> None:
    view = JobsView(queued)
    assert "paused" in view.toggle_pause().lower()
    assert Q.is_paused(queued) is True
    assert "resumed" in view.toggle_pause().lower()
    assert Q.is_paused(queued) is False


def test_an_index_off_the_end_of_the_table_is_not_an_error(queued: Database) -> None:
    """A cursor on an empty table is a real state, not an exception."""
    view = JobsView(queued)
    assert view.job_id_at(99) is None
    assert view.note_at(99) == ""
    assert "no job" in view.retry_at(99).lower()


def test_an_empty_queue_says_so_rather_than_showing_nothing(db: Database) -> None:
    view = JobsView(db)
    assert view.rows == ()
    assert "nothing queued" in view.status.lower()


# --- the screen --------------------------------------------------------

import asyncio  # noqa: E402
from collections.abc import Awaitable, Callable  # noqa: E402
from typing import Any  # noqa: E402

from textual.widgets import DataTable, Static  # noqa: E402

from rytp.tui.screens.jobs import JobsScreen  # noqa: E402


def drive_screen(
    db: Database, body: Callable[[JobsScreen, Any], Awaitable[None]]
) -> None:
    """Mount the screen on a bare app, headlessly."""
    from textual.app import App, ComposeResult

    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            return iter(())

        def on_mount(self) -> None:
            self.push_screen(JobsScreen(db))

    async def main() -> None:
        app = Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, JobsScreen)
            await body(app.screen, pilot)

    asyncio.run(main())


def test_the_screen_shows_the_queue(queued: Database) -> None:
    def check(screen: JobsScreen, pilot: Any) -> Awaitable[None]:
        async def body() -> None:
            table = screen.query_one("#jobs-table", DataTable)
            assert table.row_count == 3
            assert [str(c.label) for c in table.columns.values()] == list(
                screen.view.columns
            )
            assert str(screen.query_one("#jobs-status", Static).content)

        return body()

    drive_screen(queued, lambda screen, pilot: check(screen, pilot))


def test_one_key_cycles_the_state_filter(queued: Database) -> None:
    async def body(screen: JobsScreen, pilot: Any) -> None:
        before = screen.view.state
        screen.action_cycle_state()
        await pilot.pause()
        assert screen.view.state != before
        assert screen.query_one("#jobs-table", DataTable).row_count == len(
            screen.view.rows
        )

    drive_screen(queued, body)


def test_the_note_of_the_highlighted_job_is_shown(queued: Database) -> None:
    async def body(screen: JobsScreen, pilot: Any) -> None:
        table = screen.query_one("#jobs-table", DataTable)
        table.move_cursor(
            row=next(i for i, r in enumerate(screen.view.rows) if r[3] == "done")
        )
        await pilot.pause()
        screen.action_refresh()
        await pilot.pause()
        assert "субтитры" in str(screen.query_one("#jobs-note", Static).content)

    drive_screen(queued, body)


def test_retrying_from_the_screen_changes_the_row(queued: Database) -> None:
    async def body(screen: JobsScreen, pilot: Any) -> None:
        table = screen.query_one("#jobs-table", DataTable)
        table.move_cursor(
            row=next(i for i, r in enumerate(screen.view.rows) if r[3] == "failed")
        )
        await pilot.pause()
        screen.action_retry()
        await pilot.pause()
        assert not [row for row in screen.view.rows if row[3] == "failed"]

    drive_screen(queued, body)


def test_the_screen_refreshes_on_a_timer(queued: Database) -> None:
    """The worker is another process; nothing tells the TUI a job finished."""

    async def body(screen: JobsScreen, pilot: Any) -> None:
        assert screen._timer is not None

    drive_screen(queued, body)


def test_the_screen_holds_no_state_of_its_own(queued: Database) -> None:
    """Part 7's rule, which is why nine tests were enough for its mapper.

    Compared against a bare `Screen`, not against nothing: Textual's own
    `__init__` sets dozens of attributes and mounting sets more, so the only
    meaningful question is what *this* class adds on top.
    """
    from textual.screen import Screen

    added = set(vars(JobsScreen(queued))) - set(vars(Screen()))
    assert added == {"view", "_timer"}
