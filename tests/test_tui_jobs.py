"""The queue view: one adapter over the commands both surfaces already share."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta

import pytest

from rytp import config
from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import set_setting
from rytp.jobs import queue as Q
from rytp.jobs import worker as W
from rytp.tui import jobs_view as JV
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
        "error", "note", "progress",
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


# --- starting a worker (BUGS.md entry 41) ------------------------------


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid


def test_spawn_worker_refuses_when_a_worker_is_already_running(
    queued: Database,
) -> None:
    W.acquire_lease(queued)

    def never(argv: object, env: object) -> _FakeProcess:
        raise AssertionError("must not spawn while a lease is live")

    view = JobsView(queued, spawn=never)
    message = view.spawn_worker()
    assert "already running" in message
    assert "pid" in message


def test_spawn_worker_launches_the_worker_module_when_nothing_holds_the_lease(
    queued: Database,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def fake_spawn(argv: object, env: object) -> _FakeProcess:
        calls.append((tuple(argv), dict(env)))  # type: ignore[arg-type]
        return _FakeProcess(pid=4242)

    view = JobsView(queued, spawn=fake_spawn)
    message = view.spawn_worker()

    assert "started worker (pid 4242)" in message
    assert len(calls) == 1
    argv, env = calls[0]
    # One worker, all pools — matches `--pool all`'s own default, and is the
    # only shape that can work at all: the lease is global (BUGS.md entry 41,
    # "the worker lease is global, not per-pool").
    assert argv == (sys.executable, "-m", "rytp", "worker")
    assert env["RYTP_DATA"] == str(config.data_root())


def test_spawn_worker_treats_a_stale_lease_as_absent(queued: Database) -> None:
    """Same rule `acquire_lease` enforces (`WORKER_LEASE_STALE_S`): a
    heartbeat old enough belongs to a worker that died, not one running."""
    stale = datetime.now(UTC) - timedelta(seconds=C.WORKER_LEASE_STALE_S + 5)
    set_setting(
        queued,
        C.SETTING_WORKER_LEASE,
        json.dumps({"pid": 1, "heartbeat": stale.isoformat()}),
    )
    view = JobsView(queued, spawn=lambda argv, env: _FakeProcess(pid=1))
    assert "started worker" in view.spawn_worker()


def test_spawn_worker_process_detaches_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(JV.sys, "platform", "linux")
    captured: dict[str, object] = {}

    class FakePopen:
        def __init__(self, argv: object, **kwargs: object) -> None:
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            self.pid = 111

    monkeypatch.setattr(JV.subprocess, "Popen", FakePopen)
    process = JV._spawn_worker_process(("rytp", "worker"), {})

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["start_new_session"] is True
    assert "creationflags" not in kwargs
    assert kwargs["stdout"] is JV.subprocess.DEVNULL
    assert kwargs["stderr"] is JV.subprocess.DEVNULL
    assert kwargs["stdin"] is JV.subprocess.DEVNULL
    assert process.pid == 111


def test_spawn_worker_process_detaches_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(JV.sys, "platform", "win32")
    captured: dict[str, object] = {}

    class FakePopen:
        def __init__(self, argv: object, **kwargs: object) -> None:
            captured["kwargs"] = kwargs
            self.pid = 222

    monkeypatch.setattr(JV.subprocess, "Popen", FakePopen)
    JV._spawn_worker_process(("rytp", "worker"), {})

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["creationflags"] == JV._DETACHED_PROCESS | JV._CREATE_NEW_PROCESS_GROUP
    assert "start_new_session" not in kwargs


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


def test_the_screen_renders_a_bracketed_progress_detail_without_crashing(
    db: Database,
) -> None:
    """BUGS.md entry 20: Textual's markup parser crashes on a bare
    `Static.update`/`add_row`, and Rich's table formatter silently drops
    bracketed text — a progress detail can be a filename or anything else,
    so it must go through `plain_row`/`set_text` like every other cell."""
    vid = make_video(db, external_id="VIDEO_C", title="Третье видео")
    running_id = Q.enqueue(db, "download", vid)
    claimed = Q.claim(db, "network")
    assert claimed is not None and claimed.id == running_id
    Q.set_progress(db, running_id, "download [3/10] клип.mp4")

    def body(screen: JobsScreen, pilot: Any) -> Awaitable[None]:
        async def run() -> None:
            table = screen.query_one("#jobs-table", DataTable)
            progress_col = screen.view.columns.index("progress")
            cell = table.get_cell_at((0, progress_col))
            assert "[3/10]" in str(cell)

        return run()

    drive_screen(db, lambda screen, pilot: body(screen, pilot))


def test_the_screen_refreshes_on_a_timer(queued: Database) -> None:
    """The worker is another process; nothing tells the TUI a job finished."""

    async def body(screen: JobsScreen, pilot: Any) -> None:
        assert screen._timer is not None

    drive_screen(queued, body)


def test_the_refresh_the_timer_calls_actually_shows_another_process_s_write(
    queued: Database,
) -> None:
    """Not just that a timer exists (the test above), but that the refresh it
    fires picks up a row changed by something other than this screen — the
    same shape of change a real `rytp worker` process makes by claiming a
    job out from under the TUI."""

    async def body(screen: JobsScreen, pilot: Any) -> None:
        index = next(i for i, r in enumerate(screen.view.rows) if r[3] == "pending")
        job_id = screen.view.job_id_at(index)
        assert job_id is not None
        claimed = Q.claim(queued, "network")
        assert claimed is not None and claimed.id == job_id

        screen.action_refresh()
        await pilot.pause()

        row = next(r for r in screen.view.rows if int(r[0]) == job_id)
        assert row[3] == "running"

    drive_screen(queued, body)


def test_the_worker_key_reports_a_live_lease(queued: Database) -> None:
    W.acquire_lease(queued)

    async def body(screen: JobsScreen, pilot: Any) -> None:
        screen.action_start_worker()
        await pilot.pause()
        status = str(screen.query_one("#jobs-status", Static).content)
        assert "already running" in status

    drive_screen(queued, body)


def test_the_worker_key_starts_a_detached_worker_when_nothing_holds_the_lease(
    queued: Database,
) -> None:
    async def body(screen: JobsScreen, pilot: Any) -> None:
        screen.view._spawn = lambda argv, env: _FakeProcess(pid=777)
        screen.action_start_worker()
        await pilot.pause()
        status = str(screen.query_one("#jobs-status", Static).content)
        assert "started worker (pid 777)" in status

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
