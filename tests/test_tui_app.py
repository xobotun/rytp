"""The Textual shell, driven headlessly."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from textual.widgets import DataTable, Input, OptionList, Static

from rytp.commands import Command, CommandResult, Param
from rytp.db import Database
from rytp.models import NotFoundError
from rytp.tui.app import RytpApp

calls: list[dict[str, Any]] = []
handler_threads: list[int] = []


def listing(db: Database, **kwargs: Any) -> CommandResult:
    calls.append(kwargs)
    handler_threads.append(threading.get_ident())
    return CommandResult(
        columns=("id", "title"),
        rows=(("1", "Утренний эфир"),),
        message="1 video",
    )


def failing(db: Database, **kwargs: Any) -> CommandResult:
    raise NotFoundError("channel 7 is not in the catalog")


def syncing(db: Database, **kwargs: Any) -> CommandResult:  # pragma: no cover - never run
    raise AssertionError("a long-running command must not run inside the TUI")


def refreshing(db: Database, **kwargs: Any) -> CommandResult:
    calls.append(kwargs)
    return CommandResult(message="refreshed")


SAMPLE = {
    "videos.list": Command(
        name="videos.list",
        group="videos",
        summary="List catalogued videos.",
        params=(Param("limit", int, "Rows.", default=50),),
        handler=listing,
    ),
    "videos.boom": Command(
        name="videos.boom",
        group="videos",
        summary="Fail on purpose.",
        params=(),
        handler=failing,
    ),
    "videos.refresh": Command(
        name="videos.refresh",
        group="videos",
        summary="Refresh the catalog, no arguments needed.",
        params=(),
        handler=refreshing,
    ),
    "channel.sync": Command(
        name="channel.sync",
        group="channel",
        summary="Re-enumerate a channel.",
        params=(Param("channel", str, "Channel.", positional=True),),
        handler=syncing,
        long_running=True,
    ),
}


def drive(db: Database, body: Callable[[RytpApp, Any], Awaitable[None]]) -> None:
    """Run `body` against a mounted RytpApp, headlessly."""

    async def main() -> None:
        app = RytpApp(db, commands=SAMPLE)
        async with app.run_test() as pilot:
            await body(app, pilot)

    asyncio.run(main())


def _option_ids(palette: OptionList) -> list[str | None]:
    return [palette.get_option_at_index(i).id for i in range(palette.option_count)]


def test_the_palette_lists_every_registered_command(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        palette = app.query_one("#palette", OptionList)
        # SAMPLE has two groups (`videos`, `channel`), each of which gets a
        # heading option ahead of its commands (contracts §5, entry 6's
        # palette half) — one heading per group, so `+2`, never restated
        # for a group already being listed since `palette_entries` sorts by
        # dotted name and one group's entries are always contiguous.
        assert palette.option_count == len(SAMPLE) + 2
        ids = _option_ids(palette)
        # Every command still gets exactly one option, headings aside.
        assert {i for i in ids if i is not None and i in SAMPLE} == set(SAMPLE)
        assert {i for i in ids if i is not None} - set(SAMPLE) == {
            "group:videos",
            "group:channel",
        }

    drive(db, body)


def test_group_headings_are_disabled_and_unselectable(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        palette = app.query_one("#palette", OptionList)
        headings = [
            palette.get_option_at_index(i)
            for i in range(palette.option_count)
            if (palette.get_option_at_index(i).id or "").startswith("group:")
        ]
        assert headings, "no group heading was rendered"
        assert all(h.disabled for h in headings)
        for heading in headings:
            assert "channel" in str(heading.prompt) or "videos" in str(heading.prompt)
            # It describes the group, never restates the command list under it.
            assert "sync" not in str(heading.prompt)

    drive(db, body)


def test_typing_in_the_filter_narrows_the_palette(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.query_one("#filter", Input).value = "sync"
        await pilot.pause()
        palette = app.query_one("#palette", OptionList)
        # One heading (`channel`) plus the one matching command.
        assert palette.option_count == 2
        ids = _option_ids(palette)
        assert ids == ["group:channel", "channel.sync"]

    drive(db, body)


def test_running_a_command_fills_the_result_table(db: Database) -> None:
    calls.clear()

    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("videos.list")
        app.run_selected("limit=5")
        await pilot.pause()
        table = app.query_one("#results", DataTable)
        assert table.row_count == 1
        assert [str(c.label) for c in table.columns.values()] == ["id", "title"]
        assert app.status_text == "1 video"
        assert calls == [{"limit": 5}]

    drive(db, body)


def test_a_domain_error_shows_in_the_status_line_and_does_not_crash(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("videos.boom")
        app.run_selected("")
        await pilot.pause()
        assert app.status_text == "channel 7 is not in the catalog"
        assert app.is_running

    drive(db, body)


def test_a_long_running_command_is_not_run_inline(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("channel.sync")
        app.run_selected("CHANNEL_ONE")
        await pilot.pause()
        assert "rytp channel sync CHANNEL_ONE" in app.status_text
        assert app.query_one("#results", DataTable).row_count == 0

    drive(db, body)


def test_a_bad_argument_line_is_reported_not_raised(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("videos.list")
        app.run_selected("limit=many")
        await pilot.pause()
        assert "limit" in app.status_text

    drive(db, body)


# -- entry 22: every handler runs off the event loop ----------------------


def test_a_handler_runs_in_a_worker_thread_not_the_event_loop(db: Database) -> None:
    """`videos.add` blocks on a network call and is still `long_running=False`
    — the bug (entry 22) was calling any such handler inline. Assert the
    mechanism (a different thread ran it), not how fast it returned."""
    handler_threads.clear()

    async def body(app: RytpApp, pilot: Any) -> None:
        main_thread = threading.get_ident()
        app.select("videos.list")
        app.run_selected("limit=1")
        await pilot.pause()
        assert handler_threads, "the handler never ran"
        assert handler_threads[0] != main_thread

    drive(db, body)


def test_a_command_is_shown_running_before_its_worker_completes(db: Database) -> None:
    """The in-flight indicator entry 22 also asks for: something must say a
    command is running, not just go quiet until it finishes."""

    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("videos.list")
        app.run_selected("limit=1")
        assert "running" in app.status_text
        assert "videos list" in app.status_text

    drive(db, body)


# -- entry 14a: arrow keys reach the option list ---------------------------


def test_down_from_the_filter_moves_the_palette_highlight(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        palette = app.query_one("#palette", OptionList)
        app.query_one("#filter", Input).focus()
        await pilot.pause()
        start = palette.highlighted
        await pilot.press("down")
        await pilot.pause()
        # Not `start + 1`: a group heading between two commands is a real,
        # counted option (skipped by `OptionList`'s own cursor movement, not
        # by index arithmetic here), so the highlight may jump by more than
        # one position. What must hold is that it moved, and landed on a
        # real, selectable option rather than a disabled heading.
        assert palette.highlighted != start
        assert not palette.get_option_at_index(palette.highlighted).disabled

    drive(db, body)


def test_up_from_the_filter_moves_the_palette_highlight(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        palette = app.query_one("#palette", OptionList)
        app.query_one("#filter", Input).focus()
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("down")
        await pilot.pause()
        moved = palette.highlighted
        await pilot.press("up")
        await pilot.pause()
        assert palette.highlighted != moved
        assert not palette.get_option_at_index(palette.highlighted).disabled

    drive(db, body)


def test_arrow_keys_still_leave_the_filter_focused(db: Database) -> None:
    """Forwarding the highlight move must not steal focus — the filter stays
    the place typing narrows the list, exactly as before the key press."""

    async def body(app: RytpApp, pilot: Any) -> None:
        filter_input = app.query_one("#filter", Input)
        filter_input.focus()
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        assert app.focused is filter_input

    drive(db, body)


# -- entry 14c: Enter in the filter never does nothing ---------------------


def test_enter_in_the_filter_moves_to_arguments_for_a_parameterised_command(
    db: Database,
) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.query_one("#filter", Input).value = "channel.sync"
        await pilot.pause()
        app.query_one("#filter", Input).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.focused is app.query_one("#arguments", Input)

    drive(db, body)


def test_enter_in_the_filter_runs_a_parameterless_command_directly(db: Database) -> None:
    calls.clear()

    async def body(app: RytpApp, pilot: Any) -> None:
        app.query_one("#filter", Input).value = "videos.refresh"
        await pilot.pause()
        app.query_one("#filter", Input).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert calls == [{}]
        assert app.status_text == "refreshed"

    drive(db, body)


def test_enter_in_an_empty_filter_is_reported_not_silently_dropped(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.query_one("#filter", Input).value = "no such command anywhere"
        await pilot.pause()
        app.query_one("#filter", Input).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.status_text
        assert app.is_running

    drive(db, body)


# -- entry 14b: one spelling on screen --------------------------------------


def test_the_palette_shows_the_spaced_form_not_the_dotted_one(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        palette = app.query_one("#palette", OptionList)
        rendered = "\n".join(
            str(palette.get_option_at_index(i).prompt) for i in range(palette.option_count)
        )
        assert "channel sync" in rendered
        assert "channel.sync" not in rendered

    drive(db, body)


# -- entry 18: the arguments placeholder shows this command's shape --------


def test_selecting_a_command_updates_the_arguments_placeholder(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        arguments = app.query_one("#arguments", Input)
        before = arguments.placeholder
        app.select("channel.sync")
        palette = app.query_one("#palette", OptionList)
        palette.action_select()
        await pilot.pause()
        assert arguments.placeholder != before
        assert arguments.placeholder == "channel sync <channel>"

    drive(db, body)


# -- entry 12: the home screen says it runs things --------------------------


def test_the_home_screen_explains_how_to_run_a_command(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        home_help = app.query_one("#home-help", Static)
        rendered = str(home_help.content)
        assert "filter" in rendered.lower()
        assert "enter" in rendered.lower()

    drive(db, body)


def test_importing_the_tui_creates_no_directories(tmp_path: Path) -> None:
    import subprocess
    import sys

    from tests.test_config import child_env

    proc = subprocess.run(
        [sys.executable, "-c", "import rytp.tui.app"],
        cwd=str(tmp_path),
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert list(tmp_path.iterdir()) == []
