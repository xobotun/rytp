"""The Textual shell, driven headlessly."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from textual.widgets import DataTable, Input, OptionList

from rytp.commands import Command, CommandResult, Param
from rytp.db import Database
from rytp.models import NotFoundError
from rytp.tui.app import RytpApp

calls: list[dict[str, Any]] = []


def listing(db: Database, **kwargs: Any) -> CommandResult:
    calls.append(kwargs)
    return CommandResult(
        columns=("id", "title"),
        rows=(("1", "Утренний эфир"),),
        message="1 video",
    )


def failing(db: Database, **kwargs: Any) -> CommandResult:
    raise NotFoundError("channel 7 is not in the catalog")


def syncing(db: Database, **kwargs: Any) -> CommandResult:  # pragma: no cover - never run
    raise AssertionError("a long-running command must not run inside the TUI")


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


def test_the_palette_lists_every_registered_command(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        palette = app.query_one("#palette", OptionList)
        assert palette.option_count == len(SAMPLE)
        ids = [palette.get_option_at_index(i).id for i in range(palette.option_count)]
        assert ids == sorted(SAMPLE)

    drive(db, body)


def test_typing_in_the_filter_narrows_the_palette(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.query_one("#filter", Input).value = "sync"
        await pilot.pause()
        palette = app.query_one("#palette", OptionList)
        assert palette.option_count == 1
        assert palette.get_option_at_index(0).id == "channel.sync"

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
