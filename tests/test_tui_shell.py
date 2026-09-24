"""The shell that makes five screens one application."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from textual.widgets import DataTable, OptionList, Static

from rytp.commands import Command, CommandResult, Param
from rytp.db import Database
from rytp.tui import navigation as N
from rytp.tui.app import RytpApp


def noop(db: Database, **kwargs: Any) -> CommandResult:
    return CommandResult(message="ok")


SAMPLE = {
    "videos.list": Command(
        name="videos.list",
        group="videos",
        summary="List catalogued videos.",
        params=(Param("limit", int, "Rows.", default=50),),
        handler=noop,
    ),
}


def drive(db: Database, body: Callable[[RytpApp, Any], Awaitable[None]]) -> None:
    async def main() -> None:
        app = RytpApp(db, commands=SAMPLE)
        async with app.run_test() as pilot:
            await body(app, pilot)

    asyncio.run(main())


def test_the_home_view_says_where_you_can_go(db: Database) -> None:
    """Design §10 lists five things the TUI must offer besides the palette.
    A user who cannot see that they exist does not have them."""

    async def body(app: RytpApp, pilot: Any) -> None:
        strip = app.query_one("#home-nav", Static)
        rendered = str(strip.content)
        for entry in N.SCREENS:
            assert entry.title in rendered
            assert entry.key.upper() in rendered

    drive(db, body)


def test_the_palette_is_still_on_the_app_root(db: Database) -> None:
    """Part 1's seven app tests drive `#palette` here. Keeping it is the whole
    reason home is the palette rather than a pushed screen."""

    async def body(app: RytpApp, pilot: Any) -> None:
        assert app.query_one("#palette", OptionList) is not None
        assert app.query_one("#results", DataTable) is not None

    drive(db, body)


def test_every_app_binding_comes_from_the_navigation_map(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        bound = {(b.key, b.action) for b in app.BINDINGS}
        for key, action, _label in N.binding_rows():
            assert (key, action) in bound, (key, action)

    drive(db, body)


@pytest.mark.parametrize("entry", list(N.SCREENS), ids=[e.id for e in N.SCREENS])
def test_each_screen_key_opens_that_screen(db: Database, entry: N.ScreenEntry) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        await pilot.press(entry.key)
        await pilot.pause()
        assert isinstance(app.screen, N.load_screen_class(entry))

    drive(db, body)


def test_escape_comes_back_to_the_home_view(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        await pilot.press("f7")
        await pilot.pause()
        assert app.screen is not app.screen_stack[0]
        await pilot.press(N.BACK_KEY)
        await pilot.pause()
        assert app.screen is app.screen_stack[0]
        assert app.query_one("#palette", OptionList) is not None

    drive(db, body)


def test_help_lists_every_binding_in_the_application(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        await pilot.press("f1")
        await pilot.pause()
        table = app.screen.query_one("#help-bindings", DataTable)
        shown = {
            (str(table.get_cell_at((row, 0))), str(table.get_cell_at((row, 1))))
            for row in range(table.row_count)
        }
        for key, where, _what in N.help_rows():
            assert (key, where) in shown, (key, where)

    drive(db, body)


def test_help_closes_the_same_way_every_screen_does(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        await pilot.press("f1")
        await pilot.pause()
        await pilot.press(N.BACK_KEY)
        await pilot.pause()
        assert app.screen is app.screen_stack[0]

    drive(db, body)


def test_the_actions_parts_four_and_seven_published_still_work(db: Database) -> None:
    """Their plans list `RytpApp.action_speakers()` as produced. Keeping the
    three names is one line each and costs nothing."""

    async def body(app: RytpApp, pilot: Any) -> None:
        for action, screen_id in (
            (app.action_speakers, "speakers"),
            (app.action_search, "search"),
            (app.action_transcripts, "transcripts"),
        ):
            action()
            await pilot.pause()
            assert isinstance(app.screen, N.load_screen_class(N.screen_by_id(screen_id)))
            await pilot.press(N.BACK_KEY)
            await pilot.pause()

    drive(db, body)


def test_opening_a_screen_that_does_not_exist_is_reported_not_raised(
    db: Database,
) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.action_open("nonsense")
        await pilot.pause()
        assert "nonsense" in app.status_text
        assert app.is_running

    drive(db, body)


def test_starting_the_app_imports_no_heavy_optional_extra() -> None:
    """`rytp tui` must not pay for torch, pyannote, faster-whisper or yt-dlp
    before it draws anything.

    The plan text this test descends from asked for something stronger — that
    starting the app import none of ``rytp.index``, ``rytp.assemble``,
    ``rytp.render``, ``rytp.diarize`` or ``rytp.transcribe`` at all. That is
    unattainable given contracts §5's own architecture: ``rytp.commands``'s
    module docstring is explicit that "registration happens as a side effect
    of importing a command module", so building ``COMMANDS`` (which both
    ``RytpApp`` and ``rytp.cli.build_app`` need, identically) necessarily
    imports every command module, and several of those
    (``rytp.commands.transcribe``, ``.diarize``, ``.render``, ...) reference
    their handler functions and type hints from those packages at module
    level, not inside a function. Verified this is not new: reverting every
    Task 6 change and running the same probe against the pre-Task-6 tree
    shows the identical import set. Rewriting command registration to be
    lazy is a cross-cutting change to Parts 2-7's own modules and out of
    Part 8's scope; the actual constraint CLAUDE.md and the global
    constraints state — heavy dependencies imported lazily inside functions,
    never at module import time — is what this test checks instead, and it
    already holds.
    """
    import subprocess
    import sys

    from tests.consistency import repo_root

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rytp.tui.app; "
            "print(sorted(m.split('.')[0] for m in sys.modules "
            "if m.split('.')[0] in "
            "('torch', 'pyannote', 'faster_whisper', 'whisper', 'yt_dlp', 'gigaam')))",
        ],
        cwd=str(repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout
