"""BUGS.md entry 20: the TUI must not crash on square brackets.

`set_status`/`show_result` used to hand text straight to `Static.update`,
which parses it as Textual console markup. A usage line spells optional
parameters `[title=…]`, and square brackets are markup syntax, so any
message carrying one killed the app with `MarkupError`.

Rich's own markup parser tolerates such a string; Textual 8's stricter
parser is what actually raises, so every `Static` assertion here goes
through `textual.content.Content.from_markup` — never `rich.markup` —
to catch exactly the failure that was missed before.

`DataTable` cells turned out to have a *different* exposure: its own cell
formatter runs plain `str` cells through Rich's `Text.from_markup`, which
is lenient about some shapes (`[title=…]` renders, silently dropped) and
not about others (`[/take2]` raises `rich.errors.MarkupError` outright,
since it looks like an unmatched closing tag). Both are covered below.
"""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from textual.content import Content
from textual.markup import MarkupError
from textual.widgets import DataTable, Static

from rytp.commands import COMMANDS, Command, CommandResult, Param
from rytp.db import Database
from rytp.tui.app import RytpApp
from rytp.tui.text import plain_row, set_text

# -- helpers ---------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OWNED_MODULES = (
    _REPO_ROOT / "rytp" / "tui" / "app.py",
    _REPO_ROOT / "rytp" / "tui" / "screens" / "cutlist.py",
    _REPO_ROOT / "rytp" / "tui" / "screens" / "jobs.py",
    _REPO_ROOT / "rytp" / "tui" / "screens" / "speakers.py",
    _REPO_ROOT / "rytp" / "tui" / "screens" / "transcript.py",
    _REPO_ROOT / "rytp" / "tui" / "screens" / "search.py",
)


def _raises_markup_error(text: str) -> bool:
    """What Textual 8 actually does when it renders `text` as markup.

    This is the check the bug report says matters: `rich.markup` accepts
    strings that `Content.from_markup` rejects, so a test written against
    Rich would have passed while the app still crashed.
    """
    try:
        Content.from_markup(text)
    except MarkupError:
        return True
    return False


def drive(db: Database, body: Callable[[RytpApp, Any], Awaitable[None]]) -> None:
    async def main() -> None:
        app = RytpApp(db)
        async with app.run_test() as pilot:
            await body(app, pilot)

    asyncio.run(main())


def _query_one_receivers(tree: ast.Module) -> set[str]:
    """Names of local variables assigned straight from a `query_one(...)` call.

    `transcript.py` writes `status = self.query_one(...)` once and calls
    `.update(...)` on `status` later, rather than chaining the call — this
    finds that binding so the sweep below can still catch it.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "query_one"
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _is_static_receiver(node: ast.Call, local_names: set[str]) -> bool:
    """Best-effort: is this `.update(...)` call reached through a `Static`?

    Only flags calls whose receiver is a `query_one(...)` chain or a local
    name bound from one. A bare `dict.update(...)` or similar, which other
    tasks' hand-offs of these same files may add, is left alone — this test
    file is not theirs to fix.
    """
    assert isinstance(node.func, ast.Attribute)
    receiver = node.func.value
    if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute):
        return receiver.func.attr == "query_one"
    if isinstance(receiver, ast.Name):
        return receiver.id in local_names
    return False


# -- the bracket string itself really is markup-hostile ---------------------


def test_the_usage_line_is_confirmed_hostile_to_textuals_markup_parser() -> None:
    """Sanity check that this file is testing the right failure mode."""
    from rytp.tui.palette import usage_line

    line = usage_line(COMMANDS["videos.add"])
    assert "[" in line and "]" in line
    assert _raises_markup_error(f"too many values for videos.add: {line}")


# -- the source sweep: no bare Static.update remains -------------------------


@pytest.mark.parametrize("path", _OWNED_MODULES, ids=lambda p: p.name)
def test_no_bare_static_update_remains(path: Path) -> None:
    """A source-level scan: every `Static.update(` call must be gone.

    `set_text` is the only permitted way to put text on a `Static` widget
    in these files, so a new call site cannot reintroduce the crash without
    this test noticing. Walking the AST rather than grepping avoids false
    positives from comments or docstrings, and only flags a receiver this
    file can prove reaches a `Static` (a `query_one(...)` chain, or a local
    bound from one) so an unrelated `.update(` a later task's hand-off adds
    to one of these files is not our call to fail on.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    local_names = _query_one_receivers(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and _is_static_receiver(node, local_names)
        ):
            raise AssertionError(
                f"{path}: a Static widget's `.update(` is called directly — "
                "route it through rytp.tui.text.set_text instead"
            )


@pytest.mark.parametrize("path", _OWNED_MODULES, ids=lambda p: p.name)
def test_add_row_never_takes_a_bare_string_cell(path: Path) -> None:
    """Every `DataTable.add_row(*row)` must wrap its cells first.

    `add_row` runs a plain `str` cell through Rich's markup parser (a
    different, but equally real, exposure from `Static.update`'s). Every
    owned call site must pass through `rytp.tui.text.plain_row` — a bare
    `table.add_row(*row)` on a tuple of strings is the regression this
    guards against.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_row"
        ):
            wrapped = any(
                isinstance(arg, ast.Starred)
                and isinstance(arg.value, ast.Call)
                and isinstance(arg.value.func, ast.Name)
                and arg.value.func.id == "plain_row"
                for arg in node.args
            )
            # A call with no starred/spread argument at all (e.g. literal
            # cells passed directly) is not the `*row` shape this sweep is
            # about; only flag a starred argument that isn't `plain_row(...)`.
            has_bare_star = any(
                isinstance(arg, ast.Starred)
                and not (
                    isinstance(arg.value, ast.Call)
                    and isinstance(arg.value.func, ast.Name)
                    and arg.value.func.id == "plain_row"
                )
                for arg in node.args
            )
            if has_bare_star and not wrapped:
                raise AssertionError(
                    f"{path}: `add_row(*...)` does not go through plain_row"
                )


# -- set_text itself ---------------------------------------------------------


def test_set_text_renders_brackets_as_plain_text() -> None:
    async def main() -> None:
        from textual.app import App, ComposeResult

        class _T(App[None]):
            def compose(self) -> ComposeResult:
                yield Static("", id="status")

        app = _T()
        async with app.run_test() as pilot:
            widget = app.query_one("#status", Static)
            message = "слишком много значений [title=…] [kind=…] [channel=…]"
            set_text(widget, message)
            await pilot.pause()
            # Stays alive and renders the literal text, brackets included.
            assert app.is_running
            rendered = widget.render()
            assert isinstance(rendered, Content)
            assert rendered.plain == message

    asyncio.run(main())


# -- plain_row: the DataTable cell exposure -----------------------------------


def test_plain_row_survives_a_closing_tag_shape_that_crashes_rich() -> None:
    """`[/x]` looks like an unmatched closing tag to Rich's own parser."""
    from rich.errors import MarkupError as RichMarkupError
    from rich.text import Text

    cell = "клип [/дубль2].mp4"
    with pytest.raises(RichMarkupError):
        Text.from_markup(cell)
    (wrapped,) = plain_row([cell])
    assert wrapped.plain == cell


def test_plain_row_survives_a_bracketed_prefix_rich_would_silently_eat() -> None:
    from rich.text import Text

    # A Latin, lower-case tag shape (`[bracketed]`) is exactly what Rich's
    # parser treats as a real style tag and drops; a Cyrillic-leading one
    # (`[в скобках]`) is not a valid tag name to Rich and survives either
    # way — this test is about the shape that is silently corrupted.
    cell = "[bracketed] название.mp4"
    assert Text.from_markup(cell).plain != cell
    (wrapped,) = plain_row([cell])
    assert wrapped.plain == cell


async def _drive_datatable(cell: str) -> tuple[bool, str]:
    """Add one plain_row-wrapped cell to a live DataTable; report survival."""
    from textual.app import App, ComposeResult

    class _T(App[None]):
        def compose(self) -> ComposeResult:
            yield DataTable(id="dt")

    app = _T()
    async with app.run_test() as pilot:
        table = app.query_one("#dt", DataTable)
        table.add_columns("cell")
        table.add_row(*plain_row([cell]))
        await pilot.pause()
        app.export_screenshot()
        rendered = table.get_cell_at((0, 0))
        return app.is_running, str(rendered)


def test_a_bracketed_datatable_cell_survives_through_a_live_app() -> None:
    cell = "образец/[в скобках] название.mp4"
    is_running, rendered = asyncio.run(_drive_datatable(cell))
    assert is_running
    assert rendered == cell


def test_a_closing_tag_shaped_datatable_cell_survives_through_a_live_app() -> None:
    cell = "клип [/дубль2].mp4"
    is_running, rendered = asyncio.run(_drive_datatable(cell))
    assert is_running
    assert rendered == cell


# -- the exact reproduction from BUGS.md entry 20 ----------------------------


def test_videos_add_five_bare_words_no_longer_crashes_the_app(db: Database) -> None:
    """`videos add a b c d e` in the arguments field used to kill the app."""

    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("videos.add")
        app.run_selected("a b c d e")
        await pilot.pause()
        assert app.is_running
        assert "too many values" in app.status_text
        assert "[title=" in app.status_text
        assert "[kind=" in app.status_text
        assert "[channel=" in app.status_text

    drive(db, body)


# -- show_result: a bracketed result message and a bracketed cell -----------


def test_show_result_with_a_bracketed_message_and_cell(db: Database) -> None:
    bracketed_message = "переименовано [старое] в [новое]: слишком много значений [title=…]"
    bracketed_cell = "sample_inputs/[в скобках] название.mp4"

    def handler(database: Database, **kwargs: Any) -> CommandResult:
        return CommandResult(
            columns=("path",),
            rows=((bracketed_cell,),),
            message=bracketed_message,
        )

    commands = {
        "videos.bracketed": Command(
            name="videos.bracketed",
            group="videos",
            summary="Return a bracketed message and cell.",
            params=(Param("target", str, "Target.", positional=True),),
            handler=handler,
        ),
    }

    async def main() -> None:
        app = RytpApp(db, commands=commands)
        async with app.run_test() as pilot:
            app.select("videos.bracketed")
            app.run_selected("anything")
            await pilot.pause()
            assert app.is_running
            assert app.status_text == bracketed_message
            table = app.query_one("#results", DataTable)
            assert table.row_count == 1
            assert str(table.get_cell_at((0, 0))) == bracketed_cell

    asyncio.run(main())
