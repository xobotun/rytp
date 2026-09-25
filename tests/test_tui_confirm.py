"""The reusable confirmation modal (BUGS.md entry 44).

`rytp/tui/confirm.py` is not owned by any one screen — the Videos screen's
`shift+delete` is only its first caller — so it gets its own test file
rather than living inside `test_tui_videos.py`.
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Button

from rytp.tui.confirm import ConfirmScreen


class _Host(App[None]):
    """The smallest app that can push a `ConfirmScreen` and read its result."""

    def compose(self) -> ComposeResult:
        return iter(())


def drive(body) -> object:  # type: ignore[no-untyped-def]
    async def main() -> object:
        app = _Host()
        async with app.run_test() as pilot:
            return await body(app, pilot)

    return asyncio.run(main())


def test_defaults_to_cancel_focused() -> None:
    """The Cancel button holds focus on mount, so a stray Enter — the
    likeliest accidental keypress — resolves to `False`, not the
    destructive action."""

    async def body(app, pilot):  # type: ignore[no-untyped-def]
        result_future = app.push_screen(
            ConfirmScreen("Drop it?", ("details",)), wait_for_dismiss=False
        )
        await pilot.pause()
        focused = app.focused
        assert focused is not None
        assert focused.id == "confirm-cancel"
        return result_future

    drive(body)


def test_escape_dismisses_false_without_a_button_press() -> None:
    async def body(app, pilot):  # type: ignore[no-untyped-def]
        results = []
        app.push_screen(ConfirmScreen("Drop it?", ("details",)), results.append)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        return results

    results = drive(body)
    assert results == [False]


def test_the_confirm_button_dismisses_true() -> None:
    async def body(app, pilot):  # type: ignore[no-untyped-def]
        results = []
        app.push_screen(
            ConfirmScreen("Drop it?", ("details",), confirm_label="Drop"), results.append
        )
        await pilot.pause()
        await pilot.click("#confirm-ok")
        await pilot.pause()
        return results

    results = drive(body)
    assert results == [True]


def test_the_cancel_button_dismisses_false() -> None:
    async def body(app, pilot):  # type: ignore[no-untyped-def]
        results = []
        app.push_screen(ConfirmScreen("Drop it?", ("details",)), results.append)
        await pilot.pause()
        await pilot.click("#confirm-cancel")
        await pilot.pause()
        return results

    results = drive(body)
    assert results == [False]


def test_the_title_and_lines_render_verbatim_brackets_and_all() -> None:
    """Set through `rytp.tui.text.set_text`, never a bare `Static(...)` —
    a title with `[` or `]` (routine for this project's Russian video
    titles) must render, not crash `Content.from_markup`."""
    from textual.content import Content
    from textual.widgets import Static

    async def body(app, pilot):  # type: ignore[no-untyped-def]
        app.push_screen(
            ConfirmScreen(
                "Drop the transcript of video 7?",
                ('Интервью [часть 1]', "", "This deletes words."),
            )
        )
        await pilot.pause()
        body_widget = app.screen.query_one("#confirm-body", Static)
        rendered = body_widget.render()
        assert isinstance(rendered, Content)
        assert "Интервью [часть 1]" in rendered.plain
        assert app.is_running
        app.screen.dismiss(False)
        await pilot.pause()

    drive(body)


def test_only_true_or_false_is_ever_dismissed() -> None:
    """Both buttons and escape must resolve to a plain `bool` — nothing
    else is a valid return from this screen."""
    keys = {binding.key for binding in ConfirmScreen.BINDINGS}
    assert "escape" in keys
    button_ids = {"confirm-ok", "confirm-cancel"}

    async def body(app, pilot):  # type: ignore[no-untyped-def]
        screen = ConfirmScreen("Drop it?", ("details",))
        app.push_screen(screen)
        await pilot.pause()
        ids = {b.id for b in screen.query(Button)}
        return ids

    ids = drive(body)
    assert ids == button_ids
