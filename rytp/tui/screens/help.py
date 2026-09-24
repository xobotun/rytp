"""Every binding in the application, generated from the navigation map.

Nothing here is typed by hand, which is the point: a screen that adds a key
documents itself, and a key that is removed disappears from help in the same
commit.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from rytp.tui.navigation import BACK_KEY, help_rows

__all__ = ["HelpScreen"]


class HelpScreen(Screen[None]):
    """A table of (key, where, what)."""

    DEFAULT_CSS = """
    #help-bindings { height: 1fr; }
    #help-note { height: auto; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(BACK_KEY, "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="help-bindings")
        yield Static(
            "Every screen returns here with Escape. Long-running commands are "
            "queued, never run in the TUI — press F7 to watch the queue.",
            id="help-note",
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#help-bindings", DataTable)
        table.cursor_type = "row"
        table.add_columns("key", "where", "what")
        for key, where, what in help_rows():
            table.add_row(key, where, what)
