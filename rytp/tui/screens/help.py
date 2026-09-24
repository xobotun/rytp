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
from rytp.tui.text import plain_row, set_text

__all__ = ["HelpScreen"]

#: BUGS.md entry 19: the tier vocabulary (`caption` / `timed` / `aligned`) is
#: load-bearing and was explained nowhere in the app. F1 is the obvious home
#: because it already shows every binding. `--allow-timed` (Task 6) is named
#: here because a user cannot choose to use it without understanding the
#: difference it lets them override.
GLOSSARY = (
    "Transcript tiers (contracts §3):\n"
    "caption — downloaded subtitles. Searchable, no end times, never "
    "cuttable.\n"
    "timed — the transcriber's own word timestamps. Good text, but the "
    "boundaries are not trustworthy: measured on real data, 78.7% of "
    "Whisper's word gaps are exactly zero.\n"
    "aligned — forced alignment plus a snap to a measured energy minimum "
    "and zero crossing. The only tier that can be cut.\n"
    "'Cuttable only' (search screen, Ctrl+T) keeps aligned words and drops "
    "the rest. `assemble plan --allow-timed` is the explicit override that "
    "admits timed words anyway."
)


class HelpScreen(Screen[None]):
    """A table of (key, where, what)."""

    DEFAULT_CSS = """
    #help-bindings { height: 1fr; }
    #help-note { height: auto; padding: 0 1; }
    #help-glossary { height: auto; padding: 0 1; }
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
        yield Static("", id="help-glossary")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#help-bindings", DataTable)
        table.cursor_type = "row"
        table.add_columns("key", "where", "what")
        for key, where, what in help_rows():
            table.add_row(*plain_row((key, where, what)))
        set_text(self.query_one("#help-glossary", Static), GLOSSARY)
