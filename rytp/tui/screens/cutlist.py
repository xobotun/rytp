"""Editing a cut list without another window (design §8, §10).

Two screens, like the mapper: one to choose a cut list, one to edit it. Both
are shells — every decision belongs to `CutlistView`, and each action below is
a single call into it.

This is the only screen with no focused `Input`, so its verbs are bare letters.
They are used dozens of times in a row on one cut list and a chord for each
would be a worse tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from rytp import constants as C
from rytp.db import Database
from rytp.tui.cutlist_view import CutlistView, available_cutlists
from rytp.tui.navigation import BACK_KEY

__all__ = ["CutlistPickerScreen", "CutlistScreen"]


def _fill(
    table: DataTable, columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...]
) -> None:
    table.clear(columns=True)
    if columns:
        table.add_columns(*columns)
        for row in rows:
            table.add_row(*row)


class CutlistPickerScreen(Screen[None]):
    """Which cut list. Nobody should have to remember a name."""

    DEFAULT_CSS = """
    #cutlist-picker { height: 1fr; }
    #cutlist-picker-status { height: auto; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(BACK_KEY, "app.pop_screen", "Back"),
    ]

    def __init__(self, db: Database) -> None:
        super().__init__()
        self._db = db
        self._paths: list[Path] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="cutlist-picker")
        yield Static("", id="cutlist-picker-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#cutlist-picker", DataTable)
        table.cursor_type = "row"
        found = available_cutlists()
        self._paths = [item.path for item in found]
        _fill(
            table,
            ("name", "target", "slots", "gaps", "sources", "problem"),
            tuple(
                (
                    item.name,
                    item.target,
                    str(item.slots),
                    str(item.gaps),
                    str(item.sources),
                    item.error or "",
                )
                for item in found
            ),
        )
        self.query_one("#cutlist-picker-status", Static).update(
            f"{len(found)} cut list{'' if len(found) == 1 else 's'}"
            if found
            else "no cut lists yet — plan one with `rytp assemble plan \"…\"`"
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "cutlist-picker" and self._paths:
            self.app.push_screen(CutlistScreen(self._db, self._paths[event.cursor_row]))


class CutlistScreen(Screen[None]):
    """The timeline, the highlighted slot's options, and five verbs."""

    DEFAULT_CSS = """
    #cutlist-slots   { height: 2fr; }
    #cutlist-options { height: 1fr; }
    #cutlist-status  { height: auto; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(BACK_KEY, "app.pop_screen", "Back"),
        Binding("s", "swap", "Swap in option"),
        Binding("[", "nudge_start_back", "Start earlier"),
        Binding("]", "nudge_start_on", "Start later"),
        Binding("ctrl+left", "nudge_end_back", "End earlier"),
        Binding("ctrl+right", "nudge_end_on", "End later"),
        Binding("g", "set_gap", "Pause before"),
        Binding("c", "clear_gap", "Measured pause"),
        Binding("u", "undo", "Undo"),
        Binding("r", "reload", "Discard edits"),
        Binding("ctrl+s", "save", "Save"),
    ]

    def __init__(self, db: Database, path: Path) -> None:
        super().__init__()
        self._db = db
        self.view = CutlistView(path)

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="cutlist-slots")
        yield DataTable(id="cutlist-options")
        yield Static("", id="cutlist-status")
        yield Footer()

    def on_mount(self) -> None:
        for table_id in ("#cutlist-slots", "#cutlist-options"):
            self.query_one(table_id, DataTable).cursor_type = "row"
        self.action_redraw()

    # -- actions -----------------------------------------------------

    def action_swap(self) -> None:
        self._after(self.view.swap(self._slot(), self._option()))

    def action_nudge_start_back(self) -> None:
        self._after(
            self.view.nudge(self._slot(), edge="start", delta_ms=-C.TUI_CUTLIST_NUDGE_MS)
        )

    def action_nudge_start_on(self) -> None:
        self._after(
            self.view.nudge(self._slot(), edge="start", delta_ms=C.TUI_CUTLIST_NUDGE_MS)
        )

    def action_nudge_end_back(self) -> None:
        self._after(
            self.view.nudge(self._slot(), edge="end", delta_ms=-C.TUI_CUTLIST_NUDGE_MS)
        )

    def action_nudge_end_on(self) -> None:
        self._after(
            self.view.nudge(self._slot(), edge="end", delta_ms=C.TUI_CUTLIST_NUDGE_MS)
        )

    def action_set_gap(self) -> None:
        self._after(self.view.set_gap_before(self._slot(), C.TUI_CUTLIST_COARSE_NUDGE_MS))

    def action_clear_gap(self) -> None:
        self._after(self.view.clear_gap(self._slot()))

    def action_undo(self) -> None:
        self._after(self.view.undo())

    def action_reload(self) -> None:
        self._after(self.view.reload())

    def action_save(self) -> None:
        self.view.save()
        self._after(self.view.status)

    # -- drawing -----------------------------------------------------

    def action_redraw(self) -> None:
        """Redraw everything. Called after a mutation, never from a handler
        that a redraw itself can trigger."""
        index = self._slot()
        table = self.query_one("#cutlist-slots", DataTable)
        _fill(table, self.view.columns, self.view.rows)
        if self.view.rows:
            table.move_cursor(row=min(index, len(self.view.rows) - 1))
        self._draw_options()
        self.query_one("#cutlist-status", Static).update(self.view.status)

    def _draw_options(self) -> None:
        index = self._slot()
        _fill(
            self.query_one("#cutlist-options", DataTable),
            self.view.option_columns(index),
            self.view.option_rows(index),
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        # Only the options pane, and only from the slot table: redrawing the
        # slot table here would move its cursor, raise this message again, and
        # loop.
        if event.data_table.id == "cutlist-slots":
            self._draw_options()

    def _slot(self) -> int:
        return int(self.query_one("#cutlist-slots", DataTable).cursor_row or 0)

    def _option(self) -> int:
        return int(self.query_one("#cutlist-options", DataTable).cursor_row or 0)

    def _after(self, message: str) -> None:
        self.action_redraw()
        self.query_one("#cutlist-status", Static).update(message)

    # `_draw_slots` refills the slot table, which moves its cursor, which
    # raises RowHighlighted. If that handler refilled the slot table again the
    # screen would loop forever, so highlighting only redraws the options.
