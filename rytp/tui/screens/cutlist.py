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
from rytp.tui.text import plain_row, set_text

__all__ = ["CutlistPickerScreen", "CutlistScreen"]


def _fill(
    table: DataTable, columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...]
) -> None:
    table.clear(columns=True)
    if columns:
        table.add_columns(*columns)
        for row in rows:
            table.add_row(*plain_row(row))


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
        set_text(
            self.query_one("#cutlist-picker-status", Static),
            f"{len(found)} cut list{'' if len(found) == 1 else 's'}"
            if found
            else "no cut lists yet — plan one with `rytp assemble plan \"…\"`",
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "cutlist-picker" and self._paths:
            self.app.push_screen(CutlistScreen(self._db, self._paths[event.cursor_row]))


class CutlistScreen(Screen[None]):
    """The timeline, the highlighted slot's options, and its verbs.

    BUGS.md entry 29: this is the screen with the most bindings by some
    margin, and its footer used to run off the end of an 80-column terminal
    — recoverable only because PowerShell keeps scrollback. Most keys below
    are bound with ``show=False``: they still work, and they still appear in
    full on the help screen (``F1``, or ``?`` from here — `navigation.
    ScreenEntry.children` is what makes this screen's bindings reach that
    table), but the footer itself shows only the handful used on every pass
    over a cut list — swap, snap, undo, save — plus how to see the rest.

    This screen's own contribution to the footer fits comfortably inside an
    80-column terminal in isolation (`tests/test_tui_cutlist.py`). The
    *running app's* footer also always carries the F1-F8 navigation strip and
    the Ctrl+P command-palette key (`rytp/tui/app.py`, `rytp/tui/
    navigation.py`), and that baseline is already wider than 80 columns on
    every screen, this one included — a second, separate overflow that is not
    this task's to fix. Entry 29 was this screen's twelve keys; the app-level
    strip is Task 17's ("the footer lesson" the QA batch plan hands it).
    """

    DEFAULT_CSS = """
    #cutlist-slots   { height: 2fr; }
    #cutlist-options { height: 1fr; }
    #cutlist-status  { height: auto; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(BACK_KEY, "app.pop_screen", "Back"),
        Binding("s", "swap", "Swap"),
        # Word-edge nudging (BUGS.md entry 30): plain is
        # `TUI_CUTLIST_NUDGE_MS` (10 ms), Shift is `TUI_CUTLIST_SHIFT_NUDGE_MS`
        # (100 ms) — coarser, not finer: a wrong fragment is usually wrong by
        # a syllable, not by a millisecond. Bracket keys move the start,
        # Ctrl+arrow the end, matching the split this screen already had.
        Binding(
            "[", f"nudge('start', {-C.TUI_CUTLIST_NUDGE_MS})", "Start earlier",
            show=False,
        ),
        Binding(
            "]", f"nudge('start', {C.TUI_CUTLIST_NUDGE_MS})", "Start later",
            show=False,
        ),
        Binding(
            "shift+[", f"nudge('start', {-C.TUI_CUTLIST_SHIFT_NUDGE_MS})",
            "Start earlier (coarse)", show=False,
        ),
        Binding(
            "shift+]", f"nudge('start', {C.TUI_CUTLIST_SHIFT_NUDGE_MS})",
            "Start later (coarse)", show=False,
        ),
        Binding(
            "ctrl+left", f"nudge('end', {-C.TUI_CUTLIST_NUDGE_MS})", "End earlier",
            show=False,
        ),
        Binding(
            "ctrl+right", f"nudge('end', {C.TUI_CUTLIST_NUDGE_MS})", "End later",
            show=False,
        ),
        Binding(
            "ctrl+shift+left", f"nudge('end', {-C.TUI_CUTLIST_SHIFT_NUDGE_MS})",
            "End earlier (coarse)", show=False,
        ),
        Binding(
            "ctrl+shift+right", f"nudge('end', {C.TUI_CUTLIST_SHIFT_NUDGE_MS})",
            "End later (coarse)", show=False,
        ),
        # The snap-to-clean-boundary key (BUGS.md entry 30): one press does
        # what forty 1 ms nudges do, and does it better, because a clicking
        # seam comes from an amplitude discontinuity rather than from timing
        # precision. `z` pairs with the bracket keys (start), `x` with
        # Ctrl+arrow (end).
        Binding("z", "snap('start')", "Snap"),
        Binding("x", "snap('end')", "Snap end", show=False),
        Binding("g", "set_gap", "Pause before", show=False),
        Binding("c", "clear_gap", "Measured pause", show=False),
        Binding("u", "undo", "Undo"),
        Binding("r", "reload", "Discard edits", show=False),
        Binding("ctrl+s", "save", "Save"),
        Binding("?", "app.help", "Help"),
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

    def action_nudge(self, edge: str, delta_ms: int) -> None:
        self._after(self.view.nudge(self._slot(), edge=edge, delta_ms=delta_ms))

    def action_snap(self, edge: str) -> None:
        self._after(self.view.snap(self._slot(), edge=edge))

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
        set_text(self.query_one("#cutlist-status", Static), self.view.status)

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
        set_text(self.query_one("#cutlist-status", Static), message)

    # `_draw_slots` refills the slot table, which moves its cursor, which
    # raises RowHighlighted. If that handler refilled the slot table again the
    # screen would loop forever, so highlighting only redraws the options.
