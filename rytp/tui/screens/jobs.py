"""The queue, watched (design §10).

A shell. Every decision — which rows, which filters, what the status line says,
what retrying means — is `JobsView`'s, and every action below is one line.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from rytp import constants as C
from rytp.db import Database
from rytp.tui.jobs_view import JobsView
from rytp.tui.navigation import BACK_KEY
from rytp.tui.text import plain_row, set_text

__all__ = ["JobsScreen"]


class JobsScreen(Screen[None]):
    """What is queued, running, done and failed."""

    DEFAULT_CSS = """
    #jobs-table  { height: 1fr; }
    #jobs-note   { height: auto; padding: 0 1; }
    #jobs-status { height: auto; padding: 0 1; }
    """

    # No bare letters: consistent with every other screen, and the footer
    # stays readable next to the mapper's.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(BACK_KEY, "app.pop_screen", "Back"),
        Binding("ctrl+r", "refresh", "Refresh"),
        # Not ctrl+s: the cut-list editor saves with it, and one key that
        # means "save" on one screen and "filter" on another is the `-s`
        # finding happening again inside Part 8.
        Binding("ctrl+e", "cycle_state", "State filter"),
        Binding("ctrl+o", "cycle_pool", "Pool filter"),
        Binding("ctrl+y", "retry", "Retry job"),
        Binding("ctrl+g", "retry_all", "Retry all failed"),
        Binding("ctrl+x", "cancel", "Cancel job"),
        Binding("ctrl+b", "toggle_pause", "Pause/resume"),
        # BUGS.md entry 41: enqueueing only writes a `jobs` row; nothing
        # drains it without a separate `rytp worker` process, and this is
        # the screen someone watches while wondering why nothing moves.
        # Every ctrl+<letter> chord is already claimed somewhere in the app
        # (`test_one_key_means_one_thing_wherever_it_is_bound` requires one
        # meaning per key everywhere, not just on this screen) and f1-f9 are
        # all app-level screen keys, so this is the next function key up.
        Binding("f12", "start_worker", "Start worker"),
    ]

    def __init__(self, db: Database) -> None:
        super().__init__()
        self.view = JobsView(db)
        self._timer: object | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="jobs-table")
        yield Static("", id="jobs-note")
        yield Static("", id="jobs-status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#jobs-table", DataTable).cursor_type = "row"
        self._draw()
        # The worker is a separate process (design §5), so progress only
        # arrives by looking again.
        self._timer = self.set_interval(C.TUI_QUEUE_REFRESH_S, self.action_refresh)

    # -- actions -----------------------------------------------------

    def action_refresh(self) -> None:
        self.view.refresh()
        self._draw()

    def action_cycle_state(self) -> None:
        self.view.cycle_state()
        self._draw()

    def action_cycle_pool(self) -> None:
        self.view.cycle_pool()
        self._draw()

    def action_retry(self) -> None:
        self._announce(self.view.retry_at(self._cursor()))

    def action_retry_all(self) -> None:
        self._announce(self.view.retry_all_failed())

    def action_cancel(self) -> None:
        self._announce(self.view.cancel_at(self._cursor()))

    def action_toggle_pause(self) -> None:
        self._announce(self.view.toggle_pause())

    def action_start_worker(self) -> None:
        self._announce(self.view.spawn_worker())

    # -- drawing -----------------------------------------------------

    def _cursor(self) -> int:
        return int(self.query_one("#jobs-table", DataTable).cursor_row or 0)

    def _announce(self, message: str) -> None:
        self._draw()
        set_text(self.query_one("#jobs-status", Static), message)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        # Only the note pane: redrawing the table here would move its cursor
        # and raise this message again. Without it the note would change only
        # on the two-second timer, which is a long time to stare at a row.
        if event.data_table.id == "jobs-table":
            set_text(
                self.query_one("#jobs-note", Static),
                self.view.note_at(int(event.cursor_row)),
            )

    def _draw(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        cursor = table.cursor_row or 0
        table.clear(columns=True)
        if self.view.columns:
            table.add_columns(*self.view.columns)
            for row in self.view.rows:
                table.add_row(*plain_row(row))
        if self.view.rows:
            table.move_cursor(row=min(cursor, len(self.view.rows) - 1))
        set_text(self.query_one("#jobs-note", Static), self.view.note_at(cursor))
        set_text(self.query_one("#jobs-status", Static), self.view.status)
