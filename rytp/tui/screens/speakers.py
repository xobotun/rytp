"""The speaker mapper screen (design §10).

Design §10 calls speaker assignment "the one genuinely interactive task",
and this is the screen for it. It is deliberately thin: every decision
lives in :class:`~rytp.diarize.mapper.MappingSession`, which is tested
without a terminal, and this file only draws that object and forwards
keystrokes into it. If you find yourself adding an `if` here, it probably
belongs in the session.

Two screens. :class:`SpeakerVideosScreen` lists the videos that have
diarizer labels, because nobody remembers a video id; picking one opens
:class:`SpeakerMapperScreen`, which is the two panes.

**Key choices.** The filter `Input` holds focus, so every binding avoids a
bare letter — typing must reach the filter. The motion the screen exists
for is: type part of a name, press enter to link, and the cursor moves to
the next unnamed voice. Typing a name nobody has and pressing ctrl+n
creates that person and links them in the same gesture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from rytp import constants as C
from rytp.diarize import store
from rytp.diarize.mapper import MappingSession
from rytp.tui.text import plain_row, set_text

if TYPE_CHECKING:
    from rytp.db import Database

__all__ = ["SpeakerMapperScreen", "SpeakerVideosScreen", "run_mapper"]


def _fill(table: DataTable, columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Replace a table's contents. Columns are re-added: they can change."""
    table.clear(columns=True)
    table.add_columns(*columns)
    for row in rows:
        table.add_row(*plain_row(row))


class SpeakerMapperScreen(Screen[None]):
    """Local labels on the left, the global roster on the right."""

    DEFAULT_CSS = """
    #panes { height: 1fr; }
    #labels { width: 1fr; }
    #roster { width: 1fr; }
    #suggestions { height: auto; max-height: 10; }
    #mapper-status { height: auto; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("enter", "assign", "Name this voice"),
        Binding("ctrl+n", "new_speaker", "New person from filter"),
        Binding("ctrl+u", "unassign", "Unname"),
        Binding("f5", "toggle_suggestions", "Suggestions"),
        Binding("ctrl+down", "next_label", "Next voice"),
        Binding("ctrl+up", "previous_label", "Previous voice"),
    ]

    def __init__(self, db: Database, video_id: int) -> None:
        super().__init__()
        self.session = MappingSession(db, video_id)
        self._syncing = False

    # -- layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="filter the roster: name or alias", id="roster-filter")
        with Horizontal(id="panes"):
            yield DataTable(id="labels")
            yield DataTable(id="roster")
        yield DataTable(id="suggestions")
        yield Static("", id="mapper-status")
        yield Footer()

    def on_mount(self) -> None:
        for table_id in ("#labels", "#roster", "#suggestions"):
            self.query_one(table_id, DataTable).cursor_type = "row"
        self.query_one("#suggestions", DataTable).display = False
        self.refresh_panes()
        self.query_one("#roster-filter", Input).focus()

    # -- drawing ---------------------------------------------------------

    def refresh_panes(self) -> None:
        """Redraw both tables from the session, cursors included."""
        self._syncing = True
        try:
            labels = self.query_one("#labels", DataTable)
            _fill(labels, *self.session.label_table())
            if self.session.labels:
                labels.move_cursor(row=self.session.label_index)

            roster = self.query_one("#roster", DataTable)
            _fill(roster, *self.session.roster_table())
            if self.session.matches:
                roster.move_cursor(row=self.session.roster_index)
        finally:
            self._syncing = False
        self._show_status()

    def _show_status(self, message: str | None = None) -> None:
        text = message if message is not None else self.session.status
        remaining = self.session.n_unmapped()
        suffix = f"{remaining} voice{'' if remaining == 1 else 's'} still unnamed"
        set_text(
            self.query_one("#mapper-status", Static),
            f"{text} — {suffix}" if text else suffix,
        )

    # -- events ----------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "roster-filter":
            self.session.set_query(event.value)
            self.refresh_panes()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "roster-filter":
            self.action_assign()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Let the mouse and arrow keys move the session's cursors too."""
        if self._syncing:
            return
        if event.data_table.id == "labels":
            self.session.select_label(event.cursor_row)
            if self.query_one("#suggestions", DataTable).display:
                self._fill_suggestions()
        elif event.data_table.id == "roster":
            self.session.select_roster(event.cursor_row)

    # -- actions ---------------------------------------------------------

    def action_assign(self) -> None:
        message = self.session.assign()
        self.refresh_panes()
        self._show_status(message)

    def action_unassign(self) -> None:
        message = self.session.unassign()
        self.refresh_panes()
        self._show_status(message)

    def action_new_speaker(self) -> None:
        """Create the person named in the filter box and link them."""
        text = self.query_one("#roster-filter", Input).value.strip()
        if not text:
            self._show_status("type a name in the filter box first")
            return
        message = self.session.create_and_assign(text)
        self.query_one("#roster-filter", Input).value = ""
        self.session.set_query("")
        self.refresh_panes()
        self._show_status(message)

    def action_next_label(self) -> None:
        self.session.move_label(1)
        self.refresh_panes()

    def action_previous_label(self) -> None:
        self.session.move_label(-1)
        self.refresh_panes()

    def action_toggle_suggestions(self) -> None:
        """Show or hide the advisory pane. Showing it changes nothing."""
        table = self.query_one("#suggestions", DataTable)
        table.display = not table.display
        if table.display:
            self._fill_suggestions()

    def _fill_suggestions(self) -> None:
        self._syncing = True
        try:
            _fill(
                self.query_one("#suggestions", DataTable),
                *self.session.suggestion_table(),
            )
        finally:
            self._syncing = False


class SpeakerVideosScreen(Screen[None]):
    """The videos that have labels, so nobody has to remember an id."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, db: Database) -> None:
        super().__init__()
        self._db = db
        self._video_ids: list[int] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="videos")
        yield Static("", id="videos-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#videos", DataTable)
        table.cursor_type = "row"
        rows = store.diarized_videos(self._db)
        self._video_ids = [row.video_id for row in rows]
        _fill(
            table,
            ("id", "title", "published", "voices", "unnamed"),
            [
                (
                    str(row.video_id),
                    (row.title or "")[: C.SPEAKER_TITLE_TRUNCATE_CHARS],
                    (row.published_at or "—")[:10],
                    str(row.n_labels),
                    str(row.n_unmapped),
                )
                for row in rows
            ],
        )
        set_text(
            self.query_one("#videos-status", Static),
            f"{len(rows)} diarized video{'' if len(rows) == 1 else 's'}"
            if rows
            else "nothing diarized yet — `rytp speakers diarize <video>`",
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "videos" or not self._video_ids:
            return
        self.app.push_screen(
            SpeakerMapperScreen(self._db, self._video_ids[event.cursor_row])
        )


class _MapperApp(App[None]):
    """The one-screen app `rytp speakers map <video>` launches."""

    TITLE = "rytp — speakers"

    def __init__(self, db: Database, video_id: int) -> None:
        super().__init__()
        self._db = db
        self._video_id = video_id

    def on_mount(self) -> None:
        self.push_screen(SpeakerMapperScreen(self._db, self._video_id))


def run_mapper(db: Database, video_id: int) -> None:
    """Open the mapper for one video. Called only by `rytp speakers map`."""
    _MapperApp(db, video_id).run()
