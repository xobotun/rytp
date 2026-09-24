"""Reading a transcript in the TUI (design §10).

A shell. Every row it draws comes from `resolve("transcript.show")` — the
same handler `rytp transcript show` calls — so the two surfaces cannot
drift, which is the whole point of generating both from one registry.
The only direct query is the video picker's listing: choosing what to
look at is navigation, not an operation on data, and Part 7's speaker
picker does the same.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from rytp.commands import resolve
from rytp.index.utterances import indexed_videos
from rytp.models import RytpError

if TYPE_CHECKING:
    from rytp.db import Database

__all__ = ["TranscriptScreen", "TranscriptVideosScreen"]


def _fill(table: DataTable, columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Replace a table's contents. Columns are re-added: they can change."""
    table.clear(columns=True)
    table.add_columns(*columns)
    for row in rows:
        table.add_row(*row)


class TranscriptScreen(Screen[None]):
    """One video's blocks, scrollable, with anchors and timestamps."""

    DEFAULT_CSS = """
    #transcript-blocks { height: 1fr; }
    #transcript-status { height: auto; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, db: Database, video_id: int) -> None:
        super().__init__()
        self._db = db
        self._video_id = video_id

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="transcript-blocks")
        yield Static("", id="transcript-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#transcript-blocks", DataTable)
        table.cursor_type = "row"
        status = self.query_one("#transcript-status", Static)
        try:
            result = resolve("transcript.show").handler(self._db, video=str(self._video_id))
        except RytpError as exc:
            # A TUI that raises on a normal mistake — an unindexed video —
            # is worse than one that says what to do about it.
            _fill(table, ("anchor",), [])
            status.update(str(exc))
            return
        _fill(table, result.columns, list(result.rows))
        title = self._title()
        status.update(f"{title} — {result.message}" if title else (result.message or ""))

    def _title(self) -> str:
        row = self._db.conn.execute(
            "SELECT title FROM videos WHERE id = ?", (self._video_id,)
        ).fetchone()
        return "" if row is None else str(row["title"])


class TranscriptVideosScreen(Screen[None]):
    """Which transcript to read, so nobody has to remember an id."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, db: Database) -> None:
        super().__init__()
        self._db = db
        self._video_ids: list[int] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="transcript-videos")
        yield Static("", id="transcript-videos-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#transcript-videos", DataTable)
        table.cursor_type = "row"
        rows = indexed_videos(self._db)
        self._video_ids = [video_id for video_id, _, _ in rows]
        _fill(
            table,
            ("id", "title", "blocks"),
            [(str(video_id), title, str(count)) for video_id, title, count in rows],
        )
        self.query_one("#transcript-videos-status", Static).update(
            f"{len(rows)} indexed video{'' if len(rows) == 1 else 's'}"
            if rows
            else "nothing indexed yet — run `rytp index build`"
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "transcript-videos" or not self._video_ids:
            return
        self.app.push_screen(TranscriptScreen(self._db, self._video_ids[event.cursor_row]))
