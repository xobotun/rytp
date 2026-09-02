"""Videos screen — browse the ``videos`` table."""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable

from rytp import constants as C

if TYPE_CHECKING:
    from rytp.db import Database


class VideosScreen(Screen):
    """A read-only table over the ``videos`` table.

    The page size is :data:`C.VIDEOS_PAGE_SIZE`; titles are truncated
    to :data:`C.VIDEOS_TITLE_TRUNCATE_CHARS` characters so the table
    stays readable in a terminal window.
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, db: "Database") -> None:
        super().__init__()
        self._db = db

    def compose(self):
        yield DataTable(id="videos-table")

    def on_mount(self) -> None:
        table = self.query_one("#videos-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("id", "source", "kind", "title", "downloaded")
        for row in self._db.conn.execute(
            "SELECT id, source, kind, title, downloaded FROM videos "
            "ORDER BY id DESC LIMIT ?",
            (C.VIDEOS_PAGE_SIZE,),
        ):
            table.add_row(
                str(row["id"]),
                row["source"],
                row["kind"],
                (row["title"] or "")[: C.VIDEOS_TITLE_TRUNCATE_CHARS],
                "yes" if row["downloaded"] else "no",
            )