"""Speakers screen — pick a video, then map its raw diarizer labels."""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header

from rytp import constants as C
from rytp.speakers import SpeakerMapper

if TYPE_CHECKING:
    from rytp.db import Database


class SpeakersScreen(Screen):
    """Two-pane mapper (DESIGN §6).

    On mount, lists the videos that have any diarizer labels. Picking
    one swaps to a per-video mapper backed by :class:`SpeakerMapper`.

    Title truncation is :data:`C.SPEAKERS_TITLE_TRUNCATE_CHARS` —
    shorter than the videos screen because the speakers pane is
    narrower.
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, db: "Database") -> None:
        super().__init__()
        self._db = db

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield DataTable(id="raw-labels")
            yield DataTable(id="speakers")
        yield Footer()

    def on_mount(self) -> None:
        raw = self.query_one("#raw-labels", DataTable)
        speakers = self.query_one("#speakers", DataTable)
        raw.cursor_type = "row"
        speakers.cursor_type = "row"
        raw.add_columns("video_id", "title", "n_raw_labels")
        speakers.add_columns("id", "label", "aliases")
        # Populate from the most recent video that has any diarizer
        # labels; if there's no such video, leave both panes empty.
        row = self._db.conn.execute(
            """
            SELECT v.id, v.title, COUNT(DISTINCT w.diarizer_speaker) AS n
            FROM videos v JOIN words w ON w.video_id = v.id
            WHERE w.diarizer_speaker IS NOT NULL
            GROUP BY v.id
            ORDER BY v.id DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return
        raw.add_row(
            str(row["id"]),
            (row["title"] or "")[: C.SPEAKERS_TITLE_TRUNCATE_CHARS],
            str(row["n"]),
        )
        mapper = SpeakerMapper(self._db, row["id"])
        for s in mapper.right_pane():
            speakers.add_row(str(s.id), s.label, ", ".join(s.aliases))