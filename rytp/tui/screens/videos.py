"""The fleet overview: where each video is, registered to cuttable
(BUGS.md entry 23, plan Task 16).

Every other screen answers a question about one thing — a transcript, a
search, a cut list. Nothing showed where each video *is* on the pipeline, and
the owner asked for exactly this after losing track of the command sequence
by hand late in a working session.

This screen adds no second opinion about what "done" means. The table is
`videos list --long` (Task 15, `rytp.commands.catalog.videos_list`), which
already reads the registered `readiness` predicates rather than growing a
new rule, run again here with a filter and redrawn; the five shortcuts call
the exact commands the palette would call — `ingest`, `transcribe.run`,
`transcribe.align`, `speakers.enqueue`, `index.build` — never a parallel
path. Every shortcut enqueues rather than running (BUGS.md entry 22: a
blocking call on Textual's event loop freezes the interface), which is also
what the palette already does for every one of these commands, since all
five are `long_running=True` or, for `speakers.enqueue`, have no foreground
form at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from rytp import constants as C
from rytp.commands import CommandResult, resolve
from rytp.models import RytpError
from rytp.tui.confirm import ConfirmScreen
from rytp.tui.navigation import BACK_KEY
from rytp.tui.text import plain_row, set_text

if TYPE_CHECKING:
    from rytp.db import Database

__all__ = ["VideoFleetSession", "VideosScreen"]

#: BUGS.md entry 23's third decision. `align` and `diarize` are registered
#: `reopenable=False` (`rytp/jobs/__init__.py`): once satisfied, that job
#: kind's own `readiness` predicate never offers it again and a full
#: `reconcile` will not re-fire it — only a fresh `--enqueue` (from here, or
#: the CLI) reopens it. `videos list --long`'s tick means the same "done" on
#: every column (Task 15), so rather than invent a second meaning of "done"
#: for these two, this screen redraws exactly those ticks with a second
#: glyph: pressing the shortcut again on an already-ticked row is expected to
#: look like nothing happened, and a plain tick next to a working shortcut
#: would read as broken instead. F1 has the full explanation.
NOT_REOPENABLE_MARK: str = f"{C.CELL_TICK}*"
_FROZEN_COLUMNS: tuple[str, ...] = ("aligned", "diarized")

#: BUGS.md entry 19's tier vocabulary is what "tier" answers; the search
#: screen's `CUTTABLE_HINT` is the sibling of this line. Kept short on
#: purpose and pointed at F1 rather than restating the glossary here.
FLEET_HINT = (
    "tier: caption/timed/aligned, only aligned is cuttable — see F1 · "
    f"{NOT_REOPENABLE_MARK} = done but frozen, will not re-run under "
    "reconcile — the shortcut reopens it"
)


@dataclass
class VideoFleetSession:
    """The listing shown, its filter, and the five per-row shortcuts.

    No SQL of its own: every read runs `videos.list --long`
    (`rytp.commands.catalog.videos_list`) through the registry, and every
    write is a registered command's handler called exactly as the CLI or the
    palette would call it — `_dispatch` is the one place that happens.
    """

    db: Database
    search: str = ""
    result: CommandResult = field(default_factory=CommandResult)
    error: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        self.refresh()

    # -- reading -----------------------------------------------------

    def refresh(self) -> None:
        """Re-run the listing with the current filter.

        Never raises: a bad filter is shown, not fatal — the same rule the
        search screen's `SearchSession.run` follows.
        """
        try:
            self.result = resolve("videos.list").handler(
                self.db,
                search=self.search or None,
                long=True,
                limit=C.DEFAULT_LIST_LIMIT,
            )
            self.error = None
        except RytpError as exc:
            self.result = CommandResult()
            self.error = str(exc)

    def run(self, search: str) -> None:
        """Change the filter and reload. Called by Enter and by the tests."""
        self.search = search
        self.refresh()

    @property
    def columns(self) -> tuple[str, ...]:
        return self.result.columns

    @property
    def rows(self) -> tuple[tuple[str, ...], ...]:
        """`videos list --long`'s rows, with the frozen columns re-marked.

        Same tick, same source of truth (`_FROZEN_COLUMNS`) — only the
        glyph shown for "done" changes, and only for `aligned` and
        `diarized`. See `NOT_REOPENABLE_MARK`.
        """
        if not self.columns:
            return ()
        frozen_idx = [
            self.columns.index(name) for name in _FROZEN_COLUMNS if name in self.columns
        ]
        if not frozen_idx:
            return self.result.rows
        marked: list[tuple[str, ...]] = []
        for row in self.result.rows:
            cells = list(row)
            for i in frozen_idx:
                if cells[i] == C.CELL_TICK:
                    cells[i] = NOT_REOPENABLE_MARK
            marked.append(tuple(cells))
        return tuple(marked)

    @property
    def status(self) -> str:
        """One line: the error if there is one, else the last note (if any)
        followed by the count and the fixed hint — the search screen's
        `CUTTABLE_HINT` pattern, one filter clause at a time."""
        if self.error:
            return self.error
        parts = [part for part in (self.note, self.result.message) if part]
        parts.append(FLEET_HINT)
        return " · ".join(parts)

    def video_id_at(self, row: int) -> int | None:
        if not (0 <= row < len(self.rows)) or "id" not in self.columns:
            return None
        try:
            return int(self.rows[row][self.columns.index("id")])
        except ValueError:
            return None

    def video_summary_at(self, row: int) -> tuple[int, str] | None:
        """`(video id, title)` for a confirmation prompt, or `None` when
        nothing is selected — the pairing `ConfirmScreen` needs to name the
        exact row about to be destroyed, never "are you sure?" alone."""
        video_id = self.video_id_at(row)
        if video_id is None or "title" not in self.columns:
            return None
        return video_id, self.rows[row][self.columns.index("title")]

    # -- shortcuts: enqueue, never run (BUGS.md entries 22 and 23) ----

    def _dispatch(self, row: int, command: str, **kwargs: object) -> None:
        video_id = self.video_id_at(row)
        if video_id is None:
            self.note = "no video selected"
            return
        self.error = None
        try:
            result = resolve(command).handler(self.db, video=str(video_id), **kwargs)
            self.note = result.message or f"{command} queued for video {video_id}"
        except RytpError as exc:
            self.note = str(exc)
        self.refresh()

    def ingest(self, row: int) -> None:
        """`ingest`: download, captions, extract-wav, caption-words — the
        whole acquisition chain queued as one call. Always queued; there is
        no foreground form of `ingest` at all."""
        self._dispatch(row, "ingest")

    def transcribe(self, row: int) -> None:
        """`transcribe.run --enqueue`, with the engine `default_transcriber`
        names — the same call `rytp transcribe run <video> --enqueue` makes."""
        self._dispatch(row, "transcribe.run", enqueue=True)

    def align(self, row: int) -> None:
        """`transcribe.align --enqueue`.

        Unlike `transcribe.run`, that command has no fallback for a missing
        `--aligner` (contracts: `REQUIRED`, no default) — it is meant to be
        typed. Reading `default_aligner` here and reporting when it is unset
        keeps this shortcut from enqueueing a job that could only fail later,
        on the worker, far from whoever pressed the key.
        """
        from rytp.transcribe.registry import setting

        aligner = setting(self.db, "default_aligner", "")
        if not aligner:
            self.note = (
                "no default_aligner set — run `rytp settings set default_aligner "
                "<name>` first, or `transcribe align --aligner <name>` directly"
            )
            return
        self._dispatch(row, "transcribe.align", aligner=aligner, enqueue=True)

    def diarize(self, row: int) -> None:
        """`speakers.enqueue`."""
        self._dispatch(row, "speakers.enqueue")

    def reindex(self, row: int) -> None:
        """`index.build --enqueue`, scoped to this one video."""
        self._dispatch(row, "index.build", enqueue=True)

    def drop_transcript(self, row: int) -> None:
        """`transcribe.remove`, run (not enqueued — it only deletes rows,
        contracts §5's `--dry-run`/`--yes` gate is for commands that delete
        files) after the caller has already confirmed.

        Unlike the other five shortcuts, the word count is worth stating
        explicitly: `transcribe.remove`'s own message names the video but
        not how much was lost, and a confirmed destructive action should
        not read as if nothing happened.
        """
        video_id = self.video_id_at(row)
        if video_id is None:
            self.note = "no video selected"
            return
        self.error = None
        try:
            result = resolve("transcribe.remove").handler(self.db, video=str(video_id))
            words = result.rows[0][0] if result.rows else None
            message = result.message or f"transcribe.remove ran for video {video_id}"
            self.note = f"{message} — {words} words removed" if words is not None else message
        except RytpError as exc:
            self.note = str(exc)
        self.refresh()


class VideosScreen(Screen[None]):
    """A filterable table of the fleet, one shortcut per pipeline stage."""

    DEFAULT_CSS = """
    #videos-table  { height: 1fr; }
    #videos-status { height: auto; padding: 0 1; }
    """

    # No bare letters (the filter `Input` holds focus by default). None of
    # these five collide with a chord bound elsewhere (`test_tui_navigation.
    # py`'s `test_one_key_means_one_thing_wherever_it_is_bound`), and none of
    # them is `ctrl+h/i/j/m` or `ctrl+[` — a terminal sends those as the
    # identical byte as Backspace/Tab/Linefeed/Enter/Escape, so Textual could
    # never tell the chord from the named key apart.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(BACK_KEY, "app.pop_screen", "Back"),
        Binding("ctrl+k", "ingest", "Ingest"),
        Binding("ctrl+w", "transcribe", "Transcribe"),
        Binding("ctrl+a", "align", "Align"),
        Binding("ctrl+d", "diarize", "Diarize"),
        # Not ctrl+f: terminals and shells commonly claim it for "find".
        # ctrl+l is the only letter chord free on every screen —
        # c/v/z are reserved, and h/i/j/m are indistinguishable from
        # Backspace/Tab/Linefeed/Enter at the protocol level.
        Binding("ctrl+l", "reindex", "Index"),
        # shift+delete, not ctrl+x (BUGS.md entry 44): ctrl+x is already
        # "Cancel job" on the jobs screen, and `test_one_key_means_one_
        # thing_wherever_it_is_bound` compares binding *descriptions*
        # across every screen, so reusing it with a different label would
        # fail, correctly. shift+delete is free on every screen, is not in
        # `textual.widgets.Input.BINDINGS` (so the filter Input, which
        # holds focus by default, never swallows it — the same reason the
        # other five shortcuts are chords rather than bare letters), and is
        # the conventional "permanent delete". Confirmed, not immediate:
        # the other five re-run for free, this one costs a re-transcription.
        Binding("shift+delete", "drop_transcript", "Drop transcript"),
    ]

    def __init__(self, db: Database) -> None:
        super().__init__()
        self.session = VideoFleetSession(db)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="filter by title, then Enter", id="videos-filter")
        yield DataTable(id="videos-table")
        yield Static("", id="videos-status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#videos-table", DataTable).cursor_type = "row"
        self.query_one("#videos-filter", Input).focus()
        self._draw()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "videos-filter":
            self.filter(event.value)

    # -- actions, called by the tests too -----------------------------

    def filter(self, text: str) -> None:
        self.session.run(text)
        self._draw()

    def action_ingest(self) -> None:
        self.session.ingest(self._row())
        self._draw()

    def action_transcribe(self) -> None:
        self.session.transcribe(self._row())
        self._draw()

    def action_align(self) -> None:
        self.session.align(self._row())
        self._draw()

    def action_diarize(self) -> None:
        self.session.diarize(self._row())
        self._draw()

    def action_reindex(self) -> None:
        self.session.reindex(self._row())
        self._draw()

    @work
    async def action_drop_transcript(self) -> None:
        """Confirm, then run `transcribe.remove` on the highlighted row.

        Behind a modal, unlike the other five shortcuts: this one is not
        cheap to undo. A misfire means paying for a full re-transcription
        again — minutes of GPU time on a long video, plus re-mapping
        speakers by hand (BUGS.md entry 44).

        `@work`: `push_screen_wait` (below) requires an active Textual
        worker task — plain action dispatch does not run one — so this
        schedules itself as one rather than awaiting the modal directly.
        """
        row = self._row()
        summary = self.session.video_summary_at(row)
        if summary is None:
            self.session.note = "no video selected"
            self._draw()
            return
        video_id, title = summary
        confirmed = await self.app.push_screen_wait(
            ConfirmScreen(
                f"Drop the transcript of video {video_id}?",
                (
                    title,
                    "",
                    "This deletes its words, utterances and speaker labels.",
                    "Recovering means re-transcribing the video from scratch.",
                ),
                confirm_label="Drop transcript",
            )
        )
        if confirmed:
            self.session.drop_transcript(row)
        self._draw()

    # -- drawing -------------------------------------------------------

    def _row(self) -> int:
        return int(self.query_one("#videos-table", DataTable).cursor_row or 0)

    def _draw(self) -> None:
        """Repaint, keeping the cursor on the video it was already on.

        `clear()` resets `cursor_row` to 0, so without this a shortcut
        would bounce the selection back to the top and make working down
        a list (4, 3, 2, 1) impossible — every keypress would re-select
        the first row.

        Keyed on the video id rather than the row index, so the selection
        follows the video rather than a position: a row set that shifts
        keeps the same video highlighted, and a video that is genuinely
        gone (filtered out) falls back to the top, which is what a new
        filter should do anyway.
        """
        table = self.query_one("#videos-table", DataTable)
        keep = self._video_id_at(table.cursor_row)
        table.clear(columns=True)
        if self.session.columns:
            table.add_columns(*self.session.columns)
            for row in self.session.rows:
                table.add_row(*plain_row(row))
        if keep is not None:
            for index, row in enumerate(self.session.rows):
                if row and row[0] == keep:
                    table.move_cursor(row=index)
                    break
        set_text(self.query_one("#videos-status", Static), self.session.status)

    def _video_id_at(self, index: int | None) -> str | None:
        """The id cell of a row, or None when nothing is selected."""
        if index is None or not (0 <= index < len(self.session.rows)):
            return None
        row = self.session.rows[index]
        return str(row[0]) if row else None
