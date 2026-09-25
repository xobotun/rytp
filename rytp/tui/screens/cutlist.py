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
from typing import ClassVar, Final

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from rytp import constants as C
from rytp.commands import resolve
from rytp.db import Database
from rytp.index.export import play_clip
from rytp.models import RytpError
from rytp.tui.cutlist_view import CutlistView, available_cutlists
from rytp.tui.navigation import BACK_KEY
from rytp.tui.text import plain_row, set_text

__all__ = ["CutlistPickerScreen", "CutlistScreen", "RetargetScreen"]

#: BUGS.md entry 29 made the footer fit by hiding almost every editing key
#: (``show=False`` below); the owner then had to open F1 to learn `[`, `]`,
#: `ctrl+left` and `ctrl+right` existed at all. Putting the full list back in
#: the footer re-breaks entry 29 — this is the screen with the most bindings
#: by some margin — so it goes in the screen's own body instead, where
#: wrapping is this module's choice rather than Textual's. Literal key names,
#: matching how F1's own table spells them (`rytp/tui/screens/help.py`),
#: rather than arrow glyphs a narrow font might not have.
#:
#: Two lines, by what the owner had to hunt for versus what the footer
#: already shows: the first is the boundary-editing loop — nudge, coarse
#: nudge, snap, and play (the newest of the five, least guessable, and the
#: one the owner found only by being told it existed) — the second is
#: remove and render, plus the footer's own swap/undo/save for one place
#: that lists everything at a glance.
#:
#: Measured in the running app at 80×24 this wraps to 5 rows — 21% of the
#: screen — which the owner accepted ("I have a bigger terminal... maybe a
#: button to hide the hints?") in exchange for a key that reclaims them:
#: `h`, bound below and named right here in the hint's own last clause, so
#: the toggle is never harder to find than the thing it hides.
_EDIT_HINT: Final = (
    "[ / ]  start ±10ms   shift+[ / shift+]  start ±100ms   "
    "ctrl+left / ctrl+right  end ±10ms   "
    "ctrl+shift+left / ctrl+shift+right  end ±100ms   "
    "z / x  snap start/end   ctrl+p  play\n"
    "d  remove   t  retarget   e  render (queues; F7 to watch)   "
    "s  swap · u  undo · ctrl+s  save · ?  all keys · h  hide this hint"
)


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


class RetargetScreen(ModalScreen[str | None]):
    """Ask for a new target sentence, prefilled with the current one.

    Owner's request: edit the target — add a word, or drop one with no
    cuttable form — without losing the hand-tuning already done on the
    rest of the cut list. A `ModalScreen` rather than a bare `Input`
    dropped into `CutlistScreen`'s own body: that screen's `escape` already
    means "back to the picker" (`BACK_KEY`), and `Input` does not consume
    `escape` itself, so typing into one there would pop the whole editor
    mid-edit. A modal's own `escape` binding only dismisses the modal.

    Resolves to the typed text on Enter, or `None` on Escape — never
    raises and never talks to the database itself; `CutlistScreen` decides
    what an empty or unchanged answer means.
    """

    DEFAULT_CSS = """
    RetargetScreen {
        align: center middle;
    }
    #retarget-dialog {
        width: 80%;
        max-width: 100;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    #retarget-input {
        margin-top: 1;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, current_target: str) -> None:
        super().__init__()
        self._current = current_target

    def compose(self) -> ComposeResult:
        with Vertical(id="retarget-dialog"):
            yield Static(id="retarget-label")
            yield Input(value=self._current, id="retarget-input")

    def on_mount(self) -> None:
        set_text(
            self.query_one("#retarget-label", Static),
            "Edit the target sentence, then Enter (Escape cancels):",
        )
        self.query_one("#retarget-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


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
    #cutlist-hint    { height: auto; padding: 0 1; }
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
        # Play the current selection (owner's request): the fragment list's
        # highlighted slot, verbatim as edited, or the options pane's
        # highlighted alternative/substitution when that pane holds the
        # cursor. Same key and label as the search screen's `ctrl+p`
        # (`rytp/tui/screens/search.py`) — one span of audio is one idea,
        # whichever screen names it a "hit" or a "fragment".
        Binding("ctrl+p", "play", "Play fragment", show=False),
        Binding("u", "undo", "Undo"),
        Binding("r", "reload", "Discard edits", show=False),
        Binding("ctrl+s", "save", "Save"),
        # Remove a fragment the owner no longer wants in the splice (owner's
        # request). Not a gap, not shown as one — see CutlistView.remove's
        # docstring for why. "d" pairs with no other binding on any screen.
        Binding("d", "remove", "Remove", show=False),
        # Edit the target sentence (owner's request): add a word, or drop
        # one with no cuttable form, without discarding the hand-tuning
        # already done on the rest of the slots. See `RetargetScreen` and
        # `CutlistView.retarget`. "t" for target; free everywhere else this
        # screen's letters are checked against (`test_one_key_means_one_
        # thing_wherever_it_is_bound`).
        Binding("t", "retarget", "Retarget", show=False),
        # Reclaim the hint's own rows (owner's request, following on from
        # BUGS.md entry 29): visible by default, since the hint exists for
        # someone meeting this screen for the first time; toggled off and
        # back on for the rest of a session with no persistence — there is
        # no user-preference store in this project, and a view toggle does
        # not justify inventing one. "h" for hint; free everywhere else
        # this screen's letters are checked against
        # (`test_one_key_means_one_thing_wherever_it_is_bound`).
        Binding("h", "toggle_hint", "Toggle hint", show=False),
        # Order a render (owner's request): enqueue Part 6's `render.run`,
        # matching the Videos screen's "queue, never run inline" idiom
        # (BUGS.md entry 22) and its F7 pointer. "e" for encode; "r" already
        # means "Discard edits" here.
        Binding("e", "render", "Render", show=False),
        Binding("?", "app.help", "Help"),
    ]

    def __init__(self, db: Database, path: Path) -> None:
        super().__init__()
        self._db = db
        self.view = CutlistView(path)
        # Guards against a second `ctrl+p` while `ffplay` is still running.
        # `play_clip` blocks a whole OS thread, so there is no cheap way to
        # cut the first clip off mid-word; a second press is ignored rather
        # than queued or made to interrupt, which would otherwise overlap
        # two clips into noise.
        self._playing = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="cutlist-hint")
        yield DataTable(id="cutlist-slots")
        yield DataTable(id="cutlist-options")
        yield Static("", id="cutlist-status")
        yield Footer()

    def on_mount(self) -> None:
        for table_id in ("#cutlist-slots", "#cutlist-options"):
            self.query_one(table_id, DataTable).cursor_type = "row"
        # `_EDIT_HINT` has literal `[` / `]` in it — exactly the markup this
        # module exists to route around a bare `Static.update` (BUGS.md
        # entry 20; see rytp/tui/text.py).
        set_text(self.query_one("#cutlist-hint", Static), _EDIT_HINT)
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

    def action_remove(self) -> None:
        self._after(self.view.remove(self._slot()))

    def action_toggle_hint(self) -> None:
        """Show or hide `#cutlist-hint` — the whole point is reclaiming its
        rows, so this flips the widget's own `display` rather than clearing
        its text (which would leave the row occupied but blank) or
        rebuilding the layout. No new state of this screen's own: the
        widget's current `display` is the only record of which way it is,
        which is also why this needs no persistence — a fresh screen always
        starts from the widget's own default, visible.
        """
        hint = self.query_one("#cutlist-hint", Static)
        hint.display = not hint.display

    @work
    async def action_retarget(self) -> None:
        """Edit the target sentence through a modal text box, then fold the
        result into the same in-memory undo/dirty edit `self.view` already
        keeps for every other verb (`ctrl+s` to save, `u` to undo, `r` to
        discard) — no confirmation dialog, matching `action_render`'s
        reasoning that a reversible edit does not need one.

        `@work`: `push_screen_wait` requires an active Textual worker task
        — plain action dispatch does not run one — the same reason
        `VideosScreen.action_drop_transcript` carries it for its own modal.
        """
        new_target = await self.app.push_screen_wait(
            RetargetScreen(self.view.cutlist.target)
        )
        if new_target is None:
            return
        self._after(self.view.retarget(self._db, new_target))

    def action_undo(self) -> None:
        self._after(self.view.undo())

    def action_reload(self) -> None:
        self._after(self.view.reload())

    def action_save(self) -> None:
        self.view.save()
        self._after(self.view.status)

    def action_render(self) -> None:
        """Order a render (owner's request): enqueue `render.run`, never run
        it inline — encoding blocks, and running it on Textual's event loop
        would freeze the interface (BUGS.md entry 22). Matches the Videos
        screen's shortcuts, which all enqueue and say "F7 to watch the
        queue" (`rytp/tui/screens/videos.py`) — this does the same, through
        the same registered command the palette and the CLI use.

        `render.run` reads the cut list back off disk, so an edit still
        only in memory would be silently absent from the render. Rather
        than refuse or pop a modal — task 2's "no confirmation for a
        reversible edit" reasoning applies here too, since saving is
        exactly what ctrl+s already does — this just saves first when
        there is something unsaved, so what gets queued always matches
        what is on screen.
        """
        if self.view.dirty:
            self.view.save()
        try:
            result = resolve("render.run").handler(
                self._db, name=self.view.name, enqueue=True
            )
            note = result.message or f"render queued for {self.view.name!r}"
        except RytpError as exc:
            note = str(exc)
        self._after(note)

    def action_play(self) -> None:
        """Play whatever is selected — the current slot, or, when the
        options pane holds the cursor, the highlighted alternative or
        substitution instead.

        Silent on a gap and on nothing to play (`CutlistView.play_target`
        returns ``None`` for both): a gap is visibly a gap on screen, and a
        key pressed repeatedly while auditioning fragments should not
        narrate every miss. Run off the event loop (`@work(thread=True)`
        below) because `play_clip` blocks until `ffplay` exits — the same
        trap BUGS.md entry 22 recorded for a handler called straight from a
        screen action.
        """
        if self._playing:
            return
        options_table = self.query_one("#cutlist-options", DataTable)
        option_index = self._option() if options_table.has_focus else None
        target = self.view.play_target(self._slot(), option_index)
        if target is None:
            return
        self._playing = True
        self._play_clip(*target)

    @work(thread=True, exclusive=True, group="cutlist-play")
    def _play_clip(self, video_id: int, start_ms: int, end_ms: int) -> None:
        try:
            play_clip(self._db, video_id, start_ms, end_ms)
        except RytpError as exc:
            # No ffplay, or no cached audio for that video (contracts §7 —
            # the WAV cache is prunable). Both are one-line, actionable
            # complaints, never a traceback. `call_from_thread` lives on the
            # app, not the screen, which is why this is `self.app.…` and not
            # `self.…` — the same seam `RytpApp._post_progress` uses.
            self.app.call_from_thread(self._report_play_error, str(exc))
        finally:
            self.app.call_from_thread(self._clear_playing)

    def _report_play_error(self, message: str) -> None:
        set_text(self.query_one("#cutlist-status", Static), message)

    def _clear_playing(self) -> None:
        self._playing = False

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
