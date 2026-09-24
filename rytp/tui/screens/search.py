"""Searching and playing a hit in the TUI (design §7, §10).

Design §10 wants the two surfaces generated from one definition, so this
screen runs the registered handlers rather than reaching past them:
`search.words` for the query and `search.play` for playback, the same two
`rytp search words` and `rytp search play` invoke. The screen exists to
remove the copy-and-paste between them.

Everything that is not a widget lives in `SearchSession`, which is why
the shell below is short enough to read in one go.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from rytp import constants as C
from rytp.commands import CommandResult, resolve
from rytp.models import RytpError
from rytp.tui.text import plain_row, set_text

if TYPE_CHECKING:
    from rytp.db import Database

__all__ = ["SearchScreen", "SearchSession"]

#: Column index of the anchor in a `search.words` result. The anchor is
#: what `search.play` takes, which is what lets the two compose.
_ANCHOR = 0

#: BUGS.md entry 19: the tier vocabulary is load-bearing and nothing near the
#: filter itself said what it meant. F1 carries the full glossary; this is the
#: one-line version at the point of use. `assemble plan --allow-timed` (Task 6)
#: is the named escape hatch, so a user deciding whether to reach for it needs
#: to see it mentioned right here, not just in the help screen.
CUTTABLE_HINT = (
    "aligned tier only; timed and caption words cannot be cut — "
    "assemble plan --allow-timed overrides, see F1"
)


@dataclass
class SearchSession:
    """Query, filters, and the last result. No widgets, no SQL.

    Every database access goes through a registered command handler, so
    the screen cannot drift from the CLI and the speaker resolver — whose
    signature is still settling across parts — is touched in exactly one
    place, inside `search.words`.
    """

    db: Database
    speaker: str | None = None
    cuttable: bool = False
    limit: int = C.SEARCH_DEFAULT_LIMIT
    query: str = ""
    result: CommandResult = field(default_factory=CommandResult)
    error: str | None = None

    # -- running ---------------------------------------------------------

    def run(self, query: str) -> None:
        """Search, keeping any failure as text rather than an exception."""
        self.query = query
        self.error = None
        try:
            self.result = resolve("search.words").handler(
                self.db,
                query=query,
                speaker=self.speaker,
                cuttable=self.cuttable,
                limit=self.limit,
            )
        except RytpError as exc:
            # An empty box, or a person who is not on the roster. Both are
            # ordinary mistakes; a screen that raises on them is unusable,
            # and showing the resolver's message is also what stops
            # "unknown speaker" reading as "he never said it".
            self.result = CommandResult()
            self.error = str(exc)

    def also(self, query: str) -> tuple[tuple[str, ...], ...]:
        """Run and hand back the rows. A convenience for tests."""
        self.run(query)
        return self.rows

    def play(self, row: int) -> str:
        """Play one row through `search.play`. Returns what it reported."""
        self.error = None
        anchor = self.anchor_at(row)
        if anchor is None:
            self.error = "nothing to play"
            return self.error
        try:
            played = resolve("search.play").handler(self.db, anchor=anchor)
        except RytpError as exc:
            # No ffplay, or no audio for that video. Worth saying, not
            # worth losing the result table over.
            self.error = str(exc)
            return self.error
        return played.message or f"played {anchor}"

    def toggle_cuttable(self) -> None:
        """Flip the cuttable-only filter and re-run the current query."""
        self.cuttable = not self.cuttable
        if self.query:
            self.run(self.query)

    # -- reading ---------------------------------------------------------

    @property
    def columns(self) -> tuple[str, ...]:
        return self.result.columns

    @property
    def rows(self) -> tuple[tuple[str, ...], ...]:
        return self.result.rows

    @property
    def status(self) -> str:
        """One line: the error if there is one, else what the command said."""
        if self.error:
            return self.error
        filters = f" · cuttable only — {CUTTABLE_HINT}" if self.cuttable else ""
        if self.speaker:
            filters += f" · speaker {self.speaker}"
        return (self.result.message or "") + filters

    def anchor_at(self, row: int) -> str | None:
        if not (0 <= row < len(self.rows)):
            return None
        return self.rows[row][_ANCHOR]


class SearchScreen(Screen[None]):
    """A query box, the hits, and one key that plays the highlighted one."""

    DEFAULT_CSS = """
    #search-hits { height: 1fr; }
    #search-status { height: auto; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("ctrl+p", "play", "Play hit"),
        Binding(
            "ctrl+t",
            "toggle_cuttable",
            "Cuttable only",
            tooltip=CUTTABLE_HINT,
        ),
    ]

    def __init__(self, db: Database, speaker: str | None = None) -> None:
        super().__init__()
        self.session = SearchSession(db, speaker=speaker)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="word or phrase, then Enter", id="search-query")
        yield DataTable(id="search-hits")
        yield Static("", id="search-status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#search-hits", DataTable).cursor_type = "row"
        self.query_one("#search-query", Input).focus()
        self._draw()

    # -- actions ---------------------------------------------------------

    def search(self, query: str) -> None:
        """Run a query and redraw. Called by Enter and by the tests."""
        self.session.run(query)
        self._draw()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-query":
            self.search(event.value)

    def action_play(self) -> None:
        """Play whichever hit the cursor is on (design §7: "Playback")."""
        row = self.query_one("#search-hits", DataTable).cursor_row
        message = self.session.play(row)
        set_text(self.query_one("#search-status", Static), message)

    def action_toggle_cuttable(self) -> None:
        self.session.toggle_cuttable()
        self._draw()

    # -- drawing ---------------------------------------------------------

    def _draw(self) -> None:
        table = self.query_one("#search-hits", DataTable)
        table.clear(columns=True)
        if self.session.columns:
            table.add_columns(*self.session.columns)
            for row in self.session.rows:
                table.add_row(*plain_row(row))
        set_text(self.query_one("#search-status", Static), self.session.status)
