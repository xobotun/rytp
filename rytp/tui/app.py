"""The Textual shell around the command registry (design §10).

A palette of every registered command, a line to type arguments into,
and a table for the result. Workflow screens — the speaker mapper above
all — belong to later plan parts; this is the proof that both surfaces
come from one definition.

Long-running commands are listed but never run here: design §5 puts them
in a separate worker process, so the TUI prints the equivalent command
line instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import DataTable, Footer, Header, Input, OptionList, Static
from textual.widgets.option_list import Option

from rytp.commands import COMMANDS, Command, CommandResult
from rytp.db import Database
from rytp.models import NotFoundError, RytpError
from rytp.tui.enqueue import enqueue_plan, foreground_hint
from rytp.tui.navigation import binding_rows, home_strip, load_screen_class, screen_by_id
from rytp.tui.palette import (
    PaletteEntry,
    match_entries,
    palette_entries,
    parse_arguments,
)
from rytp.tui.text import plain_row, set_text

__all__ = ["RytpApp", "run_tui"]


class RytpApp(App[None]):
    """Command palette, argument line, result table."""

    TITLE = "rytp"
    SUB_TITLE = "find what was said, then cut it"

    CSS = """
    #palette { height: 1fr; }
    #results { height: 1fr; }
    #status  { height: auto; padding: 0 1; }
    """

    # Derived from the navigation map (Part 8). A new screen is added there,
    # not here, so it cannot exist without a key, a home-strip entry and a
    # help line. `list[BindingType]` because App declares the wider type and
    # list is invariant.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(key, action, description) for key, action, description in binding_rows()
    ]

    def __init__(
        self, db: Database, commands: Mapping[str, Command] | None = None
    ) -> None:
        super().__init__()
        self._db = db
        self._commands: Mapping[str, Command] = COMMANDS if commands is None else commands
        self._entries = palette_entries(self._commands)
        # NOT `self.visible`: Textual's DOM owns that name.
        self.shown: list[PaletteEntry] = list(self._entries)
        self.status_text = ""

    # -- layout ------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(home_strip(), id="home-nav")
        yield Input(placeholder="filter commands", id="filter")
        yield OptionList(id="palette")
        yield Input(placeholder="arguments: value name=value …", id="arguments")
        yield Static("", id="status")
        yield DataTable(id="results")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_palette("")

    # -- palette -----------------------------------------------------

    def refresh_palette(self, query: str) -> None:
        palette = self.query_one("#palette", OptionList)
        palette.clear_options()
        self.shown = match_entries(self._entries, query)
        for entry in self.shown:
            marker = " (worker)" if entry.long_running else ""
            palette.add_option(Option(f"{entry.name}  —  {entry.summary}{marker}", id=entry.name))
        if self.shown:
            palette.highlighted = 0

    def select(self, name: str) -> None:
        """Highlight a command by name. Used by the tests and by the palette."""
        for index, entry in enumerate(self.shown):
            if entry.name == name:
                self.query_one("#palette", OptionList).highlighted = index
                return

    def selected_command(self) -> Command | None:
        palette = self.query_one("#palette", OptionList)
        index = palette.highlighted
        if index is None or index >= len(self.shown):
            return None
        return self._commands[self.shown[index].name]

    # -- running -----------------------------------------------------

    def run_selected(self, argument_line: str | None = None) -> None:
        """Parse the argument line and run the highlighted command."""
        cmd = self.selected_command()
        if cmd is None:
            self.set_status("no command selected")
            return
        if argument_line is None:
            argument_line = self.query_one("#arguments", Input).value
        try:
            values = parse_arguments(cmd, argument_line)
        except RytpError as exc:
            self.set_status(str(exc))
            return
        if cmd.long_running:
            plan = enqueue_plan(cmd, values)
            if plan is None:
                self.set_status(foreground_hint(cmd, values))
                return
            # Not running the work: the handler writes one row to `jobs` and
            # returns. design §5 keeps the worker a separate process.
            try:
                result = cmd.handler(self._db, **plan.values)
            except RytpError as exc:
                self.set_status(str(exc))
                return
            self.show_result(result)
            self.set_status(f"{result.message or 'queued'} — F7 to watch the queue")
            return
        try:
            result = cmd.handler(self._db, **values)
        except RytpError as exc:
            self.set_status(str(exc))
            return
        self.show_result(result)

    def show_result(self, result: CommandResult) -> None:
        table = self.query_one("#results", DataTable)
        table.clear(columns=True)
        if result.columns:
            table.add_columns(*result.columns)
            for row in result.rows:
                table.add_row(*plain_row(row))
        self.set_status(result.message or f"{len(result.rows)} rows")

    def set_status(self, text: str) -> None:
        self.status_text = text
        set_text(self.query_one("#status", Static), text)

    # -- events ------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter":
            self.refresh_palette(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "arguments":
            self.run_selected(event.value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.query_one("#arguments", Input).focus()

    def action_focus_filter(self) -> None:
        self.query_one("#filter", Input).focus()

    # -- navigation --------------------------------------------------

    def action_open(self, screen_id: str) -> None:
        """Push one of the screens named in the navigation map.

        One action for every screen, parameterised by id, so adding a screen
        is one row in `navigation.SCREENS` and no code here. The screen class
        is imported only now, which is what keeps `rytp tui` from loading the
        index, assemble, diarize and render packages at start-up.
        """
        try:
            entry = screen_by_id(screen_id)
            screen_class = load_screen_class(entry)
        except (NotFoundError, ModuleNotFoundError) as exc:
            self.set_status(str(exc))
            return
        # Every entry screen's constructor takes (db), but the navigation
        # table only knows the common Screen base, not each concrete
        # subclass's signature.
        self.push_screen(screen_class(self._db))  # type: ignore[arg-type]

    def action_help(self) -> None:
        """Every binding in the application, on the key people try first."""
        from rytp.tui.screens.help import HelpScreen

        self.push_screen(HelpScreen())

    # The three names Parts 4 and 7 published in their Interfaces blocks.
    def action_speakers(self) -> None:
        self.action_open("speakers")

    def action_search(self) -> None:
        self.action_open("search")

    def action_transcripts(self) -> None:
        self.action_open("transcripts")


def run_tui(db: Database) -> None:
    """Launch the app. Called only by `rytp tui`."""
    RytpApp(db).run()
