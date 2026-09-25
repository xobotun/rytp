"""The Textual shell around the command registry (design §10).

A palette of every registered command, a line to type arguments into,
and a table for the result. Workflow screens — the speaker mapper above
all — belong to later plan parts; this is the proof that both surfaces
come from one definition.

Long-running commands are listed but never run here: design §5 puts the
worker itself in a separate process, so a queued command's handler only
writes one row to ``jobs`` and returns; the TUI prints the equivalent
command line for anything that cannot be queued at all (BUGS.md entry 22).

**Every** handler — queued or not — runs off the event loop, in a Textual
worker thread. `long_running` classifies whether a command has a *queued*
form, not whether its handler blocks: `videos.add` is correctly not
queueable (cataloguing is one metadata call, not per-video work) and still
shells out to yt-dlp for a network round trip, which used to freeze the
whole interface for the duration. Running everything through a worker
means that gap cannot reopen for the next handler that happens to block.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Final

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.content import Content
from textual.widgets import DataTable, Footer, Header, Input, OptionList, Static
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState

from rytp.commands import COMMANDS, Command, CommandResult
from rytp.db import Database
from rytp.models import NotFoundError, RytpError
from rytp.progress import format_line
from rytp.progress import install as install_progress_sink
from rytp.tui.enqueue import enqueue_plan, foreground_hint
from rytp.tui.navigation import binding_rows, home_strip, load_screen_class, screen_by_id
from rytp.tui.palette import (
    PaletteEntry,
    group_heading,
    match_entries,
    palette_entries,
    parse_arguments,
)
from rytp.tui.text import plain_row, set_text

__all__ = ["RytpApp", "run_tui"]

#: What the home screen says about itself (BUGS.md entry 12). One or two
#: lines, plain text — no brackets, so it never needs `set_text`'s markup
#: escaping, but it is still routed through the `Static` constructor rather
#: than `.update(...)` to keep `tests/test_tui_markup.py`'s AST sweep happy.
_HOME_HELP = (
    "Type to filter, ↓ or Tab to reach the list below, Enter to select — "
    "that fills the argument field underneath. Enter there runs the command, "
    "or queues it and points at F7 to watch it."
)


#: Measured in the real app at 80x24 (`tests/test_tui_footer.py`): the
#: app-level strip alone — F1-F9 plus Textual's own Ctrl+P command-palette
#: key — ran to 116 columns before a single screen-level binding was added,
#: which is why *every* screen's footer overflowed regardless of how few
#: keys that screen bound itself (the cut-list screen's own eleven keys were
#: never the whole story). These two are what a user reaches for on every
#: screen without thinking; the rest are one key (F1) away, and the home
#: view's nav strip (`home_strip()`) already names all of them without
#: costing a column of the footer.
_FOOTER_VISIBLE: Final = frozenset({"f1", "ctrl+q"})


class RytpApp(App[None]):
    """Command palette, argument line, result table."""

    TITLE = "rytp"
    SUB_TITLE = "find what was said, then cut it"

    # Textual's own command-palette overlay is a second, unrelated command
    # surface this project does not use — `COMMANDS` and this app's own
    # OptionList palette are the one and only palette. Left enabled, it adds
    # a Ctrl+P key to every screen's footer for a feature nothing here
    # exercises, which is part of the overflow this measures away.
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    #palette { height: 1fr; }
    #results { height: 1fr; }
    #status  { height: auto; padding: 0 1; }
    """

    # Derived from the navigation map (Part 8). A new screen is added there,
    # not here, so it cannot exist without a key, a home-strip entry and a
    # help line. `list[BindingType]` because App declares the wider type and
    # list is invariant. Every one of these is bound and reachable from any
    # screen — screen bindings stack on top, they never replace these — but
    # only the two in `_FOOTER_VISIBLE` cost a column of the footer; F1 Help
    # lists the rest, and the home view's own nav strip does too.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(key, action, description, show=key in _FOOTER_VISIBLE)
        for key, action, description in binding_rows()
    ]

    def __init__(
        self, db: Database, commands: Mapping[str, Command] | None = None
    ) -> None:
        super().__init__()
        self._db = db
        self._commands: Mapping[str, Command] = COMMANDS if commands is None else commands
        self._entries = palette_entries(self._commands)
        self._entries_by_name = {entry.name: entry for entry in self._entries}
        # NOT `self.visible`: Textual's DOM owns that name.
        self.shown: list[PaletteEntry] = list(self._entries)
        self.status_text = ""
        # Workers in flight, keyed by the `Worker` instance Textual hands
        # back from `run_worker`, valued by whether this run is a queue
        # (so completion should mention F7) or a plain foreground run.
        self._pending_runs: dict[Worker[CommandResult], bool] = {}

    # -- layout ------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(home_strip(), id="home-nav")
        yield Static(_HOME_HELP, id="home-help")
        yield Input(placeholder="filter — type to narrow the list below", id="filter")
        yield OptionList(id="palette")
        yield Input(placeholder="arguments: value name=value …", id="arguments")
        yield Static("", id="status")
        yield DataTable(id="results")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_palette("")

    # -- palette -----------------------------------------------------

    def refresh_palette(self, query: str) -> None:
        """Rebuild the option list from the filtered entries.

        contracts §5: `GROUP_SUMMARIES` is read by both surfaces — the CLI's
        `_group_help` and this palette's group heading. A heading is a
        disabled `Option`, one per non-empty group, inserted where the
        (already-sorted) entries cross a group boundary; `palette_entries`
        sorts by full dotted name, so one group's entries are always
        contiguous and a heading is never duplicated for a group that is
        still being listed. Disabled options are skipped automatically by
        `OptionList`'s own cursor movement and never post `OptionSelected`
        (Textual's `action_select` checks `disabled` before posting), so
        nothing downstream has to know they exist — except that *this*
        method no longer keys off position: an inserted heading shifts every
        real option's index, so selection and highlighting go through the
        option's `id` (a command name) instead of `self.shown[index]`.
        """
        palette = self.query_one("#palette", OptionList)
        palette.clear_options()
        self.shown = match_entries(self._entries, query)
        last_group: str | None = None
        first_command_index: int | None = None
        index = 0
        for entry in self.shown:
            if entry.group and entry.group != last_group:
                heading = group_heading(entry.group)
                palette.add_option(
                    Option(Content(heading), id=f"group:{entry.group}", disabled=True)
                )
                index += 1
            last_group = entry.group
            marker = " (worker)" if entry.long_running else ""
            # One spelling on screen (entry 14b): the spaced form, which is
            # what `usage_line` already renders and both the CLI and this
            # placeholder accept. `entry.name` (dotted) still identifies the
            # option internally — `select`/`selected_command` key off it —
            # it is simply never shown.
            label = entry.name.replace(".", " ")
            palette.add_option(
                Option(Content(f"{label}  —  {entry.summary}{marker}"), id=entry.name)
            )
            if first_command_index is None:
                first_command_index = index
            index += 1
        if first_command_index is not None:
            palette.highlighted = first_command_index

    def select(self, name: str) -> None:
        """Highlight a command by name. Used by the tests and by the palette."""
        palette = self.query_one("#palette", OptionList)
        for index in range(palette.option_count):
            if palette.get_option_at_index(index).id == name:
                palette.highlighted = index
                return

    def selected_command(self) -> Command | None:
        palette = self.query_one("#palette", OptionList)
        index = palette.highlighted
        if index is None:
            return None
        option_id = palette.get_option_at_index(index).id
        if option_id is None:
            return None
        return self._commands.get(option_id)

    # -- running -----------------------------------------------------

    def run_selected(self, argument_line: str | None = None) -> None:
        """Parse the argument line and run the highlighted command.

        The parsing above stays inline — it is pure and instant, and a bad
        argument line should be reported before anything is scheduled. The
        handler call itself never runs here (entry 22): every handler goes
        through `_start_handler`, in a worker thread, whether this run
        queues the work or does it now.
        """
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
        queued = False
        if cmd.long_running:
            plan = enqueue_plan(cmd, values)
            if plan is None:
                self.set_status(foreground_hint(cmd, values))
                return
            # Not running the work: the handler writes one row to `jobs` and
            # returns. design §5 keeps the worker a separate process.
            values = plan.values
            queued = True
        self._start_handler(cmd, values, queued=queued)

    def _start_handler(self, cmd: Command, values: Mapping[str, Any], *, queued: bool) -> None:
        """Run `cmd`'s handler in a Textual worker thread, off the event loop.

        `videos.add` is registered `long_running=False` — correctly, since
        cataloguing is one metadata call, not per-video work — and still
        shells out to yt-dlp for it. `long_running` says nothing about
        whether a handler blocks, so every handler is run this way, not
        just the ones already known to be slow.
        """
        label = cmd.name.replace(".", " ")
        self.set_status(f"running {label} …")

        def call() -> CommandResult:
            with install_progress_sink(self._post_progress):
                return cmd.handler(self._db, **values)

        worker: Worker[CommandResult] = self.run_worker(
            call, name=cmd.name, thread=True, exclusive=True, exit_on_error=False
        )
        self._pending_runs[worker] = queued

    def _post_progress(
        self, stage: str, done: int | None, total: int | None, detail: str
    ) -> None:
        """The TUI's progress sink (plan §1c). Runs on the worker thread, so
        it hands the line to the app rather than touching a widget directly —
        `call_from_thread` is what makes that safe."""
        self.call_from_thread(self.set_status, format_line(stage, done, total, detail))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Show a handler's result once its worker finishes.

        Only workers this app started for a command run are tracked here
        (`_pending_runs`); anything else is none of this handler's business.
        """
        worker = event.worker
        if worker not in self._pending_runs:
            return
        if event.state not in (WorkerState.SUCCESS, WorkerState.ERROR):
            return
        queued = self._pending_runs.pop(worker)
        if event.state is WorkerState.ERROR:
            exc = worker.error
            if isinstance(exc, RytpError):
                self.set_status(str(exc))
                return
            # Handlers never raise anything but RytpError by contract
            # (contracts §8); anything else is a bug, and re-raising here
            # surfaces it the same way an inline call would have.
            if exc is not None:
                raise exc
            return
        result = worker.result
        assert result is not None
        self.show_result(result)
        if queued:
            self.set_status(f"{result.message or 'queued'} — F7 to watch the queue")

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
        elif event.input.id == "filter":
            # 14c: Enter in the filter used to be silently discarded — focus
            # starts here, so it is the first key a new user presses. A
            # command that takes parameters moves focus to `#arguments` (the
            # same place `on_option_list_option_selected` sends a selection);
            # one that takes none just runs, since there is nothing to type.
            cmd = self.selected_command()
            if cmd is None:
                self.set_status("no command matches")
                return
            if cmd.params:
                self.query_one("#arguments", Input).focus()
            else:
                self.run_selected("")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        # 18: show this command's real shape instead of a placeholder that
        # never changes. `usage_line` (via `PaletteEntry.usage`) already
        # renders exactly `videos add <url> [audio=…]`. A group heading never
        # reaches this handler — it is `disabled`, and Textual's own
        # `action_select` refuses to post `OptionSelected` for a disabled
        # option — but the `id` lookup is a no-op rather than an error either
        # way, so this stays correct even if that ever changes.
        arguments = self.query_one("#arguments", Input)
        option_id = event.option.id
        entry = self._entries_by_name.get(option_id) if option_id else None
        if entry is not None:
            arguments.placeholder = entry.usage
        arguments.focus()

    def on_key(self, event: events.Key) -> None:
        # 14a: focus starts in `#filter`, where Up/Down would otherwise just
        # move the text cursor and the `OptionList` — which never has focus
        # yet — never sees them. Forward the highlight move instead, the way
        # every command palette behaves.
        if event.key not in ("up", "down"):
            return
        if self.focused is not self.query_one("#filter", Input):
            return
        palette = self.query_one("#palette", OptionList)
        if event.key == "up":
            palette.action_cursor_up()
        else:
            palette.action_cursor_down()
        event.stop()
        event.prevent_default()

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
