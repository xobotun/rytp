"""Textual app for rytp.

Public surface:

* :class:`RytpApp` — the main textual App subclass.

Importing this module requires ``textual`` to be installed. The
``textual`` package is an optional runtime dep; for headless operation,
use :mod:`rytp.speakers` (the headless mapper logic) instead.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from textual.app import App
    from textual.binding import Binding
except ImportError as e:  # pragma: no cover - import guard
    raise ImportError(
        "textual is not installed. Install with `pip install rytp` "
        "(which pulls in textual as a base dependency) or, if you "
        "installed the package without its dependencies, run "
        "`pip install textual rich` to use `rytp tui`."
    ) from e

from rytp.tui.screens.speakers import SpeakersScreen
from rytp.tui.screens.videos import VideosScreen

if TYPE_CHECKING:
    from rytp.db import Database


class RytpApp(App):
    """The main rytp textual app.

    Args:
        db: an open, migrated :class:`rytp.db.Database`.
    """

    TITLE = "rytp"
    SUB_TITLE = "YouTube → transcript → datamine → splice"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("v", "show_videos", "Videos"),
        Binding("s", "show_speakers", "Speakers"),
    ]

    def __init__(self, db: "Database") -> None:
        super().__init__()
        self._db = db

    def on_mount(self) -> None:
        self.push_screen(VideosScreen(self._db))

    def action_show_videos(self) -> None:
        self.push_screen(VideosScreen(self._db))

    def action_show_speakers(self) -> None:
        self.push_screen(SpeakersScreen(self._db))


def run_tui(db: "Database") -> None:
    """Launch the textual app — convenience wrapper for the CLI."""
    RytpApp(db).run()