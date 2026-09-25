"""The screen map (design §10).

Parts 1, 4 and 7 each contribute screens and each appended a key and an action
to the app by hand. This table is the one place a screen is declared: the app's
`BINDINGS`, the home view's navigation strip and the help screen are all
generated from it, so a screen cannot exist without being reachable, listed and
documented.

Deliberately free of Textual imports. Screen classes are loaded inside the
functions that need them, which keeps the table unit-testable and keeps
`rytp tui` from importing the search, index, diarize, assemble and queue
packages before it has drawn anything.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from rytp.models import NotFoundError

if TYPE_CHECKING:
    # Type-checking only: no Textual import at module import time.
    from textual.screen import Screen

__all__ = [
    "APP_ACTIONS",
    "BACK_KEY",
    "SCREENS",
    "ScreenEntry",
    "app_keys",
    "binding_rows",
    "help_rows",
    "home_strip",
    "load_screen_class",
    "load_screen_classes",
    "screen_binding_keys",
    "screen_by_id",
    "screen_by_key",
]

#: How every screen returns to the home view. One key, one meaning.
BACK_KEY: Final = "escape"


@dataclass(frozen=True)
class ScreenEntry:
    """One pushable screen, and everything both surfaces need to describe it."""

    key: str
    id: str
    title: str
    summary: str
    module: str
    factory: str
    #: Screens the entry screen pushes in turn, in the same module. Three of
    #: the five are pickers — a video, a video, a cut list — and every
    #: interesting binding lives on the screen they push. Listing them is what
    #: makes the collision test and the help screen see `f5` on the mapper,
    #: `ctrl+p` on the transcript reader and eleven keys on the cut-list
    #: editor; without it both would read the pickers, find nothing, and pass.
    children: tuple[str, ...] = ()
    #: True only for a screen whose default focus is not a text box, and which
    #: may therefore bind bare letters. The mapper and the search screen both
    #: focus an `Input`, so they may not.
    bare_letters: bool = False


#: design §10 lists what the TUI must offer: "browsing videos, reading
#: transcripts, searching, playing hits, editing cut lists, adding to the queue,
#: watching job progress", plus speaker assignment, which it calls the one
#: genuinely interactive task. Browsing videos is the palette itself
#: (`videos list` on the home view) or the fleet overview below (`f9`), which
#: adds pipeline-stage columns and shortcuts on top of the same listing; the
#: other five are here.
SCREENS: Final[tuple[ScreenEntry, ...]] = (
    ScreenEntry(
        key="f3",
        id="speakers",
        title="Speakers",
        summary="Name the voices in a diarized video.",
        module="rytp.tui.screens.speakers",
        factory="SpeakerVideosScreen",
        children=("SpeakerMapperScreen",),
    ),
    ScreenEntry(
        key="f4",
        id="search",
        title="Search",
        summary="Find where a word or phrase was said, and play it.",
        module="rytp.tui.screens.search",
        factory="SearchScreen",
    ),
    ScreenEntry(
        key="f6",
        id="transcripts",
        title="Transcripts",
        summary="Read one video's transcript.",
        module="rytp.tui.screens.transcript",
        factory="TranscriptVideosScreen",
        children=("TranscriptScreen",),
    ),
    ScreenEntry(
        key="f7",
        id="jobs",
        title="Queue",
        summary="Watch what is queued, running, done and failed.",
        module="rytp.tui.screens.jobs",
        factory="JobsScreen",
    ),
    ScreenEntry(
        key="f8",
        id="cutlists",
        title="Cut lists",
        summary="Swap a fragment for one of its alternatives, and fix timings.",
        module="rytp.tui.screens.cutlist",
        factory="CutlistPickerScreen",
        children=("CutlistScreen",),
        # No `Input` has focus on either cut-list screen, so bare letters are
        # free and the editing verbs read better as `s`, `a`, `g`.
        bare_letters=True,
    ),
    ScreenEntry(
        key="f9",
        id="videos",
        title="Videos",
        summary="See where each video is between registered and cuttable.",
        module="rytp.tui.screens.videos",
        factory="VideosScreen",
    ),
)

#: BUGS.md entry 23 named `f5` — free at app level, and it "sits naturally
#: before Search" — but Part 7's mapper already binds it locally
#: (`toggle_suggestions`, on `SpeakerMapperScreen`). An app-level `f5` would
#: still technically work (a screen's own bindings win over the app's), but
#: the footer would then show two different meanings for the same key on
#: that one screen, which `test_no_app_level_key_is_shadowed_by_a_screen`
#: refuses to allow. `f9` is the next unclaimed function key.

#: App-level keys that are not a screen. `f5` is absent on purpose and stays
#: absent: Part 7's mapper binds it. Nothing here says so — the test derives it
#: from the mapper's own BINDINGS, because a comment cannot fail.
APP_ACTIONS: Final[tuple[tuple[str, str, str], ...]] = (
    ("f1", "help", "Help"),
    ("f2", "focus_filter", "Filter"),
    ("ctrl+q", "quit", "Quit"),
)


def screen_by_id(screen_id: str) -> ScreenEntry:
    """Look a screen up by id, naming the alternatives when it is missing."""
    for entry in SCREENS:
        if entry.id == screen_id:
            return entry
    known = ", ".join(entry.id for entry in SCREENS)
    raise NotFoundError(f"no screen called {screen_id!r}; the TUI has: {known}")


def screen_by_key(key: str) -> ScreenEntry | None:
    """The screen a key opens, or None if the key is not a screen key."""
    return next((entry for entry in SCREENS if entry.key == key), None)


def load_screen_class(entry: ScreenEntry) -> type[Screen]:
    """Import one screen's module and return the class the key opens.

    Lazy on purpose: opening the search screen should not cost the cut-list
    parser, and starting the app should cost neither.
    """
    module = importlib.import_module(entry.module)
    return getattr(module, entry.factory)


def load_screen_classes(entry: ScreenEntry) -> tuple[type[Screen], ...]:
    """The entry screen and every screen it pushes, from the same module.

    Three of the five entries are pickers whose only binding is `escape`;
    everything a user actually presses is on the screen they push. Anything
    that reasons about bindings must look at both, or it reasons about nothing.
    """
    module = importlib.import_module(entry.module)
    names = (entry.factory, *entry.children)
    return tuple(getattr(module, name) for name in names)


def binding_rows() -> tuple[tuple[str, str, str], ...]:
    """Every app-level binding as ``(key, action, description)``.

    Plain tuples rather than Textual `Binding` objects, so this module stays
    import-light; `rytp/tui/app.py` does the one-line conversion.
    """
    rows = list(APP_ACTIONS)
    rows.extend((entry.key, f"open('{entry.id}')", entry.title) for entry in SCREENS)
    return tuple(rows)


def app_keys() -> frozenset[str]:
    """Every key bound at app level."""
    return frozenset(key for key, _action, _label in binding_rows())


def screen_binding_keys() -> dict[str, frozenset[str]]:
    """Screen id -> the keys that screen binds itself.

    Imports every screen, which is why only the tests and the help screen call
    it.
    """
    from textual.binding import Binding

    return {
        entry.id: frozenset(
            binding.key
            for screen in load_screen_classes(entry)
            for binding in Binding.make_bindings(screen.BINDINGS)
        )
        for entry in SCREENS
    }


def home_strip() -> str:
    """The one line the home view shows above the palette."""
    parts = [f"{key.upper()} {label}" for key, _action, label in APP_ACTIONS[:1]]
    parts.extend(f"{entry.key.upper()} {entry.title}" for entry in SCREENS)
    return "  ·  ".join(parts)


def help_rows() -> tuple[tuple[str, str, str], ...]:
    """Every binding in the application as ``(key, where, what)``.

    Generated, so a screen that adds a key documents itself.
    """
    from textual.binding import Binding

    rows: list[tuple[str, str, str]] = [
        (key, "anywhere", label) for key, _action, label in APP_ACTIONS
    ]
    rows.extend((entry.key, "anywhere", f"Open {entry.title}") for entry in SCREENS)
    for entry in SCREENS:
        for screen in load_screen_classes(entry):
            for binding in Binding.make_bindings(screen.BINDINGS):
                rows.append(
                    (binding.key, entry.title, binding.description or binding.action)
                )
    return tuple(rows)
