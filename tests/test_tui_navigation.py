"""One table of screens, and the collision check that reads the real code.

design §10 wants the TUI "aligned with" the CLI. Alignment between screens is
a separate problem, and it was solved so far by three plan authors reading each
other. This module is that reading, automated.
"""

from __future__ import annotations

import re

import pytest
from textual.screen import Screen

from rytp.tui import navigation as N

ENTRIES = list(N.SCREENS)
IDS = [entry.id for entry in ENTRIES]


def test_every_screen_is_named_once() -> None:
    assert len(IDS) == len(set(IDS))
    assert set(IDS) == {
        "speakers", "search", "transcripts", "jobs", "cutlists", "videos",
    }


def test_every_screen_has_a_title_and_a_summary() -> None:
    for entry in ENTRIES:
        assert entry.title.strip(), entry.id
        assert entry.summary.strip().endswith("."), entry.id


def test_no_key_is_claimed_twice_at_app_level() -> None:
    keys = [key for key, _action, _label in N.APP_ACTIONS] + [e.key for e in ENTRIES]
    assert len(keys) == len(set(keys)), sorted(keys)


def test_app_level_keys_are_function_keys_or_control_chords() -> None:
    """The palette filter has focus on the home view and must keep receiving
    typed text, so no app-level key may be a bare letter."""
    for key in N.app_keys():
        assert re.fullmatch(r"f\d{1,2}|ctrl\+\w+", key), key


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_every_named_screen_really_exists(entry: N.ScreenEntry) -> None:
    for cls in N.load_screen_classes(entry):
        assert isinstance(cls, type)
        assert issubclass(cls, Screen), f"{entry.module}: {cls}"


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_every_screen_goes_back_the_same_way(entry: N.ScreenEntry) -> None:
    """Rule 2 of the binding scheme: home is the hub, escape returns to it."""
    for cls in N.load_screen_classes(entry):
        keys = {binding.key for binding in cls.BINDINGS}
        assert N.BACK_KEY in keys, f"{cls.__name__} has no {N.BACK_KEY} binding"


def test_the_pushed_screens_are_the_ones_with_the_bindings() -> None:
    """Three of the five entries are pickers. `f5` is on `SpeakerMapperScreen`,
    not on `SpeakerVideosScreen`, and eleven cut-list keys are on
    `CutlistScreen`, not on the picker. A collision test that only read the
    entry screens would find nothing and pass, which is worse than no test."""
    keys = N.screen_binding_keys()
    assert "f5" in keys["speakers"], "the mapper's suggestions key is invisible"
    assert "ctrl+s" in keys["cutlists"], "the editor's save key is invisible"
    assert len(keys["cutlists"]) >= 10


def test_one_key_means_one_thing_wherever_it_is_bound() -> None:
    """The `-s` finding, indoors. A key bound on more than one screen must
    describe the same action, or muscle memory is wrong half the time."""
    meanings: dict[str, set[str]] = {}
    for entry in ENTRIES:
        for cls in N.load_screen_classes(entry):
            for binding in cls.BINDINGS:
                meanings.setdefault(binding.key, set()).add(
                    (binding.description or binding.action).lower()
                )
    clashes = {key: sorted(what) for key, what in meanings.items() if len(what) > 1}
    assert not clashes, f"one key, several meanings: {clashes}"


def test_the_videos_screen_is_f9_not_f5() -> None:
    """BUGS.md entry 23 named `f5` as the natural key, but Part 7's mapper
    already binds it locally — see `SCREENS`'s comment. `f9` is next."""
    entry = N.screen_by_id("videos")
    assert entry.key == "f9"
    assert "f5" not in N.app_keys()


def test_f5_really_would_collide_with_the_mapper() -> None:
    """The reasoning `SCREENS` records, checked against the real code rather
    than trusted as a comment: if `f5` were an app-level key it would be
    shadowed on the speakers screen, exactly as
    `test_no_app_level_key_is_shadowed_by_a_screen` would refuse."""
    assert "f5" in N.screen_binding_keys()["speakers"]


def test_no_app_level_key_is_shadowed_by_a_screen() -> None:
    """The reason `f5` is not an app-level key, derived rather than declared.

    Part 7's mapper binds `f5` to toggle suggestions. An app-level `f5` would
    still work — screen bindings win — but the footer would show two meanings
    for one key, which is worse than not having the key. A hand-maintained
    reserved list would go stale the moment a screen changed a binding; this
    reads the screens.
    """
    app = N.app_keys() - {N.BACK_KEY}
    clashes = {
        screen_id: sorted(app & keys)
        for screen_id, keys in N.screen_binding_keys().items()
        if app & keys
    }
    assert not clashes, (
        f"app-level keys shadowed by a screen: {clashes}. Move the app-level "
        f"binding; the screen's key is the more specific one."
    )


def test_a_screen_with_a_text_box_binds_no_bare_letter() -> None:
    """Part 7's rule, kept and scoped. The speaker mapper and the search screen
    both focus an `Input` by default; a bare `a` there would eat a keystroke
    the user meant to type."""
    for entry in ENTRIES:
        if entry.bare_letters:
            continue
        keys = N.screen_binding_keys()[entry.id]
        bare = sorted(key for key in keys if re.fullmatch(r"[a-z]", key))
        assert not bare, f"{entry.id} binds bare letters {bare} but has a focused Input"


def test_the_bindings_the_app_uses_come_from_this_table() -> None:
    rows = N.binding_rows()
    assert ("f1", "help", "Help") in rows
    assert ("ctrl+q", "quit", "Quit") in rows
    for entry in ENTRIES:
        assert (entry.key, f"open('{entry.id}')", entry.title) in rows


def test_the_home_strip_names_every_screen_and_its_key() -> None:
    strip = N.home_strip()
    for entry in ENTRIES:
        assert entry.key.upper() in strip, entry.id
        assert entry.title in strip, entry.id
    assert "F1" in strip


def test_help_covers_the_app_bindings_and_every_screen_binding() -> None:
    rows = N.help_rows()
    keys = {(key, where) for key, where, _what in rows}
    for key, _action, _label in N.APP_ACTIONS:
        assert (key, "anywhere") in keys, key
    for entry in ENTRIES:
        assert (entry.key, "anywhere") in keys, entry.key
        for screen_key in N.screen_binding_keys()[entry.id]:
            assert (screen_key, entry.title) in keys, (entry.id, screen_key)


def test_looking_a_screen_up_by_a_name_that_is_not_one_is_an_error() -> None:
    from rytp.models import NotFoundError

    with pytest.raises(NotFoundError, match="nonsense"):
        N.screen_by_id("nonsense")


def test_the_table_imports_no_textual_and_no_screen() -> None:
    """Keeping the map import-light is what lets the help screen list every
    binding without the app importing five workflow packages to start."""
    import subprocess
    import sys

    from tests.consistency import repo_root

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rytp.tui.navigation; "
            "print(sorted(m for m in sys.modules if m.startswith('rytp.tui.screens')))",
        ],
        cwd=str(repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout
