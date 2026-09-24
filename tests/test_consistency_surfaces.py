"""Both surfaces, generated from one registry — checked, not assumed.

design §10 claims alignment "holds by construction". Nobody had ever verified
that across all seven parts at once, and the reviews that did it by hand found
a flag spelled two ways and a registry list that had gone stale before a line
of code existed. This module is that reading, turned into assertions.
"""

from __future__ import annotations

import inspect

import pytest

from rytp.commands import HEALTH_CHECKS, REQUIRED, CommandResult, HealthResult, cli_path
from rytp.db import Database
from rytp.tui.palette import palette_entries
from tests.consistency import (
    DELETION_OWNERS,
    cli_command_paths,
    load_registries,
)

COMMANDS, JOB_KINDS = load_registries()
NAMES = sorted(COMMANDS)


def test_both_registries_are_fully_populated() -> None:
    """The floor check in load_registries() ran at import; this names the counts."""
    assert len(COMMANDS) >= 50, sorted(COMMANDS)
    assert len(JOB_KINDS) >= 9, sorted(JOB_KINDS)


@pytest.mark.parametrize("name", NAMES)
def test_every_command_is_reachable_from_the_cli(name: str) -> None:
    """Including `cli_only` ones — the CLI is the surface that launches them."""
    assert cli_path(COMMANDS[name]) in cli_command_paths()


def test_the_cli_exposes_nothing_the_registry_does_not_define() -> None:
    """contracts §5: "Both surfaces are generated from this; neither may define
    a command of its own."."""
    registered = {cli_path(cmd) for cmd in COMMANDS.values()}
    assert cli_command_paths() == registered


@pytest.mark.parametrize("name", NAMES)
def test_every_command_except_cli_only_is_in_the_tui_palette(name: str) -> None:
    entries = {entry.name for entry in palette_entries(COMMANDS)}
    if COMMANDS[name].cli_only:
        assert name not in entries, (
            f"{name} is cli_only: it launches a surface and cannot be invoked "
            "from inside the surface it launches"
        )
    else:
        assert name in entries


def test_only_surface_launchers_are_cli_only() -> None:
    """A `cli_only` command is a surface launcher, not a way to hide work.

    `tui` opens the Textual app; `speakers.map` opens its two-pane mapper
    screen (`_run_mapper` -> `rytp.tui.screens.speakers.run_mapper`) rather
    than doing the mapping itself, so it qualifies for the same reason.
    Contracts §5 and the module docstrings of both `rytp/commands/__init__.py`
    and `rytp/commands/speakers.py` name this pairing explicitly.
    """
    assert {name for name in NAMES if COMMANDS[name].cli_only} == {"tui", "speakers.map"}


@pytest.mark.parametrize("group", sorted(DELETION_OWNERS))
def test_every_group_that_creates_something_can_remove_it(group: str) -> None:
    """contracts §5: "there is no entity you can create but not get rid of"."""
    assert DELETION_OWNERS[group] in COMMANDS


def test_the_deletion_table_covers_every_group_that_has_one() -> None:
    """A new group with entities must appear in the table above, or say why not.

    `queue` and `cache` hold no entities: `queue.pause` is a switch and
    `cache.prune` deletes a regenerable cache, which contracts §7 says is
    prunable by design rather than an entity with an identity.
    """
    groups = {cmd.group for cmd in COMMANDS.values() if cmd.group}
    assert groups - set(DELETION_OWNERS) == {"queue", "cache", "search", "transcript"}


@pytest.mark.parametrize("name", NAMES)
def test_the_handler_accepts_every_parameter_declared_beside_it(name: str) -> None:
    """The defect this catches is a `Param("video")` in front of a handler whose
    keyword is `video_id`: a TypeError at run time, in one command, found by
    whoever runs that command. contracts §5 fixes the calling convention —
    "Handlers take keyword arguments matching `Param.name`" — so it is checkable
    for all of them at once.
    """
    cmd = COMMANDS[name]
    signature = inspect.signature(cmd.handler)
    accepts_anything = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if accepts_anything:
        return
    keywords = {
        p.name
        for p in signature.parameters.values()
        if p.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    declared = {param.name for param in cmd.params}
    assert declared <= keywords, (
        f"{name}: declared {sorted(declared - keywords)} but the handler "
        f"{cmd.handler.__name__} accepts {sorted(keywords)}"
    )


@pytest.mark.parametrize("name", NAMES)
def test_the_handler_takes_the_database_first(name: str) -> None:
    """contracts §5: handlers "receive an open `Database` as the first positional
    argument"."""
    positional = [
        p
        for p in inspect.signature(COMMANDS[name].handler).parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional, f"{name}: handler takes no positional database argument"
    assert positional[0].name == "db", f"{name}: first argument is {positional[0].name!r}"


@pytest.mark.parametrize("name", NAMES)
def test_every_command_carries_a_summary_both_surfaces_can_show(name: str) -> None:
    summary = COMMANDS[name].summary
    assert summary.strip(), name
    assert summary[0].isupper() or summary[0].isdigit(), f"{name}: {summary!r}"
    assert summary.rstrip().endswith("."), f"{name}: {summary!r}"


@pytest.mark.parametrize("name", NAMES)
def test_every_parameter_carries_help_text(name: str) -> None:
    for param in COMMANDS[name].params:
        assert param.help.strip(), f"{name}.{param.name}"


def test_no_command_declares_a_required_parameter_after_an_optional_one() -> None:
    """Part 1's validator already refuses this per command; asserted here across
    the whole registry so a part cannot ship one behind a conditional import."""
    for name in NAMES:
        seen_optional = False
        for param in COMMANDS[name].params:
            if param.default is REQUIRED:
                assert not seen_optional, f"{name}.{param.name} is required but comes late"
            else:
                seen_optional = True


# --- health checks (contracts §5) ------------------------------------


def test_every_health_check_the_contract_names_is_registered() -> None:
    """contracts §5 assigns the checks by part; `doctor` is useless if one is
    missing and nothing says so."""
    names = set(HEALTH_CHECKS)
    for expected in ("ffmpeg", "ffprobe", "fts5", "ffplay"):
        assert expected in names, f"{expected} missing; registered: {sorted(names)}"


def test_only_the_things_that_must_work_are_required() -> None:
    """contracts §5 is explicit about which side of the line each check sits on:
    a missing optional engine reports `ok=False` and is *advisory*, never a
    reason for `doctor` to exit non-zero."""
    required = {name for name, check in HEALTH_CHECKS.items() if check.required}
    for name in ("ffmpeg", "ffprobe", "fts5"):
        assert name in required, f"{name} must be required"
    for name in ("ffplay", "yt-dlp"):
        assert name not in required, f"{name} must be advisory"
    for name, check in HEALTH_CHECKS.items():
        if name.startswith(("transcriber:", "aligner:", "diarizer:")):
            assert not check.required, f"{name} is an engine and must be advisory"


@pytest.mark.parametrize("name", sorted(HEALTH_CHECKS))
def test_no_health_check_raises_on_an_empty_database(db: Database, name: str) -> None:
    """contracts §5: "A check never raises and never blocks"."""
    result = HEALTH_CHECKS[name].run(db)
    assert isinstance(result, HealthResult)
    assert isinstance(result.ok, bool)
    assert result.detail.strip(), f"{name} reported nothing"


#: Commands that produce a table on an empty database, with no arguments
#: beyond their defaults — one per part that lists anything.
LISTABLE = (
    "channel.list",
    "videos.list",
    "jobs.list",
    "jobs.stats",
    "speakers.list",
    "render.list",
    "doctor",
)


@pytest.mark.parametrize("name", LISTABLE)
def test_a_result_is_strings_all_the_way_down(db: Database, name: str) -> None:
    """contracts §5: "Strings only. Formatting a value for display is the
    handler's job, so the CLI and the TUI cannot render the same result
    differently." A handler that leaked an `int` would render as `3` in one
    surface and crash the other's `add_row`.
    """
    cmd = COMMANDS[name]
    from rytp.tui.palette import parse_arguments

    result = cmd.handler(db, **parse_arguments(cmd, ""))
    assert isinstance(result, CommandResult)
    for column in result.columns:
        assert isinstance(column, str), (name, column)
    for row in result.rows:
        assert len(row) == len(result.columns), (name, row)
        for cell in row:
            assert isinstance(cell, str), (name, cell)
