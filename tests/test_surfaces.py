"""The CLI and the TUI are generated from one dict, and this proves it.

This test is meant to fail whenever a later plan part adds a command to
one surface and not the other. That is the point of it. Part 7 asserts
the same `cli_only` behaviour for `speakers.map`, so the two must agree.
"""

from __future__ import annotations

from rytp.cli import build_app, command_paths
from rytp.commands import COMMANDS
from rytp.tui.palette import palette_entries


def cli_only_names() -> set[str]:
    """Surface launchers: on the CLI, deliberately absent from the palette."""
    return {name for name, cmd in COMMANDS.items() if cmd.cli_only}


def test_part_one_registers_the_commands_it_owns() -> None:
    """A registry that never got imported would make every assertion below vacuous.

    Parts 2, 3, 5 and 7 have since added their own groups; this assertion is
    updated to match, per this file's own docstring: it is meant to
    fail — and be updated — whenever a later plan part adds a command.
    """
    assert set(COMMANDS) == {
        "channel.add",
        "channel.list",
        "channel.remove",
        "channel.sync",
        "videos.add",
        "videos.list",
        "videos.remove",
        "doctor",
        "tui",
        "assemble.plan",
        "assemble.show",
        "assemble.suggest",
        "assemble.remove",
        "ingest",
        "fetch-video",
        "worker",
        "jobs.list",
        "jobs.stats",
        "jobs.retry",
        "jobs.cancel",
        "queue.pause",
        "queue.resume",
        "cache.prune",
        "assets.remove",
        "transcribe.captions",
        "transcribe.run",
        "transcribe.align",
        "transcribe.compare",
        "transcribe.engines",
        "transcribe.fingerprint",
        "transcribe.remove",
        "render.run",
        "render.pauses",
        "render.list",
        "render.remove",
        "index.build",
        "index.drop",
        "search.words",
        "search.play",
        "search.export",
        "transcript.build",
        "transcript.show",
        "speakers.add",
        "speakers.list",
        "speakers.alias",
        "speakers.remove",
        "speakers.labels",
        "speakers.link",
        "speakers.unlink",
        "speakers.diarize",
        "speakers.embed",
        "speakers.enqueue",
        "speakers.suggest",
        "speakers.engines",
        "speakers.map",
        "settings.list",
        "settings.get",
        "settings.set",
        "settings.unset",
    }


def test_every_entity_part_one_creates_can_also_be_removed() -> None:
    """Contracts §5: there is no entity you can create but not get rid of."""
    for group in ("channel", "videos"):
        assert f"{group}.add" in COMMANDS
        assert f"{group}.remove" in COMMANDS


def test_every_registered_command_is_reachable_from_the_cli() -> None:
    """Including the cli_only ones: the CLI hides nothing."""
    assert command_paths(build_app()) == set(COMMANDS)


def test_every_command_that_is_not_cli_only_is_reachable_from_the_tui() -> None:
    assert {entry.name for entry in palette_entries()} == set(COMMANDS) - cli_only_names()


def test_neither_surface_invents_a_command() -> None:
    cli = command_paths(build_app())
    tui = {entry.name for entry in palette_entries()}
    assert cli - tui == cli_only_names()
    assert tui - cli == set()


def test_the_tui_launcher_is_the_cli_only_command_part_one_registers() -> None:
    # Part 7 adds `speakers.map`, cli_only for the same reason `tui` is: it
    # launches an interactive surface and cannot sensibly launch itself.
    assert cli_only_names() == {"tui", "speakers.map"}


def test_the_two_surfaces_agree_on_summaries() -> None:
    for entry in palette_entries():
        assert entry.summary == COMMANDS[entry.name].summary


def test_every_command_documents_every_parameter() -> None:
    """Both surfaces show `help`; an empty one is a hole in the UI."""
    for name, cmd in COMMANDS.items():
        for param in cmd.params:
            assert param.help.strip(), f"{name}.{param.name} has no help text"


def test_the_cli_help_lists_every_group() -> None:
    from typer.testing import CliRunner

    result = CliRunner().invoke(build_app(), ["--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    for group in {cmd.group for cmd in COMMANDS.values() if cmd.group}:
        assert group in result.stdout
