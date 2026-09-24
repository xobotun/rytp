"""The registry, rendered for the interactive surface — no terminal involved."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp.commands import Command, CommandResult, Param
from rytp.models import InvalidInputError
from rytp.tui.palette import (
    cli_invocation,
    match_entries,
    palette_entries,
    parse_arguments,
    usage_line,
)


def noop(db: object, **kwargs: object) -> CommandResult:
    return CommandResult()


ADD = Command(
    name="videos.add",
    group="videos",
    summary="Register one video.",
    params=(
        Param("target", str, "URL or local path.", positional=True),
        Param("kind", str, "Kind.", default=None, choices=("video", "short")),
        Param("limit", int, "Rows.", default=10),
        Param("force", bool, "Overwrite.", default=False),
        Param("out", Path, "Where to write.", default=None),
        Param("speaker", str, "Roster name.", default=None),
    ),
    handler=noop,
)

SYNC = Command(
    name="channel.sync",
    group="channel",
    summary="Re-enumerate a channel.",
    params=(Param("channel", str, "Channel URL or title.", positional=True),),
    handler=noop,
    long_running=True,
)

LAUNCHER = Command(
    name="tui",
    group="",
    summary="Open the interactive surface.",
    params=(),
    handler=noop,
    cli_only=True,
)

SAMPLE = {"videos.add": ADD, "channel.sync": SYNC, "tui": LAUNCHER}


def test_every_registered_command_becomes_one_entry() -> None:
    entries = palette_entries(SAMPLE)
    assert [entry.name for entry in entries] == ["channel.sync", "videos.add"]
    assert entries[0].long_running is True
    assert entries[1].summary == "Register one video."
    assert entries[1].group == "videos"


def test_a_cli_only_command_is_absent_from_the_palette() -> None:
    """Contracts §5: a surface launcher cannot be invoked from the surface it launches."""
    assert "tui" not in {entry.name for entry in palette_entries(SAMPLE)}


def test_usage_shows_required_and_optional_parameters_differently() -> None:
    assert usage_line(ADD) == (
        "videos add <target> [kind=…] [limit=…] [force=…] [out=…] [speaker=…]"
    )
    assert usage_line(SYNC) == "channel sync <channel>"


def test_matching_is_case_insensitive_over_name_and_summary() -> None:
    entries = palette_entries(SAMPLE)
    assert [e.name for e in match_entries(entries, "SYNC")] == ["channel.sync"]
    assert [e.name for e in match_entries(entries, "register")] == ["videos.add"]
    assert [e.name for e in match_entries(entries, "")] == [e.name for e in entries]
    assert match_entries(entries, "nothing at all") == []


def test_parse_fills_positionals_in_order_then_keywords() -> None:
    values = parse_arguments(ADD, "https://example.invalid/w/VIDEO_A kind=short limit=3")
    assert values == {
        "target": "https://example.invalid/w/VIDEO_A",
        "kind": "short",
        "limit": 3,
        "force": False,
        "out": None,
        "speaker": None,
    }


def test_parse_converts_each_declared_type() -> None:
    values = parse_arguments(ADD, "x force=yes out=a/b.mp4 limit=7")
    assert values["force"] is True
    assert values["out"] == Path("a/b.mp4")
    assert values["limit"] == 7
    assert parse_arguments(ADD, "x force=no")["force"] is False


def test_parse_accepts_quoted_values() -> None:
    values = parse_arguments(SYNC, '"Channel One"')
    assert values["channel"] == "Channel One"


def test_parse_rejects_a_missing_required_parameter() -> None:
    with pytest.raises(InvalidInputError, match="target"):
        parse_arguments(ADD, "kind=short")


def test_the_global_speaker_alias_is_accepted_when_typing_arguments() -> None:
    """Contracts §5, via PARAM_ALIASES: both spellings reach the `speaker` param."""
    assert parse_arguments(ADD, "x global_speaker=Ведущий")["speaker"] == "Ведущий"
    assert parse_arguments(ADD, "x speaker=Ведущий")["speaker"] == "Ведущий"


def test_parse_rejects_an_unknown_parameter() -> None:
    with pytest.raises(InvalidInputError, match="nonsense"):
        parse_arguments(ADD, "x nonsense=1")


def test_parse_rejects_a_value_outside_choices() -> None:
    with pytest.raises(InvalidInputError, match="must be one of"):
        parse_arguments(ADD, "x kind=opera")


def test_parse_rejects_a_value_of_the_wrong_type() -> None:
    with pytest.raises(InvalidInputError, match="limit"):
        parse_arguments(ADD, "x limit=many")


def test_parse_rejects_extra_positional_values() -> None:
    with pytest.raises(InvalidInputError, match="too many"):
        parse_arguments(SYNC, "one two")


def test_a_url_with_a_query_string_is_still_a_positional_value() -> None:
    """`?v=…` must not be mistaken for a `name=value` argument."""
    values = parse_arguments(ADD, "https://example.invalid/watch?v=VIDEO_A")
    assert values["target"] == "https://example.invalid/watch?v=VIDEO_A"


def test_cli_invocation_is_copy_pasteable() -> None:
    values = parse_arguments(ADD, "https://example.invalid/w/VIDEO_A kind=short limit=3")
    assert cli_invocation(ADD, values) == (
        "rytp videos add https://example.invalid/w/VIDEO_A --kind short --limit 3"
    )


def test_cli_invocation_renders_flags_and_omits_defaults() -> None:
    values = parse_arguments(ADD, "x force=true")
    assert cli_invocation(ADD, values) == "rytp videos add x --force"
    assert cli_invocation(SYNC, {"channel": "Channel One"}) == (
        'rytp channel sync "Channel One"'
    )
