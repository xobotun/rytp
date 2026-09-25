"""The registry, rendered for the interactive surface — no terminal involved."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp.commands import Command, CommandResult, Param
from rytp.models import InvalidInputError
from rytp.tui.palette import (
    cli_invocation,
    group_heading,
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


def test_parse_accepts_single_quoted_values() -> None:
    values = parse_arguments(SYNC, "'Channel One'")
    assert values["channel"] == "Channel One"


def test_parse_accepts_a_quoted_cyrillic_filename_with_spaces() -> None:
    values = parse_arguments(ADD, '"видео «файл» с пробелами.mp4" kind=short')
    assert values["target"] == "видео «файл» с пробелами.mp4"
    assert values["kind"] == "short"


def test_parse_preserves_an_unquoted_windows_path() -> None:
    """Entry 21: plain POSIX-mode `shlex` treats backslash as an escape
    character and would mangle this into
    `D:Workrytpsample_inputsfile.mp4` — silently, with no error."""
    windows_path = r"D:\Work\rytp\sample_inputs\file.mp4"
    values = parse_arguments(ADD, windows_path)
    assert values["target"] == windows_path


def test_parse_preserves_an_unquoted_windows_path_alongside_a_named_value() -> None:
    windows_path = r"D:\Video\Ролики\клип.mp4"
    values = parse_arguments(ADD, f"{windows_path} kind=short")
    assert values["target"] == windows_path
    assert values["kind"] == "short"


def test_parse_rejects_a_missing_required_parameter() -> None:
    with pytest.raises(InvalidInputError, match="target"):
        parse_arguments(ADD, "kind=short")


def test_parse_strips_quotes_from_a_named_values_right_hand_side() -> None:
    """The tokenizer strips a quote pair as it merges `name=` with its
    value into one token, so no separate stripping step is needed here."""
    assert parse_arguments(ADD, 'x speaker="Ведущий"')["speaker"] == "Ведущий"
    assert parse_arguments(ADD, "x speaker='Ведущий'")["speaker"] == "Ведущий"


def test_parse_strips_quotes_from_a_dashed_flags_equals_value() -> None:
    """Proves the choices check sees the stripped value, not `'short'`."""
    assert parse_arguments(ADD, "x --kind='short'")["kind"] == "short"


def test_parse_accepts_a_named_value_containing_spaces_when_quoted() -> None:
    """The regression `posix=False` introduced: a quote starting *after*
    `name=` is only ever entered from POSIX mode's mid-token quote state,
    which plain non-POSIX `shlex.split` does not have — `name="value with
    spaces"` broke there even though it worked before entry 21 was fixed."""
    assert parse_arguments(ADD, 'x speaker="Ведущий Один"')["speaker"] == "Ведущий Один"
    assert parse_arguments(ADD, "x speaker='Ведущий Один'")["speaker"] == "Ведущий Один"


def test_parse_preserves_literal_quote_characters_inside_a_value() -> None:
    """A value that is itself quoted must not be double-stripped: only the
    outer, syntactic quote pair goes away."""
    values = parse_arguments(ADD, """x speaker='He said "hi"'""")
    assert values["speaker"] == 'He said "hi"'


def test_parse_rejects_an_unbalanced_quote_without_a_traceback() -> None:
    """`shlex` raises a bare `ValueError` on an unclosed quote; that must
    not escape as a traceback (the class of defect in BUGS.md entry 16)."""
    with pytest.raises(InvalidInputError):
        parse_arguments(ADD, 'x speaker="unclosed')


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


def test_parse_accepts_a_dashed_boolean_flag() -> None:
    """Entry 24: `--force` is accepted as a synonym for `force=true`."""
    values = parse_arguments(ADD, "x --force")
    assert values["force"] is True


def test_parse_accepts_a_negated_dashed_boolean_flag() -> None:
    values = parse_arguments(ADD, "x --no-force")
    assert values["force"] is False


def test_parse_accepts_a_dashed_flag_with_equals() -> None:
    values = parse_arguments(ADD, "x --kind=short")
    assert values["kind"] == "short"


def test_parse_accepts_a_dashed_flag_with_a_separate_value() -> None:
    values = parse_arguments(ADD, "x --kind short")
    assert values["kind"] == "short"


def test_parse_accepts_a_dashed_flag_with_a_quoted_separate_value() -> None:
    values = parse_arguments(SYNC, '--channel "Channel One"')
    assert values["channel"] == "Channel One"


def test_a_dashed_flag_synonym_matches_the_named_form_exactly() -> None:
    """The example from the plan: `fetch-video 1 --captions` is
    `fetch-video 1 captions=true`."""
    named = parse_arguments(ADD, "x force=true")
    dashed = parse_arguments(ADD, "x --force")
    assert dashed == named


def test_parse_rejects_an_unknown_dashed_flag() -> None:
    with pytest.raises(InvalidInputError, match="videos add"):
        parse_arguments(ADD, "x --bogus")


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


# -- entry 6, palette half: GROUP_SUMMARIES reaches the TUI ------------------


def test_group_heading_says_what_the_group_is_about() -> None:
    """contracts §5: both surfaces read `GROUP_SUMMARIES` and describe a
    group the same way — the CLI's `_group_help` and this palette heading."""
    heading = group_heading("videos", {"videos": "the local video catalog"})
    assert heading == "videos — the local video catalog"


def test_group_heading_never_restates_the_command_list() -> None:
    """The summary says what the group is *about*; the commands underneath
    it are their own rows already, so the heading must not repeat them."""
    heading = group_heading("channel", {"channel": "the channels videos are catalogued from"})
    assert "sync" not in heading
    assert "add" not in heading


def test_group_heading_falls_back_to_the_bare_group_name() -> None:
    """A group with no entry in `GROUP_SUMMARIES` is a registration error
    the consistency suite already refuses (`commands.__init__._validate`),
    but the heading itself degrades rather than raising, since it only
    renders what it is given."""
    assert group_heading("mystery", {}) == "mystery"


def test_group_heading_reads_the_real_registry_by_default() -> None:
    """Without an explicit table, both surfaces read the same
    `GROUP_SUMMARIES` the registry ships (contracts §5)."""
    assert group_heading("videos") == "videos — the local video catalog"
