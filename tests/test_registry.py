"""The one definition both surfaces are generated from (contracts §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import commands
from rytp.commands import (
    REQUIRED,
    Command,
    CommandResult,
    Param,
    cli_path,
    leaf_name,
    register,
    resolve,
)
from rytp.models import NotFoundError


@pytest.fixture()
def registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, Command]:
    """An empty COMMANDS for this test, so registrations don't leak."""
    fresh: dict[str, Command] = {}
    monkeypatch.setattr(commands, "COMMANDS", fresh)
    return fresh


def noop(db: object, **kwargs: object) -> CommandResult:
    return CommandResult(message="ok")


def make(
    name: str,
    group: str,
    *params: Param,
    long_running: bool = False,
    cli_only: bool = False,
) -> Command:
    return Command(
        name=name,
        group=group,
        summary=f"Summary for {name}.",
        params=params,
        handler=noop,
        long_running=long_running,
        cli_only=cli_only,
    )


def test_register_stores_and_returns_the_command(registry: dict[str, Command]) -> None:
    cmd = register(make("videos.list", "videos"))
    assert registry["videos.list"] is cmd
    assert resolve("videos.list") is cmd


def test_resolve_names_the_alternatives_when_it_fails(registry: dict[str, Command]) -> None:
    register(make("videos.list", "videos"))
    register(make("channel.list", "channel"))
    with pytest.raises(NotFoundError) as excinfo:
        resolve("video.lst")
    message = str(excinfo.value)
    assert "video.lst" in message
    assert "channel.list" in message
    assert "videos.list" in message


def test_a_top_level_command_has_an_empty_group(registry: dict[str, Command]) -> None:
    cmd = register(make("ingest", ""))
    assert cli_path(cmd) == ("ingest",)
    assert leaf_name(cmd) == "ingest"


def test_a_grouped_command_maps_to_two_path_segments(registry: dict[str, Command]) -> None:
    cmd = register(make("videos.add", "videos"))
    assert cli_path(cmd) == ("videos", "add")
    assert leaf_name(cmd) == "add"


def test_registering_the_same_name_twice_is_an_error(registry: dict[str, Command]) -> None:
    register(make("videos.list", "videos"))
    with pytest.raises(ValueError, match="already registered"):
        register(make("videos.list", "videos"))


def test_the_group_must_match_the_dotted_name(registry: dict[str, Command]) -> None:
    with pytest.raises(ValueError, match="group"):
        register(make("videos.list", "channel"))
    with pytest.raises(ValueError, match="group"):
        register(make("ingest", "videos"))


def test_names_are_lowercase_dotted_identifiers(registry: dict[str, Command]) -> None:
    for bad in ("Videos.List", "videos..list", "videos.", ".list", "videos list", ""):
        with pytest.raises(ValueError, match="name"):
            register(make(bad, bad.split(".")[0]))


def test_parameter_names_must_be_unique(registry: dict[str, Command]) -> None:
    with pytest.raises(ValueError, match="duplicate parameter"):
        register(
            make(
                "videos.list",
                "videos",
                Param("limit", int, "Rows.", default=50),
                Param("limit", int, "Rows again.", default=50),
            )
        )


def test_positional_parameters_come_first(registry: dict[str, Command]) -> None:
    with pytest.raises(ValueError, match="positional"):
        register(
            make(
                "videos.add",
                "videos",
                Param("title", str, "Title.", default=None),
                Param("target", str, "URL or path.", positional=True),
            )
        )


def test_a_parameter_type_must_be_one_the_surfaces_can_render(
    registry: dict[str, Command],
) -> None:
    with pytest.raises(ValueError, match="type"):
        register(make("videos.add", "videos", Param("when", dict, "Nope.", default=None)))
    register(
        make(
            "videos.add",
            "videos",
            Param("target", Path, "A path.", positional=True),
            Param("limit", int, "Rows.", default=1),
            Param("ratio", float, "A ratio.", default=1.0),
            Param("force", bool, "A flag.", default=False),
        )
    )


def test_choices_are_only_meaningful_for_strings(registry: dict[str, Command]) -> None:
    with pytest.raises(ValueError, match="choices"):
        register(
            make("videos.list", "videos", Param("kind", int, "Kind.", default=1, choices=("a",)))
        )


def test_a_short_flag_is_a_single_dash_letter(registry: dict[str, Command]) -> None:
    with pytest.raises(ValueError, match="short"):
        register(make("videos.list", "videos", Param("limit", int, "Rows.", default=1, short="n")))
    with pytest.raises(ValueError, match="short"):
        register(
            make("videos.list", "videos", Param("limit", int, "Rows.", default=1, short="--n"))
        )


def test_a_positional_parameter_may_not_carry_a_short_flag(
    registry: dict[str, Command],
) -> None:
    with pytest.raises(ValueError, match="short"):
        register(
            make("videos.add", "videos", Param("target", str, "T.", positional=True, short="-t"))
        )


def test_every_command_needs_a_summary(registry: dict[str, Command]) -> None:
    with pytest.raises(ValueError, match="summary"):
        register(
            Command(
                name="videos.list",
                group="videos",
                summary="",
                params=(),
                handler=noop,
            )
        )


def test_required_is_distinguishable_from_none(registry: dict[str, Command]) -> None:
    """`default=None` means optional-and-empty; REQUIRED means the user must supply it."""
    required = Param("target", str, "T.", positional=True)
    optional = Param("title", str, "Title.", default=None)
    assert required.default is REQUIRED
    assert optional.default is None


def test_command_result_defaults_to_empty(registry: dict[str, Command]) -> None:
    result = CommandResult()
    assert result.columns == ()
    assert result.rows == ()
    assert result.message is None


def test_long_running_defaults_to_false(registry: dict[str, Command]) -> None:
    assert register(make("videos.list", "videos")).long_running is False
    assert register(make("channel.sync", "channel", long_running=True)).long_running is True


def test_cli_only_defaults_to_false(registry: dict[str, Command]) -> None:
    """Contracts §5: a surface launcher is a registry command the TUI hides."""
    assert register(make("videos.list", "videos")).cli_only is False
    assert register(make("tui", "", cli_only=True)).cli_only is True
