"""The CLI is generated; this checks the generator, not any one command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rytp.commands import Command, CommandResult, Param, leaf_name
from rytp.db import Database, schema
from rytp.models import NotFoundError
from tests.test_config import child_env

runner = CliRunner()

seen: dict[str, object] = {}


def record(db: Database, **kwargs: object) -> CommandResult:
    seen.clear()
    seen.update(kwargs)
    seen["db_type"] = type(db).__name__
    return CommandResult(message="recorded")


def explode(db: Database, **kwargs: object) -> CommandResult:
    raise NotFoundError("channel 7 is not in the catalog")


def crash(db: Database, **kwargs: object) -> CommandResult:
    raise ZeroDivisionError("this is a bug, not a user error")


def tabulate(db: Database, **kwargs: object) -> CommandResult:
    return CommandResult(
        columns=("id", "title"),
        rows=(("1", "Утренний эфир"), ("2", "Короткое")),
        message="2 videos",
    )


SAMPLE = {
    "videos.add": Command(
        name="videos.add",
        group="videos",
        summary="Register one video.",
        params=(
            Param("target", str, "URL or local path.", positional=True),
            Param("kind", str, "Kind.", default=None, choices=("video", "short")),
            Param("limit", int, "Rows.", default=10, short="-n"),
            Param("ratio", float, "A ratio.", default=1.0),
            Param("force", bool, "Overwrite.", default=False),
            Param("out", Path, "Where to write.", default=None),
            Param("speaker", str, "Roster name.", default=None),
        ),
        handler=record,
    ),
    "videos.boom": Command(
        name="videos.boom",
        group="videos",
        summary="Raise a domain error.",
        params=(),
        handler=explode,
    ),
    "videos.crash": Command(
        name="videos.crash",
        group="videos",
        summary="Raise a bug.",
        params=(),
        handler=crash,
    ),
    "videos.table": Command(
        name="videos.table",
        group="videos",
        summary="Return a table.",
        params=(),
        handler=tabulate,
    ),
    "doctor": Command(
        name="doctor",
        group="",
        summary="A top-level command.",
        params=(),
        handler=record,
    ),
}


@pytest.fixture()
def app(data_dir: Path):
    from rytp.cli import build_app

    return build_app(SAMPLE)


def test_a_grouped_command_is_reachable_at_group_then_leaf(app) -> None:
    result = runner.invoke(app, ["videos", "add", "https://example.invalid/w/VIDEO_A"])
    assert result.exit_code == 0, result.output
    assert seen["target"] == "https://example.invalid/w/VIDEO_A"
    assert seen["db_type"] == "Database"


def test_a_top_level_command_is_reachable_directly(app) -> None:
    assert runner.invoke(app, ["doctor"]).exit_code == 0


def test_defaults_are_applied_and_types_converted(app) -> None:
    result = runner.invoke(
        app,
        [
            "videos",
            "add",
            "https://example.invalid/w/VIDEO_A",
            "-n",
            "3",
            "--ratio",
            "2.5",
            "--force",
            "--out",
            "somewhere.mp4",
            "--kind",
            "short",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["limit"] == 3
    assert seen["ratio"] == 2.5
    assert seen["force"] is True
    assert seen["out"] == Path("somewhere.mp4")
    assert seen["kind"] == "short"


def test_an_omitted_optional_arrives_as_none(app) -> None:
    runner.invoke(app, ["videos", "add", "https://example.invalid/w/VIDEO_A"])
    assert seen["kind"] is None
    assert seen["out"] is None
    assert seen["force"] is False
    assert seen["limit"] == 10


def test_a_missing_required_argument_is_a_usage_error(app) -> None:
    result = runner.invoke(app, ["videos", "add"])
    assert result.exit_code == 2
    assert "Missing argument" in result.stderr


def test_a_value_outside_choices_is_rejected_before_the_handler(app) -> None:
    seen.clear()
    result = runner.invoke(
        app, ["videos", "add", "https://example.invalid/w/VIDEO_A", "--kind", "opera"]
    )
    assert result.exit_code == 2
    assert "must be one of" in result.stderr
    assert seen == {}


def test_a_parameter_alias_is_accepted_under_both_spellings(app) -> None:
    """Contracts §5: `--global-speaker` is another spelling of `--speaker`."""
    assert (
        runner.invoke(app, ["videos", "add", "x", "--speaker", "Ведущий"]).exit_code == 0
    )
    assert seen["speaker"] == "Ведущий"
    assert (
        runner.invoke(
            app, ["videos", "add", "x", "--global-speaker", "Ведущий"]
        ).exit_code
        == 0
    )
    assert seen["speaker"] == "Ведущий"


def test_help_lists_the_summary_and_every_parameter(app) -> None:
    result = runner.invoke(app, ["videos", "add", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "Register one video." in result.stdout
    for flag in ("--kind", "--limit", "--ratio", "--force", "--out", "--global-speaker"):
        assert flag in result.stdout


def test_a_domain_error_is_one_line_on_stderr_and_exit_one(app) -> None:
    result = runner.invoke(app, ["videos", "boom"])
    assert result.exit_code == 1
    assert result.stderr.strip() == "channel 7 is not in the catalog"
    assert "Traceback" not in result.stderr


def test_an_unexpected_error_is_not_swallowed(app) -> None:
    result = runner.invoke(app, ["videos", "crash"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ZeroDivisionError)


def test_a_table_result_is_rendered_with_headers(app) -> None:
    result = runner.invoke(app, ["videos", "table"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines[0] == "2 videos"
    assert lines[1].split() == ["id", "title"]
    assert "Утренний эфир" in lines[3]


def test_render_result_handles_each_shape() -> None:
    from rytp.cli import render_result

    assert render_result(CommandResult()) == ""
    assert render_result(CommandResult(message="done")) == "done"
    rendered = render_result(
        CommandResult(columns=("a", "bb"), rows=(("1", "2"), ("333", "4")))
    )
    assert rendered.splitlines() == ["a    bb", "---  --", "1    2", "333  4"]


def test_render_result_wraps_a_multiline_cell_under_its_own_column() -> None:
    """BUGS.md entry 15: `transcript show --line-length` embeds newlines in
    a cell. The column width and continuation lines must follow the
    wrapped shape, not the raw string length — "whatever the table
    decides" is exactly what the flag exists to stop."""
    from rytp.cli import render_result

    rendered = render_result(
        CommandResult(
            columns=("id", "text"),
            rows=(("1", "line one\nline two"),),
        )
    )
    assert rendered.splitlines() == [
        "id  text",
        "--  --------",
        "1   line one",
        "    line two",
    ]


def test_the_database_is_created_and_migrated_on_first_use(app, data_dir: Path) -> None:
    assert not (data_dir / "rytp.db").exists()
    assert runner.invoke(app, ["doctor"]).exit_code == 0
    assert (data_dir / "rytp.db").exists()
    database = Database(data_dir / "rytp.db")
    try:
        assert database.schema_version() == schema.LATEST_VERSION
    finally:
        database.close()


def test_the_data_flag_overrides_rytp_data_for_this_run(app, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    result = runner.invoke(app, ["--data", str(elsewhere), "doctor"])
    assert result.exit_code == 0, result.output
    assert (elsewhere / "rytp.db").exists()


def test_the_cli_exposes_exactly_the_registered_commands(app) -> None:
    """The generator adds nothing of its own — `tui` is a registry entry too."""
    from rytp.cli import command_paths

    assert command_paths(app) == set(SAMPLE)


def test_importing_the_cli_creates_no_directories(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import rytp.cli"],
        cwd=str(tmp_path),
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert list(tmp_path.iterdir()) == []


def test_a_group_heading_names_what_it_is_about_and_lists_its_commands(
    data_dir: Path,
) -> None:
    """BUGS.md entry 6: "Commands in the videos group." said nothing. The
    heading must say what the group is *about* and list its commands, so the
    shape of the surface reads from `--help` alone."""
    from rytp.cli import build_app

    app = build_app(SAMPLE, group_summaries={"videos": "the local video catalog"})
    result = runner.invoke(app, ["videos", "--help"])
    assert result.exit_code == 0, result.output
    assert "the local video catalog" in result.stdout

    # The listing itself is asserted against `_group_help` rather than the
    # rendered page: rich wraps the panel at terminal width, so a joined
    # string can be split across lines, and bare leaf names are too short
    # to substring-match usefully ("add" matches half the page).
    from rytp.cli import _group_help

    help_text = _group_help(
        "videos", [SAMPLE[name] for name in sorted(SAMPLE)
                   if SAMPLE[name].group == "videos"],
        {"videos": "the local video catalog"},
    )
    # Rendered, not raw: the names carry a rich style, so comparing the
    # plain text proves both that the markup parses and that the content
    # is right. A malformed tag would raise here.
    from rich.text import Text

    plain = Text.from_markup(help_text).plain
    assert plain.startswith("the local video catalog — ")
    listing = plain.split(" — ", 1)[1]
    assert listing == ", ".join(
        leaf_name(SAMPLE[name]) for name in sorted(SAMPLE)
        if SAMPLE[name].group == "videos"
    )
    assert "videos " not in listing


def test_a_group_with_no_summary_falls_back_to_the_bare_listing(
    data_dir: Path,
) -> None:
    """The CLI itself never refuses to run over a missing entry — that is
    `_validate`'s job at registration time; here it degrades gracefully."""
    from rytp.cli import build_app

    app = build_app(SAMPLE, group_summaries={})
    result = runner.invoke(app, ["videos", "--help"])
    assert result.exit_code == 0, result.output

    # Asserted on the function, not the rendered page: leaf names alone are
    # too short to substring-match ("add" matches half the help text).
    from rich.text import Text

    from rytp.cli import _group_help

    cmds = [SAMPLE[n] for n in sorted(SAMPLE) if SAMPLE[n].group == "videos"]
    plain = Text.from_markup(_group_help("videos", cmds, {})).plain
    assert " — " not in plain                # no summary, so no dash
    assert plain == ", ".join(leaf_name(c) for c in cmds)


def test_the_real_registry_describes_every_group_it_uses(data_dir: Path) -> None:
    """No group ships with a blank heading (contracts §5, BUGS.md entry 6)."""
    from rytp.cli import build_app
    from rytp.commands import COMMANDS

    app = build_app()
    for cmd in COMMANDS.values():
        if not cmd.group:
            continue
        result = runner.invoke(app, [cmd.group, "--help"])
        assert result.exit_code == 0, result.output
        assert " — " in result.stdout.replace("\n", " "), (
            f"group {cmd.group!r} has no ' — ' separator between its "
            "description and its command list"
        )


def test_bare_python_m_rytp_prints_help_and_exits_zero(tmp_path: Path) -> None:
    """Click exits 2 for a group with no arguments; `python -m rytp` must not."""
    proc = subprocess.run(
        [sys.executable, "-m", "rytp"],
        cwd=str(tmp_path),
        env=child_env(),
        capture_output=True,
        text=True,
        # The CLI forces UTF-8 on its streams; decoding with the locale
        # encoding fails on a cp1252 console and leaves stdout as None.
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Usage" in proc.stdout
    assert list(tmp_path.iterdir()) == []
