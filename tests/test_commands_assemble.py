"""The assemble command group: registered once, used by both surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import rytp.commands.assemble as assemble_commands  # noqa: F401 - registers the commands
from rytp import constants as C
from rytp.assemble import cutlist_name, cutlist_path, read_cutlist
from rytp.cli import build_app
from rytp.commands import COMMANDS
from rytp.commands.assemble import (
    assemble_plan,
    assemble_remove,
    assemble_show,
    assemble_suggest,
    format_ms,
    orphaned_renders,
    parse_ids,
)
from rytp.db import Database
from rytp.models import InvalidInputError, NotFoundError, RytpError, utc_now_iso
from tests.assembly_corpus import add_video, add_words

runner = CliRunner()

NAMES = ("assemble.plan", "assemble.show", "assemble.suggest")


@pytest.fixture()
def corpus(db: Database) -> tuple[int, int]:
    first = add_video(db, external_id="VIDEO_A", title="A")
    add_words(db, first, "мы все понимаем")
    second = add_video(db, external_id="VIDEO_B", title="B")
    add_words(db, second, "это неизбежно")
    return first, second


def test_every_command_is_registered_in_the_assemble_group() -> None:
    for name in NAMES:
        assert name in COMMANDS
        assert COMMANDS[name].group == "assemble"
        assert COMMANDS[name].summary
        assert COMMANDS[name].long_running is False


def test_parse_ids_reads_a_comma_separated_flag() -> None:
    assert parse_ids("") == ()
    assert parse_ids(" 3, 7 ,3 ") == (3, 7)
    with pytest.raises(InvalidInputError, match="seven"):
        parse_ids("3,seven")


def test_format_ms_reads_like_a_timestamp() -> None:
    assert format_ms(0) == "0:00.000"
    assert format_ms(612_340) == "10:12.340"
    assert format_ms(3_723_004) == "1:02:03.004"


def test_plan_writes_the_file_and_reports_the_slots(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    result = assemble_plan(db, target="мы все понимаем это неизбежно", name="демо")
    assert cutlist_path("демо").exists()
    assert result.columns == ("#", "kind", "target", "source", "in", "out", "text")
    assert [row[1] for row in result.rows] == ["fragment", "fragment"]
    assert "демо" in (result.message or "")
    assert read_cutlist("демо").target == "мы все понимаем это неизбежно"


def test_plan_names_the_file_after_the_target_when_not_told(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    assemble_plan(db, target="Мы всё понимаем")
    assert cutlist_path("мы-все-понимаем").exists()


def test_plan_refuses_to_overwrite_without_force(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    assemble_plan(db, target="мы все понимаем", name="демо")
    with pytest.raises(RytpError, match="force"):
        assemble_plan(db, target="мы все понимаем", name="демо")
    assemble_plan(db, target="мы все понимаем", name="демо", force=True)


def test_plan_reports_gaps_in_the_table_and_the_message(
    db: Database, data_dir: Path
) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы дело")
    result = assemble_plan(db, target="мы дела", name="демо")
    assert [row[1] for row in result.rows] == ["fragment", "gap"]
    assert "1 word not found" in (result.message or "")


def test_plan_passes_the_controls_through(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    assemble_plan(
        db,
        target="мы все понимаем",
        name="демо",
        consistency=0.9,
        seed=4,
        pad=30,
        exclude="99",
        min_align=0.2,
    )
    params = read_cutlist("демо").params
    assert (params.consistency, params.seed, params.pad_ms) == (0.9, 4, 30)
    assert params.exclude == (99,)
    assert params.min_align_score == 0.2


def test_plan_rejects_a_knob_off_the_dial(db: Database, data_dir: Path) -> None:
    with pytest.raises(InvalidInputError, match="consistency"):
        assemble_plan(db, target="мы все", name="демо", consistency=3.0)


def test_show_reads_a_cut_list_back(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    assemble_plan(db, target="мы все понимаем это неизбежно", name="демо")
    result = assemble_show(db, name="демо")
    assert [row[1] for row in result.rows] == ["fragment", "fragment"]
    assert "2 fragments" in (result.message or "")


def test_show_says_a_dangling_video_makes_the_cut_list_unrenderable(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    """A hand-edit can leave a dangling id. Show it, and say render will refuse."""
    assemble_plan(db, target="мы все понимаем", name="демо")
    path = cutlist_path("демо")
    path.write_text(
        path.read_text(encoding="utf-8").replace("video_id = 1", "video_id = 4242"),
        encoding="utf-8",
        newline="\n",
    )
    result = assemble_show(db, name="демо")
    assert "4242" in (result.message or "")
    assert "NOT RENDERABLE" in (result.message or "")
    assert result.rows  # still shown: this is the file you are about to fix


def test_show_reports_a_broken_file_on_one_line(db: Database, data_dir: Path) -> None:
    path = cutlist_path("сломано")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('schema_version = 1\nname = "oops\n', encoding="utf-8", newline="\n")
    with pytest.raises(RytpError, match=r"сломано\.toml"):
        assemble_show(db, name="сломано")


def test_suggest_ranks_stand_ins_for_a_word(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "дело тела")
    result = assemble_suggest(db, word="дела")
    assert result.columns == ("word", "why", "distance", "occurrences", "source", "in", "out")
    assert [row[0] for row in result.rows] == ["дело", "тела"]
    assert [row[1] for row in result.rows] == ["stem", "edit"]


def test_suggest_says_so_when_nothing_is_close(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно")
    result = assemble_suggest(db, word="дела")
    assert result.rows == ()
    assert "nothing" in (result.message or "").lower()


def test_the_default_speaker_param_is_none_not_empty_string() -> None:
    """The plan's defect: '' would mean "filter for a speaker literally
    named the empty string", which is not what an unset flag means."""
    for name in ("assemble.plan", "assemble.suggest"):
        params = {param.name: param for param in COMMANDS[name].params}
        assert params["speaker"].default is None


def test_the_cli_runs_the_whole_group(data_dir: Path) -> None:
    app = build_app()
    add = runner.invoke(app, ["videos", "add", "--help"])
    assert add.exit_code == 0
    planned = runner.invoke(
        app, ["assemble", "plan", "мы все", "--name", "демо"], env={"COLUMNS": "200"}
    )
    assert planned.exit_code == 0, planned.output
    shown = runner.invoke(app, ["assemble", "show", "демо"], env={"COLUMNS": "200"})
    assert shown.exit_code == 0, shown.output


def test_the_cli_reports_a_bad_knob_on_one_line_and_exits_one(data_dir: Path) -> None:
    result = runner.invoke(
        build_app(), ["assemble", "plan", "мы все", "--consistency", "5"]
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output


def test_the_default_consistency_is_the_one_the_design_asks_for() -> None:
    params = {param.name: param for param in COMMANDS["assemble.plan"].params}
    assert params["consistency"].default == C.ASSEMBLE_DEFAULT_CONSISTENCY
    assert params["target"].positional is True


def add_render(db: Database, cutlist: str, *, state: str = "rendered") -> int:
    """A renders row, as part 6 would write it (contracts §3)."""
    cursor = db.conn.execute(
        "INSERT INTO renders (cutlist_name, output_path, canvas_mode, state, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (cutlist, "output/9/output.mp4", "pillarbox", state, utc_now_iso()),
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def test_remove_is_registered_and_is_not_long_running() -> None:
    assert "assemble.remove" in COMMANDS
    assert COMMANDS["assemble.remove"].group == "assemble"
    assert COMMANDS["assemble.remove"].long_running is False
    params = {param.name for param in COMMANDS["assemble.remove"].params}
    assert {"name", "dry_run", "yes"} <= params


def test_remove_refuses_without_yes_and_keeps_the_file(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    """Contracts §5: without --yes, a command that deletes files refuses."""
    assemble_plan(db, target="мы все понимаем", name="демо")
    with pytest.raises(RytpError, match="--yes"):
        assemble_remove(db, name="демо")
    assert cutlist_path("демо").exists()


def test_dry_run_reports_and_deletes_nothing(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    assemble_plan(db, target="мы все понимаем", name="демо")
    size = cutlist_path("демо").stat().st_size
    result = assemble_remove(db, name="демо", dry_run=True)
    assert cutlist_path("демо").exists()
    assert "демо.toml" in (result.message or "")
    assert str(size) in (result.message or "")
    assert "would" in (result.message or "").lower()


def test_dry_run_needs_no_yes(db: Database, data_dir: Path, corpus: tuple[int, int]) -> None:
    """A dry run deletes nothing, so demanding --yes for it would be noise."""
    assemble_plan(db, target="мы все понимаем", name="демо")
    assert assemble_remove(db, name="демо", dry_run=True).message


def test_remove_deletes_the_file(db: Database, data_dir: Path, corpus: tuple[int, int]) -> None:
    assemble_plan(db, target="мы все понимаем", name="демо")
    result = assemble_remove(db, name="демо", yes=True)
    assert not cutlist_path("демо").exists()
    assert "removed" in (result.message or "")


def test_removing_a_cut_list_that_is_not_there_says_so(db: Database, data_dir: Path) -> None:
    with pytest.raises(NotFoundError, match="нет-такого"):
        assemble_remove(db, name="нет-такого", yes=True)


def test_remove_rejects_a_traversing_name(db: Database, data_dir: Path) -> None:
    with pytest.raises(InvalidInputError):
        assemble_remove(db, name="../escape", yes=True)


def test_remove_names_the_renders_that_used_this_cut_list(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    """Contracts §5 says do not orphan silently; part 6 owns those rows."""
    assemble_plan(db, target="мы все понимаем", name="демо")
    render_id = add_render(db, "демо")
    result = assemble_remove(db, name="демо", yes=True)
    assert str(render_id) in (result.message or "")
    assert "keep their output" in (result.message or "")
    assert "render remove" in (result.message or "")


def test_remove_warns_but_does_not_refuse_when_renders_exist(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    """A rendered file stands on its own; losing the cut list costs
    reproducibility, not the artifact."""
    assemble_plan(db, target="мы все понимаем", name="демо")
    add_render(db, "демо")
    assemble_remove(db, name="демо", yes=True)
    assert not cutlist_path("демо").exists()


def test_remove_leaves_the_renders_table_completely_alone(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    """render.remove is part 6's. This command writes no rows at all."""
    assemble_plan(db, target="мы все понимаем", name="демо")
    render_id = add_render(db, "демо")
    assemble_remove(db, name="демо", yes=True)
    row = db.conn.execute("SELECT cutlist_name FROM renders WHERE id = ?", (render_id,)).fetchone()
    assert row is not None
    assert row["cutlist_name"] == "демо"


def test_a_dry_run_lists_the_renders_too(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    assemble_plan(db, target="мы все понимаем", name="демо")
    render_id = add_render(db, "демо", state="failed")
    result = assemble_remove(db, name="демо", dry_run=True)
    assert [row[0] for row in result.rows] == [str(render_id)]
    assert result.columns == ("render", "state", "output")


def test_orphaned_renders_finds_only_this_cut_lists_renders(db: Database) -> None:
    mine = add_render(db, "демо")
    add_render(db, "другое")
    assert [row[0] for row in orphaned_renders(db, "демо")] == [str(mine)]
    assert orphaned_renders(db, "ничего") == ()


def test_the_cli_runs_remove_end_to_end(data_dir: Path) -> None:
    app = build_app()
    assert runner.invoke(app, ["assemble", "plan", "мы все", "--name", "демо"]).exit_code == 0
    refused = runner.invoke(app, ["assemble", "remove", "демо"])
    assert refused.exit_code == 1
    assert "Traceback" not in refused.output
    assert cutlist_path("демо").exists()
    removed = runner.invoke(app, ["assemble", "remove", "демо", "--yes"])
    assert removed.exit_code == 0, removed.output
    assert not cutlist_path("демо").exists()


def test_cutlist_name_round_trips_through_plan_and_remove(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    """Whatever plan named the file, remove must accept the same name."""
    target = "Мы всё понимаем!"
    assemble_plan(db, target=target)
    assemble_remove(db, name=cutlist_name(target), yes=True)
    assert not cutlist_path(cutlist_name(target)).exists()
