"""The render commands: registered once, used by both surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest

import rytp.commands.render  # noqa: F401 - importing registers the commands
from rytp.commands import COMMANDS, resolve
from rytp.db import Database
from rytp.models import RytpError
from rytp.render import run as RUN
from rytp.render.ffmpeg import Tools
from tests.render_fakes import RecordingRunner, loudnorm_json, probe_json
from tests.test_render_pauses import add_speaker, add_words, evenly_spaced
from tests.test_render_plan import fragment, source_video


def test_every_command_is_registered_in_the_render_group() -> None:
    for name in ("render.run", "render.pauses", "render.list", "render.remove"):
        assert name in COMMANDS
        assert COMMANDS[name].group == "render"
        assert COMMANDS[name].summary


def test_render_run_is_marked_long_running() -> None:
    assert COMMANDS["render.run"].long_running is True


def test_render_run_exposes_every_design_knob() -> None:
    names = {p.name for p in COMMANDS["render.run"].params}
    assert {
        "name", "canvas", "height", "fps", "gap_ms", "loudnorm",
        "preset", "crf", "keep_intermediates", "render_name", "dry_run", "enqueue",
    } <= names
    canvas = next(p for p in COMMANDS["render.run"].params if p.name == "canvas")
    assert canvas.choices == ("16:9", "bbox")


def test_render_run_renders_and_reports(
    db: Database, tmp_path: Path, data_dir: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(
        name="demo", target_text="мы всё",
        fragments=(fragment(vid, 0, 1_000, text="мы всё"),),
    )
    runner = RecordingRunner(geometry=probe_json(1280, 720),
                             loudness=loudnorm_json(-23.0))
    monkeypatch.setattr(
        "rytp.commands.render._load_request", lambda _name: request
    )
    monkeypatch.setattr(
        "rytp.commands.render._tools", lambda: Tools.faked(runner)
    )
    result = resolve("render.run").handler(db, name="demo")
    assert result.rows
    assert "output.mp4" in " ".join(result.rows[0])
    assert Path(result.rows[0][1]).exists()


def test_render_run_dry_run_writes_nothing(
    db: Database, tmp_path: Path, data_dir: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 1_000),))
    runner = RecordingRunner(geometry=probe_json(1280, 720),
                             loudness=loudnorm_json(-23.0))
    monkeypatch.setattr("rytp.commands.render._load_request", lambda _name: request)
    monkeypatch.setattr("rytp.commands.render._tools", lambda: Tools.faked(runner))
    result = resolve("render.run").handler(db, name="demo", dry_run=True)
    assert "dry run" in (result.message or "")
    assert not Path(result.rows[0][1]).exists()


def test_render_run_can_queue_instead_of_rendering(
    db: Database, data_dir: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "rytp.commands.render._load_request",
        lambda _name: RUN.RenderRequest(name="demo", fragments=(fragment(1, 0, 10),)),
    )
    result = resolve("render.run").handler(db, name="demo", enqueue=True)
    assert "queued" in (result.message or "")
    render_row = db.conn.execute("SELECT id, cutlist_name, state FROM renders").fetchone()
    assert (render_row["cutlist_name"], render_row["state"]) == ("demo", "planned")
    job = db.conn.execute("SELECT kind, target_id, pool FROM jobs").fetchone()
    assert job["kind"] == "render"
    assert job["target_id"] == render_row["id"]
    assert job["pool"] == "cpu"


def test_render_list_shows_what_has_been_made(db: Database) -> None:
    first = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    RUN.finish_render(db, first, state="rendered", output_path=Path("/out/demo/o.mp4"))
    RUN.create_render(db, cutlist_name="другой", options=RUN.RenderOptions())
    result = resolve("render.list").handler(db)
    assert len(result.rows) == 2
    assert result.rows[0][1] == "другой"  # newest first
    assert "rendered" in " ".join(result.rows[1])


def test_an_unknown_cut_list_is_a_one_line_error(db: Database, data_dir: None) -> None:
    with pytest.raises(RytpError, match="demo"):
        resolve("render.run").handler(db, name="demo")


def test_render_pauses_reports_per_speaker_and_per_video(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    label = add_speaker(db, vid, label="host")
    add_words(db, vid, evenly_spaced(80, gap_ms=220), video_speaker_id=label)
    result = resolve("render.pauses").handler(db, video=str(vid))
    flat = [" ".join(row) for row in result.rows]
    assert any("host" in row for row in flat)
    assert any("220" in row for row in flat)
    assert any("video" in row for row in flat)


def test_render_pauses_flags_a_degenerate_distribution(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    add_words(db, vid, [(i * 300, (i + 1) * 300) for i in range(120)])
    result = resolve("render.pauses").handler(db, video=str(vid))
    assert any("zero" in " ".join(row) for row in result.rows)


def test_render_pauses_needs_something_to_look_at(db: Database) -> None:
    with pytest.raises(RytpError, match="--video"):
        resolve("render.pauses").handler(db)


def _render_with_output(db: Database, *, name: str = "demo", size: int = 2_048) -> int:
    from rytp.config import paths

    render_id = RUN.create_render(db, cutlist_name=name, options=RUN.RenderOptions())
    output_dir = paths().output_dir(f"{name}-1a2b3c4d")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "output.mp4").write_bytes(b"\x00" * size)
    (output_dir / "report.md").write_text("# Render", encoding="utf-8")
    db.conn.execute(
        "UPDATE renders SET output_path = ? WHERE id = ?",
        (str(output_dir / "output.mp4"), render_id),
    )
    db.conn.commit()
    return render_id


def test_render_remove_refuses_without_yes(db: Database, data_dir: None) -> None:
    render_id = _render_with_output(db)
    with pytest.raises(RytpError, match="--yes"):
        resolve("render.remove").handler(db, render_id=render_id)
    assert db.conn.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 1


def test_render_remove_dry_run_shows_the_files_and_the_cut_list(
    db: Database, data_dir: None
) -> None:
    render_id = _render_with_output(db)
    result = resolve("render.remove").handler(db, render_id=render_id, dry_run=True)
    assert "would remove" in (result.message or "")
    assert "demo" in (result.message or "")
    assert len(result.rows) == 2
    assert any("2,048" in cell for row in result.rows for cell in row)
    assert db.conn.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 1


def test_render_remove_with_yes_takes_the_directory_and_the_row(
    db: Database, data_dir: None
) -> None:
    from rytp.config import paths

    render_id = _render_with_output(db)
    result = resolve("render.remove").handler(db, render_id=render_id, yes=True)
    assert "removed render" in (result.message or "")
    assert not paths().output_dir("demo-1a2b3c4d").exists()
    assert db.conn.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 0
