"""Executing a render plan, with a fake ffmpeg (design §9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp.db import Database
from rytp.render import run as RUN
from rytp.render.ffmpeg import FfmpegFailedError, Tools
from rytp.render.report import MissingWord, SubstitutionRef
from tests.render_fakes import RecordingRunner, loudnorm_json, probe_json
from tests.test_render_plan import fragment, source_video

WIDE = probe_json(1280, 720)


def kit(**kw: object) -> tuple[Tools, RecordingRunner]:
    runner = RecordingRunner(  # type: ignore[arg-type]
        geometry=WIDE, loudness=loudnorm_json(-23.0), **kw
    )
    return Tools.faked(runner), runner


def two_fragment_request(vid: int) -> RUN.RenderRequest:
    return RUN.RenderRequest(
        name="demo",
        target_text="мы всё исправим",
        fragments=(fragment(vid, 0, 1_000, text="мы всё"),
                   fragment(vid, 5_000, 5_500, text="исправит")),
        missing=(
            MissingWord(
                text="исправим",
                position=2,
                substitutions=(
                    SubstitutionRef(
                        text="исправит", reason="edit", video_id=vid,
                        start_ms=5_000, end_ms=5_500, occurrences=3,
                    ),
                ),
            ),
        ),
    )


def test_a_render_runs_one_command_per_fragment_then_measures_then_concats(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, runner = kit()
    result = RUN.render_cutlist(db, two_fragment_request(vid), tools=tools)
    encodes = [c for c in runner.calls if "-filter_complex" in c]
    assert len(encodes) == 2
    assert len(runner.commands_containing("concat")) == 2  # measure, then join
    assert result.output_path.exists()
    assert result.fragment_count == 2


def test_each_fragment_command_carries_the_canvas_the_gap_and_the_gain(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, runner = kit()
    RUN.render_cutlist(
        db, two_fragment_request(vid), RUN.RenderOptions(gap_ms=200), tools=tools
    )
    first = next(c for c in runner.calls if "-filter_complex" in c)
    graph = first[first.index("-filter_complex") + 1]
    assert "scale=1280:720" in graph
    assert "tpad=stop_mode=clone:stop_duration=0.200" in graph
    assert "apad=pad_dur=0.200" in graph
    assert "volume=7dB" in graph  # -23 LUFS measured against a -16 target


def test_the_last_fragment_has_no_freeze_frame(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, runner = kit()
    RUN.render_cutlist(
        db, two_fragment_request(vid), RUN.RenderOptions(gap_ms=200), tools=tools
    )
    last = [c for c in runner.calls if "-filter_complex" in c][-1]
    graph = last[last.index("-filter_complex") + 1]
    assert "tpad" not in graph and "apad" not in graph


def test_the_concat_list_holds_the_intermediates_in_order(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, _ = kit()
    result = RUN.render_cutlist(
        db, two_fragment_request(vid), RUN.RenderOptions(keep_intermediates=True),
        tools=tools,
    )
    listing = (result.plan.list_file).read_text(encoding="utf-8").splitlines()
    assert listing[0].endswith("0000.mkv'")
    assert listing[1].endswith("0001.mkv'")


def test_the_report_lands_beside_the_output_and_names_everything(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, _ = kit()
    result = RUN.render_cutlist(db, two_fragment_request(vid), tools=tools)
    assert result.report_path.parent == result.output_path.parent
    text = result.report_path.read_text(encoding="utf-8")
    assert "## Fragments" in text
    assert "## Words not found" in text and "исправим" in text
    assert "## Description" in text
    assert f"video {vid}" in text


def test_a_dry_run_writes_nothing_and_runs_no_encode(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, runner = kit()
    result = RUN.render_cutlist(db, two_fragment_request(vid), tools=tools, dry_run=True)
    assert not result.output_path.exists()
    assert not result.report_path.exists()
    assert [c for c in runner.calls if "-filter_complex" in c] == []
    assert result.plan.duration_ms > 0


def test_intermediates_are_removed_on_success(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, _ = kit()
    result = RUN.render_cutlist(db, two_fragment_request(vid), tools=tools)
    assert not result.plan.fragments_dir.exists()
    assert result.output_path.exists()


def test_keep_intermediates_leaves_them_for_inspection(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, _ = kit()
    result = RUN.render_cutlist(
        db, two_fragment_request(vid), RUN.RenderOptions(keep_intermediates=True),
        tools=tools,
    )
    assert sorted(p.name for p in result.plan.fragments_dir.iterdir()) == [
        "0000.mkv", "0001.mkv"
    ]


def test_no_loudnorm_skips_both_scans_and_the_filter(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, runner = kit()
    result = RUN.render_cutlist(
        db, two_fragment_request(vid), RUN.RenderOptions(loudnorm=False), tools=tools
    )
    assert runner.measurements == []
    final = runner.calls[-1]
    assert "-af" not in final
    assert "-c:a" in final  # audio is still re-encoded; see the plan's decisions
    assert "left alone" in result.report_path.read_text(encoding="utf-8")


def test_an_unmeasurable_programme_still_renders_and_says_so(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    runner = RecordingRunner(
        geometry=WIDE,
        # Scanned in order: the programme scan is the one reading concat.txt.
        loudness={"concat.txt": '{"input_i" : "-inf"}', "-i": loudnorm_json(-23.0)},
    )
    result = RUN.render_cutlist(db, two_fragment_request(vid), tools=Tools.faked(runner))
    assert result.output_path.exists()
    assert "could not be measured" in result.report_path.read_text(encoding="utf-8")


def test_a_failed_fragment_stops_the_render_with_ffmpegs_complaint(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    runner = RecordingRunner(
        returncode=1, stderr="Invalid argument", geometry=WIDE,
        loudness=loudnorm_json(-23.0), fail_when="-filter_complex",
    )
    with pytest.raises(FfmpegFailedError, match="Invalid argument"):
        RUN.render_cutlist(db, two_fragment_request(vid), tools=Tools.faked(runner))


def test_newest_existing_prefers_the_newest_rendition_that_is_still_on_disk(
    db: Database, tmp_path: Path
) -> None:
    """design §4: upgrading a rendition is just another insert, so the
    newest row wins — but only among the ones whose file is actually there
    (contracts §3, "existence is checked"). ``assets_for`` now returns
    newest-first; this is the one caller that depends on that order, so it
    is pinned directly rather than trusted to the shared helper's own tests.
    """
    from rytp.db.queries import insert_asset
    from tests.fakes import make_video, touch

    vid = make_video(db)
    older = touch(tmp_path / "v360.mp4")
    newer = touch(tmp_path / "v720.mp4")
    insert_asset(db, video_id=vid, role="video", format_id="360", path=str(older))
    insert_asset(db, video_id=vid, role="video", format_id="720", path=str(newer))
    assert RUN._newest_existing(db, vid, "video") == newer

    newer.unlink()  # the newest rendition's file vanished outside the tool
    assert RUN._newest_existing(db, vid, "video") == older


def test_the_job_payload_carries_the_options_and_nothing_else() -> None:
    options = RUN.RenderOptions(canvas_mode="bbox", gap_ms=0, loudnorm=False, crf=24)
    payload = RUN.payload_for(options)
    assert "cutlist" not in payload  # that lives in the renders row
    assert RUN.options_from_payload(payload) == options


def test_a_render_row_opens_planned_and_closes_rendered(db: Database) -> None:
    render_id = RUN.create_render(
        db, cutlist_name="demo", options=RUN.RenderOptions(canvas_mode="bbox")
    )
    row = RUN.get_render(db, render_id)
    assert (row["cutlist_name"], row["state"], row["canvas_mode"]) == (
        "demo", "planned", "bbox"
    )
    assert row["created_at"] and row["finished_at"] is None
    RUN.finish_render(db, render_id, state="rendered", output_path=Path("/out/o.mp4"))
    row = RUN.get_render(db, render_id)
    assert row["state"] == "rendered"
    assert row["output_path"] == str(Path("/out/o.mp4"))
    assert row["finished_at"]


def test_an_unknown_render_is_a_not_found_error(db: Database) -> None:
    from rytp.models import NotFoundError

    with pytest.raises(NotFoundError, match="404404"):
        RUN.get_render(db, 404404)


def test_a_finished_render_records_its_output(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    render_id = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    tools, _ = kit()
    result = RUN.render_cutlist(
        db, two_fragment_request(vid), tools=tools, render_row_id=render_id
    )
    row = RUN.get_render(db, render_id)
    assert row["state"] == "rendered"
    assert row["output_path"] == str(result.output_path)


def test_a_failed_render_marks_its_row_failed(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    render_id = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    runner = RecordingRunner(
        returncode=1, stderr="Invalid argument", geometry=WIDE,
        loudness=loudnorm_json(-23.0), fail_when="-filter_complex",
    )
    with pytest.raises(FfmpegFailedError):
        RUN.render_cutlist(
            db, two_fragment_request(vid), tools=Tools.faked(runner),
            render_row_id=render_id,
        )
    assert RUN.get_render(db, render_id)["state"] == "failed"


def test_a_dry_run_removal_counts_the_bytes_and_deletes_nothing(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    render_id = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    tools, _ = kit()
    result = RUN.render_cutlist(
        db, two_fragment_request(vid), tools=tools, render_row_id=render_id
    )
    removal = RUN.remove_render(db, render_id, dry_run=True)
    assert removal.removed is False
    assert removal.cutlist_name == "demo"
    assert removal.total_bytes > 0
    assert any(path.endswith("output.mp4") for path, _ in removal.files)
    assert any(path.endswith("report.md") for path, _ in removal.files)
    assert result.output_path.exists()
    assert RUN.get_render(db, render_id)["id"] == render_id


def test_removal_takes_the_directory_the_row_and_any_queued_job(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    from rytp.jobs.queue import enqueue
    from rytp.models import NotFoundError

    vid = source_video(db, tmp_path, "VIDEO_A")
    render_id = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    # Enqueued while the render is still "planned" and its cut list isn't on
    # disk (this test builds a RenderRequest by hand rather than writing a
    # real cutlists/demo.toml), so the real readiness predicate parks the
    # job BLOCKED rather than resolving it SATISFIED out from under us.
    enqueue(db, "render", render_id, payload=RUN.payload_for(RUN.RenderOptions()))
    tools, _ = kit()
    result = RUN.render_cutlist(
        db, two_fragment_request(vid), tools=tools, render_row_id=render_id
    )
    removal = RUN.remove_render(db, render_id)
    assert removal.removed is True
    assert removal.jobs_cancelled == 1
    assert not result.output_path.parent.exists()
    assert db.conn.execute(
        "SELECT state FROM jobs WHERE kind = 'render' AND target_id = ?", (render_id,)
    ).fetchone()["state"] == "cancelled"
    with pytest.raises(NotFoundError):
        RUN.get_render(db, render_id)


def test_removal_notices_a_cut_list_that_is_already_gone(
    db: Database, data_dir: None
) -> None:
    """Part 5's `assemble.remove` warns and leaves the row; this is the mirror."""
    render_id = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    assert RUN.plan_removal(db, render_id).cutlist_present is False


def test_removal_refuses_to_delete_outside_the_data_tree(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    """`output_path` is text a human can edit; it is not a licence to rmtree."""
    render_id = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    elsewhere = tmp_path / "precious"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("mine", encoding="utf-8")
    db.conn.execute(
        "UPDATE renders SET output_path = ? WHERE id = ?",
        (str(elsewhere / "output.mp4"), render_id),
    )
    db.conn.commit()
    removal = RUN.remove_render(db, render_id)
    assert removal.output_dir is None
    assert removal.files == ()
    assert (elsewhere / "keep.txt").exists()  # untouched
    assert db.conn.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 0


def test_the_job_reads_its_cut_list_from_the_row(db: Database, data_dir: None) -> None:
    render_id = RUN.create_render(db, cutlist_name="ghost", options=RUN.RenderOptions())
    with pytest.raises(RUN.RenderError, match="not found"):
        RUN.run_render_job(db, render_id, RUN.payload_for(RUN.RenderOptions()))
    assert RUN.get_render(db, render_id)["state"] == "failed"
