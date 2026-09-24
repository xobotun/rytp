"""Planning a render: sources, canvas, gains, gaps (design §9)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import insert_asset
from rytp.render import run as RUN
from rytp.render.ffmpeg import Tools
from tests.fakes import make_video, touch
from tests.render_fakes import RecordingRunner, loudnorm_json, probe_json
from tests.test_render_pauses import add_words, evenly_spaced

WIDE = probe_json(1280, 720)
NARROW = probe_json(640, 480)


def fragment(video_id: int, start_ms: int, end_ms: int, **kw: object) -> RUN.RenderFragment:
    return RUN.RenderFragment(
        video_id=video_id,
        first_word_ord=kw.pop("first_word_ord", 0),  # type: ignore[arg-type]
        last_word_ord=kw.pop("last_word_ord", 1),  # type: ignore[arg-type]
        start_ms=start_ms,
        end_ms=end_ms,
        text=kw.pop("text", "текст"),  # type: ignore[arg-type]
        **kw,  # type: ignore[arg-type]
    )


def source_video(db: Database, tmp_path: Path, name: str, **kw: object) -> int:
    """A catalogued video with a rendition and an audio asset on disk."""
    vid = make_video(db, external_id=name, url=f"https://example.invalid/watch/{name}", **kw)
    insert_asset(
        db, video_id=vid, role="video", format_id="720",
        path=str(touch(tmp_path / f"{name}-video.mp4")), width=1280, height=720,
    )
    insert_asset(
        db, video_id=vid, role="audio", path=str(touch(tmp_path / f"{name}-audio.m4a"))
    )
    return vid


def tools(**kw: object) -> tuple[Tools, RecordingRunner]:
    runner = RecordingRunner(  # type: ignore[arg-type]
        geometry=WIDE, loudness=loudnorm_json(-23.0), **kw
    )
    return Tools.faked(runner), runner


def test_a_cut_list_becomes_a_request_without_importing_part_five() -> None:
    cutlist = SimpleNamespace(
        name="demo",
        target="мы всё исправим",
        slots=(
            SimpleNamespace(
                kind="fragment", target_first=0, target_last=1, text="мы всё",
                video_id=3, first_word_ord=1204, last_word_ord=1205,
                start_ms=612_340, end_ms=613_100, align_score=0.81, cost=1.42,
                video_speaker_id=11, speaker_label="host", alternatives=(),
            ),
            SimpleNamespace(
                kind="gap", target_first=2, target_last=2, text="исправим",
                substitutions=(
                    SimpleNamespace(
                        text="исправит", reason="edit", distance=1, occurrences=3,
                        video_id=9, first_word_ord=502, last_word_ord=502,
                        start_ms=220_100, end_ms=220_780,
                    ),
                ),
            ),
        ),
    )
    request = RUN.request_from_cutlist(cutlist)
    assert request.name == "demo"
    assert request.target_text == "мы всё исправим"
    assert len(request.fragments) == 1
    assert request.fragments[0].speaker_label == "host"
    assert request.fragments[0].gap_before_ms is None
    assert request.missing[0].text == "исправим"
    assert request.missing[0].substitutions[0].video_id == 9
    assert request.assembled_text == "мы всё"


def test_a_missing_rendition_names_the_command_that_fetches_it(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 1_000),))
    kit, _ = tools()
    with pytest.raises(RUN.MissingMediaError) as excinfo:
        RUN.plan_render(db, request, tools=kit)
    message = str(excinfo.value)
    assert "no video rendition" in message
    assert f"rytp fetch-video {vid}" in message


def test_every_unreadable_source_is_reported_at_once(db: Database, tmp_path: Path) -> None:
    first = make_video(db, external_id="VIDEO_A")
    second = make_video(db, external_id="VIDEO_B", url="https://example.invalid/watch/VIDEO_B")
    insert_asset(db, video_id=second, role="video", format_id="720",
                 path=str(touch(tmp_path / "b.mp4")))
    request = RUN.RenderRequest(
        name="demo",
        fragments=(fragment(first, 0, 500), fragment(second, 0, 500)),
    )
    kit, _ = tools()
    with pytest.raises(RUN.MissingMediaError) as excinfo:
        RUN.plan_render(db, request, tools=kit)
    assert str(first) in str(excinfo.value)
    assert str(second) in str(excinfo.value)


def test_a_deleted_file_is_as_missing_as_a_missing_row(db: Database, tmp_path: Path) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    Path(tmp_path / "VIDEO_A-video.mp4").unlink()
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 500),))
    kit, _ = tools()
    with pytest.raises(RUN.MissingMediaError, match="no video rendition"):
        RUN.plan_render(db, request, tools=kit)


def test_a_local_container_serves_as_both_video_and_audio(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id=str(tmp_path / "clip.mkv"),
                     url=None)
    insert_asset(db, video_id=vid, role="container",
                 path=str(touch(tmp_path / "clip.mkv")))
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 500),))
    kit, _ = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    source = plan.sources[0]
    assert source.video_path == source.audio_path


def test_each_source_is_probed_exactly_once(db: Database, tmp_path: Path) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(
        name="demo",
        fragments=(fragment(vid, 0, 500), fragment(vid, 900, 1_400)),
    )
    kit, runner = tools()
    RUN.plan_render(db, request, tools=kit)
    assert len(runner.probes) == 1


def test_the_canvas_covers_every_source(db: Database, tmp_path: Path) -> None:
    first = source_video(db, tmp_path, "VIDEO_A")
    second = source_video(db, tmp_path, "VIDEO_B")
    request = RUN.RenderRequest(
        name="demo", fragments=(fragment(first, 0, 500), fragment(second, 0, 500))
    )
    runner = RecordingRunner(
        geometry={"VIDEO_A-video": WIDE, "VIDEO_B-video": NARROW},
        loudness=loudnorm_json(-23.0),
    )
    plan = RUN.plan_render(db, request, tools=Tools.faked(runner))
    assert (plan.canvas.width, plan.canvas.height) == (1280, 720)
    assert plan.canvas.fps == 25


def test_output_offsets_accumulate_the_gaps(db: Database, tmp_path: Path) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(
        name="demo",
        fragments=(fragment(vid, 0, 1_000), fragment(vid, 5_000, 5_500)),
    )
    kit, _ = tools()
    plan = RUN.plan_render(db, request, RUN.RenderOptions(gap_ms=200), tools=kit)
    first, second = plan.fragments
    assert (first.output_start_ms, first.output_end_ms) == (0, 1_000)
    assert first.gap_after_ms == 200
    assert (second.output_start_ms, second.output_end_ms) == (1_200, 1_700)
    assert second.gap_after_ms == 0  # nothing follows the last fragment
    assert plan.duration_ms == 1_700


def test_zero_gaps_switches_the_feature_off_including_hand_written_ones(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(
        name="demo",
        fragments=(
            fragment(vid, 0, 1_000),
            fragment(vid, 5_000, 5_500, gap_before_ms=750),
        ),
    )
    kit, _ = tools()
    plan = RUN.plan_render(db, request, RUN.RenderOptions(gap_ms=0), tools=kit)
    assert plan.fragments[0].gap_after_ms == 0
    assert plan.duration_ms == 1_500


def test_a_hand_written_gap_wins_over_measurement(db: Database, tmp_path: Path) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    add_words(db, vid, evenly_spaced(100, gap_ms=200))
    request = RUN.RenderRequest(
        name="demo",
        fragments=(
            fragment(vid, 0, 1_000),
            fragment(vid, 5_000, 5_500, gap_before_ms=750),
        ),
    )
    kit, _ = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    assert plan.fragments[0].gap_after_ms == 750
    assert plan.fragments[0].gap_origin == "override"


def test_measured_gaps_use_the_speaker_who_just_stopped(db: Database, tmp_path: Path) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    add_words(db, vid, evenly_spaced(100, gap_ms=240))
    request = RUN.RenderRequest(
        name="demo",
        fragments=(fragment(vid, 0, 1_000), fragment(vid, 5_000, 5_500)),
    )
    kit, _ = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    assert plan.fragments[0].gap_after_ms == 240
    assert plan.fragments[0].gap_origin == "measured"


def test_a_degenerate_distribution_falls_back_and_leaves_a_note(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    add_words(db, vid, [(i * 300, (i + 1) * 300) for i in range(120)])
    request = RUN.RenderRequest(
        name="demo",
        fragments=(fragment(vid, 0, 1_000), fragment(vid, 60_000, 60_500)),
    )
    kit, _ = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    assert plan.fragments[0].gap_after_ms == C.PAUSE_FALLBACK_GAP_MS
    assert plan.fragments[0].gap_origin == "fallback"
    assert any("zero" in note for note in plan.notes)


def test_the_gain_comes_from_video_acoustics_without_running_anything(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    db.conn.execute(
        "INSERT INTO video_acoustics (video_id, loudness_lufs, computed_at) "
        "VALUES (?, ?, '2026-01-01T00:00:00+00:00')",
        (vid, -26.0),
    )
    db.conn.commit()
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 500),))
    kit, runner = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    assert plan.sources[0].measured_lufs == pytest.approx(-26.0)
    assert plan.sources[0].gain_db == pytest.approx(10.0)
    assert runner.measurements == []


def test_an_unmeasured_source_is_scanned_once_around_its_first_fragment(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(
        name="demo",
        fragments=(fragment(vid, 600_000, 600_500), fragment(vid, 700_000, 700_500)),
    )
    kit, runner = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    assert plan.sources[0].measured_lufs == pytest.approx(-23.0)
    assert plan.sources[0].gain_db == pytest.approx(7.0)
    assert len(runner.measurements) == 1
    scan = runner.measurements[0]
    assert scan[scan.index("-ss") + 1] == "540.000"  # 120 s window, centred


def test_an_unmeasurable_source_is_left_alone_and_noted(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 500),))
    runner = RecordingRunner(geometry=WIDE, loudness='{"input_i" : "-inf"}')
    plan = RUN.plan_render(db, request, tools=Tools.faked(runner))
    assert plan.sources[0].gain_db is None
    assert any("loudness" in note for note in plan.notes)


def test_no_loudnorm_skips_every_scan(db: Database, tmp_path: Path) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 500),))
    kit, runner = tools()
    plan = RUN.plan_render(db, request, RUN.RenderOptions(loudnorm=False), tools=kit)
    assert runner.measurements == []
    assert plan.sources[0].gain_db is None


def test_the_render_id_is_deterministic_and_option_sensitive() -> None:
    request = RUN.RenderRequest(name="demo", fragments=(fragment(3, 0, 500),))
    first = RUN.derive_render_id(request, RUN.RenderOptions())
    again = RUN.derive_render_id(request, RUN.RenderOptions())
    other = RUN.derive_render_id(request, RUN.RenderOptions(canvas_mode="bbox"))
    assert first == again
    assert first.startswith("demo-")
    assert first != other
    assert RUN.derive_render_id(request, RUN.RenderOptions(render_id="mine")) == "mine"


def test_dropping_provenance_does_not_move_the_output() -> None:
    """Word ordinals never reach ffmpeg, so they are not in the hash."""
    options = RUN.RenderOptions()
    with_ords = RUN.RenderRequest(
        name="demo", fragments=(fragment(3, 0, 500, first_word_ord=12,
                                         last_word_ord=15),)
    )
    without = RUN.RenderRequest(
        name="demo", fragments=(fragment(3, 0, 500, first_word_ord=0,
                                         last_word_ord=0),)
    )
    assert RUN.derive_render_id(with_ords, options) == RUN.derive_render_id(
        without, options
    )
    moved = RUN.RenderRequest(name="demo", fragments=(fragment(3, 0, 600),))
    assert RUN.derive_render_id(moved, options) != RUN.derive_render_id(
        with_ords, options
    )


def test_a_render_id_is_a_safe_directory_name() -> None:
    request = RUN.RenderRequest(name="../не безопасно/имя", fragments=(fragment(3, 0, 5),))
    render_id = RUN.derive_render_id(request, RUN.RenderOptions())
    assert "/" not in render_id and ".." not in render_id
    assert "не" in render_id  # Cyrillic survives; only path syntax is stripped
    assert len(render_id) <= C.RENDER_ID_MAX_CHARS


def test_an_empty_cut_list_is_refused(db: Database) -> None:
    kit, _ = tools()
    with pytest.raises(RUN.RenderError, match="no fragments"):
        RUN.plan_render(db, RUN.RenderRequest(name="demo", fragments=()), tools=kit)


def test_the_plan_points_at_the_contracted_output_paths(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 500),))
    kit, _ = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    assert plan.output_path.name == "output.mp4"
    assert plan.report_path.name == "report.md"
    assert plan.output_path.parent.name == plan.render_id
    assert plan.fragments[0].intermediate.suffix == C.RENDER_INTERMEDIATE_SUFFIX
