"""The two tests that actually run ffmpeg. Everything else fakes it.

Inputs are synthesised here — a colour pattern and a tone, two aspect
ratios, well under a second each. Nothing is committed and nothing is
downloaded.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from rytp.db import Database
from rytp.db.queries import insert_asset
from rytp.render import run as RUN
from tests.fakes import make_video

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
pytestmark = pytest.mark.skipif(
    not HAS_FFMPEG, reason="ffmpeg and ffprobe are not both on PATH"
)


def synth_source(
    db: Database, tmp_path: Path, name: str, *, size: str, freq: int
) -> int:
    """A catalogued video whose rendition and audio are generated here."""
    video = tmp_path / f"{name}.mp4"
    audio = tmp_path / f"{name}.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i",
         f"testsrc2=size={size}:rate=25:duration=5", "-pix_fmt", "yuv420p",
         "-preset", "ultrafast", "-an", str(video)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i",
         f"sine=frequency={freq}:duration=5", "-ac", "1", str(audio)],
        check=True, capture_output=True,
    )
    vid = make_video(
        db, external_id=name, url=f"https://example.invalid/watch/{name}"
    )
    insert_asset(db, video_id=vid, role="video", format_id="src", path=str(video))
    insert_asset(db, video_id=vid, role="audio", path=str(audio))
    return vid


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def test_two_shapes_join_into_one_canvas_with_a_freeze_frame_between(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    narrow = synth_source(db, tmp_path, "VIDEO_A", size="160x120", freq=440)
    wide = synth_source(db, tmp_path, "VIDEO_B", size="256x144", freq=220)
    request = RUN.RenderRequest(
        name="smoke",
        target_text="раз два",
        fragments=(
            RUN.RenderFragment(
                video_id=narrow, first_word_ord=0, last_word_ord=0,
                start_ms=200, end_ms=600, text="раз",
            ),
            RUN.RenderFragment(
                video_id=wide, first_word_ord=0, last_word_ord=0,
                start_ms=300, end_ms=700, text="два",
            ),
        ),
    )
    result = RUN.render_cutlist(
        db,
        request,
        RUN.RenderOptions(gap_ms=150, loudnorm=False, preset="ultrafast", crf=30),
    )
    assert result.output_path.exists()
    assert result.report_path.exists()

    probed = probe(result.output_path)
    stream = probed["streams"][0]
    assert (stream["width"], stream["height"]) == (
        result.plan.canvas.width,
        result.plan.canvas.height,
    )
    # 400 ms + a 150 ms freeze + 400 ms, within a container's rounding.
    assert float(probed["format"]["duration"]) == pytest.approx(0.95, abs=0.25)
    assert "## Fragments" in result.report_path.read_text(encoding="utf-8")


def test_the_two_pass_loudnorm_survives_a_real_ffmpeg(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    """The measuring pass's JSON is the fragile part; parse a real one."""
    vid = synth_source(db, tmp_path, "VIDEO_A", size="160x120", freq=440)
    request = RUN.RenderRequest(
        name="loud",
        fragments=(
            # Four seconds, not one: EBU R128's loudness range is a
            # short-term statistic over 3 s windows, and ffmpeg can
            # legitimately answer -inf for a shorter programme — which
            # the parser reads, correctly, as "not measured".
            RUN.RenderFragment(
                video_id=vid, first_word_ord=0, last_word_ord=0,
                start_ms=500, end_ms=4_000, text="раз",
            ),
        ),
    )
    result = RUN.render_cutlist(
        db, request, RUN.RenderOptions(loudnorm=True, preset="ultrafast", crf=30)
    )
    assert result.output_path.exists()
    report = result.report_path.read_text(encoding="utf-8")
    assert "normalized" in report
    assert "could not be measured" not in report
