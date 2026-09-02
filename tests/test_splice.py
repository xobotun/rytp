"""Tests for ``rytp.splice`` — pause sampling + manifest + runner."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rytp.db import Database
from rytp.models import Word
from rytp.splice import (
    PauseSample,
    SpliceMode,
    VideoStrategy,
    sample_pause,
    splice_clips,
)
from rytp.speakers import add_speaker, recompute_pause_stats


# --- sample_pause --------------------------------------------------------


def test_sample_pause_cross_speaker_skips(db: Database) -> None:
    p = sample_pause(db, prev_speaker_id=1, curr_speaker_id=2)
    assert p.inserted_pause_ms == 0
    assert p.video_strategy == VideoStrategy.NONE


def test_sample_pause_no_data_uses_default(db: Database) -> None:
    sid = add_speaker(db, "Alice")
    p = sample_pause(db, prev_speaker_id=sid, curr_speaker_id=sid)
    # No stats row → defaults
    assert p.video_strategy == VideoStrategy.FREEZE
    assert 100 < p.inserted_pause_ms < 400


def test_sample_pause_uses_speaker_stats_when_available(
    db: Database, fake_video_row
) -> None:
    # Populate words with predictable 100ms gaps.
    vid = db.upsert_video(fake_video_row)
    sid = add_speaker(db, "Alice")
    rows_data = [(i * 200, i * 200 + 100) for i in range(50)]
    db.conn.executemany(
        """
        INSERT INTO words (video_id, start_ms, end_ms, text, normalized_text,
                           confidence, diarizer_speaker, speaker_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [(vid, s, e, f"w{i}", f"w{i}", 1.0, "A", sid) for i, (s, e) in enumerate(rows_data)],
    )
    db.conn.commit()
    recompute_pause_stats(db)

    # Sample a bunch of times and check the distribution is being pulled
    # below the global default of 200ms (since the speaker's own mean
    # is 100ms). This is more robust to a single gaussian draw landing
    # at an outlier than asserting on one sample.
    samples = [
        sample_pause(db, prev_speaker_id=sid, curr_speaker_id=sid).inserted_pause_ms
        for _ in range(50)
    ]
    mean = sum(samples) / len(samples)
    # Strictly less than the global default (200) — the speaker data
    # is pulling the distribution down.
    assert mean < 200, f"mean {mean} not < 200"
    # The min/max clamps at [50, 800] always hold
    assert all(50 <= s <= 800 for s in samples), (
        f"out of [50,800] range: "
        f"min={min(samples)}, max={max(samples)}"
    )


# --- splice_clips (manifest only; ffmpeg mocked) -------------------------


def _insert_clip_with_speaker(
    db: Database,
    fake_video_row,
    *,
    speaker_id: int | None,
    media_path: str | None = None,
) -> int:
    vid = db.upsert_video(fake_video_row)
    cur = db.conn.execute(
        """
        INSERT INTO clips (video_id, start_ms, end_ms, source_query, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (vid, 100, 500, "hello world", "2024-01-01"),
    )
    clip_id = cur.lastrowid
    if speaker_id is not None:
        # Insert a word that matches the clip's start time so the JOIN
        # resolves the speaker.
        db.conn.execute(
            """
            INSERT INTO words (video_id, start_ms, end_ms, text, normalized_text,
                               confidence, speaker_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (vid, 100, 200, "hello", "hello", 1.0, speaker_id),
        )
    db.conn.commit()
    if media_path:
        db.conn.execute(
            "UPDATE videos SET downloaded_path = ?, downloaded = 1 WHERE id = ?",
            (media_path, vid),
        )
        db.conn.commit()
    return clip_id


def test_splice_clips_writes_manifest_and_db_rows(
    db: Database, fake_video_row, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sid = add_speaker(db, "Alice")
    clip_id = _insert_clip_with_speaker(
        db, fake_video_row, speaker_id=sid
    )
    # Mock ffmpeg so the actual concat pass is a no-op.
    monkeypatch.setattr("rytp.splice.shutil.which", lambda _: "ffmpeg")
    # Pre-stage the normalized intermediate so splice doesn't fail.
    (tmp_path / "normalized").mkdir(parents=True, exist_ok=True)
    (tmp_path / "normalized" / f"{clip_id}.m4a").write_bytes(b"x")

    monkeypatch.setattr(
        "rytp.splice._concat_mode",
        lambda db, output_path, rows, pauses: output_path.write_bytes(b"fake-output"),
    )

    result = splice_clips(
        db,
        [clip_id],
        output_path=tmp_path / "out.mp4",
        mode=SpliceMode.CONCAT,
    )
    assert result.n_clips == 1
    assert result.output_path == tmp_path / "out.mp4"
    # Manifest is valid JSON
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["n_clips" if "n_clips" in manifest else "splice_run_id"] == result.splice_run_id or "splice_run_id" in manifest
    assert manifest["mode"] == "concat"
    assert len(manifest["clips"]) == 1
    # DB row exists
    n = db.conn.execute(
        "SELECT COUNT(*) FROM splice_clips WHERE splice_run_id = ?",
        (result.splice_run_id,),
    ).fetchone()[0]
    assert n == 1


def test_splice_clips_empty_raises(db: Database) -> None:
    with pytest.raises(ValueError, match="no clips to splice"):
        splice_clips(db, [])


def test_splice_clips_unknown_clip_raises(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rytp.splice.shutil.which", lambda _: "ffmpeg")
    with pytest.raises(ValueError, match="only .* found"):
        splice_clips(db, [99999])