"""One video, caption tier to aligned tier, with nothing heavy installed."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import rytp.commands.transcribe  # noqa: F401 - importing registers the commands
from rytp.commands import resolve
from rytp.db import Database
from tests.fake_engines import FakeAligner, FakeTranscriber, registered
from tests.synth_audio import concat, silence, tone, write_wav

CAPTIONS = {
    "events": [
        {
            "tStartMs": 0,
            "segs": [{"utf8": "один", "tOffsetMs": 0}, {"utf8": " два", "tOffsetMs": 200}],
        },
        {"tStartMs": 460, "segs": [{"utf8": "три", "tOffsetMs": 0}]},
    ]
}


def test_a_video_goes_from_captions_to_cuttable_words(db: Database, tmp_path: Path) -> None:
    cursor = db.conn.execute(
        "INSERT INTO videos (source, kind, title, external_id, url, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "youtube",
            "video",
            "Sample",
            "VIDEO_A",
            "https://example.invalid/v/VIDEO_A",
            "2026-09-21T00:00:00+00:00",
        ),
    )
    video_id = int(cursor.lastrowid)
    captions = tmp_path / "captions.json3"
    captions.write_text(json.dumps(CAPTIONS, ensure_ascii=False), encoding="utf-8")
    db.conn.execute(
        "INSERT INTO assets (video_id, role, path, acquired_at) VALUES (?, ?, ?, ?)",
        (video_id, "captions", str(captions), "2026-09-21T00:00:00+00:00"),
    )
    db.conn.commit()
    wav = write_wav(
        tmp_path / "audio.wav",
        concat(tone(200), silence(120, amp=0.001, seed=71), tone(200),
               silence(120, amp=0.001, seed=72), tone(200)),
    )

    # Tier 1: searchable for no GPU time, and not cuttable.
    resolve("transcribe.captions").handler(db, video=str(video_id))
    caption_rows = db.conn.execute(
        "SELECT source, end_ms FROM words WHERE video_id = ?", (video_id,)
    ).fetchall()
    assert {row[0] for row in caption_rows} == {"caption"}
    assert all(row[1] is None for row in caption_rows)

    # An aligner runs here, so promotion replaces the caption words with the
    # cuttable `aligned` tier. Without one they would land as `timed`.
    with registered(FakeTranscriber, FakeAligner):
        resolve("transcribe.run").handler(
            db, video=str(video_id), transcriber="fake", aligner="fake-aligner", wav=wav
        )
        report = tmp_path / "compare.md"
        resolve("transcribe.compare").handler(
            db,
            video=str(video_id),
            transcribers="fake",
            aligners="fake-aligner",
            end_ms=0,
            out=report,
            wav=wav,
        )
    resolve("transcribe.fingerprint").handler(db, video=str(video_id), wav=wav)

    aligned = db.conn.execute(
        "SELECT ord, start_ms, end_ms, source, engine, align_score FROM words "
        "WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[0] for row in aligned] == [0, 1, 2]
    # An aligner ran, so these are cuttable. Without one they would be `timed`.
    assert all(row[3] == "aligned" for row in aligned)
    assert all(row[2] is not None and row[2] > row[1] for row in aligned)
    assert all(row[4] == "fake+fake-aligner+energy" for row in aligned)
    assert all(row[5] is not None for row in aligned)
    for left, right in itertools.pairwise(aligned):
        assert left[2] == right[1]

    assert report.exists()
    assert db.conn.execute(
        "SELECT COUNT(*) FROM video_acoustics WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 1
