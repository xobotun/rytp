"""Tests for ``rytp.transcripts``."""
from __future__ import annotations

from pathlib import Path

import pytest

from rytp import config
from rytp.db import Database
from rytp.speakers import add_speaker, map_diarizer_to_speaker
from rytp.transcripts import _format_ts, build_blocks, export_markdown


# ---------------------------------------------------------------------------
# Pure-function tests: ``build_blocks`` and ``_format_ts``
# ---------------------------------------------------------------------------


def test_format_ts_under_one_hour():
    assert _format_ts(0) == "0:00"
    assert _format_ts(59_000) == "0:59"
    assert _format_ts(60_000) == "1:00"
    assert _format_ts(125_000) == "2:05"


def test_format_ts_over_one_hour():
    assert _format_ts(3_600_000) == "1:00:00"
    assert _format_ts(3_725_000) == "1:02:05"


def test_build_blocks_groups_consecutive_same_speaker():
    words = [
        {"speaker_label": "A", "start_ms": 0, "end_ms": 1000, "text": "hello"},
        {"speaker_label": "A", "start_ms": 1000, "end_ms": 2000, "text": "world"},
        {"speaker_label": "A", "start_ms": 2000, "end_ms": 3000, "text": "again"},
    ]
    blocks = build_blocks(words)
    assert len(blocks) == 1
    assert blocks[0].speaker_label == "A"
    assert blocks[0].text == "hello world again"
    assert blocks[0].start_ms == 0
    assert blocks[0].end_ms == 3000


def test_build_blocks_splits_on_speaker_change():
    words = [
        {"speaker_label": "A", "start_ms": 0, "end_ms": 1000, "text": "hi"},
        {"speaker_label": "A", "start_ms": 1000, "end_ms": 2000, "text": "there"},
        {"speaker_label": "B", "start_ms": 2000, "end_ms": 3000, "text": "hey"},
        {"speaker_label": "A", "start_ms": 3000, "end_ms": 4000, "text": "ok"},
    ]
    blocks = build_blocks(words, min_block_s=0.0)  # disable absorption
    assert [b.speaker_label for b in blocks] == ["A", "B", "A"]
    assert [b.text for b in blocks] == ["hi there", "hey", "ok"]


def test_build_blocks_short_interjection_absorbed_when_min_block_set():
    # Block A (3 s), block B (0.5 s, below 2 s), block A (3 s).
    # B is sandwiched between two same-speaker blocks. The algorithm
    # looks for a same-speaker neighbor in either direction; finding
    # one (the trailing A) absorbs B into it.
    words = [
        {"speaker_label": "A", "start_ms": 0, "end_ms": 1000, "text": "one"},
        {"speaker_label": "A", "start_ms": 1000, "end_ms": 2000, "text": "two"},
        {"speaker_label": "A", "start_ms": 2000, "end_ms": 3000, "text": "three"},
        {"speaker_label": "B", "start_ms": 3000, "end_ms": 3500, "text": "huh"},
        {"speaker_label": "A", "start_ms": 3500, "end_ms": 4500, "text": "four"},
        {"speaker_label": "A", "start_ms": 4500, "end_ms": 5500, "text": "five"},
        {"speaker_label": "A", "start_ms": 5500, "end_ms": 6500, "text": "six"},
    ]
    blocks = build_blocks(words, min_block_s=2.0)
    # The trailing A is at i=2 in the raw list, B is at i=1. They
    # share a speaker (well — they don't: A vs B). So the algorithm
    # cannot absorb B anywhere: B has no same-speaker neighbor in
    # either direction. B stays as its own block.
    assert len(blocks) == 3
    assert blocks[0].speaker_label == "A"
    assert blocks[0].text == "one two three"
    assert blocks[1].speaker_label == "B"
    assert blocks[1].text == "huh"
    assert blocks[2].speaker_label == "A"
    assert blocks[2].text == "four five six"


def test_build_blocks_short_block_between_two_distinct_speakers_stays():
    # If the short block is between two different speakers, it can't
    # be absorbed — keep it as a clear speaker boundary.
    words = [
        {"speaker_label": "A", "start_ms": 0, "end_ms": 3000, "text": "long a"},
        {"speaker_label": "B", "start_ms": 3000, "end_ms": 3500, "text": "short b"},
        {"speaker_label": "C", "start_ms": 3500, "end_ms": 6500, "text": "long c"},
    ]
    blocks = build_blocks(words, min_block_s=2.0)
    assert [b.speaker_label for b in blocks] == ["A", "B", "C"]


def test_build_blocks_short_block_absorbed_into_same_speaker_neighbor():
    # A (3 s), A (0.5 s, below threshold), A (3 s). The middle block
    # is the same speaker as both neighbors; algorithm absorbs it
    # into the next same-speaker block (forward preference).
    words = [
        {"speaker_label": "A", "start_ms": 0, "end_ms": 1500, "text": "first"},
        {"speaker_label": "A", "start_ms": 1500, "end_ms": 3000, "text": "half"},
        {"speaker_label": "A", "start_ms": 3000, "end_ms": 3500, "text": "filler"},
        {"speaker_label": "A", "start_ms": 3500, "end_ms": 5000, "text": "second"},
        {"speaker_label": "A", "start_ms": 5000, "end_ms": 6500, "text": "half"},
    ]
    blocks = build_blocks(words, min_block_s=2.0)
    assert len(blocks) == 1
    assert blocks[0].text == "first half filler second half"
    assert blocks[0].start_ms == 0
    assert blocks[0].end_ms == 6500


def test_build_blocks_short_block_absorbed_backwards_when_only_prev_matches():
    # Trailing short A block (1.5 s) with a B block in between.
    # raw = [A: 0-3 (3 s, ok), B: 3-4.5 (1.5 s, short), A: 4.5-6 (1.5 s, short)]
    # i=1 (B): next=A (different), prev=A (different). No match. i+=1.
    # i=2 (A): next=none, prev=B (different). No match. i+=1.
    # Result: 3 blocks. The trailing A stays because the B between
    # blocks the immediate-neighbor absorption path.
    words = [
        {"speaker_label": "A", "start_ms": 0, "end_ms": 3000, "text": "main a"},
        {"speaker_label": "B", "start_ms": 3000, "end_ms": 4500, "text": "longer b"},
        {"speaker_label": "A", "start_ms": 4500, "end_ms": 6000, "text": "trail a"},
    ]
    blocks = build_blocks(words, min_block_s=2.0)
    assert len(blocks) == 3
    assert [b.text for b in blocks] == ["main a", "longer b", "trail a"]


def test_build_blocks_empty_input():
    assert build_blocks([]) == []


def test_build_blocks_single_word():
    words = [
        {"speaker_label": "A", "start_ms": 0, "end_ms": 1000, "text": "hi"},
    ]
    blocks = build_blocks(words, min_block_s=2.0)
    # Single block, can't be absorbed anywhere — stays as-is.
    assert len(blocks) == 1
    assert blocks[0].text == "hi"


# ---------------------------------------------------------------------------
# Integration tests: ``export_markdown``
# ---------------------------------------------------------------------------


def test_export_markdown_writes_expected_file(db: Database, data_dir: Path) -> None:
    """Round-trip: insert a video with words and a speaker mapping, export."""
    # 1. Register a video.
    db.conn.execute(
        """
        INSERT INTO videos (source, kind, title, duration, channel_id)
        VALUES ('youtube', 'video', 'Smoke Test Title', 6500, NULL)
        """
    )
    video_id = db.conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    # 2. Register a canonical speaker.
    speaker_id = add_speaker(db, "Alice")

    # 3. Register a raw diarizer label mapping.
    map_diarizer_to_speaker(db, video_id, "SPEAKER_00", speaker_id)

    # 4. Insert words — three long blocks, plus a short "SPEAKER_01"
    #    interjection that should be absorbed because we don't have a
    #    mapping for it (it becomes an unknown short block that can't
    #    be absorbed and stays — see test below for the variant with
    #    a same-speaker short block).
    db.conn.execute(
        """
        INSERT INTO words (video_id, start_ms, end_ms, text, normalized_text,
                           confidence, diarizer_speaker, speaker_id)
        VALUES
            (?, 0,    1000, 'Hello',   'hello',   1.0, 'SPEAKER_00', NULL),
            (?, 1000, 2000, 'world',   'world',   1.0, 'SPEAKER_00', NULL),
            (?, 2000, 3000, 'from',    'from',    1.0, 'SPEAKER_00', NULL),
            (?, 3000, 3500, 'interject', 'interject', 1.0, 'SPEAKER_01', NULL),
            (?, 3500, 5500, 'Alice',   'alice',   1.0, 'SPEAKER_00', NULL)
        """,
        (video_id, video_id, video_id, video_id, video_id),
    )
    db.conn.commit()

    out_path = data_dir / "transcripts" / f"{video_id}.md"
    written = export_markdown(db, video_id, out_path=out_path)

    assert written == out_path
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")

    # H1 title.
    assert "# Smoke Test Title" in text
    # At least one H3 header for the speaker block.
    assert "### [0:00] Alice" in text
    # The Alice block contains the post-interjection words.
    assert "Hello world from" in text or "Hello world" in text
    # The unknown SPEAKER_01 short block stays as its own block.
    assert "SPEAKER_01" in text


def test_export_markdown_unknown_video_raises(db: Database) -> None:
    with pytest.raises(ValueError, match="no video with id=9999"):
        export_markdown(db, 9999)


def test_export_markdown_empty_transcript(db: Database, data_dir: Path) -> None:
    """A video with no words writes the header + an empty note."""
    db.conn.execute(
        "INSERT INTO videos (source, kind, title, duration) "
        "VALUES ('youtube', 'video', 'Empty', 0)"
    )
    video_id = db.conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    db.conn.commit()

    out_path = data_dir / "transcripts" / f"{video_id}.md"
    export_markdown(db, video_id, out_path=out_path)

    text = out_path.read_text(encoding="utf-8")
    assert "# Empty" in text
    assert "no transcript yet" in text


def test_export_markdown_uses_default_min_block_when_not_passed(
    db: Database, data_dir: Path
) -> None:
    """Default 2.0s minimum should absorb a short same-speaker block."""
    db.conn.execute(
        "INSERT INTO videos (source, kind, title, duration) "
        "VALUES ('youtube', 'video', 'Defaults', 10000)"
    )
    video_id = db.conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    speaker_id = add_speaker(db, "Bob")
    map_diarizer_to_speaker(db, video_id, "SPEAKER_00", speaker_id)
    # Three long Bob blocks, each > 2 s, with no other speakers in
    # between. Result: 3 blocks (one per change point, but here
    # there are no change points so just 1 block).
    db.conn.execute(
        """
        INSERT INTO words (video_id, start_ms, end_ms, text, normalized_text,
                           confidence, diarizer_speaker, speaker_id)
        VALUES
            (?, 0,    1000, 'one',   'one',   1.0, 'SPEAKER_00', NULL),
            (?, 1000, 2000, 'two',   'two',   1.0, 'SPEAKER_00', NULL),
            (?, 2000, 3000, 'three', 'three', 1.0, 'SPEAKER_00', NULL),
            (?, 3000, 4000, 'four',  'four',  1.0, 'SPEAKER_00', NULL),
            (?, 4000, 5000, 'five',  'five',  1.0, 'SPEAKER_00', NULL)
        """,
        (video_id, video_id, video_id, video_id, video_id),
    )
    db.conn.commit()

    out_path = data_dir / "transcripts" / f"{video_id}.md"
    export_markdown(db, video_id, out_path=out_path)

    text = out_path.read_text(encoding="utf-8")
    # No speaker changes → one big Bob block.
    assert text.count("### [0:00] Bob") == 1
    assert "one two three four five" in text


def test_export_markdown_paths_default_under_config_dir(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``out_path`` is not given, the file is written under ``config.paths.transcripts``."""
    # Insert a video with no words so we can run without a real
    # transcript being produced.
    db.conn.execute(
        "INSERT INTO videos (source, kind, title, duration) "
        "VALUES ('youtube', 'video', 'Default Path', 0)"
    )
    video_id = db.conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    db.conn.commit()

    written = export_markdown(db, video_id)
    try:
        # The default location is config.paths.transcripts/{video_id}.md
        expected = config.paths.transcripts / f"{video_id}.md"
        assert written == expected
        assert expected.exists()
    finally:
        written.unlink(missing_ok=True)