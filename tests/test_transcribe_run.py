"""Tests for ``rytp.transcribe.run`` (merger + transcribe pipeline)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

from rytp.engines import DiarSegment, DiarizedWord, Word
from rytp.transcribe.run import (
    merge_words_with_diarization,
    transcribe_video,
)


def test_merge_words_basic_overlap() -> None:
    words = [
        Word(start_ms=0, end_ms=500, text="hello", confidence=1.0),
        Word(start_ms=500, end_ms=1000, text="world", confidence=1.0),
    ]
    segments = [DiarSegment(start_ms=0, end_ms=1000, speaker="SPEAKER_00")]
    out = list(merge_words_with_diarization(words, segments))
    assert out == [
        (words[0], "SPEAKER_00"),
        (words[1], "SPEAKER_00"),
    ]


def test_merge_words_picks_largest_overlap() -> None:
    # Word straddles two segments → segment with more overlap wins.
    words = [
        Word(start_ms=300, end_ms=700, text="x", confidence=1.0),
    ]
    segments = [
        DiarSegment(start_ms=0, end_ms=400, speaker="A"),
        DiarSegment(start_ms=400, end_ms=1000, speaker="B"),
    ]
    [(w, spk)] = list(merge_words_with_diarization(words, segments))
    assert spk == "B"  # overlap = 700-400=300 > 400-300=100


def test_merge_words_no_overlap_returns_none() -> None:
    words = [Word(start_ms=2000, end_ms=2500, text="x", confidence=1.0)]
    segments = [DiarSegment(start_ms=0, end_ms=1000, speaker="A")]
    [(w, spk)] = list(merge_words_with_diarization(words, segments))
    assert spk is None


def test_merge_words_empty_inputs() -> None:
    assert list(merge_words_with_diarization([], [])) == []


def test_merge_words_unordered_segments() -> None:
    words = [Word(start_ms=0, end_ms=1000, text="x", confidence=1.0)]
    segments = [
        DiarSegment(start_ms=500, end_ms=1500, speaker="B"),
        DiarSegment(start_ms=0, end_ms=500, speaker="A"),
    ]
    # Unsorted input — the merger must still find the right speaker.
    [(w, spk)] = list(merge_words_with_diarization(words, segments))
    # A has 500ms overlap (0-500), B has 500ms overlap (500-1000) → tie,
    # B wins because we walk forward and pick the latest segment with
    # the largest overlap so far.
    assert spk in {"A", "B"}


def test_transcribe_video_unknown_id_raises(db) -> None:
    with pytest.raises(ValueError, match="video_id .* not found"):
        transcribe_video(db, video_id=99999)


def test_transcribe_video_missing_media_raises(db, fake_video_row) -> None:
    vid = db.upsert_video(fake_video_row)
    # downloaded_path is None → should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        transcribe_video(db, video_id=vid)


def test_transcribe_video_writes_words_with_null_diarizer(
    db, fake_video_row, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end transcribe with a fake STT that yields words + NullDiarizer.

    The merge step must assign ``SPEAKER_00`` to every word (because
    NullDiarizer covers the whole audio).
    """
    from rytp.transcribe import run as run_mod
    from rytp import engines

    # Build a synthetic 1-second 16kHz mono WAV so chunking has a real
    # file to probe.
    import wave

    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    audio_for_vid = tmp_path / "fake.wav"
    audio_for_vid.write_bytes(audio.read_bytes())

    # Mock extract_audio to return our pre-built file
    monkeypatch.setattr(
        run_mod,
        "extract_audio",
        lambda *a, **kw: audio_for_vid,
    )

    # Register a fake STT engine that yields two words
    class FakeSTT:
        name = "fake-stt"
        requires_hf_token = False

        def transcribe(self, audio_path, *, language=None):
            yield Word(start_ms=0, end_ms=500, text="hello", confidence=0.9)
            yield Word(start_ms=500, end_ms=1000, text="world", confidence=0.9)

    engines.register_stt(FakeSTT)
    try:
        # The video must have a downloaded_path so transcribe_video doesn't
        # trip the FileNotFoundError path before extract_audio is called.
        fake_video_row["downloaded_path"] = str(audio)
        vid = db.upsert_video(fake_video_row)

        result = transcribe_video(
            db, video_id=vid, stt_engine="fake-stt", diarizer="none"
        )

        assert result.n_words == 2
        rows = db.conn.execute(
            "SELECT text, diarizer_speaker FROM words WHERE video_id = ? ORDER BY start_ms",
            (vid,),
        ).fetchall()
        assert [r["text"] for r in rows] == ["hello", "world"]
        # NullDiarizer covers [0, MAX) so every word is SPEAKER_00
        assert all(r["diarizer_speaker"] == "SPEAKER_00" for r in rows)
    finally:
        engines.STTS.pop("fake-stt", None)


def test_transcribe_video_clears_existing_words_on_rerun(
    db, fake_video_row, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Re-running transcribe on the same video should not create duplicate words.

    This prevents the issue where running transcribe multiple times would
    result in words being repeated 2x, 3x, etc. in the transcript export.
    """
    from rytp.transcribe import run as run_mod
    from rytp import engines

    # Build a synthetic 1-second 16kHz mono WAV
    import wave

    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    audio_for_vid = tmp_path / "fake.wav"
    audio_for_vid.write_bytes(audio.read_bytes())

    # Mock extract_audio
    monkeypatch.setattr(
        run_mod,
        "extract_audio",
        lambda *a, **kw: audio_for_vid,
    )

    # Register a fake STT engine
    class FakeSTT:
        name = "fake-stt-rerun"
        requires_hf_token = False

        def transcribe(self, audio_path, *, language=None):
            yield Word(start_ms=0, end_ms=500, text="hello", confidence=0.9)
            yield Word(start_ms=500, end_ms=1000, text="world", confidence=0.9)

    engines.register_stt(FakeSTT)
    try:
        fake_video_row["downloaded_path"] = str(audio)
        vid = db.upsert_video(fake_video_row)

        # Run transcribe first time
        result1 = transcribe_video(
            db, video_id=vid, stt_engine="fake-stt-rerun", diarizer="none"
        )
        assert result1.n_words == 2

        # Run transcribe second time - should clear and re-insert
        result2 = transcribe_video(
            db, video_id=vid, stt_engine="fake-stt-rerun", diarizer="none"
        )
        assert result2.n_words == 2

        # Verify no duplicates
        rows = db.conn.execute(
            "SELECT text FROM words WHERE video_id = ? ORDER BY start_ms",
            (vid,),
        ).fetchall()
        texts = [r["text"] for r in rows]
        assert texts == ["hello", "world"], f"Expected no duplicates, got {texts}"
        assert len(rows) == 2, f"Expected 2 words, got {len(rows)}"
    finally:
        engines.STTS.pop("fake-stt-rerun", None)