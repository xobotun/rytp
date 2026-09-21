"""Integration test: the user's real downloaded files.

The repo ships ``sample_youtube_id.f311.mp4`` (video) and
``sample_youtube_id.f251-1.webm`` (audio) — split files from an
``yt-dlp -f "worstvideo[height=720]+bestaudio[language=ru]"``
download. This test exercises the full pipeline against those files:

1. Register them as a single local video with separate audio
   (``register_local_video_with_separate_audio``).
2. ``extract_audio`` reads from the audio webm and writes a 16 kHz
   mono WAV (skipping the video decode).
3. ``transcribe_video`` runs a fake STT over the extracted WAV and
   verifies the words are written to the DB with the expected
   diarizer speaker.

Skipped if the user files aren't present (so the test stays
runnable in a clean checkout) or if ffmpeg isn't on PATH.
"""
from __future__ import annotations

import shutil
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_VIDEO = REPO_ROOT / "sample_youtube_id.f311.mp4"
REAL_AUDIO = REPO_ROOT / "sample_youtube_id.f251-1.webm"


pytestmark = pytest.mark.skipif(
    not (REAL_VIDEO.exists() and REAL_AUDIO.exists() and shutil.which("ffmpeg")),
    reason="requires the bundled real files and ffmpeg on PATH",
)


def test_real_files_extract_audio_from_webm(
    db, data_dir: Path, tmp_path: Path
) -> None:
    """End-to-end extract_audio reads the audio webm, not the video mp4."""
    from rytp.channels import register_local_video_with_separate_audio
    from rytp.transcribe.extract import extract_audio

    vid = register_local_video_with_separate_audio(
        db,
        REAL_VIDEO,
        REAL_AUDIO,
        youtube_id="sample_youtube_id",
        title="Смерть чиновника",
    )

    # Use the user's bundled audio for input. ``extract_audio`` is
    # called with the video as ``video_path`` and the audio webm as
    # ``audio_path`` — it should pick the audio webm.
    out = extract_audio(
        REAL_VIDEO,
        tmp_path / "extracted",
        video_id=str(vid),
        audio_path=REAL_AUDIO,
    )
    assert out.exists()

    # The output is a real 16 kHz mono int16 WAV. Verify the header.
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getsampwidth() == 2
        # The audio is ~701 s long.
        assert w.getnframes() / w.getframerate() > 600.0


def test_real_files_full_transcribe_pipeline(
    db, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Register the user's pair, transcribe with a fake engine, verify words."""
    from rytp import engines
    from rytp.channels import register_local_video_with_separate_audio
    from rytp.engines import Word
    from rytp.transcribe.run import transcribe_video

    # 1. Register the pair.
    vid = register_local_video_with_separate_audio(
        db,
        REAL_VIDEO,
        REAL_AUDIO,
        youtube_id="sample_youtube_id",
        title="Смерть чиновника",
    )

    # 2. The repo ships a pre-extracted 16 kHz mono WAV at
    #    ``data/audio/sample_youtube_id.wav``. We point the mocked
    #    ``extract_audio`` at it so the chunker and STT engine see a
    #    real canonical-format file.
    pre_extracted = REPO_ROOT / "data" / "audio" / "sample_youtube_id.wav"
    if not pre_extracted.exists():
        pytest.skip(
            f"pre-extracted WAV not found at {pre_extracted} — "
            "extract from the audio webm first with `rytp transcribe` "
            "or by hand"
        )

    import rytp.transcribe.run as run_mod

    monkeypatch.setattr(
        run_mod, "extract_audio", lambda *a, **kw: pre_extracted
    )

    # 3. Register a fake STT engine that yields two words.
    class FakeSTT:
        name = "fake-stt-real"
        requires_hf_token = False

        def transcribe(self, audio_path, *, language=None):
            yield Word(start_ms=0, end_ms=500, text="real", confidence=0.9)
            yield Word(start_ms=500, end_ms=1000, text="test", confidence=0.9)

    engines.register_stt(FakeSTT)
    try:
        # Make the video row's ``downloaded_path`` point to the real
        # mp4 so transcribe_video's FileNotFoundError check is
        # satisfied (we never actually read from it — extract_audio is
        # mocked).
        db.conn.execute(
            "UPDATE videos SET downloaded_path = ? WHERE id = ?",
            (str(REAL_VIDEO), vid),
        )
        db.conn.commit()

        result = transcribe_video(
            db, video_id=vid, stt_engine="fake-stt-real", diarizer="none"
        )
        assert result.n_words == 2

        rows = db.conn.execute(
            "SELECT text, diarizer_speaker FROM words WHERE video_id = ? ORDER BY start_ms",
            (vid,),
        ).fetchall()
        assert [r["text"] for r in rows] == ["real", "test"]
        assert all(r["diarizer_speaker"] == "SPEAKER_00" for r in rows)
    finally:
        engines.STTS.pop("fake-stt-real", None)