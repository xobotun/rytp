"""Tests for the regenerable WAV cache (design §4, contracts §7)."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from rytp import config
from rytp import constants as C
from rytp.acquire.policy import MissingAssetError
from rytp.audio import extract as E
from rytp.db import Database
from rytp.db.queries import insert_asset
from tests.fakes import make_video, touch

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _fake_ffmpeg(monkeypatch: pytest.MonkeyPatch, *, rc: int = 0, write: bool = True):
    """Replace the ffmpeg call with one that writes the output file itself."""
    seen: list[list[str]] = []

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        if write and rc == 0:
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"RIFF....WAVE")
        return subprocess.CompletedProcess(cmd, rc, "", "boom" if rc else "")

    monkeypatch.setattr(E, "_run_ffmpeg", run)
    return seen


def test_the_cache_path_is_the_contracted_one(db: Database) -> None:
    vid = make_video(db)
    assert E.wav_path(vid) == config.paths().cache_wav(vid)
    assert E.wav_path(vid).parent.name == "wav"


def test_extraction_writes_the_cache_file(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    seen = _fake_ffmpeg(monkeypatch)
    out = E.ensure_wav(db, vid)
    assert out == E.wav_path(vid) and out.exists()
    cmd = seen[0]
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == str(C.AUDIO_CHANNELS)
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == str(C.AUDIO_SAMPLE_RATE_HZ)
    # ffmpeg targets the .part sibling, not the final name directly: the
    # atomic os.replace() at the end is what makes a killed extraction leave
    # no plausible-looking .wav file.
    assert cmd[-1].endswith(".wav.part")


def test_nothing_records_the_wav_path_in_the_database(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    _fake_ffmpeg(monkeypatch)
    E.ensure_wav(db, vid)
    rows = db.conn.execute("SELECT path FROM assets").fetchall()
    assert all("cache" not in r["path"] for r in rows)


def test_a_second_call_is_a_no_op(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    seen = _fake_ffmpeg(monkeypatch)
    E.ensure_wav(db, vid)
    E.ensure_wav(db, vid)
    assert len(seen) == 1
    E.ensure_wav(db, vid, overwrite=True)
    assert len(seen) == 2


def test_a_container_is_used_when_there_is_no_audio_asset(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = touch(tmp_path / "clip.mkv")
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id=str(clip), url=None)
    insert_asset(db, video_id=vid, role="container", path=str(clip))
    seen = _fake_ffmpeg(monkeypatch)
    E.ensure_wav(db, vid)
    assert str(clip) in seen[0]


def test_the_audio_asset_wins_over_a_container(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = touch(tmp_path / "a.m4a")
    clip = touch(tmp_path / "clip.mkv")
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="container", path=str(clip))
    insert_asset(db, video_id=vid, role="audio", path=str(audio))
    seen = _fake_ffmpeg(monkeypatch)
    E.ensure_wav(db, vid)
    assert str(audio) in seen[0]


def test_no_source_asset_is_reported_in_one_line(db: Database) -> None:
    vid = make_video(db)
    with pytest.raises(MissingAssetError, match="no audio or container"):
        E.ensure_wav(db, vid)


def test_a_failed_ffmpeg_leaves_no_partial_file(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    _fake_ffmpeg(monkeypatch, rc=1)
    with pytest.raises(E.ExtractionError, match="boom"):
        E.ensure_wav(db, vid)
    assert not E.wav_path(vid).exists()
    assert list(E.wav_path(vid).parent.glob("*.part")) == []


def test_a_missing_ffmpeg_says_how_to_install_it(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    monkeypatch.setattr(E, "_ffmpeg_binary", lambda: None)
    with pytest.raises(E.FfmpegNotFoundError, match=r"ffmpeg\.org"):
        E.ensure_wav(db, vid)


def test_prune_removes_everything_and_reports_bytes(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    for vid in (a, b):
        touch(E.wav_path(vid), b"\x00" * 100)
    freed = E.prune_wav_cache(db)
    assert sorted(size for _, size in freed) == [100, 100]
    assert not E.wav_path(a).exists() and not E.wav_path(b).exists()


def test_prune_can_target_one_video(db: Database) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    for vid in (a, b):
        touch(E.wav_path(vid))
    E.prune_wav_cache(db, video_id=a)
    assert not E.wav_path(a).exists()
    assert E.wav_path(b).exists()


def test_prune_dry_run_deletes_nothing(db: Database) -> None:
    vid = make_video(db)
    touch(E.wav_path(vid), b"\x00" * 10)
    freed = E.prune_wav_cache(db, dry_run=True)
    assert freed == [(E.wav_path(vid), 10)]
    assert E.wav_path(vid).exists()


def test_prune_on_an_empty_cache_is_fine(db: Database) -> None:
    assert E.prune_wav_cache(db) == []


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_real_ffmpeg_produces_16k_mono_pcm(db: Database, tmp_path: Path) -> None:
    src = tmp_path / "tone.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=1", "-ac", "2", "-ar", "44100", str(src)],
        check=True, capture_output=True,
    )
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(src))
    out = E.ensure_wav(db, vid)
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == C.AUDIO_CHANNELS
        assert w.getframerate() == C.AUDIO_SAMPLE_RATE_HZ
        assert w.getsampwidth() == 2  # int16
