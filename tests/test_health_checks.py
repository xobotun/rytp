"""Tests for Part 2's doctor checks (contracts §5, Health checks)."""

from __future__ import annotations

import subprocess

import pytest

from rytp import constants as C
from rytp.commands import HEALTH_CHECKS
from rytp.commands import ingest as I
from rytp.db import Database
from rytp.db.queries import insert_asset
from tests.fakes import make_video, touch


def test_part_two_registers_its_checks_with_the_right_severity() -> None:
    for name in ("ffmpeg", "ffprobe", "yt-dlp", "disk", "disk-headroom"):
        assert name in HEALTH_CHECKS
        assert HEALTH_CHECKS[name].summary
    # doctor exits non-zero only on a required check, so this split is the
    # difference between "cannot work" and "worth knowing".
    assert HEALTH_CHECKS["ffmpeg"].required is True
    assert HEALTH_CHECKS["ffprobe"].required is True
    assert HEALTH_CHECKS["disk"].required is True
    assert HEALTH_CHECKS["yt-dlp"].required is False
    assert HEALTH_CHECKS["disk-headroom"].required is False


def test_a_present_binary_reports_its_version(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(I.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        I, "_run_version",
        lambda exe: subprocess.CompletedProcess(
            [exe], 0, "ffmpeg version 7.1 Copyright (c)\nbuilt with clang\n", ""
        ),
    )
    result = HEALTH_CHECKS["ffmpeg"].run(db)
    assert result.ok is True
    assert "7.1" in result.detail
    assert "built with clang" not in result.detail   # first line only
    assert result.remedy is None


def test_a_missing_required_binary_fails_with_an_exact_remedy(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(I.shutil, "which", lambda name: None)
    result = HEALTH_CHECKS["ffprobe"].run(db)
    assert result.ok is False
    assert "not on PATH" in result.detail
    assert result.remedy and "ffmpeg.org" in result.remedy


def test_a_binary_that_cannot_answer_is_reported_not_raised(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(I.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def explode(exe: str):
        raise OSError("Exec format error")

    monkeypatch.setattr(I, "_run_version", explode)
    result = HEALTH_CHECKS["ffmpeg"].run(db)   # must not raise
    assert result.ok is False
    assert "Exec format error" in result.detail


def test_a_missing_yt_dlp_is_reported_honestly_but_is_not_fatal(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # contracts §5: ok tells the truth about what was found; required=False on
    # the check is what makes it non-fatal.
    monkeypatch.setattr(I, "_ytdlp_version", lambda: None)
    result = HEALTH_CHECKS["yt-dlp"].run(db)
    assert result.ok is False
    assert "not installed" in result.detail
    assert result.remedy and "pip install" in result.remedy
    assert HEALTH_CHECKS["yt-dlp"].required is False


def test_a_present_yt_dlp_reports_its_version(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(I, "_ytdlp_version", lambda: "2026.9.1")
    result = HEALTH_CHECKS["yt-dlp"].run(db)
    assert result.ok is True and "2026.9.1" in result.detail


def test_a_readable_volume_passes_the_required_disk_check(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A nearly full drive still passes THIS check: the tool can work.
    monkeypatch.setattr(I, "_free_bytes", lambda: 1)
    result = HEALTH_CHECKS["disk"].run(db)
    assert result.ok is True
    assert "GiB free" in result.detail


def test_an_unreadable_data_volume_fails_the_required_disk_check(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode():
        raise OSError("device not ready")

    monkeypatch.setattr(I, "_free_bytes", explode)
    result = HEALTH_CHECKS["disk"].run(db)   # must not raise
    assert result.ok is False
    assert "device not ready" in result.detail


def test_ample_headroom_passes_and_counts_the_videos_that_fit(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(I, "_free_bytes", lambda: C.HEALTH_DISK_FREE_MIN_BYTES * 4)
    result = HEALTH_CHECKS["disk-headroom"].run(db)
    assert result.ok is True
    assert "more video" in result.detail
    assert result.remedy is None


def test_low_headroom_is_an_honest_failure_but_advisory(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(I, "_free_bytes", lambda: C.HEALTH_DISK_FREE_MIN_BYTES // 2)
    result = HEALTH_CHECKS["disk-headroom"].run(db)
    assert result.ok is False
    assert result.remedy and "cache prune" in result.remedy
    # Honest about the finding, but it must not make doctor exit non-zero.
    assert HEALTH_CHECKS["disk-headroom"].required is False


def test_the_headroom_estimate_uses_real_measured_sizes(
    db: Database, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio",
                 path=str(touch(tmp_path / "a.m4a")), size_bytes=C.BYTES_PER_GIB)
    monkeypatch.setattr(I, "_free_bytes", lambda: 10 * C.BYTES_PER_GIB)
    result = HEALTH_CHECKS["disk-headroom"].run(db)
    assert "10 more video" in result.detail


def test_an_unreadable_volume_does_not_crash_the_headroom_check(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode():
        raise OSError("device not ready")

    monkeypatch.setattr(I, "_free_bytes", explode)
    result = HEALTH_CHECKS["disk-headroom"].run(db)   # must not raise
    assert result.ok is False


def test_no_check_raises_whatever_the_machine_looks_like(db: Database) -> None:
    # The real machine, unmocked: whatever is or is not installed, doctor
    # must come back with a result per check and no traceback.
    for name in ("ffmpeg", "ffprobe", "yt-dlp", "disk", "disk-headroom"):
        result = HEALTH_CHECKS[name].run(db)
        assert isinstance(result.ok, bool)
        assert result.detail
