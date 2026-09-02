"""Tests for the yt-dlp wrapper."""

from __future__ import annotations

import pytest
from pathlib import Path
from typing import Any

from rytp.download.ytdlp import (
    ChannelVideo,
    DownloadResult,
    RealYtDlpRunner,
    VideoMetadata,
    YtDlpRunner,
)


class FakeRunner:
    """Fake YtDlpRunner implementation for testing."""

    def __init__(self) -> None:
        self.channel_videos: list[ChannelVideo] = []
        self.probe_result: VideoMetadata | None = None
        self.download_result: DownloadResult | None = None
        self.last_list_channel_url: str | None = None
        self.last_probe_url: str | None = None
        self.last_download_url: str | None = None
        self.last_download_out_dir: Path | None = None
        self.last_download_format: str | None = None
        self.last_download_resume: bool = False

    def list_channel(self, url: str) -> list[ChannelVideo]:
        self.last_list_channel_url = url
        return self.channel_videos

    def probe(self, url: str) -> VideoMetadata:
        self.last_probe_url = url
        if self.probe_result is None:
            raise ValueError("No probe result configured")
        return self.probe_result

    def download(
        self, url: str, out_dir: Path, *, format_selector: str, resume: bool
    ) -> DownloadResult:
        self.last_download_url = url
        self.last_download_out_dir = out_dir
        self.last_download_format = format_selector
        self.last_download_resume = resume
        if self.download_result is None:
            raise ValueError("No download result configured")
        return self.download_result


def test_fake_runner_list_channel() -> None:
    runner = FakeRunner()
    runner.channel_videos = [
        ChannelVideo(
            youtube_id="abc123",
            title="Test Video",
            url="https://youtube.com/watch?v=abc123",
            duration=300,
            kind="video",
            published_at="2024-01-01",
        )
    ]
    results = runner.list_channel("https://youtube.com/@test")
    assert len(results) == 1
    assert results[0].youtube_id == "abc123"
    assert results[0].kind == "video"


def test_fake_runner_probe() -> None:
    runner = FakeRunner()
    runner.probe_result = VideoMetadata(
        youtube_id="def456",
        title="Probe Video",
        duration=600,
        kind="short",
        url="https://youtube.com/shorts/def456",
        published_at="2024-02-01",
    )
    result = runner.probe("https://youtube.com/shorts/def456")
    assert result.youtube_id == "def456"
    assert result.kind == "short"


def test_fake_runner_download() -> None:
    runner = FakeRunner()
    runner.download_result = DownloadResult(
        path=Path("/tmp/video.mp4"),
        youtube_id="ghi789",
        title="Downloaded Video",
        duration=900,
        was_resumed=False,
    )
    result = runner.download(
        "https://youtube.com/watch?v=ghi789",
        Path("/tmp"),
        format_selector="test",
        resume=False,
    )
    assert result.path == Path("/tmp/video.mp4")
    assert result.was_resumed is False
    assert runner.last_download_format == "test"
    assert runner.last_download_resume is False


def test_real_runner_import_error() -> None:
    # This test verifies that RealYtDlpRunner raises ImportError with the
    # correct message when yt-dlp is not installed.
    import importlib.util

    # Check that yt_dlp is not available in this test environment
    if importlib.util.find_spec("yt_dlp") is not None:
        pytest.skip("yt_dlp is installed; skipping import error test")

    # The InstallError message is "yt-dlp is not installed. Install with `pip install rytp[yt-dlp]`."
    # — capital I. Case-insensitive match covers both.
    with pytest.raises(ImportError, match=r"(?i)install with"):
        RealYtDlpRunner()
    # Inspect the exact message
    try:
        RealYtDlpRunner()
    except ImportError as e:
        assert "yt-dlp" in str(e)
        assert "pip install" in str(e)
    else:
        pytest.fail("RealYtDlpRunner did not raise ImportError")


# Integration-style tests that would use a real runner (skipped without yt-dlp)
@pytest.mark.integration
def test_real_runner_list_channel() -> None:
    pytest.skip("Requires yt-dlp and network")


@pytest.mark.integration
def test_real_runner_probe() -> None:
    pytest.skip("Requires yt-dlp and network")


@pytest.mark.integration
def test_real_runner_download() -> None:
    pytest.skip("Requires yt-dlp and network")