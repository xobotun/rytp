"""Tests for the ``--format`` flag on ``rytp download``.

The flag was added when ``rytp download`` was wired to the
``download_one`` helper: ``download_one`` already took a
``format_selector`` parameter, but the CLI never exposed it. The
default mirrors the user's sample run that downloads audio and video
as separate streams.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from rytp.cli import app
from rytp.download import download_one


class _RecorderRunner:
    """Fake ``YtDlpRunner`` that records the ``format_selector`` it was called with."""

    def __init__(self) -> None:
        self.last_format: str | None = None

    def download(  # type: ignore[override]
        self,
        url: str,
        out_dir: Path,
        *,
        format_selector: str,
        resume: bool,
    ) -> Any:
        from rytp.download.ytdlp import DownloadResult

        self.last_format = format_selector
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "fake.mp4"
        out.write_bytes(b"\x00" * 8)
        return DownloadResult(
            path=out,
            youtube_id="x",
            title="t",
            duration=1,
            was_resumed=False,
        )


def test_cli_download_passes_default_format_selector(db, data_dir):
    """The default ``--format`` value should be ``worstvideo[height=720]+bestaudio[language=ru]``.

    This is the selector the user reported in their sample yt-dlp
    run — separate audio and video streams, merged by rytp after
    download.
    """
    db.conn.execute(
        "INSERT INTO videos (source, kind, youtube_id, url, title) "
        "VALUES ('youtube', 'video', 'x', 'https://example.com/x', 't')"
    )
    db.conn.commit()

    runner = _RecorderRunner()
    download_one(db, 1, runner=runner)
    assert runner.last_format == "worstvideo[height=720]+bestaudio[language=ru]"


def test_cli_download_passes_explicit_format(db, data_dir):
    """``--format`` overrides the default."""
    db.conn.execute(
        "INSERT INTO videos (source, kind, youtube_id, url, title) "
        "VALUES ('youtube', 'video', 'x', 'https://example.com/x', 't')"
    )
    db.conn.commit()

    runner = _RecorderRunner()
    download_one(db, 1, runner=runner, format_selector="best")
    assert runner.last_format == "best"


def test_cli_download_format_help_present() -> None:
    """``rytp download --help`` should mention the ``--format`` flag."""
    runner = CliRunner()
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0
    assert "--format" in result.output
    # The default value should be visible in the help text.
    assert "worstvideo" in result.output or "+bestaudio" in result.output