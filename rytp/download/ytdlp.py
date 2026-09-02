"""Thin wrapper around the yt-dlp Python API for channel listing, probing, and downloading."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from rytp import constants as C


@runtime_checkable
class YtDlpRunner(Protocol):
    """Pluggable runner so tests can substitute a fake."""

    def list_channel(self, url: str) -> list[ChannelVideo]: ...
    def probe(self, url: str) -> VideoMetadata: ...
    def download(self, url: str, out_dir: Path, *, format_selector: str, resume: bool) -> DownloadResult: ...


@dataclass(frozen=True)
class ChannelVideo:
    youtube_id: str
    title: str
    url: str
    duration: int | None
    kind: str  # "video" | "short" | "livestream" | "other"
    published_at: str | None


@dataclass(frozen=True)
class VideoMetadata:
    youtube_id: str | None
    title: str
    duration: int | None
    kind: str
    url: str
    published_at: str | None


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    youtube_id: str | None
    title: str
    duration: int | None
    was_resumed: bool  # True if download used partial file


def _map_kind(ie_key: str | None, type_: str | None) -> str:
    """Map yt-dlp's ie_key and _type to our kind enum."""
    if ie_key == "YoutubeShorts" or type_ == "url" or type_ == "short":
        return "short"
    if ie_key in ("YoutubeLive", "YoutubeChannel", "YoutubeTab") or type_ == "livestream":
        return "livestream"
    if ie_key in ("Youtube", "YoutubeTab") or type_ in ("video", "playlist_entry", None):
        return "video"
    return "other"


class RealYtDlpRunner:
    """Concrete implementation using the yt_dlp Python API."""

    def __init__(self) -> None:
        try:
            import yt_dlp  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "yt-dlp is not installed. Install with `pip install rytp[yt-dlp]`."
            ) from e

    def list_channel(self, url: str) -> list[ChannelVideo]:
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "extract_flat": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info or "entries" not in info:
            return []
        results: list[ChannelVideo] = []
        for entry in info["entries"]:
            if not entry:
                continue
            youtube_id = (
                entry.get("id")
                or entry.get("url", "").split("v=")[-1][: C.YOUTUBE_ID_LENGTH]
            )
            title = entry.get("title") or ""
            entry_url = entry.get("url") or f"https://www.youtube.com/watch?v={youtube_id}"
            duration = entry.get("duration")
            if isinstance(duration, (int, float)):
                duration = int(duration)
            kind = _map_kind(entry.get("ie_key"), entry.get("_type"))
            published_at = entry.get("upload_date") or entry.get("release_date")
            if published_at and isinstance(published_at, str) and len(published_at) == 8:
                published_at = f"{published_at[:4]}-{published_at[4:6]}-{published_at[6:8]}"
            results.append(
                ChannelVideo(
                    youtube_id=youtube_id,
                    title=title,
                    url=entry_url,
                    duration=duration,
                    kind=kind,
                    published_at=published_at,
                )
            )
        return results

    def probe(self, url: str) -> VideoMetadata:
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        youtube_id = info.get("id")
        title = info.get("title") or ""
        duration = info.get("duration")
        if isinstance(duration, (int, float)):
            duration = int(duration)
        kind = _map_kind(info.get("ie_key"), info.get("_type"))
        published_at = info.get("upload_date") or info.get("release_date")
        if published_at and isinstance(published_at, str) and len(published_at) == 8:
            published_at = f"{published_at[:4]}-{published_at[4:6]}-{published_at[6:8]}"
        return VideoMetadata(
            youtube_id=youtube_id,
            title=title,
            duration=duration,
            kind=kind,
            url=url,
            published_at=published_at,
        )

    def download(
        self,
        url: str,
        out_dir: Path,
        *,
        format_selector: str,
        resume: bool,
    ) -> DownloadResult:
        import yt_dlp

        # Ensure out_dir exists
        out_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing partial file before download
        partial_existed = False
        if resume:
            # We need youtube_id to check; extract from URL if possible
            probe_result = self.probe(url)
            youtube_id = probe_result.youtube_id
            if youtube_id:
                partial_existed = bool(list(out_dir.glob(f"*{youtube_id}*.*.part")))

        # Default format selector from user's sample run
        ydl_opts = {
            "format": format_selector,
            "outtmpl": str(out_dir / "%(title)s [%(id)s].%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "noprogress": True,
            "continuedl": True,  # Resume partial downloads
            "nopart": False,  # Use .part extension for incomplete downloads
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the downloaded file
        # We need to locate the actual output file; glob for the youtube_id
        youtube_id = None
        if probe_result := self._safe_probe(url):
            youtube_id = probe_result.youtube_id

        if youtube_id:
            matches = list(out_dir.glob(f"*{youtube_id}*.mp4"))
            if not matches:
                matches = list(out_dir.glob(f"*{youtube_id}*.mkv"))
            if not matches:
                matches = list(out_dir.glob(f"*{youtube_id}*"))
            if matches:
                # Pick the largest file (most likely the merged output)
                path = max(matches, key=lambda p: p.stat().st_size)
                return DownloadResult(
                    path=path,
                    youtube_id=youtube_id,
                    title=probe_result.title if probe_result else "",
                    duration=probe_result.duration if probe_result else None,
                    was_resumed=partial_existed,
                )

        # Fallback: return the most recent file in the directory
        files = list(out_dir.iterdir())
        if files:
            path = max(files, key=lambda p: p.stat().st_mtime)
            return DownloadResult(
                path=path,
                youtube_id=youtube_id,
                title="",
                duration=None,
                was_resumed=partial_existed,
            )

        raise FileNotFoundError(f"Download completed but no file found in {out_dir}")

    def _safe_probe(self, url: str) -> VideoMetadata | None:
        try:
            return self.probe(url)
        except Exception:
            return None