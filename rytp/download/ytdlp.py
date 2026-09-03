"""Thin wrapper around the yt-dlp Python API for channel listing, probing, and downloading."""

from __future__ import annotations

import shutil
import subprocess
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
    audio_path: Path | None = None  # Separate audio file if downloaded separately


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

        # Check if format_selector requests separate audio/video (contains '+')
        # If so, we'll download them separately and merge after
        merge_after_download = '+' in format_selector and not format_selector.strip().startswith('(')

        if merge_after_download:
            # Download without merge_output_format to get separate files
            ydl_opts = {
                "format": format_selector,
                "outtmpl": str(out_dir / "%(title)s [%(id)s].%(ext)s"),
                "quiet": True,
                "noprogress": True,
                "continuedl": True,
                "nopart": False,
            }
        else:
            # Original behavior: let yt-dlp merge
            ydl_opts = {
                "format": format_selector,
                "outtmpl": str(out_dir / "%(title)s [%(id)s].%(ext)s"),
                "merge_output_format": "mp4",
                "quiet": True,
                "noprogress": True,
                "continuedl": True,
                "nopart": False,
            }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the downloaded files
        youtube_id = None
        if probe_result := self._safe_probe(url):
            youtube_id = probe_result.youtube_id

        video_path = None
        audio_path = None

        if youtube_id:
            # Look for video files (mp4, mkv, webm without audio-only extensions)
            video_extensions = ['.mp4', '.mkv', '.webm', '.mov', '.avi']
            audio_extensions = ['.webm', '.m4a', '.mp3', '.opus', '.ogg']

            all_matches = list(out_dir.glob(f"*{youtube_id}*"))
            
            for match in all_matches:
                if match.suffix.lower() in video_extensions:
                    # Check if it's likely a video file (has video stream)
                    if self._has_video_stream(match):
                        video_path = match
                        break
            
            for match in all_matches:
                if match.suffix.lower() in audio_extensions:
                    # Check if it's likely an audio-only file
                    if self._has_audio_stream(match) and not self._has_video_stream(match):
                        audio_path = match
                        break

        # If we have separate audio/video, merge them
        if merge_after_download and video_path and audio_path:
            merged_path = self._merge_audio_video(out_dir, video_path, audio_path, youtube_id)
            video_path = merged_path
        elif not video_path and all_matches:
            # Fallback: pick the largest file
            video_path = max(all_matches, key=lambda p: p.stat().st_size)

        if not video_path:
            # Final fallback: most recent file
            files = list(out_dir.iterdir())
            if files:
                video_path = max(files, key=lambda p: p.stat().st_mtime)

        if not video_path:
            raise FileNotFoundError(f"Download completed but no file found in {out_dir}")

        return DownloadResult(
            path=video_path,
            youtube_id=youtube_id,
            title=probe_result.title if probe_result else "",
            duration=probe_result.duration if probe_result else None,
            was_resumed=partial_existed,
            audio_path=audio_path,
        )

    def _has_video_stream(self, path: Path) -> bool:
        """Check if a file has a video stream using ffprobe."""
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            # Can't check, assume it might have video
            return True
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v", "error",
                    "-select_streams", "v",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "video" in result.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def _has_audio_stream(self, path: Path) -> bool:
        """Check if a file has an audio stream using ffprobe."""
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            return True
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "audio" in result.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def _merge_audio_video(
        self,
        out_dir: Path,
        video_path: Path,
        audio_path: Path,
        youtube_id: str | None,  # kept for API compatibility / future use
    ) -> Path:
        """Merge separate audio and video files into a single container.

        The output name is ``{video_path.stem}_merged{video_path.suffix}``,
        e.g. ``Смерть чиновника [oSYPC3cc_4A].f311_merged.mp4``. We
        keep the original audio and video files on disk so the audio
        extraction stage can read ``audio_path`` directly without
        re-decoding the merged mp4; ``DownloadResult.audio_path``
        returns the unmerged audio file.

        The merge uses ``-c:v copy`` (no video re-encode) and
        ``-c:a aac`` (audio re-encode for compatibility with downstream
        consumers). The merge does NOT delete the source files; the
        caller decides whether to.
        """
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg not found on PATH, required to merge audio/video"
            )

        merged_path = out_dir / f"{video_path.stem}_merged{video_path.suffix}"

        cmd = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",  # take the video stream from input 0
            "-map",
            "1:a:0",  # take the audio stream from input 1
            "-c:v",
            "copy",
            "-c:a",
            "aac",  # Re-encode audio to AAC for player compatibility
            "-shortest",
            str(merged_path),
        ]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg merge failed: {result.stderr[-2000:]}"
            )

        if not merged_path.exists():
            raise RuntimeError(
                f"ffmpeg merge succeeded but output not found: {merged_path}"
            )

        return merged_path

    def _safe_probe(self, url: str) -> VideoMetadata | None:
        try:
            return self.probe(url)
        except Exception:
            return None