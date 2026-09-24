"""The yt-dlp seam. design §5, §13.

Everything that talks to the network in this project goes through
:class:`YtDlpRunner`, and only :class:`RealYtDlpRunner` implements it for
real — tests always pass a fake. ``yt_dlp`` itself is imported inside
methods, never at module scope, so the CLI stays importable without the
optional extra (contracts §1).

No browser cookies anywhere in this file. design §5: yt-dlp's own guidance
warns that cookie-authenticated bulk access risks account and IP bans, and
anonymous access is sufficient for everything here, captions included.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rytp import constants as C
from rytp import progress
from rytp.acquire.policy import (
    AcquireError,
    ErrorKind,
    RateLimited,
    VideoUnavailable,
    classify_error,
)
from rytp.models import ChannelEntry

#: yt-dlp ``live_status`` values that mean "this was a stream".
_LIVE_STATUSES: frozenset[str] = frozenset({"is_live", "was_live", "post_live", "is_upcoming"})


@dataclass(frozen=True)
class FetchedFile:
    """One file yt-dlp actually wrote, as described by its format dict."""

    path: Path
    format_id: str | None
    ext: str
    vcodec: str | None
    acodec: str | None
    width: int | None = None
    height: int | None = None
    abr: float | None = None
    size_bytes: int = 0
    language: str | None = None

    @property
    def has_video(self) -> bool:
        return bool(self.vcodec) and self.vcodec != "none"

    @property
    def has_audio(self) -> bool:
        return bool(self.acodec) and self.acodec != "none"


@runtime_checkable
class YtDlpRunner(Protocol):
    """Pluggable runner so tests can substitute a fake."""

    def list_entries(self, url: str) -> list[dict[str, Any]]: ...

    def probe(self, url: str) -> dict[str, Any]: ...

    def download(
        self,
        url: str,
        *,
        out_template: str,
        format_selector: str,
        rate_limit_bps: int | None,
        sleep_requests_s: float,
    ) -> list[FetchedFile]: ...

    def download_captions(
        self,
        url: str,
        *,
        out_template: str,
        langs: tuple[str, ...],
        sub_format: str,
        sleep_requests_s: float,
    ) -> list[FetchedFile]: ...


def translate_error(exc: Exception) -> AcquireError:
    """Turn any runner failure into the exception the worker knows."""
    message = str(exc)
    kind = classify_error(message)
    if kind is ErrorKind.THROTTLED:
        return RateLimited(message)
    if kind is ErrorKind.UNAVAILABLE:
        return VideoUnavailable(message)
    return AcquireError(message)


def tab_url(channel_url: str, tab: str) -> str:
    """Build a channel's per-listing URL, tolerating a pasted tab suffix."""
    base = channel_url.rstrip("/")
    for known in C.CHANNEL_TABS:
        if base.endswith(f"/{known}"):
            base = base[: -(len(known) + 1)]
            break
    return f"{base}/{tab}"


def _published_at(info: dict[str, Any]) -> str | None:
    raw = info.get("upload_date") or info.get("release_date")
    if isinstance(raw, str) and len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return None


def _duration_ms(info: dict[str, Any]) -> int | None:
    raw = info.get("duration")
    if isinstance(raw, int | float):
        return round(raw * C.MS_PER_SECOND)
    return None


def _to_entry(info: dict[str, Any], *, kind: str, fallback_url: str) -> ChannelEntry | None:
    external_id = info.get("id")
    if not external_id:
        return None
    return ChannelEntry(
        external_id=str(external_id),
        title=str(info.get("title") or external_id),
        url=str(info.get("webpage_url") or info.get("url") or fallback_url),
        duration_ms=_duration_ms(info),
        kind=kind,
        published_at=_published_at(info),
    )


def enumerate_channel(
    channel_url: str,
    *,
    tabs: Sequence[str] = C.CHANNEL_TABS,
    runner: YtDlpRunner | None = None,
) -> list[ChannelEntry]:
    """List a channel's videos, live streams and shorts, in that order.

    Entries are **not** deduplicated: a stream that appears in both
    ``/videos`` and ``/streams`` is returned twice, streams last, so the
    caller's upsert by ``(source, external_id)`` settles on the more
    specific ``kind``. A tab a channel does not have is skipped silently —
    that is not a failure — but a throttle response aborts immediately
    rather than walking into the next tab.
    """
    runner = runner or RealYtDlpRunner()
    out: list[ChannelEntry] = []
    for tab in tabs:
        url = tab_url(channel_url, tab)
        try:
            raw = runner.list_entries(url)
        except Exception as exc:
            translated = translate_error(exc)
            if isinstance(translated, RateLimited):
                raise translated from exc
            continue
        # Which listing a video came from is a far better kind signal than
        # guessing from the entry dict. design §13: streams are listed
        # separately, and missing them undercounts the corpus.
        kind = C.CHANNEL_TAB_KINDS.get(tab, "other")
        for info in raw:
            entry = _to_entry(info, kind=kind, fallback_url=url)
            if entry is not None:
                out.append(entry)
    return out


def probe_video(url: str, *, runner: YtDlpRunner | None = None) -> ChannelEntry:
    """One metadata request for ``rytp videos add <url>``. No download."""
    runner = runner or RealYtDlpRunner()
    try:
        info = runner.probe(url)
    except Exception as exc:
        raise translate_error(exc) from exc
    kind = "livestream" if info.get("live_status") in _LIVE_STATUSES else "video"
    entry = _to_entry(info, kind=kind, fallback_url=url)
    if entry is None:
        raise AcquireError(f"no video id in the metadata for {url}")
    return entry


def _report_ytdlp_progress(info: dict[str, Any]) -> None:
    """A yt-dlp ``progress_hooks`` callback: forward bytes to the seam.

    BUGS.md entry 3 — a download sat silently for minutes, indistinguishable
    from a hang. yt-dlp already knows how far along it is; this is the one
    place that call reaches :func:`rytp.progress.report`, so every caller of
    :meth:`RealYtDlpRunner.download` and
    :meth:`RealYtDlpRunner.download_captions` gets it for free.
    """
    status = info.get("status")
    if status not in ("downloading", "finished"):
        return
    done = info.get("downloaded_bytes")
    total = info.get("total_bytes") or info.get("total_bytes_estimate")
    if status == "finished" and total is None:
        total = done
    filename = info.get("filename") or info.get("info_dict", {}).get("filename")
    detail = Path(filename).name if filename else ""
    progress.report(
        "download",
        done=int(done) if isinstance(done, int | float) else None,
        total=int(total) if isinstance(total, int | float) else None,
        detail=detail,
    )


class RealYtDlpRunner:
    """The only implementation that touches the network."""

    def _ydl(self, opts: dict[str, Any]) -> Any:
        try:
            import yt_dlp
        except ImportError as exc:  # pragma: no cover - exercised by hand
            raise AcquireError(
                "yt-dlp is not installed. Install it with "
                "`pip install -e '.[yt-dlp]'`"
            ) from exc
        base: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "continuedl": True,
            "retries": C.YTDLP_RETRIES,
            "fragment_retries": C.YTDLP_RETRIES,
            "concurrent_fragment_downloads": 1,
            "progress_hooks": [_report_ytdlp_progress],
        }
        base.update(opts)
        return yt_dlp.YoutubeDL(base)

    def list_entries(self, url: str) -> list[dict[str, Any]]:
        with self._ydl({"extract_flat": "in_playlist", "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        entries = (info or {}).get("entries") or []
        return [e for e in entries if e]

    def probe(self, url: str) -> dict[str, Any]:
        with self._ydl({"skip_download": True}) as ydl:
            return dict(ydl.extract_info(url, download=False) or {})

    def download(
        self,
        url: str,
        *,
        out_template: str,
        format_selector: str,
        rate_limit_bps: int | None,
        sleep_requests_s: float,
    ) -> list[FetchedFile]:
        opts: dict[str, Any] = {
            "format": format_selector,
            "outtmpl": {"default": out_template},
            "sleep_interval_requests": sleep_requests_s,
        }
        if rate_limit_bps:
            opts["ratelimit"] = rate_limit_bps
        with self._ydl(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        return _fetched_files(info or {})

    def download_captions(
        self,
        url: str,
        *,
        out_template: str,
        langs: tuple[str, ...],
        sub_format: str,
        sleep_requests_s: float,
    ) -> list[FetchedFile]:
        opts: dict[str, Any] = {
            "skip_download": True,
            "writesubtitles": True,
            # design §6: the "-orig" track is auto-generated, so this must
            # be on or ru-orig is never offered.
            "writeautomaticsub": True,
            "subtitleslangs": list(langs),
            "subtitlesformat": sub_format,
            "outtmpl": {"default": out_template},
            "sleep_interval_requests": sleep_requests_s,
        }
        with self._ydl(opts) as ydl:
            info = ydl.extract_info(url, download=True) or {}
        out: list[FetchedFile] = []
        for lang, sub in (info.get("requested_subtitles") or {}).items():
            filepath = sub.get("filepath")
            if not filepath:
                continue
            path = Path(filepath)
            out.append(
                FetchedFile(
                    path=path,
                    format_id=lang,
                    ext=sub.get("ext") or path.suffix.lstrip("."),
                    vcodec=None,
                    acodec=None,
                    size_bytes=path.stat().st_size if path.exists() else 0,
                    language=lang,
                )
            )
        if out:
            return out
        # Older yt-dlp builds do not put ``filepath`` on the subtitle dicts.
        # The files are still written next to the template as
        # ``<stem>.<lang>.<ext>``, so find them rather than fail.
        stem = Path(out_template).name.split(".")[0]
        for path in sorted(Path(out_template).parent.glob(f"{stem}.*.{sub_format}")):
            lang = path.name[len(stem) + 1 : -(len(sub_format) + 1)]
            out.append(
                FetchedFile(
                    path=path,
                    format_id=lang,
                    ext=sub_format,
                    vcodec=None,
                    acodec=None,
                    size_bytes=path.stat().st_size,
                    language=lang,
                )
            )
        return out


def _fetched_files(info: dict[str, Any]) -> list[FetchedFile]:
    """Read the per-format results out of a completed download.

    With a comma-joined selector yt-dlp does not merge, so every entry of
    ``requested_downloads`` is one real file with its own ``filepath``.
    """
    out: list[FetchedFile] = []
    for fmt in info.get("requested_downloads") or []:
        filepath = fmt.get("filepath")
        if not filepath:
            continue
        path = Path(filepath)
        out.append(
            FetchedFile(
                path=path,
                format_id=fmt.get("format_id"),
                ext=fmt.get("ext") or path.suffix.lstrip("."),
                vcodec=fmt.get("vcodec"),
                acodec=fmt.get("acodec"),
                width=fmt.get("width"),
                height=fmt.get("height"),
                abr=fmt.get("abr"),
                size_bytes=int(fmt.get("filesize") or fmt.get("filesize_approx") or
                               (path.stat().st_size if path.exists() else 0)),
                language=fmt.get("language"),
            )
        )
    return out
