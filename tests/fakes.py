"""Fakes and factories shared by the Part 2 test modules.

Nothing here touches the network or an external binary.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.db import Database

VIDEO_A_URL = "https://example.invalid/watch/VIDEO_A"
CHANNEL_ONE_URL = "https://example.invalid/@CHANNEL_ONE"


def make_video(db: Database, **overrides: Any) -> int:
    """Insert one catalog row and return its id."""
    row: dict[str, Any] = {
        "source": C.REMOTE_SOURCE,
        "kind": "video",
        "channel_id": None,
        "external_id": "VIDEO_A",
        "url": VIDEO_A_URL,
        "title": "Video A",
        "duration_ms": 3_600_000,
        "published_at": "2020-01-01",
        "metadata_json": "{}",
        "created_at": datetime.now(UTC).isoformat(),
    }
    row.update(overrides)
    cur = db.conn.execute(
        "INSERT INTO videos (source, kind, channel_id, external_id, url, title, "
        "duration_ms, published_at, metadata_json, created_at) "
        "VALUES (:source, :kind, :channel_id, :external_id, :url, :title, "
        ":duration_ms, :published_at, :metadata_json, :created_at)",
        row,
    )
    return int(cur.lastrowid)


def touch(path: Path, content: bytes = b"\x00" * 16) -> Path:
    """Create a small real file, parents included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@contextmanager
def temp_job_kind(
    name: str,
    pool: str,
    handler: Callable[[Any, int, dict[str, Any]], None],
    readiness: Callable[[Any, int], Any] | None = None,
    *,
    target_kind: str = "video",
    reopenable: bool = True,
) -> Iterator[None]:
    """Register a throwaway job kind for one test, then undo exactly that.

    JOB_KINDS and JOB_HANDLERS are module-level registries, so a test that
    added to them permanently would leak into every later test. Later parts
    (Part 3 onward) register real kinds under names this helper also uses in
    tests predating them (``transcribe``, ``align``, ``fingerprint``), so this
    restores whatever was there before rather than unconditionally popping —
    unconditionally popping would permanently delete a real, already-landed
    job kind for the rest of the test session.
    """
    from rytp.jobs import JOB_HANDLERS, JOB_KINDS, JobKind, Readiness, register_job_kind

    _missing = object()
    previous_kind = JOB_KINDS.get(name, _missing)
    previous_handler = JOB_HANDLERS.get(name, _missing)
    # register_job_kind refuses a duplicate name, which is right in
    # production but wrong for a test overriding an already-landed real
    # kind, so drop it first and let register_job_kind re-add it clean.
    JOB_KINDS.pop(name, None)
    JOB_HANDLERS.pop(name, None)
    register_job_kind(
        JobKind(
            name=name,
            pool=pool,
            readiness=readiness or (lambda db, target: Readiness.READY),
            handler=handler,
            summary="test kind",
            target_kind=target_kind,
            reopenable=reopenable,
        )
    )
    try:
        yield
    finally:
        if previous_kind is _missing:
            JOB_KINDS.pop(name, None)
        else:
            JOB_KINDS[name] = previous_kind  # type: ignore[assignment]
        if previous_handler is _missing:
            JOB_HANDLERS.pop(name, None)
        else:
            JOB_HANDLERS[name] = previous_handler  # type: ignore[assignment]


AUDIO_FILE_SPEC = {
    "name": "140.m4a", "format_id": "140", "ext": "m4a",
    "vcodec": "none", "acodec": "mp4a.40.2", "abr": 129.5, "language": "ru",
}
VIDEO_FILE_SPEC = {
    "name": "136.mp4", "format_id": "136", "ext": "mp4",
    "vcodec": "avc1.4d401f", "acodec": "none", "width": 1280, "height": 720,
}
CAPTION_FILE_SPEC = {
    "name": "captions.ru-orig.json3", "format_id": "ru-orig", "ext": "json3",
    "vcodec": None, "acodec": None, "language": "ru-orig",
}

#: Real, minimal json3 content — not just a placeholder file. Part 3's
#: caption_words job actually parses this (`rytp.transcribe.captions`), where
#: earlier fixtures only ever asserted the asset row existed.
_MINIMAL_JSON3_BYTES = json.dumps(
    {"events": [{"tStartMs": 0, "segs": [{"utf8": "один"}]}]}, ensure_ascii=False
).encode("utf-8")


class FakeYtDlpRunner:
    """A YtDlpRunner that writes small real files and never opens a socket.

    ``entries`` maps a tab URL to the raw yt-dlp entry dicts it should
    return; ``raises`` makes the next download blow up, which is how the
    throttle and unavailable paths are tested.
    """

    def __init__(
        self,
        *,
        entries: dict[str, list[dict[str, Any]]] | None = None,
        info: dict[str, Any] | None = None,
        files: list[dict[str, Any]] | None = None,
        captions: list[dict[str, Any]] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.entries = entries or {}
        self.info = info or {}
        self.files = [AUDIO_FILE_SPEC, VIDEO_FILE_SPEC] if files is None else files
        self.captions = [CAPTION_FILE_SPEC] if captions is None else captions
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def list_entries(self, url: str) -> list[dict[str, Any]]:
        self.calls.append({"op": "list_entries", "url": url})
        if self.raises is not None:
            raise self.raises
        if url not in self.entries:
            raise RuntimeError("ERROR: This channel does not have a videos tab")
        return self.entries[url]

    def probe(self, url: str) -> dict[str, Any]:
        self.calls.append({"op": "probe", "url": url})
        if self.raises is not None:
            raise self.raises
        return self.info

    def _emit(self, out_template: str, specs: list[dict[str, Any]]) -> list[Any]:
        from rytp.acquire.ytdlp import FetchedFile

        parent = Path(out_template).parent
        made = []
        for spec in specs:
            content = _MINIMAL_JSON3_BYTES if spec.get("ext") == "json3" else b"\x00" * 16
            path = touch(parent / spec["name"], content)
            made.append(
                FetchedFile(
                    path=path,
                    format_id=spec.get("format_id"),
                    ext=spec.get("ext", path.suffix.lstrip(".")),
                    vcodec=spec.get("vcodec"),
                    acodec=spec.get("acodec"),
                    width=spec.get("width"),
                    height=spec.get("height"),
                    abr=spec.get("abr"),
                    size_bytes=path.stat().st_size,
                    language=spec.get("language"),
                )
            )
        return made

    def download(
        self, url: str, *, out_template: str, format_selector: str,
        rate_limit_bps: int | None, sleep_requests_s: float,
    ) -> list[Any]:
        self.calls.append({
            "op": "download", "url": url, "out_template": out_template,
            "format_selector": format_selector, "rate_limit_bps": rate_limit_bps,
            "sleep_requests_s": sleep_requests_s,
        })
        if self.raises is not None:
            raise self.raises
        return self._emit(out_template, self.files)

    def download_captions(
        self, url: str, *, out_template: str, langs: tuple[str, ...],
        sub_format: str, sleep_requests_s: float,
    ) -> list[Any]:
        self.calls.append({
            "op": "download_captions", "url": url, "out_template": out_template,
            "langs": langs, "sub_format": sub_format,
        })
        if self.raises is not None:
            raise self.raises
        return self._emit(out_template, self.captions)
