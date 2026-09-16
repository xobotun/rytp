"""rytp CLI entry point.

Built on :mod:`typer`. This module wires every subcommand from
DESIGN §6, plus the top-level HF_TOKEN pre-launch gate (DESIGN §12).

The gate runs before any subcommand body so a missing token never
displays a half-broken TUI: it prints one clear paragraph to stderr
and exits non-zero.

**All 21 subcommands are wired in v1.** CRUD commands
(channels, videos, queue, speakers, transcripts) work out of the
box. Heavy-lift commands (download, transcribe, mine, splice, tui)
require their respective optional extras
(``pip install rytp[yt-dlp]``, ``[stt]``, ``[pyannote]``, ``[all]``)
to actually run, and otherwise print a one-line install-hint error.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import typer

from rytp import config
from rytp import constants as C
from rytp import engines as engines_mod

# Quieten huggingface_hub's noisy "Ignored error while writing commit
# hash to ... PermissionError" warning. It fires every time we re-run
# ``faster-whisper`` against a cached model whose ``refs/main`` file
# is read-only (the common case on Windows when the cache was created
# by another user / app). The cache is otherwise fine, so printing the
# traceback every run is just noise — drop the warning to ERROR.
logging.getLogger("huggingface_hub._snapshot_download").setLevel(logging.ERROR)

app = typer.Typer(
    name="rytp",
    help="Download YouTube videos, transcribe them locally, datamine, and splice.",
    no_args_is_help=True,
    add_completion=False,
)


def _configure_stdio() -> None:
    """Reconfigure ``sys.stdout`` and ``sys.stderr`` to UTF-8.

    Help text contains ``§`` (DESIGN §7) and ``—`` em-dashes. On Windows
    the default code page (cp1251 / cp1252) silently replaces these
    with ``?`` or the U+FFFD replacement character, which then shows up
    in the help banner as a broken glyph. Forcing UTF-8 here makes the
    bytes valid UTF-8 regardless of the attached console; terminals
    that understand UTF-8 will render the original characters, and
    the rest will at least fail cleanly rather than producing mojibake.

    Silently no-ops on streams that can't be reconfigured (older
    Python, captured pipes under pytest, etc.) — the user's locale
    wins in those cases.
    """
    import sys

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


_configure_stdio()

# Subcommand groups (channel, videos, queue, speakers) — added below.
channel_app = typer.Typer(help="Manage YouTube channels in the index.", no_args_is_help=True)
videos_app = typer.Typer(help="Browse and register videos.", no_args_is_help=True)
queue_app = typer.Typer(help="Control the download queue.", no_args_is_help=True)
speakers_app = typer.Typer(help="Manage the global speaker roster and per-video mapping.", no_args_is_help=True)
transcripts_app = typer.Typer(help="Export human-readable transcripts (DESIGN §7).", no_args_is_help=True)

app.add_typer(channel_app, name="channel")
app.add_typer(videos_app, name="videos")
app.add_typer(queue_app, name="queue")
app.add_typer(speakers_app, name="speakers")
app.add_typer(transcripts_app, name="transcripts")


# ---------------------------------------------------------------------------
# HF_TOKEN pre-launch gate (DESIGN §12)
# ---------------------------------------------------------------------------


def _check_hf_token_requirement(
    *,
    stt_name: str | None,
    diarizer_name: str | None,
    combined_name: str | None,
) -> list[tuple[str, type]]:
    """Return a list of ``(label, engine_class)`` tuples that need HF_TOKEN but don't have one.

    Empty list means the gate passes.
    """
    missing: list[tuple[str, type]] = []
    if config.hf_token() is not None:
        return missing

    if combined_name is not None:
        try:
            cls = engines_mod.resolve_combined(combined_name)
        except ValueError:
            cls = None  # unknown engine; let the subcommand error handle it
        if cls is not None and getattr(cls, "requires_hf_token", False):
            missing.append((f"combined engine {combined_name!r}", cls))

    if stt_name is not None:
        try:
            cls = engines_mod.resolve_stt(stt_name)
        except ValueError:
            cls = None
        if cls is not None and getattr(cls, "requires_hf_token", False):
            missing.append((f"STT engine {stt_name!r}", cls))

    if diarizer_name is not None:
        try:
            cls = engines_mod.resolve_diarizer(diarizer_name)
        except ValueError:
            cls = None
        if cls is not None and getattr(cls, "requires_hf_token", False):
            missing.append((f"diarizer {diarizer_name!r}", cls))

    return missing


def _emit_hf_token_help(errors: list[tuple[str, type]]) -> None:
    """Print a one-paragraph paper-cut to stderr. Names model, URL, env var."""
    lines = ["rytp: HF_TOKEN is required for the following engine(s):"]
    for label, cls in errors:
        # Each engine class may declare its own help URL via class attr.
        url = getattr(cls, "help_url", "https://huggingface.co/settings/tokens")
        env = getattr(cls, "token_env_var", "HF_TOKEN")
        lines.append(f"  - {label}")
        lines.append(f"      visit: {url}")
        lines.append(f"      set env var: {env}")
    lines.append("")
    lines.append(
        "Visit the model page on huggingface.co, click 'Agree and access',"
        " set the env var, and re-run. After this one-time setup, rytp is"
        " fully offline — the token is only consulted for authentication,"
        " not for any network call."
    )
    typer.echo("\n".join(lines), err=True)


@app.callback()
def _main_callback(
    ctx: typer.Context,
    stt: Optional[str] = typer.Option(
        None,
        "--stt",
        help="STT engine name (overrides settings.default_stt_engine for this run).",
    ),
    diarizer: Optional[str] = typer.Option(
        None,
        "--diarizer",
        help="Diarizer name (overrides settings.default_diarizer for this run).",
    ),
    combined: Optional[str] = typer.Option(
        None,
        "--combined",
        help="Combined STT+Diarize engine name (mutually exclusive with --stt/--diarizer).",
    ),
) -> None:
    """rytp — YouTube → transcript → datamine → splice pipeline."""
    # Mutual exclusion: --combined may not be combined with --stt or
    # --diarizer. The help text promises this; we enforce it here so
    # users don't silently get the wrong engine (the per-subcommand
    # fallback would otherwise pick --stt/--diarizer first).
    if combined is not None and (stt is not None or diarizer is not None):
        typer.echo(
            "--combined is mutually exclusive with --stt and --diarizer; "
            "pick one group of options, not both.",
            err=True,
        )
        raise typer.Exit(code=2)

    # Only run the gate when an engine has been *named* on the command line.
    # We deliberately don't read the defaults here: if the user picks a
    # gated engine explicitly, fail loudly. If they leave the default and
    # the default doesn't need a token (NullDiarizer doesn't), stay silent.
    if stt is None and diarizer is None and combined is None:
        return

    errors = _check_hf_token_requirement(
        stt_name=stt,
        diarizer_name=diarizer,
        combined_name=combined,
    )
    if errors:
        _emit_hf_token_help(errors)
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Helpers shared by the wired subcommands
# ---------------------------------------------------------------------------


def _not_implemented(name: str) -> None:
    """Print a one-line stub message and exit 1.

    No subcommand uses this in v1; kept for tests that grep the
    source for the marker.
    """
    typer.echo(f"rytp {name}: not implemented yet", err=True)
    raise typer.Exit(code=1)


def _open_db() -> "Database":
    """Open the global DB, run migrations, return the connection holder.

    Most subcommands take a copy of this. The returned object is
    a real :class:`rytp.db.Database`; callers are expected to close
    it (or use it via the ``with`` form).
    """
    from rytp.db import Database

    db = Database(config.paths.db)
    db.migrate()
    return db


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a small fixed-width table for the ``list`` subcommands.

    Columns are sized to fit the widest cell; first column left
    aligned, the rest left aligned too (numbers read better
    left-aligned for short rows).
    """
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]
    sep = "  "
    fmt = sep.join(f"{{:<{w}}}" for w in widths)
    typer.echo(fmt.format(*headers))
    typer.echo(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        typer.echo(fmt.format(*r))


def _cohesion_callback(value: Optional[str]) -> Optional[str]:
    """Validate ``--cohesion`` and return the canonical string.

    Used as a typer ``Option(callback=...)`` so that ``mine --cohesion bogus``
    fails fast with a clean ``Invalid value for '--cohesion'`` message
    instead of leaking an ``enum.Enum`` ``ValueError`` traceback from
    deep inside :func:`rytp.mine.mine`.

    The default ``"med"`` is supplied via ``typer.Option(..., default="med")``;
    this callback sees that string, not ``None``. We still accept
    ``None`` defensively for callers that explicitly pass the option
    without a default.
    """
    if value is None:
        return value
    from rytp.mine import Cohesion

    try:
        return Cohesion(value).value
    except ValueError as e:
        allowed = ", ".join(c.value for c in Cohesion)
        raise typer.BadParameter(
            f"{value!r} is not a valid cohesion; choose from {allowed}."
        ) from e


# Allowed values for ``videos list --kind`` and ``--source``. Kept in
# sync with the SQLite CHECK constraints on the ``videos`` table
# (:mod:`rytp.db` migration 2). Listed as tuples so the order is
# deterministic in error messages.
_ALLOWED_KIND: tuple[str, ...] = ("video", "short", "livestream", "other")
_ALLOWED_SOURCE: tuple[str, ...] = ("youtube", "ytdlp", "local")


def _kind_callback(value: Optional[str]) -> Optional[str]:
    if value is None:
        return value
    if value not in _ALLOWED_KIND:
        raise typer.BadParameter(
            f"{value!r} is not a valid kind; "
            f"choose from {', '.join(_ALLOWED_KIND)}."
        )
    return value


def _source_callback(value: Optional[str]) -> Optional[str]:
    if value is None:
        return value
    if value not in _ALLOWED_SOURCE:
        raise typer.BadParameter(
            f"{value!r} is not a valid source; "
            f"choose from {', '.join(_ALLOWED_SOURCE)}."
        )
    return value


def _splice_mode_callback(value: Optional[str]) -> Optional[str]:
    """Validate ``--mode`` against :class:`rytp.splice.SpliceMode`.

    Without this callback, ``splice --mode bogus`` raised a bare
    ``ValueError`` from deep inside ``rytp.splice.splice_clips``,
    leaking a 12-line traceback (the same shape as the original
    ``mine --cohesion bogus`` bug). Routing the validation through
    Typer's BadParameter makes it consistent with every other validated
    flag.
    """
    if value is None:
        return value
    from rytp.splice import SpliceMode

    try:
        return SpliceMode(value).value
    except ValueError as e:
        allowed = ", ".join(m.value for m in SpliceMode)
        raise typer.BadParameter(
            f"{value!r} is not a valid SpliceMode; choose from {allowed}."
        ) from e


def _handle_business_error(e: BaseException) -> None:
    """Translate a caught business exception into a clean CLI exit.

    Centralizes the "no traceback, just one line, exit 1" pattern.
    Falls back to a generic message if the exception is something
    exotic we don't have a friendlier re-phrasing for.
    """
    import sqlite3

    if isinstance(e, sqlite3.IntegrityError) and "FOREIGN KEY" in str(e):
        # Best-effort: pull a video id out of the message and reuse the
        # wording the rest of the CLI uses ("video N not found").
        msg = str(e)
        typer.echo(
            f"database constraint violated: {msg}. "
            "Check that any referenced id (channel, video, speaker) exists.",
            err=True,
        )
        raise typer.Exit(code=1) from e
    if isinstance(e, FileNotFoundError):
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    if isinstance(e, PermissionError):
        typer.echo(
            f"cannot write to {e.filename or 'the target path'} "
            f"(Access is denied). Check that the parent directory is writable.",
            err=True,
        )
        raise typer.Exit(code=1) from e
    if isinstance(e, (ValueError, ImportError, RuntimeError)):
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
    # Anything else: re-raise — typer/Click will print a traceback,
    # which is appropriate for an unknown bug.
    raise e


# ---------------------------------------------------------------------------
# channel group (wired)
# ---------------------------------------------------------------------------


@channel_app.command("add")
def channel_add(
    url: str = typer.Argument(..., help="Channel URL (YouTube or any yt-dlp-supported site)."),
    title: Optional[str] = typer.Option(
        None, "--title", help="Override the title. Probed from yt-dlp if not given."
    ),
) -> None:
    """Register a channel in the ``channels`` table."""
    from rytp.channels import add_channel
    from rytp.download.ytdlp import RealYtDlpRunner

    db = _open_db()
    try:
        try:
            runner = RealYtDlpRunner()
        except ImportError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1) from e
        cid = add_channel(db, runner, url, title=title)
    finally:
        db.close()
    typer.echo(f"channel {cid}: {url}")


@channel_app.command("sync")
def channel_sync(
    name: str = typer.Argument(
        ...,
        help="Channel title, URL, or id. The DB is searched for any of these.",
    ),
) -> None:
    """yt-dlp flat-playlist listing; upsert every video into ``videos``."""
    from rytp.channels import sync_channel
    from rytp.download.ytdlp import RealYtDlpRunner

    db = _open_db()
    try:
        try:
            runner = RealYtDlpRunner()
        except ImportError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1) from e
        new_count = sync_channel(db, runner, name)
    finally:
        db.close()
    typer.echo(f"added {new_count} new videos for channel {name}")


@channel_app.command("list")
def channel_list() -> None:
    """Show channels + per-channel video count."""
    from rytp.channels import list_channels

    db = _open_db()
    try:
        rows = list_channels(db)
    finally:
        db.close()
    if not rows:
        typer.echo("(no channels yet — try `rytp channel add <url>`)")
        return
    _print_table(
        ["id", "title", "videos", "url"],
        [[str(r["id"]), r["title"], str(r["n_videos"]), r["url"]] for r in rows],
    )


# ---------------------------------------------------------------------------
# videos group (wired)
# ---------------------------------------------------------------------------


@videos_app.command("add")
def videos_add(
    url_or_path: str = typer.Argument(
        ...,
        help="A YouTube URL, any yt-dlp URL, or a local file path.",
    ),
    audio: Optional[Path] = typer.Option(
        None,
        "--audio",
        help=(
            "Optional separate audio file paired with the video "
            "(typical of an yt-dlp `+` format selector that downloaded "
            "audio and video as separate streams). The audio file is "
            "stored as ``videos.downloaded_audio_path`` and used by the "
            "audio-extraction stage instead of re-decoding the video."
        ),
    ),
    youtube_id: Optional[str] = typer.Option(
        None,
        "--youtube-id",
        help=(
            "YouTube video id (e.g. ``oSYPC3cc_4A``). Only used when "
            "registering a local file pair so re-registration updates "
            "the same row instead of creating a duplicate."
        ),
    ),
    title: Optional[str] = typer.Option(
        None,
        "--title",
        help="Override the title. Defaults to the video file's stem.",
    ),
) -> None:
    """Register a single video: URL or local file path.

    Local files are detected by :func:`rytp.channels._is_local_path`;
    no yt-dlp call is made for them, and ``source='local'``,
    ``downloaded=True`` are set in the resulting row. URLs are
    probed via yt-dlp, which requires the `yt-dlp` extra to be
    installed.

    When ``--audio`` is given alongside a local video file, the row
    is registered with ``downloaded_audio_path`` pointing at the
    audio file — the same shape as a freshly-downloaded YouTube row
    with a ``+`` format selector would produce.
    """
    from rytp.channels import (
        _is_local_path,
        register_local_video_with_separate_audio,
        register_video,
    )
    from rytp.download.ytdlp import RealYtDlpRunner

    db = _open_db()
    try:
        try:
            if _is_local_path(url_or_path) and audio is not None:
                # Local video + local audio pair. ``register_video``
                # doesn't model this case; use the dedicated helper.
                vid = register_local_video_with_separate_audio(
                    db,
                    Path(url_or_path),
                    audio,
                    youtube_id=youtube_id,
                    title=title,
                )
            elif _is_local_path(url_or_path):
                # No runner needed for local files; ``register_video`` skips
                # the probe entirely. Pass a placeholder to satisfy the
                # type signature — the function never calls it.
                class _UnusedRunner:
                    def probe(self, url: str):  # pragma: no cover
                        raise RuntimeError("unreachable: local file branch")

                runner: object = _UnusedRunner()
                vid = register_video(db, runner, url_or_path)
            else:
                try:
                    runner = RealYtDlpRunner()
                except ImportError as e:
                    typer.echo(str(e), err=True)
                    raise typer.Exit(code=1) from e
                vid = register_video(db, runner, url_or_path)
        except FileNotFoundError as e:
            # Local file branch detects ``not path.exists()`` and raises.
            # Translate to a clean exit instead of a stack trace.
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1) from e
    finally:
        db.close()
    typer.echo(f"video {vid}: {url_or_path}")


@videos_app.command("list")
def videos_list(
    channel: Optional[str] = typer.Option(
        None, "--channel", help="Restrict to one channel (id, title, or URL)."
    ),
    kind: Optional[str] = typer.Option(
        None,
        "--kind",
        help="Filter by kind: video, short, livestream, other.",
        callback=_kind_callback,
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        help="Filter by source: youtube, ytdlp, local.",
        callback=_source_callback,
    ),
) -> None:
    """Browse the ``videos`` index."""
    from rytp.channels import list_videos, resolve_channel_id

    db = _open_db()
    try:
        channel_id: Optional[int] = None
        if channel is not None:
            channel_id = resolve_channel_id(db, channel)
            if channel_id is None:
                typer.echo(f"channel not found: {channel}", err=True)
                raise typer.Exit(code=1)
        rows = list_videos(db, channel_id=channel_id, kind=kind, source=source)
    finally:
        db.close()
    if not rows:
        typer.echo("(no videos match these filters)")
        return
    table_rows: list[list[str]] = []
    for v in rows:
        table_rows.append(
            [
                str(v.id),
                v.source,
                v.kind,
                v.title or "",
                "yes" if v.downloaded else "no",
            ]
        )
    _print_table(
        ["id", "source", "kind", "title", "downloaded"],
        table_rows,
    )


# ---------------------------------------------------------------------------
# singleton: download
# ---------------------------------------------------------------------------


@app.command("download")
def download_cmd(
    video_id_or_url: str = typer.Argument(
        ...,
        help=(
            "Either a numeric ``videos.id`` for a row already in the "
            "index, or the URL of such a row. For brand-new URLs, "
            "register them first with `rytp videos add <url>`."
        ),
    ),
    format_selector: str = typer.Option(
        # The default mirrors the user's sample run: a separate-stream
        # selector so audio and video land as two files which the
        # download stage merges into a single mp4.
        "worstvideo[height=720]+bestaudio[language=ru]",
        "--format",
        "-f",
        help=(
            "yt-dlp format selector. When the string contains a `+`, "
            "the audio and video streams are downloaded as two separate "
            "files and merged with ffmpeg; the merged file is "
            "``videos.downloaded_path`` and the original audio file is "
            "``videos.downloaded_audio_path``."
        ),
    ),
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        help="Don't attempt to resume a partial download.",
    ),
) -> None:
    """Download a single video and mark ``videos.downloaded = 1``.

    The media file lands under ``data/media/``; the resulting path
    is recorded in ``videos.downloaded_path`` so the transcribe
    stage can find it. When the ``--format`` selector contains ``+``
    (separate audio/video streams), the two are downloaded as
    separate files and then merged with ffmpeg into a single mp4 —
    the merge is transparent to the rest of the pipeline.

    Requires the ``[yt-dlp]`` extra (``pip install rytp[yt-dlp]``).
    """
    from rytp.download import download_one

    db = _open_db()
    try:
        try:
            vid = download_one(
                db,
                video_id_or_url,
                resume=not no_resume,
                format_selector=format_selector,
            )
        except ImportError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1) from e
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1) from e
    finally:
        db.close()
    typer.echo(f"video {vid}: downloaded")


# ---------------------------------------------------------------------------
# queue group (mostly wired; worker is a stub)
# ---------------------------------------------------------------------------


@queue_app.command("add")
def queue_add(
    video_ids: list[int] = typer.Argument(..., help="Video row ids to enqueue."),
) -> None:
    """Enqueue videos for the download worker."""
    from rytp.download.queue import Queue

    db = _open_db()
    try:
        try:
            q = Queue(db)
            new_ids = q.enqueue(video_ids)
        except sqlite3.IntegrityError as e:
            # Foreign key failure — one of the video ids doesn't exist.
            # Match the wording the rest of the CLI uses for video
            # lookups so users see a consistent message.
            missing = ", ".join(str(v) for v in video_ids)
            typer.echo(
                f"video {missing} not found in videos table "
                "(the queue table references videos.id)",
                err=True,
            )
            raise typer.Exit(code=1) from e
    finally:
        db.close()
    typer.echo(f"enqueued {len(new_ids)} (already pending/running: {len(video_ids) - len(new_ids)})")


@queue_app.command("worker")
def queue_worker(
    once: bool = typer.Option(
        False,
        "--once",
        help=(
            "Process at most one queue item then exit. Useful for "
            "testing or for `cron`-style invocation; the long-running "
            "default is to keep polling until the queue is empty."
        ),
    ),
    poll_s: float = typer.Option(
        C.QUEUE_POLL_INTERVAL_S,
        "--poll-s",
        help="Seconds to sleep between claims when the queue is empty.",
    ),
) -> None:
    """Long-running download worker; respects ``queue_paused``.

    Loops: claim a pending item, run :func:`rytp.download.download_one`,
    mark the item done/failed, repeat. Stops cleanly when:

    * the queue has no more pending items **and** ``--once`` was
      passed (one-shot mode);
    * the user hits Ctrl-C (KeyboardInterrupt — graceful exit).
    """
    from rytp.download import run_queued_item
    from rytp.download.queue import Queue

    db = _open_db()
    try:
        q = Queue(db)
        try:
            while True:
                if q.is_paused():
                    typer.echo("queue paused; exiting")
                    return
                item = q.claim_next()
                if item is None:
                    if once:
                        return
                    db.close()  # release the DB during the sleep
                    import time as _time

                    _time.sleep(poll_s)
                    db = _open_db()
                    q = Queue(db)
                    continue
                typer.echo(f"claimed queue item {item['id']} (video {item['video_id']})")
                ok, msg = run_queued_item(db, item["id"])
                typer.echo(("ok: " if ok else "fail: ") + msg)
                if once:
                    return
        except KeyboardInterrupt:
            typer.echo("\ninterrupted — leaving running items in 'running' state", err=True)
    finally:
        try:
            db.close()
        except sqlite3.ProgrammingError:
            pass


@queue_app.command("pause")
def queue_pause() -> None:
    """Pause the queue: the worker stops claiming new items."""
    from rytp.download.queue import Queue

    db = _open_db()
    try:
        Queue(db).pause()
    finally:
        db.close()
    typer.echo("queue paused")


@queue_app.command("resume")
def queue_resume() -> None:
    """Resume the queue: the worker starts claiming new items again."""
    from rytp.download.queue import Queue

    db = _open_db()
    try:
        Queue(db).resume()
    finally:
        db.close()
    typer.echo("queue resumed")


@queue_app.command("list")
def queue_list() -> None:
    """Show queue contents (pending, running, done, failed counts + rows)."""
    from rytp.download.queue import Queue

    db = _open_db()
    try:
        q = Queue(db)
        stats = q.stats()
        rows = q.list_pending(limit=C.QUEUE_LIST_LIMIT)
    finally:
        db.close()
    typer.echo(
        f"pending={stats.get('pending', 0)} "
        f"running={stats.get('running', 0)} "
        f"done={stats.get('done', 0)} "
        f"failed={stats.get('failed', 0)}"
    )
    if not rows:
        return
    _print_table(
        ["id", "video_id", "status", "started_at", "attempts"],
        [
            [
                str(r["id"]),
                str(r["video_id"]),
                r["status"],
                r["started_at"] or "",
                str(r["attempts"]),
            ]
            for r in rows
        ],
    )


# ---------------------------------------------------------------------------
# singleton: transcribe (stub)
# ---------------------------------------------------------------------------


@app.command("transcribe")
def transcribe_cmd(
    video_id: int = typer.Argument(..., help="FK to the videos table."),
    stt: Optional[str] = typer.Option(
        None,
        "--stt",
        help="STT engine name. Falls back to settings.default_stt_engine.",
    ),
    diarizer: Optional[str] = typer.Option(
        None,
        "--diarizer",
        help="Diarizer name. Falls back to settings.default_diarizer.",
    ),
    combined: Optional[str] = typer.Option(
        None,
        "--combined",
        help="Combined STT+Diarize engine (mutually exclusive with --stt/--diarizer).",
    ),
    language: Optional[str] = typer.Option(
        None, "--language", help="BCP-47 language code (e.g. 'ru', 'en')."
    ),
    diarizer_sensitivity: Optional[float] = typer.Option(
        None,
        "--diarizer-sensitivity",
        help="Diarizer sensitivity (0.0-1.0, only used with --diarizer energy).",
    ),
) -> None:
    """Extract audio WAV, run STT (chunked) + Diarizer, write ``words``.

    Pipeline (see DESIGN §5 / §11):

    1. Extract the per-video 16 kHz mono WAV into ``data/audio/``.
    2. Plan chunks: below the 30-min threshold the audio is
       transcribed whole; above it the audio is sliced into
       25-min chunks with 5-min overlap and each chunk is
       transcribed separately.
    3. Run the named STT engine (or combined engine) and produce
       ``Word`` (or ``DiarizedWord``) records.
    4. If a STT+Diarize split path is in use, run the named
       Diarizer on the **full** audio and merge.
    5. Bulk-insert ``words`` rows with diarizer labels set.

    The HF_TOKEN pre-launch gate (DESIGN §12) runs at the top
    level and gates any engine whose ``requires_hf_token`` is True.
    The ``null`` STT engine is registered for tests and writes
    nothing; it's also the default when no engine is configured.
    """
    from rytp.config import load_settings
    from rytp.transcribe.run import transcribe_video

    settings = load_settings()
    stt_name = stt or settings.default_stt_engine
    diarizer_name = diarizer or settings.default_diarizer
    combined_name = combined or settings.default_combined_engine

    db = _open_db()
    try:
        try:
            result = transcribe_video(
                db,
                video_id,
                stt_engine=stt_name,
                diarizer=diarizer_name,
                combined_engine=combined_name,
                language=language,
                diarizer_sensitivity=diarizer_sensitivity,
            )
        except (ImportError, ValueError, FileNotFoundError, RuntimeError) as e:
            _handle_business_error(e)
    finally:
        try:
            db.close()
        except sqlite3.ProgrammingError:
            # Connection already closed (e.g. via context-manager); don't
            # let a double-close mask the real result.
            pass
    typer.echo(
        f"video {video_id}: wrote {result.n_words} words "
        f"(run {result.transcribe_run_id}, diarizer={diarizer_name})"
    )


# ---------------------------------------------------------------------------
# speakers group (mostly wired; map is a stub)
# ---------------------------------------------------------------------------


@speakers_app.command("add")
def speakers_add(
    label: str = typer.Argument(..., help="Canonical speaker label (e.g. 'Alice')."),
    alias: list[str] = typer.Option(
        [], "--alias", help="Add an alias for the speaker. Repeatable."
    ),
    notes: Optional[str] = typer.Option(None, "--notes"),
) -> None:
    """Add to the global ``speakers`` roster. Idempotent on ``label``.

    If the label already exists, ``--alias`` and ``--notes`` updates are
    silently ignored by ``add_speaker``; this command prints a clear
    warning so the user knows their second invocation was a no-op. Use
    a future ``speakers update`` (planned for v2) to mutate an existing
    speaker's aliases/notes.
    """
    from rytp.speakers import add_speaker, find_alias_collisions

    db = _open_db()
    try:
        sid, was_inserted = add_speaker(db, label, aliases=alias, notes=notes)
        # Only check for alias collisions when the alias was actually
        # stored. If ``add_speaker`` returned ``was_inserted=False``,
        # the new aliases were dropped on the floor and any collision
        # warning would be misleading (the alias isn't in the roster
        # yet).
        collisions: list[tuple[str, int]] = []
        if was_inserted and alias:
            collisions = find_alias_collisions(
                db, aliases=alias, exclude_speaker_id=sid
            )
    finally:
        db.close()
    if was_inserted:
        typer.echo(f"speaker {sid}: {label} (added)")
    else:
        typer.echo(f"speaker {sid}: {label}")
    # Tell the user when the idempotent path silently dropped their
    # --alias / --notes input. ``add_speaker`` is documented as
    # idempotent, but "idempotent" doesn't explain that *modifiers* are
    # dropped on re-call -- this warning is what closes that gap.
    if not was_inserted and (alias or notes is not None):
        dropped: list[str] = []
        if alias:
            dropped.append(f"--alias ({len(alias)} value(s))")
        if notes is not None:
            dropped.append("--notes")
        typer.echo(
            "warning: speaker " + f'"{label}"' + " already exists; "
            + ", ".join(dropped)
            + " were not applied because `add` is idempotent on label. "
            "Use `rytp speakers update` (planned v2) to mutate existing "
            "speakers' aliases/notes.",
            err=True,
        )
    if collisions:
        # De-dupe by conflicting speaker for the message.
        unique_targets = sorted({sid for _, sid in collisions})
        target_list = ", ".join(
            f"speaker {sid}" for sid in unique_targets
        )
        typer.echo(
            f"warning: {len(collisions)} alias(es) collide with "
            f"{target_list} — the speaker-mapping step may pick the "
            "wrong speaker for matching labels.",
            err=True,
        )


@speakers_app.command("list")
def speakers_list() -> None:
    """List the global roster."""
    from rytp.speakers import list_speakers

    db = _open_db()
    try:
        rows = list_speakers(db)
    finally:
        db.close()
    if not rows:
        typer.echo("(no speakers yet — try `rytp speakers add <label>`)")
        return
    _print_table(
        ["id", "label", "aliases", "notes"],
        [
            [str(s.id), s.label, ", ".join(s.aliases), s.notes or ""]
            for s in rows
        ],
    )


@speakers_app.command("recompute-pauses")
def speakers_recompute_pauses() -> None:
    """Recompute ``speaker_pause_stats`` for every speaker."""
    from rytp.speakers import recompute_pause_stats

    db = _open_db()
    try:
        n = recompute_pause_stats(db)
    finally:
        db.close()
    typer.echo(f"recomputed pause stats for {n} speaker" + ("s" if n != 1 else ""))


@speakers_app.command("map")
def speakers_map(
    video_id: int = typer.Argument(
        ...,
        help="FK to the videos table to map. Lists all raw diarizer labels in the video.",
    ),
) -> None:
    """Headless one-shot mapping: list every raw diarizer label in a video.

    A full TUI mapper (with fuzzy match against the existing
    roster + per-label confirm) is on the v2 roadmap. For now
    this command prints the labels so the user can call
    :func:`rytp.speakers.map_diarizer_to_speaker` from Python
    or via future interactive plumbing.

    See :class:`rytp.speakers.SpeakerMapper` for the in-process
    mapper object that the future TUI will be built on.
    """
    from rytp.speakers import SpeakerMapper

    db = _open_db()
    try:
        # Validate the video id first -- ``distinct_diarizer_speakers``
        # returns an empty list for any unknown id, which the previous
        # version of this command conflated with "no labels yet" and
        # silently printed the global roster instead. A nonexistent
        # video_id is an error, not an empty result.
        video_row = db.conn.execute(
            "SELECT id FROM videos WHERE id = ?", (video_id,)
        ).fetchone()
        if video_row is None:
            typer.echo(
                f"video {video_id} not found in videos table",
                err=True,
            )
            raise typer.Exit(code=1)

        mapper = SpeakerMapper(db, video_id)
        labels = mapper.left_pane()
    finally:
        db.close()
    if not labels:
        typer.echo(f"video {video_id} has no diarizer labels yet")
        return
    typer.echo(f"raw diarizer labels in video {video_id}:")
    for label in labels:
        typer.echo(f"  - {label!r}")


# --- singleton: mine -------------------------------------------------------


@app.command("mine")
def mine_cmd(
    query: str = typer.Argument(..., help="Search string (substring via FTS5 prefix)."),
    cohesion: str = typer.Option(
        "med",
        "--cohesion",
        help="Window cohesion: low (single word), med (5 s window), high (15 s window).",
        callback=_cohesion_callback,
    ),
    max_clips: int = typer.Option(
        C.DEFAULT_MAX_CLIPS,
        "--max-clips",
        help="Maximum number of clips to write.",
    ),
) -> None:
    """Produce a ``clips`` set from the FTS5-indexed ``words`` table.

    See DESIGN §7 and :func:`rytp.mine.mine` for the algorithm.
    """
    from rytp.mine import mine

    db = _open_db()
    try:
        clip_ids = mine(db, query, cohesion=cohesion, max_clips=max_clips)
    finally:
        db.close()
    if not clip_ids:
        typer.echo(f"no matches for {query!r}")
    else:
        typer.echo(f"wrote {len(clip_ids)} clips: {list(clip_ids)}")


# --- singleton: splice -----------------------------------------------------


@app.command("splice")
def splice_cmd(
    clip_set_id: int = typer.Argument(
        ...,
        help=(
            "Splice-set identifier. Today, pass a single clip's id and "
            "the stage will splice that one clip; for full v2 you would "
            "pass a clip-set id that groups a top-N result set."
        ),
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Output video path. Defaults to data/output/splice-{timestamp}.mp4.",
    ),
    mode: str = typer.Option(
        "concat",
        "--mode",
        help="Splice mode: 'concat' (re-encode + loudnorm) or 'stream-copy'.",
        callback=_splice_mode_callback,
    ),
) -> None:
    """Run ffmpeg; write the spliced output + a sidecar manifest JSON.

    The current implementation takes a single ``clips.id`` and
    splices just that clip. DESIGN §7 calls for a "clip-set" notion
    that groups a mine-result set; v1 ships the single-clip path
    so the heavy-lift splice is exercised end-to-end. v2 will
    accept a list / set id.
    """
    from rytp.splice import SpliceMode, splice_clips

    db = _open_db()
    try:
        try:
            result = splice_clips(
                db, [clip_set_id], output_path=out, mode=mode
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            _handle_business_error(e)
    finally:
        try:
            db.close()
        except sqlite3.ProgrammingError:
            pass
    typer.echo(
        f"wrote {result.output_path} (manifest: {result.manifest_path}, "
        f"{result.n_clips} clip)"
    )


# --- singleton: tui --------------------------------------------------------


@app.command("tui")
def tui_cmd() -> None:
    """Launch the interactive textual TUI.

    The TUI exposes two read-only screens (videos, speakers) and
    is the v1 baseline; per DESIGN §6 the full §6 subcommand set
    in the TUI is v2.

    Requires the optional ``textual`` dependency; install with
    ``pip install rytp[all]`` (or just ``pip install textual rich``)
    if you see an ``ImportError`` about a missing module on launch.
    """
    db = _open_db()
    try:
        try:
            # Import inside the try so a missing textual package
            # surfaces as the friendly ``ImportError`` from
            # ``rytp.tui.app`` instead of a raw traceback at module
            # import time.
            from rytp.tui.app import run_tui

            run_tui(db)
        except ImportError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1) from e
    finally:
        try:
            db.close()
        except sqlite3.ProgrammingError:
            pass


# --- transcripts group ------------------------------------------------------


@transcripts_app.command("export")
def transcripts_export(
    video_id: int = typer.Argument(..., help="FK to the videos table."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output path. Defaults to data/transcripts/{video_id}.md.",
    ),
    min_block_s: float = typer.Option(
        C.MIN_BLOCK_DURATION_S,
        "--min-block-s",
        help=(
            "Minimum block duration in seconds. Blocks shorter than "
            "this are absorbed into a same-speaker neighbor when "
            "possible."
        ),
    ),
) -> None:
    """Export a speaker-grouped markdown transcript for a single video.

    The output is a markdown file with one ``### [HH:MM:SS] Speaker``
    header per turn followed by the turn's text. This is the
    DESIGN §7 "Output formats" human-readable export.
    """
    from rytp.config import paths as config_paths
    from rytp.db import Database
    from rytp.transcripts import export_markdown

    db = Database(config_paths.db)
    try:
        try:
            db.migrate()
            written = export_markdown(
                db, video_id, min_block_s=min_block_s, out_path=out
            )
        except ValueError as e:
            # ``export_markdown`` raises ``ValueError`` when the video
            # id doesn't exist — turn that into a clean exit instead
            # of leaking the traceback.
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1) from e
        except PermissionError as e:
            typer.echo(
                f"cannot write to {e.filename or 'the target path'} "
                f"(Access is denied). Check that the parent directory is writable.",
                err=True,
            )
            raise typer.Exit(code=1) from e
    finally:
        try:
            db.close()
        except sqlite3.ProgrammingError:
            pass
    typer.echo(str(written))


if __name__ == "__main__":
    app()