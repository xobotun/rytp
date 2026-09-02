"""rytp CLI entry point.

Built on :mod:`typer`. This module wires every subcommand from
DESIGN §6, plus the top-level HF_TOKEN pre-launch gate (DESIGN §12).

The gate runs before any subcommand body so a missing token never
displays a half-broken TUI: it prints one clear paragraph to stderr
and exits non-zero.

**v1 subcommand status** (DESIGN §6):

* **Wired (have a backend):** ``channel add`` / ``sync`` / ``list``,
  ``videos add`` / ``list``, ``queue add`` / ``pause`` / ``resume`` /
  ``list``, ``speakers add`` / ``list`` / ``recompute-pauses``,
  ``transcripts export``.
* **Stubs (no backend yet — v2 work):** ``download``, ``queue worker``,
  ``transcribe``, ``speakers map`` (TUI), ``mine``, ``splice``, ``tui``.

A stub exits 1 with a clear ``"not implemented yet"`` message so
the user always knows where they stand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from rytp import config
from rytp import constants as C
from rytp import engines as engines_mod

app = typer.Typer(
    name="rytp",
    help="Download YouTube videos, transcribe them locally, datamine, and splice.",
    no_args_is_help=True,
    add_completion=False,
)

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

    Used by the v2 subcommands (``download``, ``transcribe``, etc.)
    that have no working backend yet.
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
) -> None:
    """Register a single video: URL or local file path.

    Local files are detected by :func:`rytp.channels._is_local_path`;
    no yt-dlp call is made for them, and ``source='local'``,
    ``downloaded=True`` are set in the resulting row. URLs are
    probed via yt-dlp, which requires the ``[yt-dlp]`` extra to be
    installed.
    """
    from rytp.channels import _is_local_path, register_video
    from rytp.download.ytdlp import RealYtDlpRunner

    db = _open_db()
    try:
        if _is_local_path(url_or_path):
            # No runner needed for local files; ``register_video`` skips
            # the probe entirely. Pass a placeholder to satisfy the
            # type signature — the function never calls it.
            class _UnusedRunner:
                def probe(self, url: str):  # pragma: no cover
                    raise RuntimeError("unreachable: local file branch")

            runner: object = _UnusedRunner()
        else:
            try:
                runner = RealYtDlpRunner()
            except ImportError as e:
                typer.echo(str(e), err=True)
                raise typer.Exit(code=1) from e
        vid = register_video(db, runner, url_or_path)
    finally:
        db.close()
    typer.echo(f"video {vid}: {url_or_path}")


@videos_app.command("list")
def videos_list(
    channel: Optional[str] = typer.Option(
        None, "--channel", help="Restrict to one channel (id, title, or URL)."
    ),
    kind: Optional[str] = typer.Option(
        None, "--kind", help="Filter by kind: video, short, livestream, other."
    ),
    source: Optional[str] = typer.Option(
        None, "--source", help="Filter by source: youtube, ytdlp, local."
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
# singleton: download (stub)
# ---------------------------------------------------------------------------


@app.command("download")
def download_cmd(
    video_id_or_url: str = typer.Argument(...),
) -> None:
    """Download a single video, mark ``downloaded``.

    Stub: backend wires up in v2. Use the ``yt-dlp`` Python API
    directly (or the CLI) for now.
    """
    _not_implemented("download")


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
        q = Queue(db)
        new_ids = q.enqueue(video_ids)
    finally:
        db.close()
    typer.echo(f"enqueued {len(new_ids)} (already pending/running: {len(video_ids) - len(new_ids)})")


@queue_app.command("worker")
def queue_worker() -> None:
    """Long-running worker; respects ``queue_paused``."""
    _not_implemented("queue worker")


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
    video_id: int = typer.Argument(...),
    stt: Optional[str] = typer.Option(None, "--stt"),
    diarizer: Optional[str] = typer.Option(None, "--diarizer"),
    combined: Optional[str] = typer.Option(None, "--combined"),
) -> None:
    """Extract audio WAV, run STT (chunked) + Diarizer (full audio), write ``words``.

    Stub: backend wires up in v2. The HF_TOKEN pre-launch gate
    (DESIGN §12) still runs in the top-level callback for the
    engines you named.
    """
    _not_implemented("transcribe")


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
    """Add to the global ``speakers`` roster. Idempotent on ``label``."""
    from rytp.speakers import add_speaker

    db = _open_db()
    try:
        sid = add_speaker(db, label, aliases=alias, notes=notes)
    finally:
        db.close()
    typer.echo(f"speaker {sid}: {label}")


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
    typer.echo(f"recomputed pause stats for {n} speaker(s)")


@speakers_app.command("map")
def speakers_map(
    video_id: int = typer.Argument(...),
) -> None:
    """Open the TUI mapper for this video (stub: v2)."""
    _not_implemented("speakers map")


# --- singleton: mine -------------------------------------------------------


@app.command("mine")
def mine_cmd(
    query: str = typer.Argument(...),
    cohesion: str = typer.Option("med", "--cohesion"),
    max_clips: int = typer.Option(C.DEFAULT_MAX_CLIPS, "--max-clips"),
) -> None:
    """Produce a ``clips`` set."""
    _not_implemented("mine")


# --- singleton: splice -----------------------------------------------------


@app.command("splice")
def splice_cmd(
    clip_set_id: int = typer.Argument(...),
    out: str = typer.Option(..., "--out"),
    mode: str = typer.Option("concat", "--mode"),
) -> None:
    """Run ffmpeg, write output + manifest under ``data/output/``."""
    _not_implemented("splice")


# --- singleton: tui --------------------------------------------------------


@app.command("tui")
def tui_cmd() -> None:
    """Launch the interactive TUI."""
    _not_implemented("tui")


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
        db.migrate()
        written = export_markdown(
            db, video_id, min_block_s=min_block_s, out_path=out
        )
    finally:
        db.close()
    typer.echo(str(written))


if __name__ == "__main__":
    app()