"""rytp CLI entry point.

Built on :mod:`typer`. This module wires every subcommand from
DESIGN §6, plus the top-level HF_TOKEN pre-launch gate (DESIGN §12).

The gate runs before any subcommand body so a missing token never
displays a half-broken TUI: it prints one clear paragraph to stderr
and exits non-zero.

Each subcommand body is currently a stub. Later segments replace
them in place — the wiring (group nesting, argument names, help
text) is what this module commits to.
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
# Subcommand stubs (DESIGN §6)
# ---------------------------------------------------------------------------


def _not_implemented(name: str) -> None:
    typer.echo(f"rytp {name}: not implemented yet", err=True)
    raise typer.Exit(code=1)


# --- channel group ---------------------------------------------------------


@channel_app.command("add")
def channel_add(url: str) -> None:
    """Register a channel."""
    _not_implemented("channel add")


@channel_app.command("sync")
def channel_sync(name: str) -> None:
    """yt-dlp flat-playlist, upsert into ``videos``."""
    _not_implemented("channel sync")


@channel_app.command("list")
def channel_list() -> None:
    """Show channels + counts."""
    _not_implemented("channel list")


# --- videos group ----------------------------------------------------------


@videos_app.command("add")
def videos_add(url_or_path: str) -> None:
    """Register a single video: URL, any yt-dlp URL, or a local file path."""
    _not_implemented("videos add")


@videos_app.command("list")
def videos_list(
    channel: Optional[str] = typer.Option(None, "--channel"),
    kind: Optional[str] = typer.Option(None, "--kind"),
    source: Optional[str] = typer.Option(None, "--source"),
) -> None:
    """Browse the index."""
    _not_implemented("videos list")


# --- singleton: download ---------------------------------------------------


@app.command("download")
def download_cmd(
    video_id_or_url: str = typer.Argument(...),
) -> None:
    """Download a single video, mark ``downloaded``."""
    _not_implemented("download")


# --- queue group -----------------------------------------------------------


@queue_app.command("add")
def queue_add(
    video_ids: list[int] = typer.Argument(...),
) -> None:
    """Enqueue videos."""
    _not_implemented("queue add")


@queue_app.command("worker")
def queue_worker() -> None:
    """Long-running worker; respects ``queue_paused``."""
    _not_implemented("queue worker")


@queue_app.command("pause")
def queue_pause() -> None:
    """Flip the global pause flag to True."""
    _not_implemented("queue pause")


@queue_app.command("resume")
def queue_resume() -> None:
    """Flip the global pause flag to False."""
    _not_implemented("queue resume")


@queue_app.command("list")
def queue_list() -> None:
    """Show queue contents."""
    _not_implemented("queue list")


# --- singleton: transcribe -------------------------------------------------


@app.command("transcribe")
def transcribe_cmd(
    video_id: int = typer.Argument(...),
    stt: Optional[str] = typer.Option(None, "--stt"),
    diarizer: Optional[str] = typer.Option(None, "--diarizer"),
    combined: Optional[str] = typer.Option(None, "--combined"),
) -> None:
    """Extract audio WAV, run STT (chunked) + Diarizer (full audio), write ``words``."""
    # The HF_TOKEN gate runs in the top-level callback. By the time we
    # reach this body, either the token is present or the user picked
    # a non-gated engine.
    _not_implemented("transcribe")


# --- speakers group --------------------------------------------------------


@speakers_app.command("add")
def speakers_add(
    label: str = typer.Argument(...),
    alias: list[str] = typer.Option([], "--alias"),
    notes: Optional[str] = typer.Option(None, "--notes"),
) -> None:
    """Add to the global ``speakers`` roster."""
    _not_implemented("speakers add")


@speakers_app.command("list")
def speakers_list() -> None:
    """List the roster."""
    _not_implemented("speakers list")


@speakers_app.command("recompute-pauses")
def speakers_recompute_pauses() -> None:
    """Recompute ``speaker_pause_stats`` for every speaker that has changed."""
    _not_implemented("speakers recompute-pauses")


@speakers_app.command("map")
def speakers_map(
    video_id: int = typer.Argument(...),
) -> None:
    """Open the TUI mapper for this video."""
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