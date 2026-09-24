"""The transcribe command group (contracts §5).

Thin handlers. Each takes an open ``Database`` first, returns a
``CommandResult``, and raises ``RytpError`` rather than printing or exiting —
the surface formats the error. Both the CLI and the TUI are generated from
these definitions, so neither may add a command of its own.
"""
from __future__ import annotations

from pathlib import Path

from rytp import constants as C
from rytp.audio.acoustics import Acoustics, fingerprint_video
from rytp.audio.energy import read_wav_mono
from rytp.commands import REQUIRED, Command, CommandResult, Param, register, resolve_video_id
from rytp.db import Database
from rytp.models import RytpError
from rytp.transcribe import health  # noqa: F401 - importing registers the checks
from rytp.transcribe.captions import ingest_captions
from rytp.transcribe.compare import comparison_rows, render_report, run_comparison
from rytp.transcribe.pipeline import (
    TranscribeOutcome,
    enqueue_index,
    invalidate_transcript,
    realign_video,
    speaker_loss_warning,
    transcribe_video,
)
from rytp.transcribe.registry import default_transcriber, engine_rows, resolve_transcriber, setting


def _load_engine_modules() -> None:
    """Import the adapter packages, which is what makes their names resolve.

    Deferred to call time so importing this module stays cheap for the TUI.
    """
    import rytp.transcribe.align
    import rytp.transcribe.engines  # noqa: F401


def _wav_for(video_id: int, override: Path | None) -> Path:
    """The cached WAV Part 2 produced (contracts §7, ``cache/wav/{id}.wav``).

    The single import site of Part 2's path helper. Part 3 never builds that
    path itself — the cache is Part 2's to place and to prune.
    """
    if override is not None:
        path = Path(override)
    else:
        from rytp.audio.extract import wav_path

        path = wav_path(video_id)
    if not path.exists():
        raise RytpError(f"no cached WAV for video {video_id} at {path}; run ingest first")
    return path


def _queue(db: Database, kind: str, video_id: int, payload: dict[str, object]) -> CommandResult:
    """Hand the work to the queue instead of doing it here (contracts §5).

    Every long-running command in this group needs this: the TUI may not run
    them inline, and without a producer the gpu pool would never receive
    transcription work at all. ``enqueue`` is idempotent on
    ``UNIQUE (kind, target_id)``, so this cannot collide with Part 2's chain.
    """
    from rytp.jobs.queue import enqueue

    job_id = enqueue(db, kind, video_id, payload=payload)
    return CommandResult(message=f"queued {kind} job {job_id} for video {video_id}")


def _split(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _score_heading(scale: str | None) -> str:
    """Name what a run's score column actually measures (BUGS.md entry 13).

    ``median align score`` printed the same heading over an energy measure,
    a log-probability and nothing at all — the exact defect this batch fixes.
    A run that produced no score at all (no aligner, no refine; or a
    scoreless aligner with refine off) keeps the generic heading, which pairs
    with the always-blank cell :func:`_number` prints for a ``None`` median.
    """
    if scale is None:
        return "median align score"
    return f"median {scale} score"


def _device_note(outcome: TranscribeOutcome) -> str | None:
    return None if outcome.align_device is None else f"aligner device: {outcome.align_device}"


def _captions_handler(db: Database, *, video: str, path: Path | None = None) -> CommandResult:
    video_id = resolve_video_id(db, video)
    count = ingest_captions(db, video_id, Path(path) if path is not None else None)
    return CommandResult(message=f"{count} caption words for video {video_id}")


def _run_handler(
    db: Database,
    *,
    video: str,
    transcriber: str = "",
    aligner: str = "",
    language: str = C.DEFAULT_LANGUAGE,
    refine: bool = True,
    wav: Path | None = None,
    enqueue: bool = False,
) -> CommandResult:
    video_id = resolve_video_id(db, video)
    if enqueue:
        # Resolve the engine here rather than leaving it to the worker.
        # contracts §3 wants an unregistered name rejected at enqueue time,
        # and wants the choice announced when it came from the setting —
        # neither is possible once this is a row in a table nobody is
        # watching. `resolve_transcriber` is a *name* lookup, deliberately
        # not `check_available`: enqueueing must not require the engine to be
        # installed on the machine doing the enqueueing, only on the worker.
        _load_engine_modules()
        chosen_transcriber = transcriber.strip() or default_transcriber(db)
        resolve_transcriber(chosen_transcriber)
        result = _queue(
            db,
            "transcribe",
            video_id,
            {
                "transcriber": chosen_transcriber,
                "aligner": aligner,
                "language": language,
                "refine": refine,
            },
        )
        if not transcriber.strip():
            return CommandResult(
                message=(
                    f"{result.message}; transcriber {chosen_transcriber!r}, from "
                    f"setting {C.SETTING_DEFAULT_TRANSCRIBER} "
                    f"(pass --transcriber to override)"
                )
            )
        return result
    _load_engine_modules()
    # An empty --transcriber falls back to the configured default, and says
    # which engine that was. Contracts §3: the owner chose "set a default and
    # inform" over "refuse until configured", so this line is the entire
    # mechanism keeping the choice visible instead of accidental — 35-90
    # GPU-hours is expensive to discover late. If that engine is not
    # installed, `load_transcriber` raises `EngineUnavailable` with the pip
    # hint and nothing here catches it: a silent substitution would leave the
    # corpus holding rows from two engines with nothing recording which.
    chosen_transcriber = transcriber.strip() or default_transcriber(db)
    # An empty --aligner falls back to the configured default. Contracts §3:
    # with no aligner at all the result is the `timed` tier, which is
    # searchable but not cuttable, so the setting is how an owner who always
    # wants cuttable words stops having to remember the flag.
    chosen_aligner = aligner or setting(db, "default_aligner", "") or None
    outcome = transcribe_video(
        db,
        video_id,
        wav_path=_wav_for(video_id, wav),
        transcriber=chosen_transcriber,
        aligner=chosen_aligner,
        language=language or None,
        refine=refine,
    )
    # The words just changed, so the video's utterances are gone and its
    # search index is stale until Part 4 rebuilds it.
    enqueue_index(db, video_id)
    notes = [f"video {video_id}: {outcome.n_words} {outcome.source} words"]
    if not transcriber.strip():
        notes.append(
            f"transcriber {chosen_transcriber!r}, from setting "
            f"{C.SETTING_DEFAULT_TRANSCRIBER} (pass --transcriber to override)"
        )
    if outcome.source == "timed":
        notes.append(
            "not cuttable — run `rytp transcribe align` with an aligner to upgrade them"
        )
    warning = speaker_loss_warning(video_id, outcome.speakers_lost)
    if warning:
        notes.append(warning)
    device_note = _device_note(outcome)
    if device_note:
        notes.append(device_note)
    notes.extend(outcome.align_notes)
    return CommandResult(
        columns=(
            "video", "words", "chunks", "tier", "engine",
            _score_heading(outcome.align_scale),
        ),
        rows=(
            (
                str(outcome.video_id),
                str(outcome.n_words),
                str(outcome.n_chunks),
                outcome.source,
                outcome.engine,
                _number(outcome.median_align_score),
            ),
        ),
        message=". ".join(notes),
    )


def _align_handler(
    db: Database,
    *,
    video: str,
    aligner: str,
    refine: bool = True,
    wav: Path | None = None,
    enqueue: bool = False,
) -> CommandResult:
    video_id = resolve_video_id(db, video)
    if enqueue:
        return _queue(db, "align", video_id, {"aligner": aligner, "refine": refine})
    _load_engine_modules()
    outcome = realign_video(
        db,
        video_id,
        wav_path=_wav_for(video_id, wav),
        aligner=aligner,
        refine=refine,
    )
    enqueue_index(db, video_id)
    notes = [f"video {video_id}: {outcome.n_words} words are now cuttable"]
    device_note = _device_note(outcome)
    if device_note:
        notes.append(device_note)
    notes.extend(outcome.align_notes)
    return CommandResult(
        columns=(
            "video", "words", "chunks", "tier", "engine",
            _score_heading(outcome.align_scale),
        ),
        rows=(
            (
                str(outcome.video_id),
                str(outcome.n_words),
                str(outcome.n_chunks),
                outcome.source,
                outcome.engine,
                _number(outcome.median_align_score),
            ),
        ),
        message=". ".join(notes),
    )


def _compare_handler(
    db: Database,
    *,
    video: str,
    transcribers: str,
    aligners: str = "",
    start_ms: int = 0,
    end_ms: int = C.COMPARE_DEFAULT_WINDOW_MS,
    out: Path | None = None,
    wav: Path | None = None,
) -> CommandResult:
    video_id = resolve_video_id(db, video)
    _load_engine_modules()
    names = _split(transcribers)
    if not names:
        raise RytpError("pass at least one transcriber, comma separated")
    wav_path = _wav_for(video_id, wav)
    runs = run_comparison(
        db,
        wav_path=wav_path,
        transcribers=names,
        aligners=_split(aligners),
        start_ms=start_ms,
        end_ms=end_ms or None,
    )
    samples, sr = read_wav_mono(wav_path)
    columns, rows = comparison_rows(runs, samples, sr)
    message = None
    if out is not None:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            render_report(runs, samples, sr, source=str(wav_path)), encoding="utf-8"
        )
        message = f"report written to {target}"
    return CommandResult(columns=columns, rows=tuple(rows), message=message)


def _engines_handler(db: Database) -> CommandResult:
    _load_engine_modules()
    return CommandResult(
        columns=("name", "kind", "hf token", "out of process", "state"),
        rows=tuple(engine_rows(db)),
    )


def _fingerprint_handler(
    db: Database, *, video: str, wav: Path | None = None, enqueue: bool = False
) -> CommandResult:
    video_id = resolve_video_id(db, video)
    if enqueue:
        return _queue(db, "fingerprint", video_id, {})
    acoustics: Acoustics = fingerprint_video(
        db, video_id, wav_path=_wav_for(video_id, wav)
    )
    return CommandResult(
        columns=("f0 mean", "f0 std", "tilt dB/dec", "noise floor dB", "reverb dB", "LUFS"),
        rows=(
            (
                _number(acoustics.f0_mean),
                _number(acoustics.f0_std),
                _number(acoustics.spectral_tilt),
                _number(acoustics.noise_floor_db),
                _number(acoustics.reverb_proxy),
                _number(acoustics.loudness_lufs),
            ),
        ),
    )


_VIDEO_ID = Param(
    name="video", type=str, help="Video row id or external id.", default=REQUIRED,
    positional=True,
)
_WAV = Param(
    name="wav",
    type=Path,
    help="Use this WAV instead of the cached one.",
    default=None,
)
_ENQUEUE = Param(
    name="enqueue",
    type=bool,
    help="Queue the work for the worker instead of running it now.",
    default=False,
)

register(
    Command(
        name="transcribe.captions",
        group="transcribe",
        summary="Ingest downloaded auto-captions as caption-tier words (not cuttable).",
        params=(
            _VIDEO_ID,
            Param(
                name="path",
                type=Path,
                help="Captions file to read instead of the registered asset.",
                default=None,
            ),
        ),
        handler=_captions_handler,
    )
)

register(
    Command(
        name="transcribe.run",
        group="transcribe",
        summary="Transcribe, align and refine a video into cuttable aligned words.",
        params=(
            _VIDEO_ID,
            Param(
                name="transcriber",
                type=str,
                help=(
                    "Registered transcriber name. Empty falls back to the "
                    "default_transcriber setting ('gigaam' out of the box), "
                    "and the result says which engine that chose. If it is "
                    "not installed the command fails with the install hint "
                    "rather than running a different one."
                ),
                default="",
            ),
            Param(
                name="aligner",
                type=str,
                help=(
                    "Registered aligner name. Empty falls back to the "
                    "default_aligner setting; with neither, the words are "
                    "written as the searchable-but-not-cuttable 'timed' tier."
                ),
                default="",
            ),
            Param(name="language", type=str, help="Spoken language.", default=C.DEFAULT_LANGUAGE),
            Param(
                name="refine",
                type=bool,
                help="Place boundaries at the measured energy minimum.",
                default=True,
            ),
            _WAV,
            _ENQUEUE,
        ),
        handler=_run_handler,
        long_running=True,
    )
)

register(
    Command(
        name="transcribe.align",
        group="transcribe",
        summary="Re-time an already transcribed video with a different aligner.",
        params=(
            _VIDEO_ID,
            Param(name="aligner", type=str, help="Registered aligner name.", default=REQUIRED),
            Param(
                name="refine",
                type=bool,
                help="Place boundaries at the measured energy minimum.",
                default=True,
            ),
            _WAV,
            _ENQUEUE,
        ),
        handler=_align_handler,
        long_running=True,
    )
)

register(
    Command(
        name="transcribe.compare",
        group="transcribe",
        summary="Run several engines over the same audio and report where they disagree.",
        params=(
            _VIDEO_ID,
            Param(
                name="transcribers",
                type=str,
                help="Comma-separated transcriber names.",
                default=REQUIRED,
            ),
            Param(
                name="aligners",
                type=str,
                help="Comma-separated aligner names to pair with each transcriber.",
                default="",
            ),
            Param(name="start_ms", type=int, help="Window start.", default=0),
            Param(
                name="end_ms",
                type=int,
                help="Window end; 0 means to the end of the audio.",
                default=C.COMPARE_DEFAULT_WINDOW_MS,
            ),
            Param(name="out", type=Path, help="Write the markdown report here.", default=None),
            _WAV,
        ),
        handler=_compare_handler,
        long_running=True,
    )
)

register(
    Command(
        name="transcribe.engines",
        group="transcribe",
        summary="List registered transcribers and aligners and whether they can run.",
        params=(),
        handler=_engines_handler,
    )
)

register(
    Command(
        name="transcribe.fingerprint",
        group="transcribe",
        summary="Measure a video's acoustic fingerprint into video_acoustics.",
        params=(_VIDEO_ID, _WAV, _ENQUEUE),
        handler=_fingerprint_handler,
        long_running=True,
    )
)


def _remove_handler(db: Database, *, video: str) -> CommandResult:
    """Drop a video's transcript and everything derived from it.

    Takes no ``--dry-run`` and no ``--yes``: contracts §5 requires those of a
    command that deletes files, and this one deletes only derived database
    rows. The media, the captions asset and the cached WAV all survive, so
    ``transcribe run`` rebuilds exactly what this removed.
    """
    video_id = resolve_video_id(db, video)
    removed = invalidate_transcript(db, video_id)
    notes = [f"removed the transcript of video {video_id}"]
    warning = speaker_loss_warning(video_id, removed.video_speakers)
    if warning:
        notes.append(warning)
    return CommandResult(
        columns=("words", "utterances", "speaker labels"),
        rows=((str(removed.words), str(removed.utterances), str(removed.video_speakers)),),
        message=". ".join(notes),
    )


register(
    Command(
        name="transcribe.remove",
        group="transcribe",
        summary="Remove a video's words, and with them its utterances and speaker labels.",
        params=(_VIDEO_ID,),
        handler=_remove_handler,
    )
)
