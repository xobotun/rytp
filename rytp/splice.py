"""Splice stage — concat clips into a single output + manifest.

DESIGN §7 / §8: for ``mode=concat``, splice reads per-clip normalized
intermediates (produced by :mod:`rytp.loudnorm`) and concatenates
them with same-speaker inter-word pauses. For ``mode=stream-copy``,
splice re-reads the original media files with ``-c copy``.

The pause-sampling logic implements DESIGN §8's soft-blend between
the speaker's own pause distribution and a global default: with
fewer than :data:`rytp.constants.PAUSE_BLEND_SATURATION_SAMPLES`
samples, the global default carries weight; past that the speaker's
own stats dominate. Cross-speaker transitions skip the pause
entirely (DESIGN §8: "real cross-speaker turns are often backchannel
with no gap").

Public surface:

* :func:`splice_clips` — run the full splice pipeline for a clips set.
* :func:`sample_pause` — sample one inter-word pause between two clips.
* :class:`SpliceMode` — ``CONCAT`` or ``STREAM_COPY``.
* :class:`VideoStrategy` — ``FREEZE`` | ``BORROW`` | ``NONE``.
* :class:`SpliceResult` — return value from :func:`splice_clips`.
"""
from __future__ import annotations

import datetime as _dt
import enum
import json
import random
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rytp import constants as C
from rytp.config import paths as config_paths
from rytp.db import Database
from rytp.models import Speaker  # noqa: F401  (kept for re-export symmetry)
from rytp.speakers import get_pause_stats, list_speakers


class SpliceMode(str, enum.Enum):
    """Splice output mode (DESIGN §7).

    * ``CONCAT`` — re-encoded, normalized, smooth audio. Default.
      Reads per-clip intermediates from ``loudnorm``; same-speaker
      pauses are inserted as silent WAV segments.
    * ``STREAM_COPY`` — lossless and faster (just ``-c copy``), but
      with audible discontinuities at clip boundaries. Reads the
      original media files.
    """

    CONCAT = "concat"
    STREAM_COPY = "stream-copy"


class VideoStrategy(str, enum.Enum):
    """How the video stream covers an inserted silence gap (DESIGN §8).

    * ``NONE`` — no pause was inserted (stream-copy or cross-speaker).
    * ``FREEZE`` — last frame is held for the pause duration. Default
      and visually correct for talking-head content.
    * ``BORROW`` — the next clip's leading edge is trimmed and used as
      the visual cover. Reserved for v2; v1 always uses FREEZE.
    """

    NONE = "none"
    FREEZE = "freeze"
    BORROW = "borrow"


# ---------------------------------------------------------------------------
# Pause sampling (DESIGN §8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PauseSample:
    """One sampled inter-word pause between two consecutive clips.

    Attributes:
        inserted_pause_ms: How much silence to insert between the
            clips. ``0`` means "no pause" (cross-speaker, or no data).
        video_strategy: How to cover the silence on the video stream.
        speaker_id: The canonical speaker id of the *next* clip (the
            one whose onset the pause is leading into).
        adjusted_in_ms: For ``BORROW`` strategy, the trimmed leading
            edge of the next clip. ``None`` for FREEZE.
        adjusted_out_ms: For ``BORROW`` strategy, the trimmed trailing
            edge of the previous clip. ``None`` for FREEZE.
    """

    inserted_pause_ms: int
    video_strategy: VideoStrategy
    speaker_id: int | None
    adjusted_in_ms: int | None = None
    adjusted_out_ms: int | None = None


def sample_pause(
    db: Database,
    *,
    prev_speaker_id: int | None,
    curr_speaker_id: int | None,
    min_pause_ms: int = C.MIN_PAUSE_MS,
    max_pause_ms: int = C.MAX_PAUSE_MS,
) -> PauseSample:
    """Sample a pause for the gap between two consecutive clips.

    Implements DESIGN §8's pause modeling:

    * **Cross-speaker transitions** skip the pause entirely — real
      backchannel turns in conversational data are often gapless, and
      forcing one sounds stilted.
    * **Same speaker, no cached stats**: use the global defaults
      (:data:`C.GLOBAL_PAUSE_MEAN_MS`, / ``GLOBAL_PAUSE_STD_MS``) and
      FREEZE strategy.
    * **Same speaker, with stats**: a soft blend of the speaker's own
      distribution and the global default, weighted by sample count
      (``w = min(n / PAUSE_BLEND_SATURATION_SAMPLES, 1.0)``). At
      ``w=1`` the speaker's own distribution is used exclusively;
      below that the global default fills in. The blend prevents a
      hard cliff where 199 samples falls back to the global default
      but 201 samples uses the speaker's own stats with no smoothing.

    The clamp to ``[p10, p90]`` is applied only when ``w > 0.5`` — at
    low ``w`` the global default is dominant and clamping it into the
    speaker's range would defeat the purpose.

    Args:
        db: Database to look up ``speaker_pause_stats``.
        prev_speaker_id: Resolved canonical speaker of the previous clip.
        curr_speaker_id: Resolved canonical speaker of the current clip.
        min_pause_ms: Hard floor for the sampled pause (default
            :data:`C.MIN_PAUSE_MS`).
        max_pause_ms: Hard ceiling for the sampled pause (default
            :data:`C.MAX_PAUSE_MS`).

    Returns:
        A :class:`PauseSample` describing the pause to insert.
    """
    if prev_speaker_id != curr_speaker_id:
        return PauseSample(0, VideoStrategy.NONE, curr_speaker_id)
    if curr_speaker_id is None:
        # No resolved speaker → skip pause.
        return PauseSample(0, VideoStrategy.NONE, None)

    stats = get_pause_stats(db, curr_speaker_id)
    if stats is None or stats.n_samples < C.MIN_SAMPLES_FOR_PAUSE_STATS:
        # No data → fall back to a sensible default
        return PauseSample(
            inserted_pause_ms=int(C.GLOBAL_PAUSE_MEAN_MS),
            video_strategy=VideoStrategy.FREEZE,
            speaker_id=curr_speaker_id,
        )

    # Blend weight (DESIGN §8).
    w = min(1.0, stats.n_samples / float(C.PAUSE_BLEND_SATURATION_SAMPLES))
    mean = w * stats.mean_ms + (1.0 - w) * C.GLOBAL_PAUSE_MEAN_MS
    std = w * stats.std_ms + (1.0 - w) * C.GLOBAL_PAUSE_STD_MS

    sampled = random.gauss(mean, std)
    clamped = max(min_pause_ms, min(max_pause_ms, sampled))

    # At high speaker-data weight, also clamp into [p10, p90].
    if w > 0.5:
        clamped = max(stats.p10_ms, min(stats.p90_ms, clamped))

    return PauseSample(
        inserted_pause_ms=int(clamped),
        video_strategy=VideoStrategy.FREEZE,
        speaker_id=curr_speaker_id,
    )


# ---------------------------------------------------------------------------
# Splice runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpliceResult:
    """Result of a splice invocation.

    Attributes:
        splice_run_id: Row id in the ``splice_runs`` table.
        output_path: The spliced output video.
        manifest_path: Sidecar JSON describing every clip in order.
        n_clips: How many clips were spliced.
    """

    splice_run_id: int
    output_path: Path
    manifest_path: Path
    n_clips: int


def splice_clips(
    db: Database,
    clip_ids: Iterable[int],
    *,
    output_path: Path | None = None,
    mode: SpliceMode | str = SpliceMode.CONCAT,
) -> SpliceResult:
    """Splice a set of clips into one output file + manifest.

    Looks up each clip in the ``clips`` table, joins against
    ``videos``, resolves per-clip ``speaker_id`` (via the
    ``videos_speaker_map`` join), samples pauses, then runs ffmpeg.

    Writes:

    * the output video at ``output_path`` (default
      ``data/output/splice-{timestamp}.mp4``).
    * the manifest at ``output_path.with_suffix('.manifest.json')``.
    * one ``splice_runs`` row + ``splice_clips`` rows for audit.

    The ``splice_clips`` rows are written *before* ffmpeg runs, so a
    crash mid-splice still leaves an audit trail in the DB. The
    ffmpeg intermediate files (``_silence-*.wav``, ``_streamclip-*``)
    are not cleaned up automatically — they're cheap and useful for
    debugging, but a future cleanup pass is in DESIGN §10.

    Args:
        db: Open Database.
        clip_ids: Clip row ids, in splice order.
        output_path: Where the output is written. Defaults to
            ``data/output/splice-{timestamp}.mp4``.
        mode: :class:`SpliceMode.CONCAT` (default) or
            :class:`SpliceMode.STREAM_COPY`.

    Raises:
        ValueError: if no clips are passed, or some clip ids don't
            resolve to rows in the ``clips`` table.
        FileNotFoundError: if a referenced media file or normalized
            intermediate is missing.
        RuntimeError: if ffmpeg isn't on PATH.
    """
    if isinstance(mode, str):
        mode = SpliceMode(mode)

    clip_ids = list(clip_ids)
    if not clip_ids:
        raise ValueError("no clips to splice")

    now = _dt.datetime.utcnow().isoformat()
    if output_path is None:
        stamp = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        output_path = config_paths.output / f"splice-{stamp}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open the splice_runs row up front so failures are visible.
    cur = db.conn.execute(
        """
        INSERT INTO splice_runs (output_path, mode, created_at)
        VALUES (?, ?, ?)
        """,
        (str(output_path), mode.value, now),
    )
    splice_run_id = cur.lastrowid
    db.conn.commit()

    # Load clip rows, in order (caller's order).
    placeholders = ",".join("?" * len(clip_ids))
    rows = db.conn.execute(
        f"""
        SELECT c.id AS clip_id, c.video_id, c.start_ms, c.end_ms,
               c.source_query, v.downloaded_path, v.local_path,
               COALESCE(m.speaker_id, w.speaker_id) AS resolved_speaker_id
        FROM clips c
        JOIN videos v ON v.id = c.video_id
        LEFT JOIN videos_speaker_map m
            ON m.video_id = c.video_id AND m.diarizer_speaker = (
                SELECT diarizer_speaker FROM words
                WHERE video_id = c.video_id AND start_ms = c.start_ms
                LIMIT 1
            )
        LEFT JOIN words w
            ON w.video_id = c.video_id AND w.start_ms = c.start_ms
        WHERE c.id IN ({placeholders})
        ORDER BY c.id
        """,
        tuple(clip_ids),
    ).fetchall()
    if len(rows) != len(set(clip_ids)):
        raise ValueError(
            f"requested {len(clip_ids)} clips, only {len(rows)} found in DB"
        )

    # Compute per-clip pauses.
    pauses: list[PauseSample] = []
    prev_speaker_id: int | None = None
    for r in rows:
        sid = r["resolved_speaker_id"]
        p = sample_pause(
            db, prev_speaker_id=prev_speaker_id, curr_speaker_id=sid
        )
        pauses.append(p)
        prev_speaker_id = sid

    # Persist splice_clips rows (audit) before running ffmpeg.
    for ord_, (r, p) in enumerate(zip(rows, pauses)):
        db.conn.execute(
            """
            INSERT INTO splice_clips (
                splice_run_id, ord, video_id, in_ms, out_ms, text,
                adjusted_in_ms, adjusted_out_ms, inserted_pause_ms, video_strategy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                splice_run_id,
                ord_,
                r["video_id"],
                r["start_ms"],
                r["end_ms"],
                r["source_query"],
                p.adjusted_in_ms,
                p.adjusted_out_ms,
                p.inserted_pause_ms,
                p.video_strategy.value,
            ),
        )
    db.conn.commit()

    # Run ffmpeg.
    if mode == SpliceMode.CONCAT:
        _concat_mode(db, output_path, rows, pauses)
    else:
        _stream_copy_mode(output_path, rows, pauses)

    # Write the manifest next to the output.
    manifest_path = output_path.with_suffix(".manifest.json")
    speakers_by_id = {s.id: s for s in list_speakers(db)}
    manifest = {
        "splice_run_id": splice_run_id,
        "output_path": str(output_path),
        "mode": mode.value,
        "created_at": now,
        "clips": [
            {
                "ord": ord_,
                "video_id": r["video_id"],
                "in_ms": r["start_ms"],
                "out_ms": r["end_ms"],
                "text": r["source_query"],
                "resolved_speaker_id": r["resolved_speaker_id"],
                "speaker_label": (
                    speakers_by_id[r["resolved_speaker_id"]].label
                    if r["resolved_speaker_id"] in speakers_by_id
                    else None
                ),
                "inserted_pause_ms": p.inserted_pause_ms,
                "video_strategy": p.video_strategy.value,
                "adjusted_in_ms": p.adjusted_in_ms,
                "adjusted_out_ms": p.adjusted_out_ms,
            }
            for ord_, (r, p) in enumerate(zip(rows, pauses))
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    return SpliceResult(
        splice_run_id=splice_run_id,
        output_path=output_path,
        manifest_path=manifest_path,
        n_clips=len(rows),
    )


def _concat_mode(
    db: Database,
    output_path: Path,
    rows: list,
    pauses: list[PauseSample],
) -> None:
    """Run ffmpeg concat on the per-clip normalized intermediates."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")

    list_file = output_path.with_suffix(".concat.txt")
    lines: list[str] = []
    for r, p in zip(rows, pauses):
        # The intermediate for each clip is data/output/normalized/{clip_id}.m4a
        intermediate = config_paths.normalized / f"{r['clip_id']}.m4a"
        if not intermediate.exists():
            raise FileNotFoundError(
                f"normalized intermediate missing for clip {r['clip_id']}: "
                f"{intermediate}. Run `rytp splice` only after `rytp loudnorm`."
            )
        lines.append(f"file '{intermediate.as_posix()}'")
        if p.inserted_pause_ms > 0:
            # Insert silence via a small WAV file in the concat list.
            # (Stream-copy can't synthesize audio, so we go via a file
            # even in concat mode — DESIGN §8.)
            silence = config_paths.output / f"_silence-{r['clip_id']}.wav"
            _write_silence(ffmpeg, silence, p.inserted_pause_ms)
            lines.append(f"file '{silence.as_posix()}'")
    list_file.write_text("\n".join(lines))

    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _stream_copy_mode(
    output_path: Path,
    rows: list,
    pauses: list[PauseSample],
) -> None:
    """Run ffmpeg concat with ``-c copy`` from the original media files.

    ffmpeg's concat demuxer doesn't support per-entry ``-ss`` / ``-t``,
    so each clip is pre-extracted into its own intermediate file
    before the final concat pass. This costs one extra file per clip
    but keeps the implementation trivial and reuses the same
    ``_write_silence`` helper for pauses.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")

    list_file = output_path.with_suffix(".concat.txt")
    lines: list[str] = []
    for r, p in zip(rows, pauses):
        media = r["downloaded_path"] or r["local_path"]
        if not media:
            raise FileNotFoundError(f"no media for video {r['video_id']}")
        media_p = Path(media)
        if not media_p.exists():
            raise FileNotFoundError(f"media file missing: {media_p}")
        extract = config_paths.output / f"_streamclip-{r['clip_id']}.mp4"
        extract.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-ss",
            f"{r['start_ms'] / C.SECONDS_TO_MS:.3f}",
            "-i",
            str(media_p),
            "-t",
            f"{max(C.MIN_CLIP_DURATION_S, (r['end_ms'] - r['start_ms']) / C.SECONDS_TO_MS):.3f}",
            "-c",
            "copy",
            str(extract),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        lines.append(f"file '{extract.as_posix()}'")
        if p.inserted_pause_ms > 0:
            silence = config_paths.output / f"_silence-{r['clip_id']}.wav"
            _write_silence(ffmpeg, silence, p.inserted_pause_ms)
            lines.append(f"file '{silence.as_posix()}'")
    list_file.write_text("\n".join(lines))

    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _write_silence(ffmpeg: str, out: Path, duration_ms: int) -> None:
    """Write a silent WAV of ``duration_ms`` milliseconds.

    Uses ffmpeg's ``anullsrc`` lavfi source — cheap, deterministic,
    matches the canonical 16 kHz mono int16 format produced by
    :mod:`rytp.transcribe.extract`.
    """
    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        (
            f"anullsrc=channel_layout=mono:"
            f"sample_rate={C.AUDIO_SAMPLE_RATE_HZ}"
        ),
        "-t",
        f"{duration_ms / C.SECONDS_TO_MS:.3f}",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)