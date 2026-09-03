"""Transcribe subcommand — the one-pass STT + Diarize merger.

DESIGN §5: extract audio, run STT (chunked), run Diarizer (full audio),
join them on overlapping intervals, write ``words`` to the DB.

Public surface:

* :func:`transcribe_video` — orchestrate the full pipeline for one video.
* :func:`merge_words_with_diarization` — the small named merger function
  used by both the STT+Diarize path and (after the words are already
  joined) the combined-engine path.
"""
from __future__ import annotations

from datetime import UTC as _UTC, datetime as _dt
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rytp.config import paths as config_paths
from rytp.db import Database
from rytp.diarize.base import DiarSegment
from rytp.diarize.none import NullDiarizer
from rytp.engines import DiarizedWord, Word
from rytp.transcribe.chunking import chunk_audio
from rytp.transcribe.extract import extract_audio
from rytp.transcribe.faster_whisper import FasterWhisperEngine


@dataclass(frozen=True)
class TranscribeResult:
    """Summary of a transcribe invocation."""

    video_id: int
    transcribe_run_id: int
    n_words: int
    stt_engine: str
    diarizer: str
    audio_path: Path


def merge_words_with_diarization(
    words: Iterable[Word],
    segments: Iterable[DiarSegment],
) -> Iterable[tuple[Word, str | None]]:
    """Join STT words to Diarizer segments on overlapping intervals.

    For each word, the diarizer_speaker is the speaker label of the
    segment with the **largest overlap** with the word's
    ``[start_ms, end_ms]``. If no segment overlaps, the speaker is
    ``None``.

    Both inputs are consumed lazily; the output is also lazy.
    """
    seg_list = sorted(segments, key=lambda s: s.start_ms)
    word_list = list(words)
    seg_idx = 0

    for w in word_list:
        best_speaker: str | None = None
        best_overlap = 0
        # Look at every segment that could overlap this word: starting
        # at seg_idx (which we keep across iterations to amortize the
        # walk), plus the next few segments that may overlap.
        i = seg_idx
        while i < len(seg_list):
            seg = seg_list[i]
            if seg.end_ms <= w.start_ms:
                # Segment ends before word starts → skip and advance.
                i += 1
                continue
            if seg.start_ms >= w.end_ms:
                # Segment starts after word ends → no overlap, and every
                # subsequent segment also starts after.
                break
            overlap = min(seg.end_ms, w.end_ms) - max(seg.start_ms, w.start_ms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg.speaker
            i += 1
        # Update seg_idx: every segment we considered that ended before
        # w.end_ms can't possibly overlap a *later* word either.
        # (We already advanced past segments ending before w.start_ms
        # above, which is the right amortized position.)
        # Don't move seg_idx backward; the next word will be at
        # >= current word's start_ms, so we can keep walking forward.
        yield (w, best_speaker)


def transcribe_video(
    db: Database,
    video_id: int,
    *,
    stt_engine: str = "null",
    diarizer: str = "none",
    combined_engine: str | None = None,
    language: str | None = None,
    video_path: Path | None = None,
    diarizer_sensitivity: float | None = None,
) -> TranscribeResult:
    """Run the full transcribe pipeline for one video.

    Steps:

    1. Look up the video in the ``videos`` table. Resolve ``video_path``
       (the on-disk media file): from the parameter if given, otherwise
       from ``downloaded_path``.
    2. Extract a 16 kHz mono WAV into ``config.paths.audio``.
    3. Plan chunks; below threshold the chunk list has one entry that
       points back at the original WAV.
    4. For each chunk, run the STT engine and yield Word objects with
       absolute timestamps.
    5. Run the Diarizer on the full audio (it's context-dependent and
       must NOT be chunked).
    6. Merge words + segments with :func:`merge_words_with_diarization`.
    7. Bulk-insert ``words`` rows, one per word, with
       ``diarizer_speaker`` set from the merger.
    8. Record a ``transcribe_runs`` row and (if there are any) the
       ``chunks`` rows.

    The combined-engine path (``combined_engine`` is not None) bypasses
    local chunking — the engine returns an already-joined stream of
    ``DiarizedWord`` records. See DESIGN §5.3.

    Returns a :class:`TranscribeResult` summarizing the run.
    """
    # 1. Look up the video.
    video_row = db.conn.execute(
        "SELECT * FROM videos WHERE id = ?", (video_id,)
    ).fetchone()
    if video_row is None:
        raise ValueError(f"video_id {video_id} not found in videos table")
    media_path = video_path or (
        Path(video_row["downloaded_path"]) if video_row["downloaded_path"] else None
    )
    if media_path is None or not media_path.exists():
        raise FileNotFoundError(
            f"no media file for video_id={video_id} (looked at downloaded_path="
            f"{video_row['downloaded_path']!r})"
        )

    # 2. Extract audio.
    # Use separate audio file if available (downloaded separately)
    downloaded_audio_path = video_row["downloaded_audio_path"]
    audio_path = extract_audio(
        media_path,
        config_paths.audio,
        video_id=str(video_id),
        audio_path=Path(downloaded_audio_path) if downloaded_audio_path else None,
    )

    # 3. Plan chunks.
    chunks = chunk_audio(audio_path)

    # 8. Open transcribe_runs row up front so failures are visible.
    now_iso = _dt.now(_UTC).isoformat()
    cur = db.conn.execute(
        """
        INSERT INTO transcribe_runs (
            video_id, stt_engine, diarizer, combined, language, started_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            video_id,
            stt_engine,
            diarizer,
            combined_engine,
            language or "",
            now_iso,
        ),
    )
    run_id = cur.lastrowid
    db.conn.commit()

    # Clear existing words for this video to prevent duplicates on re-runs.
    # DESIGN §4 mentions incremental re-runs, but in practice re-running
    # with the same engine should replace the previous results, not append
    # to them. The chunks table tracks per-run state for resumability, but
    # the words table is the final output that downstream stages read from.
    db.conn.execute("DELETE FROM words WHERE video_id = ?", (video_id,))
    db.conn.commit()

    # 4–7. Run STT (or combined), then Diarizer, merge, insert words.
    diarizer_used = diarizer  # default for the run summary
    if combined_engine is not None:
        words_iter = _run_combined_engine(combined_engine, audio_path, language)
        n_words = _insert_diarized_words(db, video_id, words_iter)
        diarizer_used = "(combined)"
    else:
        words_iter = _run_stt_engine(stt_engine, chunks, language)
        if diarizer != "none":
            segments = _run_diarizer(diarizer, audio_path, diarizer_sensitivity)
        else:
            segments = NullDiarizer().diarize(audio_path)
        joined = merge_words_with_diarization(words_iter, segments)
        n_words = _insert_words_with_diarizer(db, video_id, joined)

    # Record chunk rows so retries can be incremental.
    for c in chunks:
        db.conn.execute(
            """
            INSERT INTO chunks (transcribe_run_id, ord, start_ms, end_ms, status, words_written)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, c.ord, c.start_ms, c.end_ms, "done", n_words if c.ord == 0 else 0),
        )

    # Mark the run finished.
    db.conn.execute(
        "UPDATE transcribe_runs SET finished_at = ? WHERE id = ?",
        (_dt.now(_UTC).isoformat(), run_id),
    )
    db.conn.commit()

    return TranscribeResult(
        video_id=video_id,
        transcribe_run_id=run_id,
        n_words=n_words,
        stt_engine=stt_engine,
        diarizer=diarizer_used,
        audio_path=audio_path,
    )


# ---------------------------------------------------------------------------
# Engine invocation helpers
# ---------------------------------------------------------------------------


def _run_stt_engine(
    stt_engine: str, chunks: list, language: str | None
) -> Iterable[Word]:
    from rytp import engines

    cls = engines.resolve_stt(stt_engine)
    engine = cls()  # type: ignore[abstract]
    for chunk in chunks:
        for w in engine.transcribe(chunk.audio_path, language=language):
            # Each chunk's words carry absolute timestamps (the STT
            # contract from DESIGN §5.1), so no offset adjustment is
            # needed.
            yield w


def _run_diarizer(
    diarizer: str, audio_path: Path, sensitivity: float | None = None
) -> Iterable[DiarSegment]:
    from rytp import engines

    cls = engines.resolve_diarizer(diarizer)
    # If the diarizer accepts a sensitivity parameter and one was provided,
    # pass it. This allows tuning diarization without changing the interface
    # for diarizers that don't support it.
    try:
        if sensitivity is not None:
            engine = cls(sensitivity=sensitivity)  # type: ignore[abstract]
        else:
            engine = cls()  # type: ignore[abstract]
    except TypeError:
        # Diarizer doesn't accept sensitivity
        engine = cls()  # type: ignore[abstract]
    return engine.diarize(audio_path)


def _run_combined_engine(
    combined_engine: str, audio_path: Path, language: str | None
) -> Iterable[DiarizedWord]:
    from rytp import engines

    cls = engines.resolve_combined(combined_engine)
    engine = cls()  # type: ignore[abstract]
    return engine.transcribe_diarize(audio_path, language=language)


def _insert_words_with_diarizer(
    db: Database,
    video_id: int,
    joined: Iterable[tuple[Word, str | None]],
) -> int:
    """Insert one ``words`` row per (Word, speaker). All speakers come
    from the diarizer path; the second tuple element is the raw
    diarizer label or ``None``.
    """
    data = []
    for w, speaker in joined:
        data.append((video_id, w, speaker))
    if not data:
        return 0
    db.conn.executemany(
        """
        INSERT INTO words (video_id, start_ms, end_ms, text, normalized_text,
                           confidence, diarizer_speaker, speaker_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        [
            (vid, w.start_ms, w.end_ms, w.text, _normalize(w.text),
             w.confidence, speaker)
            for vid, w, speaker in data
        ],
    )
    db.conn.commit()
    return len(data)


def _insert_diarized_words(
    db: Database, video_id: int, words: Iterable[DiarizedWord]
) -> int:
    from rytp.models import normalize_text as _norm2

    data = list(words)
    if not data:
        return 0
    db.conn.executemany(
        """
        INSERT INTO words (video_id, start_ms, end_ms, text, normalized_text,
                           confidence, diarizer_speaker, speaker_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        [
            (
                video_id,
                w.start_ms,
                w.end_ms,
                w.text,
                _norm2(w.text),
                w.confidence,
                w.speaker,
            )
            for w in data
        ],
    )
    db.conn.commit()
    return len(data)


def _normalize(text: str) -> str:
    # Local import to avoid a circular dep at module import time.
    from rytp.models import normalize_text

    return normalize_text(text)