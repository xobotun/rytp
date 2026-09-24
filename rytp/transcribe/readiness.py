"""Readiness predicates for Part 3's job kinds (design §5, contracts §5).

``rytp/jobs/__init__.py`` imports this module at load time, so it stays light:
the standard library, :mod:`rytp.config`, :mod:`rytp.constants`,
:mod:`rytp.db` and :mod:`rytp.audio.extract`. No numpy, no engines, no
pipeline.

Readiness is derived from the database and the filesystem and **never** from
the job's payload. A predicate that changed its mind based on how a job was
enqueued would stop being a statement about the world, and the self-healing
property in design §5 would go with it: prune a cached WAV and `transcribe`
becomes blocked while `extract_wav` becomes available, with no edge table
anywhere to keep consistent.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rytp.audio.extract import wav_path

if TYPE_CHECKING:
    from rytp.db import Database
    from rytp.jobs import Readiness


def _has_transcribed_words(db: Database, video_id: int) -> bool:
    """Any non-caption words: the video has been transcribed, at either tier."""
    row = db.conn.execute(
        "SELECT 1 FROM words WHERE video_id = ? AND source <> 'caption' LIMIT 1",
        (video_id,),
    ).fetchone()
    return row is not None


def _has_any_words(db: Database, video_id: int) -> bool:
    row = db.conn.execute(
        "SELECT 1 FROM words WHERE video_id = ? LIMIT 1", (video_id,)
    ).fetchone()
    return row is not None


def caption_words_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable once the captions asset is on disk; satisfied once words exist.

    Satisfied on *any* word, not only caption-tier ones: a video already
    promoted to the aligned tier must not have its real boundaries replaced
    by caption starts, which is the same refusal
    :func:`~rytp.transcribe.captions.ingest_captions` makes for itself.
    """
    from rytp.jobs import Readiness

    if _has_any_words(db, video_id):
        return Readiness.SATISFIED
    row = db.conn.execute(
        "SELECT 1 FROM assets WHERE video_id = ? AND role = 'captions' LIMIT 1",
        (video_id,),
    ).fetchone()
    return Readiness.READY if row is not None else Readiness.BLOCKED


def transcribe_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable once the cached WAV exists; satisfied once the video has words.

    Satisfied at either transcript tier: `timed` means transcription already
    happened, and getting from there to `aligned` is the `align` job's work,
    not a reason to transcribe again.

    ``Readiness`` is imported inside the function because ``rytp.jobs``
    imports this module: a module-level import would close the cycle.
    """
    from rytp.jobs import Readiness

    if _has_transcribed_words(db, video_id):
        return Readiness.SATISFIED
    return Readiness.READY if wav_path(video_id).exists() else Readiness.BLOCKED


def align_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable when there is a transcript to align and a WAV to align it to.

    Either tier qualifies: `timed` rows get upgraded to `aligned`, and
    `aligned` rows get re-timed by a different aligner. Never *satisfied*,
    because "already aligned" does not mean "aligned by the engine you just
    asked for" — which is also why the kind sets ``reopenable=False``.
    Enqueue it again to run it again.
    """
    from rytp.jobs import Readiness

    if not _has_transcribed_words(db, video_id):
        return Readiness.BLOCKED
    return Readiness.READY if wav_path(video_id).exists() else Readiness.BLOCKED


def fingerprint_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable once the WAV exists; satisfied once the one row is written."""
    from rytp.jobs import Readiness

    row = db.conn.execute(
        "SELECT 1 FROM video_acoustics WHERE video_id = ?", (video_id,)
    ).fetchone()
    if row is not None:
        return Readiness.SATISFIED
    return Readiness.READY if wav_path(video_id).exists() else Readiness.BLOCKED
