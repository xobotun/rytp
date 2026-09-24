"""When a video is ready to be diarized (design §5).

A predicate over the database and the filesystem as they are *now*. It
never sees the job's payload, so which engine somebody asked for cannot
change the answer — a predicate that did would stop being a statement about
the world and the self-healing property would go with it.

Deliberately light: `rytp/jobs/__init__.py` imports this at module load, so
nothing from the rest of `rytp.diarize` may be imported here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rytp.config import paths
from rytp.db import Database

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rytp.jobs import Readiness


def diarize_readiness(db: Database, video_id: int) -> Readiness:
    """READY once there are aligned words and a cached WAV to measure them against.

    Aligned words, not any words: caption-tier rows have no end times
    (contracts §3), so their overlap with a diarizer segment would be
    guesswork, and a caption-tier video is not one anybody is cutting from.

    SATISFIED as soon as the video has labels. Re-transcribing deletes them
    (contracts §4), which puts this back to READY — but the kind is
    registered `reopenable=False`, so that only matters when a human asks
    for diarization again.

    The ``Readiness`` import is **inside** the function on purpose, and
    must stay there: `rytp/jobs/__init__.py` imports this module at load
    time to register the `diarize` kind, so a module-level
    ``from rytp.jobs import Readiness`` would be a real cycle.
    """
    from rytp.jobs import Readiness

    exists = db.conn.execute("SELECT 1 FROM videos WHERE id = ?", (video_id,)).fetchone()
    if exists is None:
        return Readiness.BLOCKED
    labelled = db.conn.execute(
        "SELECT 1 FROM video_speakers WHERE video_id = ? LIMIT 1", (video_id,)
    ).fetchone()
    if labelled is not None:
        return Readiness.SATISFIED
    aligned = db.conn.execute(
        "SELECT 1 FROM words WHERE video_id = ? AND source = 'aligned' LIMIT 1",
        (video_id,),
    ).fetchone()
    if aligned is None:
        return Readiness.BLOCKED
    if not paths().cache_wav(video_id).exists():
        return Readiness.BLOCKED
    return Readiness.READY
