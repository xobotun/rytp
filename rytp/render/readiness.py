"""Readiness for the ``render`` job kind. design §5.

Import-light on purpose: ``rytp/jobs/__init__.py`` imports this at
module level, so it may not reach for ffmpeg, the canvas, or anything
else the render stage needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rytp.config import paths
from rytp.db import Database

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rytp.jobs import Readiness


def render_readiness(db: Database, target_id: int) -> Readiness:
    """What the database and the disk say about one render right now.

    ``target_id`` is a ``renders.id``, so this is an ordinary predicate
    over the world, with no need for the payload design §5 forbids it
    from reading: the row names the cut list, the row records the
    output. Delete the output file and the job becomes runnable again —
    the same self-healing behaviour as pruning a cached WAV.

    The ``Readiness`` import is **inside** the function on purpose, and
    must stay there. Part 2's own ``rytp/jobs/readiness.py`` can import
    it at module level because its parent package is the importer —
    ``rytp.jobs`` is always initialised first. This module's parent is
    ``rytp.render``, which imports nothing, so a cold
    ``import rytp.render.readiness`` would run ``rytp.jobs``' bottom
    block, which imports this half-initialised module straight back and
    fails. Do not let a cleanup pass hoist it.
    """
    from rytp.jobs import Readiness

    row = db.conn.execute(
        "SELECT cutlist_name, output_path, state FROM renders WHERE id = ?",
        (target_id,),
    ).fetchone()
    if row is None:
        return Readiness.BLOCKED
    output = row["output_path"]
    if row["state"] == "rendered" and output and Path(output).exists():
        return Readiness.SATISFIED
    if not paths().cutlist(str(row["cutlist_name"])).exists():
        return Readiness.BLOCKED
    return Readiness.READY
