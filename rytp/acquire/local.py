"""Local files. design §4, §5.

A local file with embedded audio is a single ``container`` asset, referenced
where it already lives — nothing is copied into the data tree. It never gets
a download job; the only work it needs is WAV extraction.

Contracts §3 dropped every path column from ``videos``, so Part 1 stores the
resolved absolute path in ``videos.external_id`` for ``source='local'``
rows, which also makes ``UNIQUE (source, external_id)`` deduplicate
re-registration of the same file.
"""

from __future__ import annotations

from pathlib import Path

from rytp import constants as C
from rytp.acquire.policy import AcquireError, MissingAssetError
from rytp.db import Database
from rytp.db.queries import insert_asset


def register_local_container(db: Database, video_id: int) -> str:
    """Record a catalogued local file as its video's container asset."""
    row = db.conn.execute(
        "SELECT id, source, external_id FROM videos WHERE id = ?", (video_id,)
    ).fetchone()
    if row is None:
        raise AcquireError(f"video {video_id} is not in the catalog")
    if row["source"] != C.LOCAL_SOURCE:
        raise AcquireError(
            f"video {video_id} is not a local file (source={row['source']!r})"
        )
    path = Path(row["external_id"] or "")
    if not path.is_file():
        raise MissingAssetError(f"video {video_id}: {path} is not a file")
    insert_asset(
        db,
        video_id=video_id,
        role="container",
        path=str(path),
        size_bytes=path.stat().st_size,
    )
    return f"video {video_id}: registered container {path.name}"
