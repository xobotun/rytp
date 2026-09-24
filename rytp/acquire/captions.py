"""Caption acquisition. design §6.

Captions are pulled for every catalogued video, early and always: they are
tiny, they cost no GPU time, and they are the first thing to vanish when a
video is delisted. This module only fetches the file and records the asset —
turning json3 into ``words`` rows is Part 3's job.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from rytp import config, progress
from rytp.acquire import fetchable_video
from rytp.acquire.policy import DownloadPolicy, NoCaptions
from rytp.acquire.ytdlp import RealYtDlpRunner, YtDlpRunner, translate_error
from rytp.db import Database
from rytp.db.queries import insert_asset


@dataclass(frozen=True)
class CaptionResult:
    """What the caption stage did, and anything the operator should know.

    ``fallback_note`` is what the job handler returns as the job's note: on a
    bulk run nobody sees stdout, so "this video only had a translated track"
    has to live on the job row or it is lost.
    """

    message: str
    fallback_note: str | None = None


def acquire_captions(
    db: Database,
    video_id: int,
    *,
    runner: YtDlpRunner | None = None,
    policy: DownloadPolicy | None = None,
) -> CaptionResult:
    """Fetch the caption track and record it as the video's captions asset."""
    row = fetchable_video(db, video_id)
    policy = policy or DownloadPolicy.from_settings(db)
    runner = runner or RealYtDlpRunner()

    out_dir = config.ensure_dir(config.paths().media_dir(video_id))
    progress.report("captions", detail=f"video {video_id}")
    try:
        tracks = runner.download_captions(
            row["url"],
            out_template=str(out_dir / "captions.%(ext)s"),
            langs=policy.caption_langs,
            sub_format=policy.caption_format,
            sleep_requests_s=policy.sleep_requests_s,
        )
    except Exception as exc:
        raise translate_error(exc) from exc
    progress.report("captions", done=1, total=1, detail=f"video {video_id}")

    if not tracks:
        raise NoCaptions(
            f"video {video_id}: no caption track in any of "
            f"{', '.join(policy.caption_langs)}"
        )

    order = {lang: i for i, lang in enumerate(policy.caption_langs)}
    chosen = min(tracks, key=lambda t: order.get(t.format_id or "", len(order)))
    preferred = policy.caption_langs[0] if policy.caption_langs else ""
    fallback_note = (
        None if chosen.format_id == preferred
        else f"caption language {chosen.format_id!r}, not the preferred "
             f"{preferred!r}; these words may be a translation"
    )

    # Contracts §7 names the file captions.json3 regardless of which language
    # tag won, so downstream never has to guess at the suffix.
    dest = out_dir / f"captions.{chosen.ext}"
    if chosen.path != dest:
        os.replace(chosen.path, dest)
    insert_asset(
        db,
        video_id=video_id,
        role="captions",
        path=str(dest),
        format_id=chosen.format_id,
        size_bytes=chosen.size_bytes or dest.stat().st_size,
    )
    return CaptionResult(
        message=f"video {video_id}: {dest.name} ({chosen.format_id})",
        fallback_note=fallback_note,
    )
