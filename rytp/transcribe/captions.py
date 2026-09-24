"""Caption tier: YouTube json3 auto-captions become searchable words (design §6).

Tier 1 exists because it is nearly free. Captions are tiny, cost no GPU time,
and are the first thing to disappear when a video is delisted, so they are
pulled for everything and turned into `words` rows with ``source='caption'``.

What they are not is cuttable. A json3 caption carries a per-word **start** on
a 40 ms grid and no end at all, and its alignment is loose. ``words.end_ms``
stays null, which the schema's ``CHECK (source = 'caption' OR end_ms IS NOT
NULL)`` turns into a hard guarantee that nothing downstream can cut on them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.db import Database
from rytp.models import RytpError, normalize_text, stem_text
from rytp.transcribe.base import split_token


class CaptionDowngrade(RytpError):  # noqa: N818 - domain name, not a generic error type
    """Refused: the video already has aligned words, which captions would replace."""


@dataclass(frozen=True)
class CaptionWord:
    """One caption token. No end time exists — see the module docstring."""

    start_ms: int
    text: str


_WORD_COLUMNS = (
    "video_id, ord, start_ms, end_ms, text, normalized_text, stem, "
    "confidence, align_score, source, engine, video_speaker_id"
)
_INSERT_WORD = f"INSERT INTO words ({_WORD_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"


def load_json3(path: Path) -> dict[str, Any]:
    """Read a json3 caption file, with a plain error when it is something else."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RytpError(f"cannot read captions at {path}: {exc}") from exc
    except ValueError as exc:
        raise RytpError(f"{path} is not json3 captions: {exc}") from exc
    if not isinstance(payload, dict) or "events" not in payload:
        raise RytpError(f"{path} is not json3 captions: no 'events' key")
    return payload


def parse_json3(payload: dict[str, Any]) -> list[CaptionWord]:
    """Flatten json3 events into caption words.

    A word's start is its event's ``tStartMs`` plus its segment's
    ``tOffsetMs``. Events marked ``aAppend`` are roll-up repeats of the line
    already emitted and are skipped, as are the whitespace segments that
    separate roll-up lines. A segment holding several words — which manual
    captions do and auto-captions occasionally do — gives every word the
    segment's start; captions are a search resource, not a cutting one. The
    same applies to a hyphenated word: :func:`~rytp.transcribe.base.split_token`
    makes it two rows sharing one start, because a row whose normalized text
    holds a space could never be found.
    """
    words: list[CaptionWord] = []
    previous = 0
    for event in payload.get("events") or ():
        if event.get("aAppend"):
            continue
        base = int(event.get("tStartMs", 0))
        for segment in event.get("segs") or ():
            raw = str(segment.get("utf8", ""))
            if not raw.strip():
                continue
            start = max(base + int(segment.get("tOffsetMs", 0)), previous)
            words.extend(
                CaptionWord(start_ms=start, text=surface)
                for surface, _normalized in split_token(raw)
            )
            previous = start
    return words


def captions_asset_path(db: Database, video_id: int) -> Path:
    """Path of the video's captions asset (contracts §3, role ``captions``)."""
    row = db.conn.execute(
        "SELECT path FROM assets WHERE video_id = ? AND role = 'captions' "
        "ORDER BY acquired_at DESC, id DESC LIMIT 1",
        (video_id,),
    ).fetchone()
    if row is None:
        raise RytpError(f"video {video_id} has no captions asset; fetch captions first")
    return Path(str(row[0]))


def ingest_captions(db: Database, video_id: int, path: Path | None = None) -> int:
    """Replace the video's words with caption-tier rows. Returns how many.

    Refuses to run on a video that already has `timed` or `aligned` words:
    either is better text than captions, so overwriting would be a downgrade.
    Promotion goes the other way, through
    :func:`rytp.transcribe.pipeline.transcribe_video`.
    """
    source = path if path is not None else captions_asset_path(db, video_id)
    transcribed = db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ? AND source <> 'caption'",
        (video_id,),
    ).fetchone()[0]
    if transcribed:
        raise CaptionDowngrade(
            f"video {video_id} already has {transcribed} transcribed words; "
            "captions would be a downgrade"
        )
    caption_words = parse_json3(load_json3(source))
    rows = []
    ordinal = 0
    for word in caption_words:
        normalized = normalize_text(word.text)
        if not normalized:
            continue
        rows.append(
            (
                video_id,
                ordinal,
                word.start_ms,
                None,
                word.text,
                normalized,
                stem_text(normalized),
                None,
                None,
                "caption",
                C.CAPTION_ENGINE,
                None,
            )
        )
        ordinal += 1
    # Imported here rather than at module level: pipeline pulls in numpy and
    # the audio stack, which the caption tier has no use for until this point.
    from rytp.transcribe.pipeline import invalidate_transcript

    with db.transaction():
        # Contracts §4's cross-part invariant, in the one shared function that
        # implements it: words, utterances and speaker labels go together.
        invalidate_transcript(db, video_id)
        db.conn.executemany(_INSERT_WORD, rows)
    return len(rows)
