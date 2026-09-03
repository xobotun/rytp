"""Human-readable transcript export.

DESIGN §7 "Output formats" — markdown transcript with speaker labels.
A long-form transcript is more useful than per-word rows for editing,
review, and presentation; the per-word rows stay in the ``words``
table for the mine stage.

Algorithm: walk the ``words`` table for a video in time order, join
against ``videos_speaker_map`` to get the canonical speaker id, and
group consecutive same-speaker words into *blocks*. A new block
starts whenever the speaker changes.

:func:`export_markdown` is the public entry point; the ``min_block_s``
parameter (default ``MIN_BLOCK_DURATION_S``) is a soft lower bound on
block duration — blocks shorter than this are absorbed into a
neighboring same-speaker block when possible. This is the "minimum
1-2 seconds" heuristic from the design discussion: it stops the
output from being chopped into one-block-per-interjection when
backchannel is dense, without ever artificially splitting a real
speaker change.

Public surface:

* :func:`export_markdown` — write a markdown transcript to disk.
* :func:`build_blocks` — turn ``words`` rows into speaker blocks
  (testable, no I/O).
* :class:`Block` — value type.
"""
from __future__ import annotations

from datetime import UTC as _UTC, datetime as _dt
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rytp import constants as C
from rytp.db import Database
from rytp.speakers import list_speakers


@dataclass(frozen=True)
class Block:
    """One contiguous stretch of words from one speaker.

    Attributes:
        speaker_label: Display label (canonical name if known, raw
            ``SPEAKER_xx`` otherwise, or ``"(unknown)"`` if neither
            is available).
        start_ms: Block start (ms) — first word's ``start_ms``.
        end_ms: Block end (ms) — last word's ``end_ms``.
        text: The concatenated, whitespace-collapsed transcript.
    """

    speaker_label: str
    start_ms: int
    end_ms: int
    text: str


def _fetch_words(db: Database, video_id: int) -> list[dict[str, object]]:
    """Pull every ``words`` row for ``video_id`` joined to its canonical
    speaker label.

    Returns dicts (not :class:`sqlite3.Row`) so the function is easy
    to test against a plain list. Keys:

    * ``start_ms`` / ``end_ms``: int
    * ``text``: str
    * ``speaker_label``: str (canonical name, or raw ``SPEAKER_xx``,
      or ``"(unknown)"``)
    """
    rows = db.conn.execute(
        """
        SELECT w.start_ms, w.end_ms, w.text, w.diarizer_speaker,
               COALESCE(m.speaker_id, w.speaker_id) AS speaker_id
        FROM words w
        LEFT JOIN videos_speaker_map m
            ON m.video_id = w.video_id AND m.diarizer_speaker = w.diarizer_speaker
        WHERE w.video_id = ?
        ORDER BY w.start_ms
        """,
        (video_id,),
    ).fetchall()

    # Build a label lookup once; ``speakers`` is small (one entry per
    # canonical person), so a dict is the right shape.
    label_by_id: dict[int, str] = {s.id: s.label for s in list_speakers(db)}

    out: list[dict[str, object]] = []
    for r in rows:
        sid = r["speaker_id"]
        if sid is not None and sid in label_by_id:
            label = label_by_id[sid]
        elif r["diarizer_speaker"]:
            label = r["diarizer_speaker"]
        else:
            label = "(unknown)"
        out.append(
            {
                "start_ms": r["start_ms"],
                "end_ms": r["end_ms"],
                "text": (r["text"] or "").strip(),
                "speaker_label": label,
            }
        )
    return out


def build_blocks(
    words: Iterable[dict[str, object]],
    *,
    min_block_s: float = C.MIN_BLOCK_DURATION_S,
) -> list[Block]:
    """Group consecutive same-speaker words into blocks.

    Rules:

    1. Walk words in order; whenever the speaker changes, commit the
       running block and start a new one.
    2. After all words are grouped, any block whose duration is
       **less than** ``min_block_s`` is *absorbed* into an immediate
       same-speaker neighbor — preferring the next block, falling
       back to the previous. If neither immediate neighbor shares
       the speaker, the short block stays as a clear speaker
       boundary.

    The "immediate neighbor" check is deliberate: when a short
    block sits between two same-speaker blocks with a different
    speaker in between (e.g. ``[A 3s, B 0.5s, A 3s]``), the B
    block is **not** auto-absorbed into the trailing A. That would
    require either dropping B's text (loses data) or stretching
    the trailing A's time range over B (fakes a speaker change).
    v1's policy: leave the short block in place. A user can still
    see what B said; they can also re-merge manually if desired.

    Args:
        words: Iterable of word dicts from :func:`_fetch_words` (or
            any compatible shape).
        min_block_s: Lower bound on block duration in seconds.

    Returns:
        Speaker blocks in time order.
    """
    min_block_ms = int(min_block_s * C.MS_PER_SECOND)
    raw: list[Block] = []
    current: Block | None = None
    for w in words:
        label = str(w["speaker_label"])
        start_ms = int(w["start_ms"])  # type: ignore[arg-type]
        end_ms = int(w["end_ms"])  # type: ignore[arg-type]
        text = str(w["text"])
        if current is not None and current.speaker_label == label:
            # Extend the running block.
            current = Block(
                speaker_label=current.speaker_label,
                start_ms=current.start_ms,
                end_ms=end_ms,
                text=(current.text + " " + text).strip(),
            )
        else:
            if current is not None:
                raw.append(current)
            current = Block(
                speaker_label=label,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
            )
    if current is not None:
        raw.append(current)

    # Post-process: absorb short same-speaker blocks into a neighbor.
    # We loop until no changes are made, so a chain of very short
    # blocks (e.g. 0.3 s interjections) all collapse into the
    # surrounding block. The loop is cheap (block count is O(turns),
    # not O(words)).
    #
    # For each short block, we prefer the *next* neighbor (most common
    # case: a trailing interjection that the same speaker continues
    # past). If the next neighbor shares the speaker, the short block
    # is absorbed into it. If only the *previous* neighbor shares the
    # speaker, the short block is absorbed backwards. If neither
    # neighbor shares the speaker, the short block stays — a real
    # speaker boundary takes priority over the absorption heuristic
    # (DESIGN §8: "don't fake a speaker change").
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(raw):
            b = raw[i]
            if b.end_ms - b.start_ms >= min_block_ms or len(raw) == 1:
                i += 1
                continue
            absorbed = False
            if i + 1 < len(raw) and raw[i + 1].speaker_label == b.speaker_label:
                raw[i + 1] = Block(
                    speaker_label=b.speaker_label,
                    start_ms=b.start_ms,
                    end_ms=raw[i + 1].end_ms,
                    text=(b.text + " " + raw[i + 1].text).strip(),
                )
                del raw[i]
                absorbed = True
            elif i - 1 >= 0 and raw[i - 1].speaker_label == b.speaker_label:
                raw[i - 1] = Block(
                    speaker_label=raw[i - 1].speaker_label,
                    start_ms=raw[i - 1].start_ms,
                    end_ms=b.end_ms,
                    text=(raw[i - 1].text + " " + b.text).strip(),
                )
                del raw[i]
                absorbed = True
            if absorbed:
                changed = True
                # Don't advance i; the new raw[i] could also be short.
            else:
                i += 1

    return raw


def _format_ts(ms: int) -> str:
    """Format ``ms`` as ``HH:MM:SS`` (or ``M:SS`` if under one hour)."""
    s = ms // C.MS_PER_SECOND
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def export_markdown(
    db: Database,
    video_id: int,
    *,
    min_block_s: float = C.MIN_BLOCK_DURATION_S,
    out_path: Path | None = None,
    title: str | None = None,
) -> Path:
    """Export a speaker-grouped markdown transcript for ``video_id``.

    The output format is a list of ``### [HH:MM:SS] Speaker`` headers
    followed by the block's text, with one blank line between blocks.
    A top-level H1 with the video title (when known) is included for
    context.

    Args:
        db: Open Database.
        video_id: FK to the ``videos`` table.
        min_block_s: Lower bound on block duration; see
            :func:`build_blocks`.
        out_path: Where to write the markdown. Defaults to
            ``config.paths.transcripts / f"{video_id}.md"``.
        title: Optional H1 title. Defaults to the video's title
            from the ``videos`` table.

    Returns:
        The path that was written.

    Raises:
        ValueError: if the video doesn't exist.
    """
    from rytp.config import paths as config_paths

    video_row = db.conn.execute(
        "SELECT id, title, duration, source, youtube_id, local_path FROM videos "
        "WHERE id = ?",
        (video_id,),
    ).fetchone()
    if video_row is None:
        raise ValueError(f"no video with id={video_id}")

    if title is None:
        title = video_row["title"] or f"video {video_id}"

    if out_path is None:
        out_path = config_paths.transcripts / f"{video_id}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    words = _fetch_words(db, video_id)
    blocks = build_blocks(words, min_block_s=min_block_s)

    now = _dt.now(_UTC).isoformat()
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        f"_video_id={video_id} • "
        f"duration={_format_ts(video_row['duration'] or 0)} • "
        f"generated={now}Z_"
    )
    lines.append("")

    if not blocks:
        lines.append("_(no transcript yet)_")
    else:
        for b in blocks:
            lines.append(
                f"### [{_format_ts(b.start_ms)}] {b.speaker_label}"
            )
            lines.append("")
            lines.append(b.text)
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path