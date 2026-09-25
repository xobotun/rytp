"""Getting index rows back out: audio clips, playback, and transcripts.

Design §7. Two kinds of export, one module, because both answer the same
question — "render a row of the index as something outside the database".

Everything that decides *what* to run is a pure function returning a
`list[str]`; exactly one function runs it. That is what lets the whole
test suite assert on command shape without ever opening a player, and it
is the same seam pattern Part 2 uses for the WAV cache.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import asset_for
from rytp.index.search import anchor_for
from rytp.models import NotFoundError, RytpError, utc_now_iso
from rytp.timefmt import format_cue

__all__ = [
    "ClipError",
    "MediaToolMissing",
    "TranscriptBlock",
    "build_clip_command",
    "build_play_command",
    "clip_source",
    "export_clip",
    "padded",
    "play_clip",
    "render_transcript",
    "timestamp",
    "transcript_blocks",
    "write_transcript",
]


class MediaToolMissing(RytpError):  # noqa: N818 - exact name pinned by the plan
    """ffmpeg or ffplay is not on PATH."""


class ClipError(RytpError):
    """There was nothing to cut from, or ffmpeg refused to cut it."""


def _ffmpeg_binary() -> str | None:
    """Where ffmpeg is. A module attribute so a test can replace just this."""
    return shutil.which("ffmpeg")


def _ffplay_binary() -> str | None:
    """Where ffplay is. Same reason."""
    return shutil.which("ffplay")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """The only subprocess call in this module. Tests replace it."""
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def clip_source(db: Database, video_id: int) -> Path:
    """The audio to cut from.

    The cached 16 kHz mono WAV first, because it is already the audio
    every other stage works from. Contracts §7 makes that cache prunable,
    so the asset roles are the fallback and `cache prune` cannot break
    playback.
    """
    cached = config.paths().cache_wav(video_id)
    if cached.exists():
        return cached
    for role in ("audio", "container"):
        row = asset_for(db, video_id, role)
        if row is not None and Path(str(row["path"])).exists():
            return Path(str(row["path"]))
    raise ClipError(
        f"video {video_id}: no cached WAV and no audio on disk; "
        "run `rytp ingest` for it first"
    )


def padded(start_ms: int, end_ms: int, pad_ms: int) -> tuple[int, int]:
    """Widen a span by `pad_ms` on each side, never before the recording."""
    return max(0, start_ms - pad_ms), end_ms + pad_ms


def _seconds(ms: int) -> str:
    """Milliseconds as the seconds string ffmpeg takes."""
    return f"{ms / C.MS_PER_SECOND:.3f}"


def _duration(start_ms: int, end_ms: int) -> str:
    """Span length, never zero — ffmpeg would write an empty file."""
    return _seconds(max(end_ms - start_ms, 1))


def build_clip_command(
    ffmpeg: str, source: Path, out: Path, start_ms: int, end_ms: int
) -> list[str]:
    """Cut one span to a WAV.

    `-ss` goes before `-i`: input-side seek, which current ffmpeg does
    both quickly and accurately. The length is `-t`, not `-to`, because
    `-to` is an absolute output timestamp and the origin has moved.
    """
    return [
        ffmpeg,
        "-nostdin",
        "-y",
        "-ss",
        _seconds(start_ms),
        "-t",
        _duration(start_ms, end_ms),
        "-i",
        str(source),
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        str(C.AUDIO_CHANNELS),
        "-ar",
        str(C.AUDIO_SAMPLE_RATE_HZ),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(out),
    ]


def build_play_command(ffplay: str, source: Path, start_ms: int, end_ms: int) -> list[str]:
    """Play one span: no window, no banner, quits when the span ends."""
    return [
        ffplay,
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "error",
        "-ss",
        _seconds(start_ms),
        "-t",
        _duration(start_ms, end_ms),
        str(source),
    ]


def _invoke(cmd: list[str], tool: str) -> subprocess.CompletedProcess[str]:
    try:
        return _run(cmd)
    except FileNotFoundError as exc:
        # which() found it and exec() did not: a moved or half-installed
        # binary. One line, not a traceback (contracts §8).
        raise MediaToolMissing(f"{tool} could not be run: {exc}") from exc


def export_clip(
    db: Database,
    video_id: int,
    start_ms: int,
    end_ms: int,
    out: Path,
    *,
    pad_ms: int = C.CLIP_PAD_MS,
) -> Path:
    """Write one span to `out` as a WAV. Returns the path."""
    source = clip_source(db, video_id)
    ffmpeg = _ffmpeg_binary()
    if ffmpeg is None:
        raise MediaToolMissing(
            "ffmpeg not found on PATH. Install it from https://ffmpeg.org/ "
            "(on Windows, unzip the release and add its bin/ directory to PATH)"
        )
    config.ensure_dir(out.parent)
    lo, hi = padded(start_ms, end_ms, pad_ms)
    result = _invoke(build_clip_command(ffmpeg, source, out, lo, hi), "ffmpeg")
    if result.returncode != 0:
        raise ClipError(
            f"ffmpeg failed (rc={result.returncode}): "
            f"{result.stderr[-C.FFMPEG_ERROR_TAIL_CHARS:]}"
        )
    return out


def play_clip(
    db: Database,
    video_id: int,
    start_ms: int,
    end_ms: int,
    *,
    pad_ms: int = C.CLIP_PAD_MS,
) -> None:
    """Play one span through ffplay. Blocks until it finishes."""
    source = clip_source(db, video_id)
    ffplay = _ffplay_binary()
    if ffplay is None:
        raise MediaToolMissing(
            "ffplay not found on PATH. It ships with ffmpeg; install that."
        )
    lo, hi = padded(start_ms, end_ms, pad_ms)
    result = _invoke(build_play_command(ffplay, source, lo, hi), "ffplay")
    if result.returncode != 0:
        raise ClipError(
            f"ffplay failed (rc={result.returncode}): "
            f"{result.stderr[-C.FFMPEG_ERROR_TAIL_CHARS:]}"
        )


# -- markdown transcripts ---------------------------------------------


@dataclass(frozen=True)
class TranscriptBlock:
    """One utterance, ready to render. `anchor` is the handle back."""

    anchor: str
    start_ms: int
    end_ms: int
    speaker: str
    text: str


def timestamp(ms: int) -> str:
    """Milliseconds as HH:MM:SS.mmm, which sorts and greps cleanly.

    A transcript cue's own purpose (`rytp.timefmt.format_cue`) — a
    two-digit hour, distinct from a seek target's unpadded one, so a column
    of cues stays a fixed width whatever the video's length.
    """
    return format_cue(ms)


_TRANSCRIPT_SQL = (
    "SELECT u.video_id AS video_id, u.start_ms AS start_ms, u.end_ms AS end_ms,"
    " u.first_word_ord AS first_word_ord, u.last_word_ord AS last_word_ord,"
    " u.text AS text, COALESCE(s.label, vs.local_label) AS speaker"
    " FROM utterances u"
    " LEFT JOIN video_speakers vs ON vs.id = u.video_speaker_id"
    " LEFT JOIN speakers s ON s.id = vs.speaker_id"
    " WHERE u.video_id = ? ORDER BY u.first_word_ord"
)

#: One utterance's own words, in ordinal order — what a split needs that
#: `u.text` (already joined) cannot give back: each word's own ordinal and
#: timing. `end_ms` is null for a caption-tier word (contracts §3); `_split`
#: falls back to the next word's start the same way `implied_end_ms` does
#: for a whole utterance (`rytp.index.utterances`), just scoped to one.
_WORDS_IN_RANGE_SQL = (
    "SELECT ord, start_ms, end_ms, text FROM words"
    " WHERE video_id = ? AND ord BETWEEN ? AND ? ORDER BY ord"
)


def _split_words(words: Sequence[sqlite3.Row], max_chars: int) -> list[list[sqlite3.Row]]:
    """Group consecutive words into rows of at most `max_chars`, joined text.

    Never splits a word: a single word longer than `max_chars` becomes its
    own row rather than being cut mid-token (entry 37). Greedy and
    single-pass, since a row only ever needs to know the length of what it
    already holds plus the next word.
    """
    groups: list[list[sqlite3.Row]] = []
    current: list[sqlite3.Row] = []
    current_len = 0
    for word in words:
        text = str(word["text"])
        grown = len(text) if not current else current_len + 1 + len(text)
        if current and grown > max_chars:
            groups.append(current)
            current = [word]
            current_len = len(text)
        else:
            current.append(word)
            current_len = grown
    if current:
        groups.append(current)
    return groups


def transcript_blocks(
    db: Database, video_id: int, *, max_chars: int | None = None
) -> list[TranscriptBlock]:
    """One block per utterance, or — given `max_chars` — several.

    With `max_chars` left at `None` (`transcript.build`'s durable file),
    this is one block per utterance, unchanged: the anchor printed here and
    the anchor printed by `search words` describe the same span because
    they come from the same row.

    With `max_chars` set (`transcript.show --line-length`, BUGS.md entry
    37), an utterance longer than that is split at a word boundary into
    several rows instead of the caller wrapping its text for display. Every
    word carries its own `start_ms`, so a split row gets a real start and
    end rather than being visual spill from a row that started earlier —
    which is the whole point: a wrapped continuation line could not be
    played, a split row can.

    **Sub-anchors, deliberately** (BUGS.md entry 37's second decision): a
    row that used to be one third of an utterance gets its own anchor
    (`anchor_for` over just that row's word-ordinal range), not the
    utterance's anchor repeated three times. `search play` resolves an
    anchor against the `words` table by ordinal range alone
    (`span_for_anchor`), never against `utterances`, so a sub-range was
    always a valid anchor — this only starts handing one out. The
    alternative (one shared anchor per utterance, however many rows it
    became) would make `search play` on the second or third row seek to
    the start of the first, which is exactly the "spill with no timestamp
    of its own" problem entry 37 is about. A block that fits under
    `max_chars` in one row keeps the whole-utterance anchor exactly as
    before — splitting is a no-op there, not a second code path.
    """
    blocks: list[TranscriptBlock] = []
    for row in db.conn.execute(_TRANSCRIPT_SQL, (video_id,)):
        speaker = C.NULL_CELL if row["speaker"] is None else str(row["speaker"])
        first_ord = int(row["first_word_ord"])
        last_ord = int(row["last_word_ord"])
        utterance_end_ms = int(row["end_ms"])
        if max_chars is None:
            blocks.append(
                TranscriptBlock(
                    anchor=anchor_for(video_id, first_ord, last_ord),
                    start_ms=int(row["start_ms"]),
                    end_ms=utterance_end_ms,
                    speaker=speaker,
                    text=str(row["text"]),
                )
            )
            continue
        words = list(db.conn.execute(_WORDS_IN_RANGE_SQL, (video_id, first_ord, last_ord)))
        groups = _split_words(words, max_chars)
        for group_index, group in enumerate(groups):
            start_ms = int(group[0]["start_ms"])
            last_word = group[-1]
            if last_word["end_ms"] is not None:
                end_ms = int(last_word["end_ms"])
            elif group_index + 1 < len(groups):
                # Still inside this utterance: the next row's first word.
                end_ms = int(groups[group_index + 1][0]["start_ms"])
            else:
                # The last row of the utterance: its own end, already
                # computed with the next *utterance*'s first word in mind
                # (`implied_end_ms`), which a purely local fallback cannot see.
                end_ms = utterance_end_ms
            blocks.append(
                TranscriptBlock(
                    anchor=anchor_for(
                        video_id, int(group[0]["ord"]), int(last_word["ord"])
                    ),
                    start_ms=start_ms,
                    end_ms=max(end_ms, start_ms),
                    speaker=speaker,
                    text=" ".join(str(word["text"]) for word in group),
                )
            )
    return blocks


def _video_row(db: Database, video_id: int) -> tuple[str, str, int]:
    """(title, tier, word count) for the header."""
    row = db.conn.execute("SELECT title FROM videos WHERE id = ?", (video_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"no video with id {video_id}")
    counts = db.conn.execute(
        "SELECT COUNT(*) AS n, COUNT(DISTINCT source) AS tiers,"
        " MIN(source) AS tier FROM words WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    tier = "none" if not counts["n"] else (
        str(counts["tier"]) if counts["tiers"] == 1 else "mixed"
    )
    return str(row["title"]), tier, int(counts["n"])


def render_transcript(db: Database, video_id: int) -> str:
    """The whole markdown document for one video, as a string."""
    title, tier, word_count = _video_row(db, video_id)
    blocks = transcript_blocks(db, video_id)
    if not blocks:
        raise RytpError(
            f"video {video_id} has no utterances; run `rytp index build "
            f"--video {video_id}` first"
        )
    lines = [
        f"# {title}",
        "",
        f"- video: {video_id}",
        f"- source: {tier}",
        f"- words: {word_count}",
        f"- utterances: {len(blocks)}",
        f"- generated: {utc_now_iso()}",
        "",
        "Regenerable output — safe to delete, rebuilt by "
        f"`rytp transcript build {video_id}`. Never read back by any stage.",
        "",
        "Each heading carries an anchor of the form `v<video>:<first>-<last>`: "
        "the video id and the word-ordinal range the block covers. "
        f"`SELECT * FROM words WHERE video_id = {video_id} AND ord BETWEEN "
        "<first> AND <last>` returns exactly those words, and "
        "`rytp search play <anchor>` plays them.",
        "",
    ]
    for block in blocks:
        lines.append(
            f"## [{block.anchor}] {timestamp(block.start_ms)} - "
            f"{timestamp(block.end_ms)} - {block.speaker}"
        )
        lines.append("")
        lines.append(block.text)
        lines.append("")
    return "\n".join(lines)


def write_transcript(db: Database, video_id: int) -> Path:
    """Render and write `data/transcripts/{video_id}.md`. Returns the path."""
    text = render_transcript(db, video_id)
    path = config.paths().transcript(video_id)
    config.ensure_dir(path.parent)
    # newline="\n" explicitly: this runs on Windows and the file should
    # not pick up CRLF depending on where it was generated.
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
