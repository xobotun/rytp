"""Duration formatting, one function per purpose (BUGS.md entry 31).

Before this module, the same millisecond quantity printed differently
depending on which command asked: `assemble plan` said `0:02.340`,
`render run`'s report said `0:00:02.700` — same measurement, and the only
visual difference was a colon. Several *shapes* are legitimate — a
paste-ready description wants `M:SS`, a seek target wants milliseconds —
but two commands in one pipeline must agree whenever they print the same
*purpose*. This module is the one place each purpose is decided; every
call site delegates to it instead of building its own `f"{...:02d}"`.

Four purposes, so far:

- :func:`format_seek` — a target precise enough to paste into a player's
  seek box. Both a plan's fragment offsets and a render's report offsets
  are seek targets, so both use this.
- :func:`format_spoken` — the form a paste-ready description wants:
  `M:SS`, or `H:MM:SS` past an hour. Never sub-second — nobody types
  milliseconds into a description.
- :func:`format_length` — a duration in a listing (a video's total
  runtime, a source's total used time): `H:MM:SS`, no milliseconds.
- :func:`format_cue` — a subtitle/transcript cue timestamp: `HH:MM:SS.mmm`
  with the hour always two digits. Distinct from `format_seek`, which
  leaves the hour unpadded (`0:00:02.340`, not `00:00:02.340`) — a cue
  sorts and greps cleanly at any hour precisely because every field is a
  fixed width, which a seek target has no reason to promise.

All four clamp a negative input to zero rather than producing a
negative or wrapped-around string.
"""

from __future__ import annotations

from rytp import constants as C

_MS_PER_MINUTE = 60 * C.MS_PER_SECOND
_MS_PER_HOUR = 60 * _MS_PER_MINUTE


def format_seek(ms: int) -> str:
    """``H:MM:SS.mmm`` — precise enough to seek to in a player.

    The hour field is always present, even when it is ``0``, so that a
    plan's fragment offsets and a render's report offsets are always the
    same shape and can be compared by eye.
    """
    ms = max(0, int(ms))
    hours, rest = divmod(ms, _MS_PER_HOUR)
    minutes, rest = divmod(rest, _MS_PER_MINUTE)
    secs, millis = divmod(rest, C.MS_PER_SECOND)
    return f"{hours}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_spoken(ms: int) -> str:
    """``M:SS``, or ``H:MM:SS`` past an hour — the form a description wants."""
    ms = max(0, int(ms))
    hours, rest = divmod(ms, _MS_PER_HOUR)
    minutes, rest = divmod(rest, _MS_PER_MINUTE)
    secs = rest // C.MS_PER_SECOND
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_length(ms: int) -> str:
    """``H:MM:SS`` — a duration in a listing, no milliseconds."""
    ms = max(0, int(ms))
    seconds, _ = divmod(ms, C.MS_PER_SECOND)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def format_cue(ms: int) -> str:
    """``HH:MM:SS.mmm`` — a transcript/subtitle cue, hour always two digits.

    A transcript heading or row is read down a column, so every cue should
    be the same width regardless of how long the video runs — `format_seek`
    would print `0:12:03.000` and `10:12:03.000` at different widths, which
    is fine for a one-off seek target and wrong for a column of them.
    """
    ms = max(0, int(ms))
    hours, rest = divmod(ms, _MS_PER_HOUR)
    minutes, rest = divmod(rest, _MS_PER_MINUTE)
    secs, millis = divmod(rest, C.MS_PER_SECOND)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
