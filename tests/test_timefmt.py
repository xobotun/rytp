"""One duration formatter per purpose (BUGS.md entry 31).

Before this module, `assemble plan` printed a fragment offset as
`0:02.340` while `render run`'s report printed the same kind of quantity
as `0:00:02.700` — same measurement, different shape. These tests pin
each formatter's boundaries and, in the spirit of the flag-vocabulary
consistency suite, scan the source tree for a locally-defined formatter
that should have used one of these instead.
"""

from __future__ import annotations

from pathlib import Path

import rytp
from rytp import timefmt as T
from rytp.commands.assemble import format_ms
from rytp.render.report import format_timecode

# A formatter shape this module owns: two zero-padded fields joined by a
# colon, e.g. `f"{minutes:02d}:{seconds:02d}"`. Any module outside
# `rytp/timefmt.py` that builds this pattern locally is either duplicating a
# purpose this module already names, or has found a purpose that belongs
# here too.
_PRIVATE_FORMATTER = ":02d}:{"

# Modules with a private formatter this batch could not consolidate. Each is
# under concurrent edit outside Task 14 (see the task brief: rytp/tui/*,
# rytp/commands/search.py, rytp/diarize/*, rytp/transcribe/engines/*,
# rytp/jobs/__init__.py are not this task's files). Remove an entry once its
# module is pointed at rytp.timefmt; do not add an entry to make a
# newly-introduced private formatter pass.
#
# `rytp/index/export.py`'s `timestamp()` used to be here: it turned out to
# be a genuinely distinct fourth purpose (a subtitle/cue timestamp, always a
# two-digit hour, unlike `format_seek`'s unpadded one) rather than a module
# that just needed pointing at an existing formatter — `format_cue` was
# added for it and `timestamp()` now delegates.
_KNOWN_PRIVATE: set[Path] = set()


def _rytp_root() -> Path:
    return Path(rytp.__file__).resolve().parent


def test_format_seek_is_always_hours_minutes_seconds_milliseconds() -> None:
    assert T.format_seek(0) == "0:00:00.000"
    assert T.format_seek(2_340) == "0:00:02.340"
    assert T.format_seek(612_340) == "0:10:12.340"
    assert T.format_seek(3_600_000) == "1:00:00.000"
    assert T.format_seek(3_661_001) == "1:01:01.001"
    assert T.format_seek(-5) == "0:00:00.000"


def test_format_spoken_drops_the_hour_when_there_is_none() -> None:
    assert T.format_spoken(0) == "0:00"
    assert T.format_spoken(2_340) == "0:02"
    assert T.format_spoken(612_340) == "10:12"
    assert T.format_spoken(3_600_000) == "1:00:00"
    assert T.format_spoken(3_661_001) == "1:01:01"
    assert T.format_spoken(-5) == "0:00"


def test_format_length_has_no_milliseconds() -> None:
    assert T.format_length(0) == "0:00:00"
    assert T.format_length(999) == "0:00:00"
    assert T.format_length(612_340) == "0:10:12"
    assert T.format_length(3_661_000) == "1:01:01"
    assert T.format_length(-5) == "0:00:00"


def test_format_cue_always_pads_the_hour_to_two_digits() -> None:
    """Distinct from `format_seek`: a column of cues (a transcript's rows)
    must stay one width, which an unpadded hour would not guarantee."""
    assert T.format_cue(0) == "00:00:00.000"
    assert T.format_cue(62_345) == "00:01:02.345"
    assert T.format_cue(3_723_004) == "01:02:03.004"
    assert T.format_cue(-5) == "00:00:00.000"


def test_export_timestamp_is_format_cue() -> None:
    """`rytp/index/export.py`'s `timestamp()` used to build this shape
    itself; it now delegates, closing the `_KNOWN_PRIVATE` entry this
    module's Task 14 left for it."""
    from rytp.index.export import timestamp

    for ms in (0, 62_345, 3_723_004):
        assert timestamp(ms) == T.format_cue(ms)


def test_a_plan_and_a_render_report_print_the_same_seek_target_alike() -> None:
    """BUGS.md entry 31's central complaint: `assemble plan`'s fragment
    offsets and `render run`'s report offsets are the same purpose and
    must be the same string for the same millisecond value."""
    for ms in (0, 2_340, 612_340, 2_700, 3_661_001):
        assert format_ms(ms) == format_timecode(ms) == T.format_seek(ms)


def test_no_module_outside_timefmt_builds_a_private_formatter() -> None:
    root = _rytp_root()
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "timefmt.py":
            continue
        relative = path.relative_to(root)
        if relative in _KNOWN_PRIVATE:
            continue
        text = path.read_text(encoding="utf-8")
        if _PRIVATE_FORMATTER in text:
            offenders.append(str(relative))
    assert offenders == []
