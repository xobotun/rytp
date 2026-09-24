"""The cut list: the durable thing assembly produces (design §8).

    "The cut list is a plain editable file — fragments, source ids,
    in/out points in milliseconds. It is the durable representation:
    hand-edit it and re-render."

The file is one ordered ``[[slot]]`` array. Document order **is** the
output timeline, and a slot is either a fragment (something to cut) or a
gap (a word the corpus never says). There is deliberately no index field:
a number that duplicates position is a number a hand-edit can falsify.

Optional keys are omitted rather than written as an empty value, because
TOML has no null and "absent" carries meaning here — an absent
``gap_before_ms`` means the renderer decides the pause, a present one
means a human already did.

Reading is ``tomllib`` from the standard library. Writing is the small
canonical emitter at the bottom of this module rather than ``tomli-w``:
the header comment is the file's instructions to its own reader and no
TOML writer emits comments, and a canonical emitter makes the output
byte-stable, which is half of design §8's determinism requirement.
"""

from __future__ import annotations

import difflib
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final, cast

from rytp import constants as C
from rytp.assemble.match import SLOT_FRAGMENT, SLOT_GAP, Plan, SubstitutionHit
from rytp.config import ensure_dir, paths
from rytp.models import Fragment, InvalidInputError, RytpError, normalize_text

__all__ = [
    "Alternative",
    "CutList",
    "CutlistError",
    "CutlistParams",
    "Slot",
    "Substitution",
    "cutlist_name",
    "cutlist_path",
    "dumps_cutlist",
    "from_plan",
    "load_cutlist",
    "read_cutlist",
    "validate_name",
    "write_cutlist",
]

_HEADER = (
    "# rytp cut list — hand-edit this file and re-render; the timings below\n"
    "# are authoritative and are never recomputed.\n"
    "# Slots run in document order: that order is the output timeline.\n"
    "# To fill a gap, copy a substitution's fields into its slot and change\n"
    "# kind = \"gap\" to kind = \"fragment\".\n"
    "# To swap a fragment, copy one of its alternatives over its own fields.\n"
    "# gap_before_ms is optional: add it to force a pause, leave it out to\n"
    "# let the renderer decide.\n"
)


@dataclass(frozen=True)
class Alternative:
    """A source this slot could have used instead.

    Carries the same provenance as the chosen fragment — alignment and
    speaker included — so swapping one in gives a human editing the file
    the same information the chosen fragment had, rather than a bare
    timing they would have to look up again.
    """

    video_id: int
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    text: str
    cost: float
    align_score: float | None = None
    video_speaker_id: int | None = None
    speaker_label: str | None = None


@dataclass(frozen=True)
class Substitution:
    """A word the corpus does say, offered for one it does not.

    Carries full fragment fields on purpose: filling a gap is then a
    copy-paste, not another run of the assembler.
    """

    text: str
    reason: str  # "stem" | "edit"
    distance: int
    occurrences: int
    video_id: int
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Slot:
    """One position on the output timeline."""

    kind: str  # SLOT_FRAGMENT | SLOT_GAP
    target_first: int
    target_last: int  # inclusive
    text: str
    video_id: int | None = None
    first_word_ord: int | None = None
    last_word_ord: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    align_score: float | None = None
    cost: float | None = None
    video_speaker_id: int | None = None
    speaker_label: str | None = None
    gap_before_ms: int | None = None
    alternatives: tuple[Alternative, ...] = ()
    substitutions: tuple[Substitution, ...] = ()

    def as_fragment(self) -> Fragment:
        """This slot as the contracts §4 ``Fragment``.

        No asserts: ``-O`` strips those, and this is the boundary where a
        hand-edited file turns into something the renderer will cut.
        """
        if self.kind != SLOT_FRAGMENT:
            raise InvalidInputError(
                f"slot at target word {self.target_first} is a gap ({self.text!r}); "
                "a gap has nothing to cut"
            )
        if (
            self.video_id is None
            or self.first_word_ord is None
            or self.last_word_ord is None
            or self.start_ms is None
            or self.end_ms is None
        ):
            raise InvalidInputError(
                f"fragment slot at target word {self.target_first} is missing a "
                "required field; it needs video_id, first_word_ord, last_word_ord, "
                "start_ms and end_ms"
            )
        return Fragment(
            video_id=self.video_id,
            first_word_ord=self.first_word_ord,
            last_word_ord=self.last_word_ord,
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            text=self.text,
        )


@dataclass(frozen=True)
class CutlistParams:
    """The knob positions this cut list was produced with, for provenance."""

    consistency: float
    seed: int
    pad_ms: int
    speaker: str
    exclude: tuple[int, ...]
    min_align_score: float


@dataclass(frozen=True)
class CutList:
    """A whole cut list, in memory."""

    schema_version: int
    name: str
    target: str
    created_at: str
    params: CutlistParams
    slots: tuple[Slot, ...]

    @property
    def fragments(self) -> tuple[Slot, ...]:
        return tuple(slot for slot in self.slots if slot.kind == SLOT_FRAGMENT)

    @property
    def gaps(self) -> tuple[Slot, ...]:
        return tuple(slot for slot in self.slots if slot.kind == SLOT_GAP)

    @property
    def duration_ms(self) -> int:
        """Total cut length, pauses excluded — the renderer adds those."""
        return sum(
            (slot.end_ms or 0) - (slot.start_ms or 0)
            for slot in self.fragments
            if slot.start_ms is not None and slot.end_ms is not None
        )


# -- naming -----------------------------------------------------------

_SLUG_STRIP = re.compile(r"[^\w-]+", re.UNICODE)
_SLUG_RUNS = re.compile(r"-{2,}")


def cutlist_name(target: str) -> str:
    """A filename-safe, deterministic name derived from the target text."""
    slug = _SLUG_STRIP.sub("-", normalize_text(target))
    slug = _SLUG_RUNS.sub("-", slug).strip("-")
    slug = slug[: C.ASSEMBLE_NAME_MAX_CHARS].strip("-")
    return slug or C.ASSEMBLE_FALLBACK_NAME


def validate_name(name: str) -> str:
    """Reject anything that would write outside ``cutlists/``.

    The name reaches this from the command line, so a separator or a
    ``..`` in it is a path traversal, not a naming preference. Windows
    also treats ``C:name`` as a drive-relative path, hence the colon.
    """
    cleaned = name.strip()
    if not cleaned:
        raise InvalidInputError("a cut list needs a name")
    if cleaned in {".", ".."} or any(bad in cleaned for bad in ("/", "\\", ":")):
        raise InvalidInputError(
            f"bad cut list name {name!r}: no path separators, no '..', no drive letters"
        )
    return cleaned


def cutlist_path(name: str) -> Path:
    """Where a cut list of this name lives (contracts §7)."""
    return paths().cutlist(validate_name(name))


# -- building from a plan ---------------------------------------------


def from_plan(
    plan: Plan,
    *,
    name: str,
    params: CutlistParams,
    created_at: str,
    substitutions: Mapping[int, Sequence[SubstitutionHit]],
    speaker_labels: Mapping[int, str],
) -> CutList:
    """Turn a coverage plan into the artifact.

    ``substitutions`` is keyed by the gap's first target word;
    ``speaker_labels`` maps ``video_speakers.id`` to a roster label, and
    is simply empty for an undiarized corpus.
    """
    slots: list[Slot] = []
    for slot in plan.slots:
        run = slot.run
        if slot.kind == SLOT_GAP or run is None:
            slots.append(
                Slot(
                    kind=SLOT_GAP,
                    target_first=slot.target_first,
                    target_last=slot.target_last,
                    text=slot.text,
                    substitutions=tuple(
                        Substitution(
                            text=hit.text,
                            reason=hit.reason,
                            distance=hit.distance,
                            occurrences=hit.occurrences,
                            video_id=hit.run.video_id,
                            first_word_ord=hit.run.first_word_ord,
                            last_word_ord=hit.run.last_word_ord,
                            start_ms=hit.run.start_ms,
                            end_ms=hit.run.end_ms,
                        )
                        for hit in substitutions.get(slot.target_first, ())
                    ),
                )
            )
            continue
        slots.append(
            Slot(
                kind=SLOT_FRAGMENT,
                target_first=slot.target_first,
                target_last=slot.target_last,
                text=run.text,
                video_id=run.video_id,
                first_word_ord=run.first_word_ord,
                last_word_ord=run.last_word_ord,
                start_ms=run.start_ms,
                end_ms=run.end_ms,
                align_score=run.mean_align,
                cost=slot.cost,
                video_speaker_id=run.video_speaker_id,
                speaker_label=(
                    None
                    if run.video_speaker_id is None
                    else speaker_labels.get(run.video_speaker_id)
                ),
                alternatives=tuple(
                    Alternative(
                        video_id=scored.run.video_id,
                        first_word_ord=scored.run.first_word_ord,
                        last_word_ord=scored.run.last_word_ord,
                        start_ms=scored.run.start_ms,
                        end_ms=scored.run.end_ms,
                        text=scored.run.text,
                        cost=scored.cost,
                        align_score=scored.run.mean_align,
                        video_speaker_id=scored.run.video_speaker_id,
                        speaker_label=(
                            None
                            if scored.run.video_speaker_id is None
                            else speaker_labels.get(scored.run.video_speaker_id)
                        ),
                    )
                    for scored in slot.alternatives
                ),
            )
        )
    return CutList(
        schema_version=C.CUTLIST_SCHEMA_VERSION,
        name=name,
        target=plan.target,
        created_at=created_at,
        params=params,
        slots=tuple(slots),
    )


# -- the emitter ------------------------------------------------------

_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_string(value: str) -> str:
    """A TOML basic string. Cyrillic passes through; controls are escaped."""
    out: list[str] = ['"']
    for char in value:
        if char in _ESCAPES:
            out.append(_ESCAPES[char])
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append(f"\\u{ord(char):04X}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def _toml_value(value: object) -> str:
    """One scalar or one integer array, canonically."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(round(value, C.ASSEMBLE_COST_DECIMALS))
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"a cut list cannot hold {type(value).__name__}")


def _emit(lines: list[str], header: str, record: object, skip: Sequence[str] = ()) -> None:
    """One TOML table from a dataclass, omitting None and the named fields."""
    lines.append(header)
    for field in fields(record):  # type: ignore[arg-type]
        if field.name in skip:
            continue
        value = getattr(record, field.name)
        if value is None:
            continue
        lines.append(f"{field.name} = {_toml_value(value)}")
    lines.append("")


def dumps_cutlist(cutlist: CutList) -> str:
    """The whole cut list as canonical TOML text."""
    lines: list[str] = [_HEADER.rstrip("\n"), ""]
    lines.append(f"schema_version = {cutlist.schema_version}")
    lines.append(f"name = {_toml_string(cutlist.name)}")
    lines.append(f"target = {_toml_string(cutlist.target)}")
    lines.append(f"created_at = {_toml_string(cutlist.created_at)}")
    lines.append("")
    _emit(lines, "[params]", cutlist.params)
    for slot in cutlist.slots:
        _emit(lines, "[[slot]]", slot, skip=("alternatives", "substitutions"))
        for alternative in slot.alternatives:
            _emit(lines, "[[slot.alternative]]", alternative)
        for substitution in slot.substitutions:
            _emit(lines, "[[slot.substitution]]", substitution)
    return "\n".join(lines).rstrip("\n") + "\n"


def write_cutlist(cutlist: CutList, path: Path) -> Path:
    """Write the cut list, creating its directory. Returns the path written."""
    ensure_dir(path.parent)
    path.write_text(dumps_cutlist(cutlist), encoding="utf-8", newline="\n")
    return path


# -- loading ------------------------------------------------------------


class CutlistError(RytpError):
    """A cut list file that cannot be read as one.

    Always names the file and, where it applies, the slot and the field,
    because the person who will fix it is looking at a text editor.
    """


#: Target position of a slot a hand-edit inserted without saying which
#: word of the target it covers. Informational only.
_UNPLACED: Final = -1

#: Provenance a hand-edit may drop: the cut still works without it.
_ORD_DEFAULT: Final = -1


def _keys_of(record: type) -> tuple[str, ...]:
    """Field names of a dataclass, as they appear in the file."""
    return tuple(field.name for field in fields(record))


_SLOT_KEYS: Final = (
    *(name for name in _keys_of(Slot) if name not in {"alternatives", "substitutions"}),
    "alternative",
    "substitution",
)


def _reject(where: str, detail: str) -> CutlistError:
    return CutlistError(f"{where}: {detail}")


def _check_keys(where: str, table: Mapping[str, object], allowed: Sequence[str]) -> None:
    """Refuse an unknown key, guessing what was meant.

    The realistic failure is a typo or a hyphen where an underscore
    belongs, so a bare "unknown key" would be a worse message than the
    file deserves.
    """
    for key in table:
        if key in allowed:
            continue
        close = difflib.get_close_matches(key, allowed, n=1)
        hint = f"; did you mean {close[0]!r}?" if close else f"; known keys: {', '.join(allowed)}"
        raise _reject(where, f"unknown key {key!r}{hint}")


_TYPE_NAMES: Final = {int: "integer", float: "number", str: "string"}


def _raw(
    where: str, table: Mapping[str, object], key: str, kind: type, *, required: bool
) -> object | None:
    """One field, type-checked once, so the readers below need no casts of their own."""
    if key not in table:
        if required:
            raise _reject(where, f"missing required key {key!r}")
        return None
    value = table[key]
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)  # TOML separates 1 from 1.0; a person editing does not
    if isinstance(value, bool) or not isinstance(value, kind):
        raise _reject(where, f"{key} must be a {_TYPE_NAMES[kind]}, got {value!r}")
    return value


def _int_or(where: str, table: Mapping[str, object], key: str, default: int) -> int:
    """An integer field with a fallback. Never ``or``: a stored 0 is a real 0."""
    value = _raw(where, table, key, int, required=False)
    return default if value is None else cast(int, value)


def _opt_int(where: str, table: Mapping[str, object], key: str) -> int | None:
    value = _raw(where, table, key, int, required=False)
    return None if value is None else cast(int, value)


def _req_int(where: str, table: Mapping[str, object], key: str) -> int:
    return cast(int, _raw(where, table, key, int, required=True))


def _float_or(where: str, table: Mapping[str, object], key: str, default: float) -> float:
    value = _raw(where, table, key, float, required=False)
    return default if value is None else cast(float, value)


def _opt_float(where: str, table: Mapping[str, object], key: str) -> float | None:
    value = _raw(where, table, key, float, required=False)
    return None if value is None else cast(float, value)


def _str_or(where: str, table: Mapping[str, object], key: str, default: str = "") -> str:
    value = _raw(where, table, key, str, required=False)
    return default if value is None else cast(str, value)


def _opt_str(where: str, table: Mapping[str, object], key: str) -> str | None:
    value = _raw(where, table, key, str, required=False)
    return None if value is None else cast(str, value)


def _req_str(where: str, table: Mapping[str, object], key: str) -> str:
    return cast(str, _raw(where, table, key, str, required=True))


def _non_negative(where: str, key: str, value: int) -> int:
    if value < 0:
        raise _reject(where, f"{key} must be zero or more, got {value}")
    return value


def _span(where: str, table: Mapping[str, object]) -> tuple[int, int]:
    """The in and out points, validated as a real span.

    Shared by fragments, alternatives and substitutions: all three are
    something to cut, and all three fail the same way.
    """
    start = _non_negative(where, "start_ms", _req_int(where, table, "start_ms"))
    end = _non_negative(where, "end_ms", _req_int(where, table, "end_ms"))
    if end <= start:
        raise _reject(where, f"end_ms ({end}) must be greater than start_ms ({start})")
    return start, end


def _alternative(where: str, table: Mapping[str, object]) -> Alternative:
    _check_keys(where, table, _keys_of(Alternative))
    start, end = _span(where, table)
    return Alternative(
        video_id=_req_int(where, table, "video_id"),
        first_word_ord=_int_or(where, table, "first_word_ord", _ORD_DEFAULT),
        last_word_ord=_int_or(where, table, "last_word_ord", _ORD_DEFAULT),
        start_ms=start,
        end_ms=end,
        text=_str_or(where, table, "text"),
        cost=_float_or(where, table, "cost", 0.0),
        align_score=_opt_float(where, table, "align_score"),
        video_speaker_id=_opt_int(where, table, "video_speaker_id"),
        speaker_label=_opt_str(where, table, "speaker_label"),
    )


def _substitution(where: str, table: Mapping[str, object]) -> Substitution:
    _check_keys(where, table, _keys_of(Substitution))
    start, end = _span(where, table)
    return Substitution(
        text=_req_str(where, table, "text"),
        reason=_str_or(where, table, "reason", "edit"),
        distance=_int_or(where, table, "distance", 0),
        occurrences=_int_or(where, table, "occurrences", 0),
        video_id=_req_int(where, table, "video_id"),
        first_word_ord=_int_or(where, table, "first_word_ord", _ORD_DEFAULT),
        last_word_ord=_int_or(where, table, "last_word_ord", _ORD_DEFAULT),
        start_ms=start,
        end_ms=end,
    )


def _nested(where: str, table: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    raw = table.get(key, [])
    if not isinstance(raw, list):
        raise _reject(where, f"{key} must be written as [[slot.{key}]] tables")
    return [item for item in raw if isinstance(item, dict)]


def _slot(where: str, table: Mapping[str, object]) -> Slot:
    _check_keys(where, table, _SLOT_KEYS)
    kind = _req_str(where, table, "kind")
    if kind not in {SLOT_FRAGMENT, SLOT_GAP}:
        raise _reject(where, f"kind must be {SLOT_FRAGMENT!r} or {SLOT_GAP!r}, got {kind!r}")

    first = _int_or(where, table, "target_first", _UNPLACED)
    last = _int_or(where, table, "target_last", _UNPLACED)
    alternatives = tuple(
        _alternative(f"{where}: alternative {index}", item)
        for index, item in enumerate(_nested(where, table, "alternative"), start=1)
    )
    substitutions = tuple(
        _substitution(f"{where}: substitution {index}", item)
        for index, item in enumerate(_nested(where, table, "substitution"), start=1)
    )

    if kind == SLOT_GAP:
        gap_before = _opt_int(where, table, "gap_before_ms")
        return Slot(
            kind=SLOT_GAP,
            target_first=first,
            target_last=last,
            text=_str_or(where, table, "text"),
            gap_before_ms=(
                None if gap_before is None
                else _non_negative(where, "gap_before_ms", gap_before)
            ),
            alternatives=alternatives,
            substitutions=substitutions,
        )

    start, end = _span(where, table)
    gap_before = _opt_int(where, table, "gap_before_ms")
    return Slot(
        kind=SLOT_FRAGMENT,
        target_first=first,
        target_last=last,
        text=_str_or(where, table, "text"),
        video_id=_req_int(where, table, "video_id"),
        first_word_ord=_int_or(where, table, "first_word_ord", _ORD_DEFAULT),
        last_word_ord=_int_or(where, table, "last_word_ord", _ORD_DEFAULT),
        start_ms=start,
        end_ms=end,
        align_score=_opt_float(where, table, "align_score"),
        cost=_opt_float(where, table, "cost"),
        video_speaker_id=_opt_int(where, table, "video_speaker_id"),
        speaker_label=_opt_str(where, table, "speaker_label"),
        gap_before_ms=(
            None if gap_before is None else _non_negative(where, "gap_before_ms", gap_before)
        ),
        alternatives=alternatives,
        substitutions=substitutions,
    )


def _params(where: str, table: Mapping[str, object]) -> CutlistParams:
    _check_keys(where, table, _keys_of(CutlistParams))
    exclude = table.get("exclude", [])
    if not isinstance(exclude, list) or any(not isinstance(item, int) for item in exclude):
        raise _reject(where, "exclude must be a list of video ids, for example [3, 7]")
    return CutlistParams(
        consistency=_float_or(where, table, "consistency", C.ASSEMBLE_DEFAULT_CONSISTENCY),
        seed=_int_or(where, table, "seed", 0),
        pad_ms=_int_or(where, table, "pad_ms", C.ASSEMBLE_DEFAULT_PAD_MS),
        speaker=_str_or(where, table, "speaker"),
        exclude=tuple(int(item) for item in exclude),
        min_align_score=_float_or(where, table, "min_align_score", C.ASSEMBLE_MIN_ALIGN_SCORE),
    )


def load_cutlist(path: Path) -> CutList:
    """Read a cut list, tolerating a hand-edit and naming what it cannot read.

    Structure is required and provenance is not: a fragment with no
    ``video_id`` or no ``end_ms`` is not a cut and fails, while a missing
    ``first_word_ord`` is filled in and forgotten. Nothing here reads the
    database — whether a ``video_id`` still exists is the caller's
    question.
    """
    where = path.name
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CutlistError(f"cannot read the cut list at {path}: {exc}") from exc
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise CutlistError(f"{where}: not UTF-8 text: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CutlistError(f"{where}: not valid TOML: {exc}") from exc

    version = _req_int(where, document, "schema_version")
    if version != C.CUTLIST_SCHEMA_VERSION:
        raise _reject(
            where,
            f"schema_version is {version}; this build reads "
            f"{C.CUTLIST_SCHEMA_VERSION}",
        )

    raw_slots = document.get("slot", [])
    if not isinstance(raw_slots, list) or not raw_slots:
        raise _reject(where, "no slots; a cut list needs at least one [[slot]] table")

    params_table = document.get("params", {})
    if not isinstance(params_table, dict):
        raise _reject(where, "params must be a [params] table")

    return CutList(
        schema_version=C.CUTLIST_SCHEMA_VERSION,
        name=_str_or(where, document, "name", path.stem),
        target=_str_or(where, document, "target"),
        created_at=_str_or(where, document, "created_at"),
        params=_params(f"{where}: [params]", params_table),
        slots=tuple(
            _slot(f"{where}: slot {index}", table)
            for index, table in enumerate(raw_slots, start=1)
            if isinstance(table, dict)
        ),
    )


def read_cutlist(name: str) -> CutList:
    """Load the cut list of this name from under the data tree."""
    return load_cutlist(cutlist_path(name))
