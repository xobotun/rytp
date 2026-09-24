"""Editing a cut list (design §8, design §10).

design §8 makes the cut list "a plain editable file… hand-edit it and re-render",
and the owner asked for exactly that: adjusting which fragment is used and when
it starts. Part 5 wrote the emitter and a loader built for a file a person has
changed. What was missing is a place to make the change that is not another
window with a text editor in it.

Four verbs — swap, nudge, gap, undo — and a save. No database and no re-planning:
the loader does not look at the database either, and a screen that quietly
re-ran the assembler would discard the hand edits it exists to keep.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.assemble.cutlist import (
    Alternative,
    CutList,
    CutlistError,
    Slot,
    load_cutlist,
    write_cutlist,
)

__all__ = ["CutlistSummary", "CutlistView", "available_cutlists"]


@dataclass(frozen=True)
class CutlistSummary:
    """One line of the picker."""

    name: str
    path: Path
    target: str
    slots: int
    gaps: int
    sources: int
    duration_ms: int
    error: str | None = None


def available_cutlists() -> tuple[CutlistSummary, ...]:
    """Every cut list on disk, broken ones included.

    A file the owner mistyped is the one he most needs to find, so a load
    failure becomes a row with its complaint rather than a missing row.
    """
    directory = config.paths().root / C.CUTLISTS_DIRNAME
    if not directory.is_dir():
        return ()
    found: list[CutlistSummary] = []
    for path in sorted(directory.glob("*.toml")):
        try:
            cutlist = load_cutlist(path)
        # `ValueError` because `tomllib.TOMLDecodeError` subclasses it, and a
        # file that is not TOML at all may never reach Part 5's own error type.
        # A picker that raises on one bad file shows none of the good ones.
        except (CutlistError, ValueError, OSError) as exc:
            found.append(
                CutlistSummary(
                    name=path.stem,
                    path=path,
                    target="",
                    slots=0,
                    gaps=0,
                    sources=0,
                    duration_ms=0,
                    error=str(exc),
                )
            )
            continue
        found.append(
            CutlistSummary(
                name=cutlist.name or path.stem,
                path=path,
                target=cutlist.target,
                slots=len(cutlist.slots),
                gaps=len(cutlist.gaps),
                sources=len({slot.video_id for slot in cutlist.fragments}),
                duration_ms=cutlist.duration_ms,
            )
        )
    return tuple(found)


def _ms(value: int | None) -> str:
    if value is None:
        return C.NULL_CELL
    seconds, milliseconds = divmod(int(value), 1000)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:d}:{seconds:02d}.{milliseconds:03d}"


class CutlistView:
    """The slots of one cut list, and the four edits worth making by hand."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.cutlist: CutList = load_cutlist(path)
        self.name = self.cutlist.name or path.stem
        self.dirty = False
        self._undo: list[CutList] = []
        self.status = self._status()

    # -- reading -----------------------------------------------------

    @property
    def columns(self) -> tuple[str, ...]:
        return ("#", "kind", "text", "source", "in", "out", "gap", "alts")

    @property
    def rows(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            (
                str(index),
                slot.kind,
                slot.text,
                f"v{slot.video_id}" if slot.video_id is not None else C.NULL_CELL,
                _ms(slot.start_ms),
                _ms(slot.end_ms),
                _ms(slot.gap_before_ms),
                str(len(self._options(slot)[1])),
            )
            for index, slot in enumerate(self.cutlist.slots)
        )

    def slot_at(self, index: int) -> Slot | None:
        slots = self.cutlist.slots
        return slots[index] if 0 <= index < len(slots) else None

    @staticmethod
    def _options(slot: Slot) -> tuple[str, Sequence[object]]:
        """What this slot can be swapped to, and which kind of thing that is.

        Not simply "alternatives for a fragment": adopting a substitution
        turns a gap into a fragment, and its ranked substitutions are exactly
        what a second thought needs. So the ranked list that *has* entries
        wins, and a fragment that was always a fragment shows alternatives.
        """
        if slot.kind == "fragment" and slot.alternatives:
            return "alternative", slot.alternatives
        if slot.substitutions:
            return "substitution", slot.substitutions
        return "alternative", slot.alternatives

    def option_columns(self, index: int) -> tuple[str, ...]:
        slot = self.slot_at(index)
        if slot is None:
            return ()
        if self._options(slot)[0] == "alternative":
            return ("#", "text", "source", "in", "out", "cost")
        return ("#", "text", "reason", "source", "in", "out")

    def option_rows(self, index: int) -> tuple[tuple[str, ...], ...]:
        slot = self.slot_at(index)
        if slot is None:
            return ()
        if self._options(slot)[0] == "alternative":
            return tuple(
                (
                    str(position),
                    alt.text,
                    f"v{alt.video_id}",
                    _ms(alt.start_ms),
                    _ms(alt.end_ms),
                    f"{alt.cost:.2f}",
                )
                for position, alt in enumerate(slot.alternatives)
            )
        return tuple(
            (
                str(position),
                sub.text,
                sub.reason,
                f"v{sub.video_id}",
                _ms(sub.start_ms),
                _ms(sub.end_ms),
            )
            for position, sub in enumerate(slot.substitutions)
        )

    # -- editing -----------------------------------------------------

    def _replace_slot(self, index: int, slot: Slot) -> None:
        self._undo.append(self.cutlist)
        slots = list(self.cutlist.slots)
        slots[index] = slot
        self.cutlist = dataclasses.replace(self.cutlist, slots=tuple(slots))
        self.dirty = True

    def swap(self, slot_index: int, option_index: int) -> str:
        """Promote a ranked alternative, or adopt a ranked substitution."""
        slot = self.slot_at(slot_index)
        if slot is None:
            return f"no slot {slot_index}"
        what, options = self._options(slot)
        if not (0 <= option_index < len(options)):
            return f"slot {slot_index} has no {what} {option_index}"
        if what == "alternative":
            return self._promote(slot_index, slot, option_index)
        return self._adopt(slot_index, slot, option_index)

    def _promote(self, index: int, slot: Slot, option_index: int) -> str:
        chosen = slot.alternatives[option_index]

        # `is not None`, not `or`: ordinals are 0-based, so `ord == 0` is the
        # first word of a video and not a missing value. Part 5's loader uses
        # -1 for "whoever retyped a timing dropped the provenance".
        def _or(value: int | None, absent: int) -> int:
            return absent if value is None else value

        displaced = Alternative(
            video_id=_or(slot.video_id, -1),
            first_word_ord=_or(slot.first_word_ord, -1),
            last_word_ord=_or(slot.last_word_ord, -1),
            start_ms=_or(slot.start_ms, 0),
            end_ms=_or(slot.end_ms, 0),
            text=slot.text,
            cost=slot.cost if slot.cost is not None else 0.0,
        )
        alternatives = list(slot.alternatives)
        alternatives[option_index] = displaced
        self._replace_slot(
            index,
            dataclasses.replace(
                slot,
                video_id=chosen.video_id,
                first_word_ord=chosen.first_word_ord,
                last_word_ord=chosen.last_word_ord,
                start_ms=chosen.start_ms,
                end_ms=chosen.end_ms,
                text=chosen.text,
                cost=chosen.cost,
                # An Alternative carries none of these, and copying the
                # displaced fragment's would claim a clip from another video
                # was said by a person identified in this one.
                align_score=None,
                video_speaker_id=None,
                speaker_label=None,
                alternatives=tuple(alternatives),
            ),
        )
        self.status = self._status(
            f"slot {index} now comes from v{chosen.video_id}; its speaker and "
            f"alignment score are unknown, so the render will use this video's "
            f"pause statistics rather than a speaker's"
        )
        return self.status

    def _adopt(self, index: int, slot: Slot, option_index: int) -> str:
        chosen = slot.substitutions[option_index]
        self._replace_slot(
            index,
            dataclasses.replace(
                slot,
                kind="fragment",
                text=chosen.text,
                video_id=chosen.video_id,
                first_word_ord=chosen.first_word_ord,
                last_word_ord=chosen.last_word_ord,
                start_ms=chosen.start_ms,
                end_ms=chosen.end_ms,
                align_score=None,
                video_speaker_id=None,
                speaker_label=None,
            ),
        )
        self.status = self._status(
            f'slot {index} now says "{chosen.text}" instead of "{slot.text}"'
        )
        return self.status

    def nudge(self, slot_index: int, *, edge: str, delta_ms: int) -> str:
        """Move one boundary. design §8 prefers this to a clever heuristic."""
        slot = self.slot_at(slot_index)
        if slot is None:
            return f"no slot {slot_index}"
        if slot.kind != "fragment" or slot.start_ms is None or slot.end_ms is None:
            return f"slot {slot_index} is a gap: there is no boundary to move"
        start, end = slot.start_ms, slot.end_ms
        if edge == "start":
            start = max(0, start + delta_ms)
        else:
            end = max(0, end + delta_ms)
        if end - start < C.TUI_CUTLIST_NUDGE_MS:
            return (
                f"that would leave slot {slot_index} shorter than "
                f"{C.TUI_CUTLIST_NUDGE_MS} ms"
            )
        self._replace_slot(
            slot_index, dataclasses.replace(slot, start_ms=start, end_ms=end)
        )
        self.status = self._status(
            f"slot {slot_index} is now {_ms(start)} to {_ms(end)}"
        )
        return self.status

    def set_gap_before(self, slot_index: int, gap_ms: int) -> str:
        slot = self.slot_at(slot_index)
        if slot is None:
            return f"no slot {slot_index}"
        if gap_ms < 0:
            return "a pause cannot be negative"
        self._replace_slot(slot_index, dataclasses.replace(slot, gap_before_ms=gap_ms))
        self.status = self._status(f"slot {slot_index} is preceded by {gap_ms} ms")
        return self.status

    def clear_gap(self, slot_index: int) -> str:
        slot = self.slot_at(slot_index)
        if slot is None:
            return f"no slot {slot_index}"
        self._replace_slot(slot_index, dataclasses.replace(slot, gap_before_ms=None))
        self.status = self._status(f"slot {slot_index} uses the measured pause again")
        return self.status

    def undo(self) -> str:
        if not self._undo:
            return "nothing to undo"
        self.cutlist = self._undo.pop()
        self.dirty = bool(self._undo)
        self.status = self._status("undone")
        return self.status

    # -- the file ----------------------------------------------------

    def save(self) -> Path:
        """Write it back, through Part 5's emitter, so Part 6 can read it."""
        config.ensure_dir(self.path.parent)
        written = write_cutlist(self.cutlist, self.path)
        self._undo.clear()
        self.dirty = False
        self.status = self._status(f"saved to {written}")
        return written

    def reload(self) -> str:
        self.cutlist = load_cutlist(self.path)
        self._undo.clear()
        self.dirty = False
        self.status = self._status("discarded the unsaved edits")
        return self.status

    def _status(self, note: str = "") -> str:
        fragments = len(self.cutlist.fragments)
        gaps = len(self.cutlist.gaps)
        sources = len({slot.video_id for slot in self.cutlist.fragments})
        parts = [
            f"{fragments} fragment{'' if fragments == 1 else 's'}",
            f"{gaps} gap{'' if gaps == 1 else 's'}",
            f"{sources} source{'' if sources == 1 else 's'}",
            _ms(self.cutlist.duration_ms),
        ]
        if self.dirty:
            parts.append("unsaved")
        if note:
            parts.append(note)
        return " · ".join(parts)
