"""The two-pane speaker mapper, with no terminal in it (design §10).

Design §10 calls speaker assignment "the one genuinely interactive task",
which is exactly why the interaction lives here rather than in a Textual
screen: a session is an object with two lists, two cursors, a filter and
four verbs, and every one of those can be asserted on without a TTY. The
screen in `rytp/tui/screens/speakers.py` draws this object and forwards
keystrokes; it holds no state of its own.

Both panes matter and they are not symmetrical. The left pane is this
video's diarizer labels — local, citable, possibly nameless. The right pane
is the global roster of real people. Assignment writes a single
`video_speakers.speaker_id`, and the session then re-reads, so what the
panes show is always what the database says rather than what the last
keystroke intended.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.store import LabelRow, RosterRow

if TYPE_CHECKING:
    from rytp.diarize.link import Suggestion

#: The dash shown where a column has no value. Two spellings of "unknown"
#: in one table would read as two different states.
_EMPTY = "—"


def format_ms(ms: int) -> str:
    """Milliseconds as ``m:ss`` — the mapper's only unit of time."""
    seconds = max(0, ms) // 1000
    return f"{seconds // 60}:{seconds % 60:02d}"


def _rank(row: RosterRow, folded: str) -> int | None:
    """How well one roster entry matches a query: lower is better, None is no."""
    label = row.label.casefold()
    aliases = [alias.casefold() for alias in row.aliases]
    if label.startswith(folded):
        return 0
    if any(alias.startswith(folded) for alias in aliases):
        return 1
    if folded in label:
        return 2
    if any(folded in alias for alias in aliases):
        return 3
    return None


def match_roster(rows: Sequence[RosterRow], query: str) -> list[RosterRow]:
    """Rank the roster against a typed query, aliases included.

    Prefixes beat substrings and the label beats an alias, so typing the
    first letters of somebody's name puts them at the top even when an
    unrelated alias happens to contain those letters. Ties go to the
    alphabetically earlier label, which keeps the order stable between
    keystrokes — a list that reshuffles under the cursor is unusable.
    """
    folded = query.strip().casefold()
    if not folded:
        return list(rows)
    scored: list[tuple[int, str, RosterRow]] = []
    for row in rows:
        rank = _rank(row, folded)
        if rank is not None:
            scored.append((rank, row.label.casefold(), row))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [row for _rank_, _label, row in scored]


@dataclass
class MappingSession:
    """Everything the mapper does, as state plus verbs."""

    db: Database
    video_id: int

    labels: list[LabelRow] = field(default_factory=list, init=False)
    roster: list[RosterRow] = field(default_factory=list, init=False)
    matches: list[RosterRow] = field(default_factory=list, init=False)
    query: str = field(default="", init=False)
    status: str = field(default="", init=False)
    label_index: int = field(default=0, init=False)
    roster_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.reload()

    # -- state -----------------------------------------------------------

    def reload(self) -> None:
        """Re-read both panes from the database and clamp the cursors."""
        self.labels = store.label_rows(self.db, self.video_id)
        self.roster = store.roster_rows(self.db)
        self.matches = match_roster(self.roster, self.query)
        self.label_index = _clamp(self.label_index, len(self.labels))
        self.roster_index = _clamp(self.roster_index, len(self.matches))

    def set_query(self, text: str) -> None:
        """Filter the right pane and put its cursor back at the top."""
        self.query = text
        self.matches = match_roster(self.roster, text)
        self.roster_index = 0

    def select_label(self, index: int) -> None:
        self.label_index = _clamp(index, len(self.labels))

    def select_roster(self, index: int) -> None:
        self.roster_index = _clamp(index, len(self.matches))

    def move_label(self, delta: int) -> None:
        self.select_label(self.label_index + delta)

    def move_roster(self, delta: int) -> None:
        self.select_roster(self.roster_index + delta)

    @property
    def selected_label(self) -> LabelRow | None:
        return self.labels[self.label_index] if self.labels else None

    @property
    def selected_roster(self) -> RosterRow | None:
        return self.matches[self.roster_index] if self.matches else None

    def n_unmapped(self) -> int:
        """How many local labels still belong to nobody."""
        return sum(1 for row in self.labels if row.speaker_id is None)

    # -- verbs -----------------------------------------------------------

    def assign(self) -> str:
        """Link the highlighted label to the highlighted person."""
        label = self.selected_label
        if label is None:
            return self._say("no label selected")
        person = self.selected_roster
        if person is None:
            return self._say("no speaker selected")
        return self._link(label, person.speaker_id, person.label)

    def assign_to(self, ref: str) -> str:
        """Link the highlighted label to a person named by id, label or alias."""
        label = self.selected_label
        if label is None:
            return self._say("no label selected")
        person = store.resolve_speaker(self.db, ref)
        return self._link(label, person.id, person.label)

    def unassign(self) -> str:
        """Put the highlighted label back to "a voice nobody named"."""
        label = self.selected_label
        if label is None:
            return self._say("no label selected")
        with self.db.transaction():
            store.link_video_speaker(self.db, label.video_speaker_id, None)
        self.reload()
        return self._say(f"{label.local_label} is nobody again")

    def create_and_assign(self, label_text: str) -> str:
        """Add a roster entry (or find it) and link the highlighted label to it."""
        label = self.selected_label
        if label is None:
            return self._say("no label selected")
        with self.db.transaction():
            speaker_id, _created = store.add_speaker(self.db, label_text)
            store.link_video_speaker(self.db, label.video_speaker_id, speaker_id)
        self._after_assign(label)
        return self._say(f"{label.local_label} is {label_text.strip()}")

    # -- rendering -------------------------------------------------------

    def label_table(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        """The left pane: this video's voices, heaviest information first."""
        rows: list[tuple[str, ...]] = [
            (
                row.local_label,
                str(row.n_words),
                format_ms(row.speech_ms),
                row.speaker_label or _EMPTY,
                "yes" if row.has_embedding else "no",
            )
            for row in self.labels
        ]
        return ("label", "words", "speech", "person", "voice"), rows

    def roster_table(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        """The right pane: the roster, filtered by the current query."""
        rows: list[tuple[str, ...]] = [
            (row.label, ", ".join(row.aliases) or _EMPTY, str(row.n_videos))
            for row in self.matches
        ]
        return ("person", "aliases", "videos"), rows

    def suggestions(self) -> list[Suggestion]:
        """Roster candidates for the highlighted label. Advisory only."""
        label = self.selected_label
        if label is None:
            return []
        from rytp.diarize.link import suggest_for_label

        return suggest_for_label(self.db, label.video_speaker_id)

    def suggestion_table(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        """The suggestion pane: who it might be, how sure, and why only that sure."""
        rows: list[tuple[str, ...]] = [
            (
                item.speaker_label,
                f"{item.similarity:.2f}",
                item.verdict,
                item.era,
                "same era" if item.same_era else "other era",
                "similar sound" if item.acoustically_close else "unlike or unmeasured",
            )
            for item in self.suggestions()
        ]
        return ("person", "score", "verdict", "era", "when", "sound"), rows

    # -- internals -------------------------------------------------------

    def _link(self, label: LabelRow, speaker_id: int, speaker_label: str) -> str:
        with self.db.transaction():
            store.link_video_speaker(self.db, label.video_speaker_id, speaker_id)
        self._after_assign(label)
        return self._say(f"{label.local_label} is {speaker_label}")

    def _after_assign(self, label: LabelRow) -> None:
        """Re-read, then jump to the next label that still needs a person.

        Moving the cursor on is what makes the keyboard flow work: two to
        five labels per video, assigned one after another, without reaching
        for the arrow keys in between.
        """
        self.reload()
        for index, row in enumerate(self.labels):
            if row.speaker_id is None and row.video_speaker_id != label.video_speaker_id:
                self.label_index = index
                return

    def _say(self, message: str) -> str:
        self.status = message
        return message


def _clamp(index: int, length: int) -> int:
    if length <= 0:
        return 0
    return max(0, min(index, length - 1))
