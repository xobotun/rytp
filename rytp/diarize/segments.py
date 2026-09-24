"""Time arithmetic over diarizer output (design §4, §6).

No database, no engine, no file. Everything here is a function of a list of
:class:`~rytp.models.DiarSegment` and, at most, a list of word timings —
which is what makes it the part of diarization that can be tested to death.

Three ideas carry the rest of this plan:

* **A word belongs to the label it overlaps most.** Not the label at its
  midpoint, which mis-assigns a long word at a turn boundary, and not the
  first match, which depends on segment order.
* **A word that overlaps no segment stays unlabelled.** A gap in the
  diarizer's output means "nobody was labelled here", which is information,
  not an error. It is exactly the case `video_speakers.speaker_id IS NULL`
  exists to describe one level up.
* **Overlap totals are computed on merged segments.** A diarizer that emits
  two overlapping turns for one voice must not make that voice look like it
  spoke for twice as long.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from rytp import constants as C
from rytp.models import DiarSegment

#: One word, as the labeller needs it: its row id and its timings. ``end_ms``
#: is optional because a caption-tier row has none (contracts §3).
WordSpan = tuple[int, int, "int | None"]

__all__ = [
    "WordSpan",
    "label_words",
    "local_labels",
    "merge_segments",
    "speech_ms_by_label",
    "windows_for_label",
]


def merge_segments(segments: Iterable[DiarSegment]) -> list[DiarSegment]:
    """Merge touching and overlapping segments **of the same label**.

    Returned sorted by start, then by label, so the result is a stable
    function of the input set rather than of its order.
    """
    by_label: dict[str, list[DiarSegment]] = {}
    for segment in segments:
        by_label.setdefault(segment.local_label, []).append(segment)

    merged: list[DiarSegment] = []
    for label, group in by_label.items():
        group.sort(key=lambda s: (s.start_ms, s.end_ms))
        start, end = group[0].start_ms, group[0].end_ms
        for segment in group[1:]:
            if segment.start_ms <= end:
                end = max(end, segment.end_ms)
                continue
            merged.append(DiarSegment(start_ms=start, end_ms=end, local_label=label))
            start, end = segment.start_ms, segment.end_ms
        merged.append(DiarSegment(start_ms=start, end_ms=end, local_label=label))

    merged.sort(key=lambda s: (s.start_ms, s.local_label))
    return merged


def local_labels(segments: Iterable[DiarSegment]) -> tuple[str, ...]:
    """Every distinct label the diarizer emitted, sorted."""
    return tuple(sorted({segment.local_label for segment in segments}))


def speech_ms_by_label(segments: Iterable[DiarSegment]) -> dict[str, int]:
    """How long each label actually spoke, overlaps counted once."""
    totals: dict[str, int] = {}
    for segment in merge_segments(segments):
        totals[segment.local_label] = (
            totals.get(segment.local_label, 0) + segment.end_ms - segment.start_ms
        )
    return totals


def label_words(
    words: Iterable[WordSpan], segments: Sequence[DiarSegment]
) -> dict[int, str]:
    """Map word id to local label by maximum overlap.

    A sweep, not a search: both sides are sorted once and a small active set
    is carried forward, so an hour of words against thousands of segments
    stays linear. Words unlabelled by every segment are simply absent from
    the result, which is what the caller writes as NULL.
    """
    ordered = sorted(segments, key=lambda s: (s.start_ms, s.end_ms, s.local_label))
    if not ordered:
        return {}

    assigned: dict[int, str] = {}
    active: list[DiarSegment] = []
    cursor = 0

    for word_id, raw_start, raw_end in sorted(words, key=lambda w: (w[1], w[0])):
        # A zero-length or end-less word behaves as one millisecond wide, so
        # "which segment contains its start" falls out of the same maths.
        start = raw_start
        end = max(raw_end if raw_end is not None else start, start + 1)

        while cursor < len(ordered) and ordered[cursor].start_ms <= end:
            active.append(ordered[cursor])
            cursor += 1
        active = [s for s in active if s.end_ms > start]

        best_label: str | None = None
        best_overlap = 0
        for segment in active:
            overlap = min(segment.end_ms, end) - max(segment.start_ms, start)
            # Strictly greater: a tie keeps the earlier segment, because
            # `active` is in start order.
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = segment.local_label
        if best_label is not None:
            assigned[word_id] = best_label

    return assigned


def windows_for_label(
    segments: Iterable[DiarSegment],
    label: str,
    *,
    min_segment_ms: int | None = None,
    max_total_ms: int | None = None,
) -> list[tuple[int, int]]:
    """Choose up to ``max_total_ms`` of this label's speech to embed.

    Longest segments first, because a long turn is a cleaner sample of a
    voice than a scatter of interjections, then returned in chronological
    order so the concatenated audio sounds like speech. Deterministic: ties
    on length are broken by start time.

    If every segment is shorter than ``min_segment_ms`` the filter is
    dropped rather than returning nothing — whether there is *enough* speech
    to embed at all is the caller's decision (`C.EMBED_MIN_SPEECH_MS`).
    """
    floor = C.EMBED_MIN_SEGMENT_MS if min_segment_ms is None else min_segment_ms
    budget = C.EMBED_MAX_SPEECH_MS if max_total_ms is None else max_total_ms

    mine = [s for s in merge_segments(segments) if s.local_label == label]
    if not mine:
        return []
    long_enough = [s for s in mine if s.end_ms - s.start_ms >= floor]
    candidates = long_enough or mine

    chosen: list[tuple[int, int]] = []
    remaining = budget
    for segment in sorted(candidates, key=lambda s: (-(s.end_ms - s.start_ms), s.start_ms)):
        if remaining <= 0:
            break
        take = min(segment.end_ms - segment.start_ms, remaining)
        chosen.append((segment.start_ms, segment.start_ms + take))
        remaining -= take

    chosen.sort()
    return chosen
