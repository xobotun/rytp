"""What a cut list costs (design §8 "Scoring").

One knob, ``consistency``, runs from 0.0 "fewest seams" to 1.0 "most
consistent sound". It does not parameterise a formula; it picks a point on
a straight line between two named weight profiles in
:mod:`rytp.constants`, so retuning later — design §11, M4 — is editing two
triples rather than re-deriving anything.

Three costs, all in the same units, all minimised together:

``seam``
    Paid once per fragment. This is the "fewer and longer fragments" force.
``switch``
    Paid whenever the cut list changes source video. This is design §8's
    "preferring few distinct source videos is a strong and cheap proxy",
    priced as an audible event rather than counted as a set size.
``acoustic``
    Multiplies how differently the two videos actually sound, measured
    from ``video_acoustics``. Zero at the "fewest seams" end: a short mix
    tolerates many sources.

This module is a leaf. It must not import :mod:`rytp.assemble.match` —
what scoring needs from a run is the read-only :class:`RunLike` protocol
below, which ``CandidateRun`` satisfies structurally.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from rytp import constants as C
from rytp.db import Database
from rytp.models import InvalidInputError

__all__ = [
    "AcousticsMap",
    "RunLike",
    "Weights",
    "acoustic_distance",
    "fragment_cost",
    "jitter",
    "load_acoustics",
    "rounded",
    "transition_cost",
    "weights_for",
]

#: video_id -> the acoustic fields that video actually has measured.
AcousticsMap = dict[int, dict[str, float]]


class RunLike(Protocol):
    """The part of a candidate run that scoring looks at."""

    @property
    def n_words(self) -> int: ...

    @property
    def first_align(self) -> float: ...

    @property
    def last_align(self) -> float: ...

    @property
    def mean_align(self) -> float: ...


@dataclass(frozen=True)
class Weights:
    """One point on the knob."""

    seam: float
    switch: float
    acoustic: float


def _lerp(low: float, high: float, position: float) -> float:
    return low + (high - low) * position


def weights_for(consistency: float) -> Weights:
    """The weight profile at ``consistency`` on the 0..1 dial."""
    if not 0.0 <= consistency <= 1.0:
        raise InvalidInputError(
            f"consistency must be between 0.0 (fewest seams) and 1.0 "
            f"(most consistent sound); got {consistency}"
        )
    low = Weights(*C.ASSEMBLE_WEIGHTS_FEWEST_SEAMS)
    high = Weights(*C.ASSEMBLE_WEIGHTS_MOST_CONSISTENT)
    return Weights(
        seam=_lerp(low.seam, high.seam, consistency),
        switch=_lerp(low.switch, high.switch, consistency),
        acoustic=_lerp(low.acoustic, high.acoustic, consistency),
    )


def load_acoustics(db: Database, video_ids: Iterable[int]) -> AcousticsMap:
    """Read ``video_acoustics`` for the videos in play.

    A video with no row, or a row whose column is NULL, simply does not
    appear — an unmeasured source is unknown, not bad, and
    :func:`acoustic_distance` treats it that way.
    """
    wanted = sorted({int(video_id) for video_id in video_ids})
    if not wanted:
        return {}
    inline = ", ".join(str(video_id) for video_id in wanted)
    columns = ", ".join(C.ASSEMBLE_ACOUSTIC_SCALES)
    loaded: AcousticsMap = {}
    for row in db.conn.execute(
        f"SELECT video_id, {columns} FROM video_acoustics WHERE video_id IN ({inline})"
    ):
        profile = {
            name: float(row[name])
            for name in C.ASSEMBLE_ACOUSTIC_SCALES
            if row[name] is not None
        }
        if profile:
            loaded[int(row["video_id"])] = profile
    return loaded


def acoustic_distance(
    first: Mapping[str, float] | None, second: Mapping[str, float] | None
) -> float:
    """How differently two videos sound, on a 0..1 scale.

    Each field is scaled by the difference that counts as one unit
    (``ASSEMBLE_ACOUSTIC_SCALES``) and clamped *before* averaging, so one
    wild field cannot swamp the rest — a bad pitch estimate should not
    make two otherwise identical recordings look incompatible. Only
    fields both videos have are compared.
    """
    if not first or not second:
        return C.ASSEMBLE_UNKNOWN_ACOUSTIC_DISTANCE
    shared = [name for name in C.ASSEMBLE_ACOUSTIC_SCALES if name in first and name in second]
    if not shared:
        return C.ASSEMBLE_UNKNOWN_ACOUSTIC_DISTANCE
    total = 0.0
    for name in shared:
        raw = abs(first[name] - second[name]) / C.ASSEMBLE_ACOUSTIC_SCALES[name]
        total += min(raw, C.ASSEMBLE_ACOUSTIC_DISTANCE_CAP)
    return total / len(shared)


def fragment_cost(run: RunLike, weights: Weights) -> float:
    """What taking this run as one fragment costs.

    One seam, plus what its alignment is worth. The two align terms are
    separate on purpose: the first and last words are the boundaries that
    actually get cut, while the interior only tells us how much to trust
    that the fragment says what the index claims (design §3, "Precision
    beats recall for assembly").
    """
    edges = (2.0 - run.first_align - run.last_align) / 2.0
    interior = 1.0 - run.mean_align
    return (
        weights.seam
        + C.ASSEMBLE_EDGE_ALIGN_WEIGHT * edges
        + C.ASSEMBLE_MEAN_ALIGN_WEIGHT * interior
    )


def transition_cost(
    previous_video: int | None,
    video: int,
    weights: Weights,
    acoustics: AcousticsMap,
) -> float:
    """What following ``previous_video`` with ``video`` costs.

    Zero for the first fragment and for staying put. This is the term
    that makes "few distinct sources" an objective rather than a wish.
    """
    if previous_video is None or previous_video == video:
        return 0.0
    distance = acoustic_distance(acoustics.get(previous_video), acoustics.get(video))
    return weights.switch + weights.acoustic * distance


def jitter(seed: int, video_id: int, first_word_ord: int, position: int) -> float:
    """A tiny, reproducible cost nudge keyed by the seed and the candidate.

    Design §8: same input, same output; "a user-supplied seed shakes up
    choices among near-equal candidates". Seed 0 disables it entirely.
    ``blake2b`` rather than ``hash()`` because Python salts string hashing
    per process, which would make the same seed give different answers in
    different runs — the exact thing this is meant to prevent.
    """
    if seed == 0:
        return 0.0
    payload = f"{seed}:{video_id}:{first_word_ord}:{position}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    fraction = int.from_bytes(digest, "big") / float(1 << 64)
    return C.ASSEMBLE_SEED_JITTER * fraction


def rounded(cost: float) -> float:
    """Costs, rounded so near-equal candidates genuinely tie.

    Without this, float noise silently decides between two candidates the
    scoring considers equivalent, and the explicit tie-break — or the
    seed — never gets a say.
    """
    return round(cost, C.ASSEMBLE_COST_DECIMALS)
