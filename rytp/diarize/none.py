"""The null diarizer: every word belongs to one voice (design §6).

Not a placeholder. Most videos in this corpus are one person talking, and
for those this is the *correct* answer, produced for free. It is also the
default, which is what keeps diarization opt-in.

The segment covers exactly the audio, read from the WAV header, rather than
some sentinel large number: a segment that claims to run past the end of the
file would make every "how much did this label speak" figure a lie.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rytp import constants as C
from rytp.diarize.base import register_diarizer, wav_duration_ms
from rytp.models import DiarSegment


@register_diarizer
class NullDiarizer:
    """One label for the whole file."""

    name = "none"
    requires_hf_token = False
    out_of_process = False
    required_module: str | None = None
    extra: str | None = None
    #: Contracts §6: a diarizer has no `words.align_score` to report, so
    #: this is a fixed, honest placeholder — see
    #: :class:`rytp.transcribe.engines.whisper.FasterWhisperTranscriber`.
    score_scale = C.ALIGN_SCALE_NONE
    #: No model, no GPU path — same as MFA (plan §1b).
    device = "n/a"

    def __init__(self, device: str = "n/a") -> None:
        # Accepted and ignored: plan §1b has every adapter take the
        # parameter uniformly (a caller may pass device= without special-
        # casing the null diarizer), but there is no model here to place on
        # a device, so the instance attribute stays "n/a" (MfaAligner's
        # pattern).
        del device
        #: The null diarizer never has a non-fatal finding to report. Set
        #: per instance (never a class-level default) so it satisfies the
        #: `Diarizer` protocol's `notes: list[str]` as a genuine, mutable
        #: instance attribute.
        self.notes: list[str] = []

    def diarize(self, audio: Path) -> Iterable[DiarSegment]:
        yield DiarSegment(
            start_ms=0,
            end_ms=wav_duration_ms(audio),
            local_label=C.NULL_DIARIZER_LABEL,
        )
