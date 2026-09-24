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

    def diarize(self, audio: Path) -> Iterable[DiarSegment]:
        yield DiarSegment(
            start_ms=0,
            end_ms=wav_duration_ms(audio),
            local_label=C.NULL_DIARIZER_LABEL,
        )
