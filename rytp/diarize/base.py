"""Diarizer interface — see DESIGN §5.2.

Public re-exports plus a built-in :class:`NullDiarizer` that assigns
``SPEAKER_00`` to everything (the default for v1, no HF token needed).

The :class:`NullDiarizer` here is the canonical definition; other
modules (``rytp.diarize.none``, ``rytp.engines``) import it from here.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rytp import constants as C
from rytp import engines
from rytp.engines import DiarSegment, Diarizer

__all__ = ["DiarSegment", "Diarizer", "NullDiarizer"]


class NullDiarizer:
    """Diarizer that assigns ``SPEAKER_00`` to everything.

    ``requires_hf_token = False``. Used when the user skips diarization
    entirely and just wants a working transcript pipeline.

    Yields a single segment covering ``[0, NULL_DIARIZER_END_MS)`` so
    any word timestamp the merger checks will fall inside it. The
    transcribe stage's merger handles this correctly because the
    segment uses ``[start, end]`` overlap semantics.
    """

    name = "none"
    requires_hf_token = False

    def diarize(self, audio_path: Path) -> Iterable[DiarSegment]:
        yield DiarSegment(
            start_ms=0,
            end_ms=C.NULL_DIARIZER_END_MS,
            speaker=C.NULL_DIARIZER_SPEAKER_LABEL,
        )


# Register so ``engines.resolve_diarizer("none")`` works. The
# registration is idempotent — engines.py also registers at module
# load — so this is safe even if both modules import each other.
engines.register_diarizer(NullDiarizer)