"""Concrete :class:`PyannoteDiarizer` — pyannote-audio 3.x.

Gated on Hugging Face. Requires the ``rytp[pyannote]`` extra and an
``HF_TOKEN`` env var. See DESIGN §5.4.

The package is lazy-imported inside ``__init__`` so the rest of the
codebase can be imported in environments where pyannote isn't
installed.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rytp import engines
from rytp.engines import DiarSegment


class PyannoteDiarizer:
    """Diarizer backed by the pyannote-audio 3.x ``speaker-diarization-3.1`` pipeline.

    Args:
        model_id: Hugging Face model id (defaults to the recommended
            3.1 pipeline).
        auth_token: Hugging Face token. If ``None``, falls back to the
            ``HF_TOKEN`` / ``HUGGINGFACE_TOKEN`` env vars.

    Raises:
        ImportError: if ``pyannote.audio`` isn't installed.
        RuntimeError: if no auth token is available.
    """

    name = "pyannote"
    requires_hf_token = True
    help_url = "https://huggingface.co/pyannote/speaker-diarization-3.1"
    token_env_var = "HF_TOKEN"

    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-3.1",
        auth_token: str | None = None,
    ) -> None:
        try:
            from pyannote.audio import Pipeline  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "pyannote-audio is not installed. Install with "
                "`pip install rytp[pyannote]`."
            ) from e

        token = auth_token or os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGINGFACE_TOKEN"
        )
        if not token:
            raise RuntimeError(
                "HF_TOKEN is required for the pyannote diarizer. "
                "Visit https://huggingface.co/pyannote/speaker-diarization-3.1, "
                "click 'Agree and access', and set HF_TOKEN in your environment."
            )

        # Pipeline.from_pretrained takes use_auth_token= in pyannote-audio 3.x
        self._pipeline: Any = Pipeline.from_pretrained(
            model_id, use_auth_token=token
        )
        self._model_id = model_id

    def diarize(self, audio_path: Path) -> Iterable[DiarSegment]:
        """Yield :class:`DiarSegment` records from the pyannote pipeline.

        pyannote's ``Annotation`` object exposes ``itertracks(yield_label=True)``
        which yields ``(segment, track, label)`` triples. We convert each
        ``Segment`` to milliseconds and use ``label`` as the raw speaker
        name (e.g. ``"SPEAKER_00"``).
        """
        annotation = self._pipeline(str(audio_path))
        for segment, _track, label in annotation.itertracks(yield_label=True):
            yield DiarSegment(
                start_ms=int(segment.start * 1000),
                end_ms=int(segment.end * 1000),
                speaker=str(label),
            )


# Register so ``engines.resolve_diarizer("pyannote")`` works. The class
# itself is safe to import (it has no top-level pyannote dependency);
# instantiation is what fails when pyannote isn't installed.
engines.register_diarizer(PyannoteDiarizer)