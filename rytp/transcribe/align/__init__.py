"""Aligner adapters.

Importing this package is what makes an aligner's name resolvable: each module
registers its class at import time.
"""

from __future__ import annotations

from rytp.transcribe.align import mfa, wav2vec2  # noqa: F401
