"""Transcriber adapters.

Importing this package is what makes an engine's name resolvable: each module
registers its class at import time, so a name that is never imported is a name
that does not exist.
"""

from __future__ import annotations

from rytp.transcribe.engines import gigaam, whisper  # noqa: F401
