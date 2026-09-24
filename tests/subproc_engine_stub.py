"""A stand-in engine module for the out-of-process seam test.

Imported by the child process by dotted name, exactly as a real adapter is.
Keeps to the standard library so it loads under any interpreter.
"""
from __future__ import annotations

from typing import Any


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Echo the request, or fail on demand, or print noise to stdout first."""
    if request.get("noise"):
        print("loading model 100%|##########|")  # the noise is the point
    if request.get("boom"):
        raise RuntimeError("engine exploded")
    if request.get("missing_dependency"):
        raise ModuleNotFoundError(
            "No module named 'definitely_not_a_real_engine_dep'",
            name="definitely_not_a_real_engine_dep",
        )
    if request.get("missing_rytp_module"):
        # Shaped like a broken import *inside our own package* — never a
        # missing extra, so this must never get the install-hint treatment.
        raise ModuleNotFoundError(
            "No module named 'rytp.transcribe.engines.nonexistent'",
            name="rytp.transcribe.engines.nonexistent",
        )
    return {"words": [{"start_ms": 0, "end_ms": 100, "text": "ok", "confidence": 1.0}]}
