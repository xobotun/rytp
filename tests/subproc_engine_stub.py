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
    return {"words": [{"start_ms": 0, "end_ms": 100, "text": "ok", "confidence": 1.0}]}
