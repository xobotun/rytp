"""A stand-in engine module for the out-of-process seam test.

Imported by the child process by dotted name, exactly as a real adapter is.
Keeps to the standard library so it loads under any interpreter.
"""
from __future__ import annotations

from typing import Any

#: Incremented only when ``_load_stub_model`` itself runs — i.e. only on a
#: cache miss. Module-level and process-global rather than per instance:
#: the whole point is to observe how many times the *loader* ran in this
#: resident child, across however many `child_main` calls arrived.
_MODEL_LOAD_COUNT = 0


def _load_stub_model() -> int:
    """Pretends to be an expensive model load — counted, never repeated per key."""
    global _MODEL_LOAD_COUNT
    _MODEL_LOAD_COUNT += 1
    return _MODEL_LOAD_COUNT


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Echo the request, or fail on demand, or print noise to stdout first."""
    if request.get("noise"):
        print("loading model 100%|##########|")  # the noise is the point
    # Task 8b (BUGS.md entry 10): a download or a long call writes its own
    # progress to the child's stderr, which `run_child` now streams line by
    # line rather than buffering whole. Additive — every other flag here is
    # untouched.
    stderr_lines = request.get("stderr_lines")
    if stderr_lines:
        import sys

        for line in stderr_lines:
            print(line, file=sys.stderr, flush=True)
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
    if request.get("align"):
        # Proves the persistent-worker fix at the level the real aligners
        # care about: the "model" is loaded at most once per resident
        # child, no matter how many chunks call `align`.
        from rytp.transcribe.subproc import load_cached

        model_loads = load_cached(("stub-align-model",), _load_stub_model)
        words = [str(w) for w in request.get("words") or []]
        start_ms = int(request.get("start_ms", 0))
        end_ms = int(request.get("end_ms", start_ms))
        span_ms = max(1, (end_ms - start_ms) // max(1, len(words)))
        spans = []
        for index in range(len(words)):
            span_start = start_ms + index * span_ms
            span_end = span_start + span_ms
            spans.append({"start_ms": span_start, "end_ms": span_end, "score": None})
        return {"spans": spans, "model_loads": model_loads}
    return {"words": [{"start_ms": 0, "end_ms": 100, "text": "ok", "confidence": 1.0}]}
