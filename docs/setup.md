# Setup: getting an engine to actually use the GPU

`README.md` describes the pre-rewrite tree and is stale; `CLAUDE.md` is
written for an agent, not a person. This document is for a person installing
rytp on the machine it was designed for (Windows, 32 GiB RAM, an RTX 3080
Laptop 16 GiB, an i7-11800H) and wanting the engines to use that card.

It exists because following the obvious install path — `pip install -e
".[gigaam]"` and so on — produces engines that either cannot run at all, or
run on the CPU while the GPU sits idle. `rytp doctor` and `rytp transcribe
engines` now say so directly; this document is the fix for what they say.

## 1. The extras install a CPU-only torch on Windows

`pip install -e ".[wav2vec2]"` (and `.[redimnet]`, and anything else that
pulls in `torch`) resolves `torch>=2.1` from PyPI. On Windows, PyPI's `torch`
wheel is CPU-only — the CUDA build is not published there. So the documented
install path silently yields an engine that works, reports `ready`, and never
touches the card.

`rytp transcribe engines` and `rytp doctor` now probe the configured
interpreter and report this directly: an engine whose interpreter has torch
but no CUDA build shows `torch <version> has no CUDA` next to the ordinary
availability line, rather than staying silent about it.

The fix is an explicit index, matching the installed driver:

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Replace `cu124` with the build that matches the installed NVIDIA driver (see
`nvidia-smi`, or the PyTorch install matrix). Do this **inside the engine's
own interpreter** (§2 below) — installing a CUDA build into the interpreter
running `rytp` itself does nothing for an out-of-process engine, and doing it
into the wrong one is exactly the kind of silent mismatch this document
exists to head off.

## 2. The separate-interpreter workflow

`gigaam`, `mfa`, `wav2vec2`, `pyannote` and `redimnet` all run **out of
process**, each under its own Python interpreter, because their dependencies
pin versions that conflict with each other and (for MFA) because the
aligner is a conda package, not a `pip` one. Point rytp at each interpreter
with:

```
rytp settings set engine.interpreter.<name> <path to that interpreter>
```

for example `rytp settings set engine.interpreter.gigaam
/path/to/gigaam-venv/Scripts/python.exe`. A setting left unset falls back to
`engine.interpreter.default`, and that falls back to the interpreter running
`rytp` itself — which is correct only for an engine whose dependencies
happen not to conflict with rytp's own.

To build one of these environments:

1. Create a virtual environment with a Python version the engine's
   dependencies actually publish wheels for (§3 — on Python 3.14 this is not
   optional).
2. Install the engine's extra into *that* environment, e.g.
   `pip install rytp[gigaam]` (or, from a source checkout,
   `pip install -e ".[gigaam]"`), then follow §1 above for a CUDA build if
   the extra pulls in torch.
3. For MFA specifically: install the Montreal Forced Aligner itself via
   conda into that environment (it is not a `pip` package), then download
   its Russian models once:
   `mfa model download acoustic russian_mfa` and
   `mfa model download dictionary russian_mfa`. rytp finds the `mfa`
   binary beside the interpreter you point `engine.interpreter.mfa` at.
4. Point the setting at that environment's interpreter (above).
5. Run `rytp transcribe engines` (or `rytp speakers engines`) and confirm the
   engine now reports `ready` rather than a missing-module or
   missing-interpreter state.

## 3. On Python 3.14, a separate interpreter is not optional

`torch` (and everything that depends on it — `wav2vec2`, `redimnet`, and
`pyannote.audio` through its own dependencies) has no Python 3.14 wheels yet,
the same wall `sentencepiece` hits for `gigaam`. If the interpreter running
`rytp` itself is 3.14, **none** of `gigaam`, `wav2vec2`, `pyannote` or
`redimnet` can run in it at all, installed or not.

The separate-interpreter workflow in §2 is the only way to run any of them
today: create each engine's virtual environment with Python 3.12 (or another
version with published wheels for that engine's dependencies), not with
whatever interpreter happens to be running `rytp`.

MFA is unaffected by this — it is a conda environment, not a `pip` one — but
still needs its own interpreter per §2.
