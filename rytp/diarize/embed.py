"""Speaker embeddings: the protocol, the registry, the codec and the maths.

A speaker embedding is a numeric fingerprint of a voice (design §2), used
to judge whether two recordings are the same person. Everything in this
module except the adapter at the bottom is plain arithmetic on lists of
floats — no numpy, no torch — for two reasons: the vectors are a couple of
hundred values and the roster is tens of people, so vectorising buys
nothing; and this module is imported inside a *foreign* interpreter by the
out-of-process seam, where only the standard library is guaranteed.

Storage is `video_speakers.embedding`, a little-endian float32 BLOB.
float32 is half the size of float64 and further below the noise floor of
any speaker model than a cosine comparison can notice.

Vectors are stored **already L2-normalised**, so a cosine similarity is a
dot product and a centroid is a mean. Nothing outside this module needs to
remember that.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rytp import constants as C
from rytp.models import RytpError, UnknownEngineError
from rytp.transcribe.registry import check_available, interpreter_for
from rytp.transcribe.subproc import load_cached, resolve_device, run_child

if TYPE_CHECKING:
    from rytp.db import Database

__all__ = [
    "EMBEDDERS",
    "EmbeddingError",
    "SpeakerEmbedder",
    "centroid",
    "comparable",
    "cosine_similarity",
    "embedder_rows",
    "l2_normalize",
    "load_embedder",
    "pack_embedding",
    "register_embedder",
    "resolve_embedder",
    "unpack_embedding",
]


class EmbeddingError(RytpError):
    """A vector is empty, malformed, or being compared with an incompatible one."""


class SpeakerEmbedder(Protocol):
    """Audio plus the windows to listen to, one vector out."""

    name: str
    requires_hf_token: bool
    out_of_process: bool
    dim: int

    def embed(
        self, audio: Path, windows: Sequence[tuple[int, int]]
    ) -> Sequence[float]: ...


EMBEDDERS: dict[str, type[SpeakerEmbedder]] = {}
"""Registered embedder classes, by ``cls.name``."""


def register_embedder(cls: type[SpeakerEmbedder]) -> type[SpeakerEmbedder]:
    """Class decorator: make ``cls`` resolvable as ``cls.name``."""
    EMBEDDERS[cls.name] = cls
    return cls


def resolve_embedder(name: str) -> type[SpeakerEmbedder]:
    """Look up an embedder class. Raises :class:`~rytp.models.UnknownEngineError`.

    Same fix as ``resolve_transcriber``/``resolve_aligner``/``resolve_diarizer``
    (BUGS.md entry 16): the message was already right, only a bare
    ``ValueError`` escaped the CLI's ``RytpError`` funnel as a traceback.
    """
    try:
        return EMBEDDERS[name]
    except KeyError:
        available = ", ".join(sorted(EMBEDDERS)) or "(none registered)"
        raise UnknownEngineError(
            f"unknown embedder {name!r}; available: {available}"
        ) from None


def load_embedder(db: Database, name: str, **kwargs: Any) -> SpeakerEmbedder:
    """Resolve, gate, and construct an embedder."""
    cls = resolve_embedder(name)
    interpreter = interpreter_for(db, name) if getattr(cls, "out_of_process", False) else None
    check_available(cls, interpreter=interpreter)  # type: ignore[arg-type]
    if interpreter is not None:
        kwargs.setdefault("interpreter", interpreter)
    return cls(**kwargs)


def embedder_rows(db: Database) -> list[tuple[str, str, str]]:
    """Rows for the engine listing: name, out-of-process, availability."""
    from rytp.transcribe.registry import availability

    return [
        (
            name,
            "yes" if getattr(cls, "out_of_process", False) else "no",
            availability(db, cls),  # type: ignore[arg-type]
        )
        for name, cls in sorted(EMBEDDERS.items())
    ]


# -- codec -----------------------------------------------------------------


def pack_embedding(values: Sequence[float]) -> bytes:
    """A vector as the little-endian float32 BLOB the column stores."""
    if not values:
        raise EmbeddingError("refusing to store an empty embedding")
    return struct.pack(f"<{len(values)}f", *(float(v) for v in values))


def unpack_embedding(blob: bytes) -> list[float]:
    """A stored BLOB back into a vector.

    A blob whose length is not a whole number of floats is corruption, not
    a shorter vector — say so rather than truncating silently.
    """
    if not blob:
        raise EmbeddingError("refusing to read an empty embedding")
    per_value = C.EMBEDDING_BYTES_PER_VALUE
    if len(blob) % per_value:
        raise EmbeddingError(
            f"embedding blob is {len(blob)} bytes, not a multiple of {per_value}"
        )
    return list(struct.unpack(f"<{len(blob) // per_value}f", blob))


# -- arithmetic ------------------------------------------------------------


def l2_normalize(values: Sequence[float]) -> list[float]:
    """Scale a vector to unit length."""
    norm = math.sqrt(sum(float(v) * float(v) for v in values))
    if norm == 0.0:
        raise EmbeddingError("cannot normalise a zero-length embedding")
    return [float(v) / norm for v in values]


def comparable(left: Sequence[float], right: Sequence[float]) -> bool:
    """Whether two vectors can be compared at all.

    Two embedders in one corpus produce two dimensionalities. Asking this
    first is how the suggestion pass skips such a pair instead of dying on
    it halfway through a listing.
    """
    return bool(left) and len(left) == len(right)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine of the angle between two vectors: 1 identical, -1 opposite."""
    if not comparable(left, right):
        raise EmbeddingError(
            f"cannot compare embeddings of length {len(left)} and {len(right)}"
        )
    unit_left = l2_normalize(left)
    unit_right = l2_normalize(right)
    return sum(a * b for a, b in zip(unit_left, unit_right, strict=True))


def centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    """The average direction of several vectors, as a unit vector.

    This is the enrolment centroid of design §6: several recordings of one
    person, reduced to one point to compare a new voice against.
    """
    if not vectors:
        raise EmbeddingError("cannot take the centroid of no embeddings")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise EmbeddingError("cannot take the centroid of embeddings of different lengths")
    units = [l2_normalize(vector) for vector in vectors]
    mean = [sum(unit[i] for unit in units) / len(units) for i in range(width)]
    return l2_normalize(mean)


def concat_wav_windows(
    src: Path, dst: Path, windows: Sequence[tuple[int, int]]
) -> Path:
    """Copy several time ranges of a PCM WAV into one file, in order.

    Standard library only: this also runs inside the embedder's own
    interpreter, which has torch but need not have anything of ours beyond
    :mod:`rytp.models`.
    """
    import wave

    if not windows:
        raise EmbeddingError("refusing to embed with no audio selected")
    with wave.open(str(src), "rb") as reader:
        rate = reader.getframerate()
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        total = reader.getnframes()
        chunks: list[bytes] = []
        for start_ms, end_ms in windows:
            first = min(max(0, start_ms * rate // 1000), total)
            last = min(max(first, end_ms * rate // 1000), total)
            reader.setpos(first)
            chunks.append(reader.readframes(last - first))
    with wave.open(str(dst), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(b"".join(chunks))
    return dst


@register_embedder
class ReDimNetEmbedder:
    """ReDimNet B6 under its own interpreter (design §6).

    Substantially better than pyannote's internal embedding model, and it
    needs torch — whose pins are the reason engines run out of process at
    all. :func:`child_main` below is the only place the library is
    imported, and it runs inside the interpreter named by the
    `engine.interpreter.redimnet` setting.

    **The library call is unverified.** Nobody here has run ReDimNet. Its
    *contract* — the dictionary `child_main` returns — is what the rest of
    this part depends on, and that is covered by tests. When you install
    the real thing, expect to rewrite the body of one function and nothing
    else.
    """

    name = "redimnet"
    requires_hf_token = False
    out_of_process = True
    required_module = "torch"
    extra = "redimnet"
    dim = 192
    #: Class-level default (plan §1b, contracts §6); see
    #: :class:`~rytp.transcribe.engines.gigaam.GigaAMTranscriber`.
    device = C.ENGINE_DEFAULT_DEVICE

    def __init__(
        self,
        interpreter: str,
        model: str = C.REDIMNET_DEFAULT_MODEL,
        device: str = C.ENGINE_DEFAULT_DEVICE,
    ) -> None:
        self._interpreter = interpreter
        self._model = model
        #: Requested device, then the concrete device the last call used
        #: (BUGS.md entry 34).
        self.device = device

    def embed(self, audio: Path, windows: Sequence[tuple[int, int]]) -> Sequence[float]:
        result = run_child(
            interpreter=self._interpreter,
            module="rytp.diarize.embed",
            request={
                "audio": str(audio),
                "model": self._model,
                "windows": [[int(start), int(end)] for start, end in windows],
                "device": self.device,
            },
        )
        self.device = str(result.get("device") or self.device)
        return [float(value) for value in result["vector"]]


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Runs inside ReDimNet's own interpreter. The only import of torch.

    The model is cached across calls in this process, keyed by ``(model,
    device)``. Part 7 already calls this once per speaker label rather than
    per chunk, but a corpus with many labels still paid a reload each time
    before the persistent worker seam existed.
    """
    import tempfile

    import redimnet
    import torch
    import torchaudio

    device = resolve_device(str(request.get("device") or C.ENGINE_DEFAULT_DEVICE))
    model_name = str(request["model"])
    windows = [(int(start), int(end)) for start, end in request["windows"]]
    model = load_cached(
        ("redimnet", model_name, device),
        lambda: redimnet.ReDimNet.from_pretrained(model_name).to(device).eval(),
    )
    with tempfile.TemporaryDirectory(prefix="rytp-embed-") as tmp:
        clip = concat_wav_windows(Path(request["audio"]), Path(tmp) / "clip.wav", windows)
        waveform, _rate = torchaudio.load(str(clip))
        waveform = waveform.to(device)
        with torch.no_grad():
            vector = model(waveform).squeeze().tolist()
    return {"vector": [float(value) for value in vector], "device": device}
