"""Long-video chunking for STT.

Most videos in the corpus are well under an hour, but channels include
occasional 2–4 hour livestreams. Whisper-class STT engines degrade on
inputs much longer than their effective context window (around
:data:`rytp.constants.COHESION_LOW_MAX_GAP_MS` = 30 min for stable
quality), so the transcribe subcommand needs a chunking strategy — but
only for STT. The Diarizer runs on the full audio.

DESIGN §11: 25-minute chunks with 5-minute overlap (see
:data:`rytp.constants.CHUNK_LENGTH_MIN` and
:data:`rytp.constants.OVERLAP_MIN`). The overlap is wide enough that
any utterance near a chunk boundary is fully contained in two chunks,
enabling confidence-based de-duplication in a future v2.

Why these specific numbers? Whisper's reported context is ~30 minutes;
we set the threshold at 30 (DESIGN §11) so anything under 30 minutes is
passed to Whisper whole, and we start chunking only past that. The
25/5 split keeps each chunk well under the context window (25 min) and
makes the overlap exactly 20% of the chunk length — large enough to
be useful, small enough not to bloat storage.

Public surface:

* :class:`Chunk` — one planned chunk.
* :func:`probe_duration_ms` — audio length in milliseconds.
* :func:`chunk_audio` — plan chunks (and optionally slice the WAVs).
"""
from __future__ import annotations

import shutil
import struct  # noqa: F401  (re-exported for wave-format readers)
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from rytp import constants as C


@dataclass(frozen=True)
class Chunk:
    """One planned STT chunk.

    Attributes:
        ord: Zero-based chunk index in the plan.
        start_ms: Absolute start time into the original audio (ms).
        end_ms: Absolute end time into the original audio (ms,
            exclusive).
        audio_path: File the STT engine should consume. For an
            unchunked audio (below :data:`C.COHUNK_THRESHOLD_MIN`)
            it is the original audio file unchanged. For chunked
            audio it is the sliced WAV that ``chunk_audio`` wrote
            under ``out_dir``.
    """

    ord: int
    start_ms: int
    end_ms: int  # exclusive
    audio_path: Path


class ProbeError(RuntimeError):
    """Audio duration could not be determined by either ffprobe or the WAV header."""


def probe_duration_ms(audio_path: Path) -> int:
    """Return the audio length in milliseconds.

    Tries ``ffprobe`` first (most accurate, supports many formats),
    falls back to parsing a canonical WAV header. Raises
    :class:`ProbeError` if neither works — the caller (typically the
    transcribe subcommand) decides how to surface the error.

    The WAV fallback is limited to the PCM/``WAVE_FORMAT`` shape;
    for non-PCM formats (extensible, float) the byte-rate math may
    be off, but the typical use case (16 kHz mono int16) is always
    PCM.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            seconds = float(result.stdout.strip())
            return int(seconds * C.MS_PER_SECOND)
        except (subprocess.CalledProcessError, ValueError, OSError):
            pass  # fall through to the WAV header path

    if _looks_like_wav(audio_path):
        try:
            return _wav_duration_ms(audio_path)
        except (OSError, struct.error, ValueError):
            pass

    raise ProbeError(
        f"could not determine duration of {audio_path}: ffprobe unavailable "
        "and file is not a parseable WAV"
    )


def _looks_like_wav(p: Path) -> bool:
    """Cheap RIFF/WAVE magic check."""
    try:
        with p.open("rb") as f:
            return f.read(4) == b"RIFF" and f.read(4)  # any non-zero RIFF size
    except OSError:
        return False


def _wav_duration_ms(p: Path) -> int:
    """Parse a canonical PCM WAV header to get duration_ms.

    The use case (16 kHz mono int16) is always PCM; for non-PCM
    formats the byte-rate math may be off but the pipeline contract
    guarantees we never see those.
    """
    with wave.open(str(p), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        if rate <= 0:
            raise ValueError("wav has zero sample rate")
        return int(frames * C.MS_PER_SECOND / rate)


def chunk_audio(
    audio_path: Path,
    *,
    chunk_length_min: int = C.CHUNK_LENGTH_MIN,
    overlap_min: int = C.OVERLAP_MIN,
    chunk_threshold_min: int = C.CHUNK_THRESHOLD_MIN,
    out_dir: Path | None = None,
) -> list[Chunk]:
    """Plan chunks for the audio file.

    Below :data:`C.CHUNK_THRESHOLD_MIN` minutes (default 30), returns
    a single :class:`Chunk` covering the full audio; no on-disk
    slicing is done.

    Above the threshold, returns chunks of
    :data:`C.CHUNK_LENGTH_MIN` minutes with
    :data:`C.OVERLAP_MIN` minutes overlap. If ``out_dir`` is provided,
    each chunk is sliced into a separate WAV file named
    ``{stem}.chunk{ord:03d}.wav`` (``03`` = zero-padded width per
    :data:`C.CHUNK_FILENAME_ORD_WIDTH`); otherwise ``Chunk.audio_path``
    for chunked output points back at the original audio and the
    caller is responsible for slicing.

    Returns an empty list if the audio is zero-length.

    Args:
        audio_path: Path to the 16 kHz mono WAV (or any format
            ffprobe understands).
        chunk_length_min: Override for the chunk length.
        overlap_min: Override for the overlap.
        chunk_threshold_min: Override for the "no chunking below"
            threshold.
        out_dir: If given, sliced WAVs are written here. Otherwise
            only the plan is returned.

    Raises:
        ValueError: if ``chunk_length_min <= overlap_min`` (the step
            would be zero or negative).
        ProbeError: if the audio duration can't be determined.
    """
    duration_ms = probe_duration_ms(audio_path)
    if duration_ms <= 0:
        return []

    threshold_ms = chunk_threshold_min * C.MS_PER_MINUTE
    if duration_ms <= threshold_ms:
        # Single full-audio chunk — no slicing.
        return [Chunk(ord=0, start_ms=0, end_ms=duration_ms, audio_path=audio_path)]

    chunk_ms = chunk_length_min * C.MS_PER_MINUTE
    overlap_ms = overlap_min * C.MS_PER_MINUTE
    step_ms = chunk_ms - overlap_ms

    if step_ms <= 0:
        raise ValueError("chunk_length_min must exceed overlap_min")

    chunks: list[Chunk] = []
    ord_ = 0
    start = 0
    while start < duration_ms:
        end = min(start + chunk_ms, duration_ms)
        if out_dir is None:
            chunk_path = audio_path
        else:
            chunk_path = (
                out_dir
                / f"{audio_path.stem}.chunk{ord_:0{C.CHUNK_FILENAME_ORD_WIDTH}d}.wav"
            )
            _slice_wav(audio_path, chunk_path, start_ms=start, end_ms=end)
        chunks.append(
            Chunk(ord=ord_, start_ms=start, end_ms=end, audio_path=chunk_path)
        )
        if end >= duration_ms:
            break
        start += step_ms
        ord_ += 1

    return chunks


def _slice_wav(src: Path, dst: Path, *, start_ms: int, end_ms: int) -> None:
    """Slice a 16 kHz mono WAV into ``dst``.

    Uses ffmpeg if available (precise, supports any WAV); otherwise
    falls back to stdlib ``wave`` (PCM only). Both modes overwrite
    ``dst`` and create its parent directory.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        start_s = start_ms / C.SECONDS_TO_MS
        duration_s = (end_ms - start_ms) / C.SECONDS_TO_MS
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-ss",
                f"{start_s:.3f}",
                "-i",
                str(src),
                "-t",
                f"{duration_s:.3f}",
                str(dst),
            ],
            check=True,
            capture_output=True,
        )
        return

    # stdlib fallback: copy bytes from the data chunk. Assumes PCM.
    with wave.open(str(src), "rb") as w:
        rate = w.getframerate()
        start_frame = int(start_ms * rate / C.MS_PER_SECOND)
        end_frame = int(end_ms * rate / C.MS_PER_SECOND)
        n_frames = max(0, end_frame - start_frame)
        w.setpos(start_frame)
        frames = w.readframes(n_frames)
    with wave.open(str(dst), "wb") as out:
        out.setnchannels(w.getnchannels())
        out.setsampwidth(w.getsampwidth())
        out.setframerate(w.getnframes() and rate or 0)
        out.writeframes(frames)