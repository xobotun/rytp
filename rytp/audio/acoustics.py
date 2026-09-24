"""One acoustic fingerprint per video (design §4, "Acoustics").

Five numbers, one pass over the cached WAV, one row in ``video_acoustics``.
Two consumers: the assembler asks which sources will blend, and cross-video
speaker matching asks which recordings are comparable enough to compare
voices in. Both want "does this sound like the same room and the same
microphone", which is a far cheaper question than the per-clip MFCC machinery
this replaces.

Everything here is numpy and the standard library. Loudness is the exception
and lives in the next section of this module, because only ffmpeg measures
EBU R128 properly.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rytp import constants as C
from rytp.audio.energy import frame_rms, ms_to_index, read_wav_mono, to_db
from rytp.audio.vad import SpeechSegment, detect_speech
from rytp.models import RytpError, utc_now_iso

if TYPE_CHECKING:
    from rytp.db import Database


@dataclass(frozen=True)
class Acoustics:
    """The fingerprint. Any field may be ``None`` when the audio cannot answer."""

    f0_mean: float | None
    f0_std: float | None
    spectral_tilt: float | None
    noise_floor_db: float
    reverb_proxy: float | None
    loudness_lufs: float | None


def noise_floor_db(samples: np.ndarray, sr: int) -> float:
    """The quiet-end percentile of frame energy, in dBFS."""
    frame_len = max(1, ms_to_index(C.ACOUSTICS_FRAME_MS, sr))
    hop = max(1, ms_to_index(C.ACOUSTICS_HOP_MS, sr))
    db = to_db(frame_rms(samples, frame_len, hop))
    if db.size == 0:
        return C.DB_FLOOR
    return float(np.percentile(db, C.VAD_NOISE_PERCENTILE))


def _frame_starts(
    samples: np.ndarray, sr: int, speech: Sequence[SpeechSegment] | None
) -> np.ndarray:
    """Sample indexes of the frames worth analysing: loud, and inside speech.

    Capped at :data:`rytp.constants.ACOUSTICS_MAX_FRAMES` by an even stride —
    a summary of an hour gains nothing from the other 175,000 frames, and the
    stride keeps the result deterministic.
    """
    frame_len = max(1, ms_to_index(C.ACOUSTICS_FRAME_MS, sr))
    hop = max(1, ms_to_index(C.ACOUSTICS_HOP_MS, sr))
    db = to_db(frame_rms(samples, frame_len, hop))
    if db.size == 0:
        return np.zeros(0, dtype=np.int64)
    floor = float(np.percentile(db, C.VAD_NOISE_PERCENTILE))
    loud = db >= floor + C.F0_VOICED_ABOVE_FLOOR_DB
    if not bool(np.any(loud)) and float(np.max(db)) > C.DB_FLOOR:
        # Uniform level throughout: the percentile floor has landed on the
        # signal itself, so nothing clears it by 10 dB. Everything audible is
        # as loud as this recording gets, so analyse all of it. Digital
        # silence still falls through to an empty result.
        loud = db >= floor
    if speech is not None:
        starts_ms = np.arange(db.size, dtype=np.float64) * (hop * 1000.0 / sr)
        inside: np.ndarray = np.zeros(db.size, dtype=bool)
        for segment in speech:
            inside |= (starts_ms >= segment.start_ms) & (starts_ms < segment.end_ms)
        loud &= inside
    indexes: np.ndarray = np.flatnonzero(loud).astype(np.int64) * hop
    if indexes.size > C.ACOUSTICS_MAX_FRAMES:
        stride = math.ceil(indexes.size / C.ACOUSTICS_MAX_FRAMES)
        indexes = indexes[::stride]
    return indexes


def estimate_f0(
    samples: np.ndarray, sr: int, *, speech: Sequence[SpeechSegment] | None = None
) -> tuple[float | None, float | None]:
    """Mean and spread of speaking pitch, in Hz, over voiced frames.

    Autocorrelation via FFT: the direct product would be quadratic in the
    frame length and this runs over thousands of frames.
    """
    frame_len = max(1, ms_to_index(C.ACOUSTICS_FRAME_MS, sr))
    starts = _frame_starts(samples, sr, speech)
    if starts.size == 0:
        return None, None
    lag_min = max(1, sr // C.F0_MAX_HZ)
    lag_max = min(frame_len - 1, sr // C.F0_MIN_HZ)
    if lag_max <= lag_min:
        return None, None
    size = 1 << (2 * frame_len - 1).bit_length()
    values: list[float] = []
    for start in starts:
        frame: np.ndarray = samples[start : start + frame_len].astype(np.float64)
        if frame.size < frame_len:
            continue
        frame = frame - frame.mean()
        energy = float(np.dot(frame, frame))
        if energy <= 0.0:
            continue
        spectrum = np.fft.rfft(frame, size)
        auto = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:frame_len]
        band = auto[lag_min : lag_max + 1]
        if band.size == 0:
            continue
        peak = int(np.argmax(band))
        if float(band[peak]) / energy < C.F0_VOICED_AUTOCORR:
            continue
        values.append(sr / float(lag_min + peak))
    if not values:
        return None, None
    array: np.ndarray = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def spectral_tilt(
    samples: np.ndarray, sr: int, *, speech: Sequence[SpeechSegment] | None = None
) -> float | None:
    """Slope of the long-term average spectrum, in dB per decade.

    Negative and steep means dull — a distant microphone, a lossy encode, a
    muffled room. Near zero means bright. It is the single number that most
    often explains why two sources refuse to blend.
    """
    frame_len = max(1, ms_to_index(C.ACOUSTICS_FRAME_MS, sr))
    starts = _frame_starts(samples, sr, speech)
    if starts.size == 0:
        return None
    window = np.hanning(frame_len)
    accumulated: np.ndarray = np.zeros(frame_len // 2 + 1, dtype=np.float64)
    used = 0
    for start in starts:
        frame = samples[start : start + frame_len]
        if frame.size < frame_len:
            continue
        accumulated += np.abs(np.fft.rfft(frame * window)) ** 2
        used += 1
    if used == 0:
        return None
    psd = accumulated / used
    freqs = np.fft.rfftfreq(frame_len, 1.0 / sr)
    band = (freqs >= C.SPECTRAL_TILT_LO_HZ) & (freqs <= min(C.SPECTRAL_TILT_HI_HZ, sr / 2))
    if int(np.count_nonzero(band)) < 4:
        return None
    x = np.log10(freqs[band])
    y = 10.0 * np.log10(np.maximum(psd[band], 1e-20))
    return float(np.polyfit(x, y, 1)[0])


def _window_rms(samples: np.ndarray, sr: int, lo_ms: float, hi_ms: float) -> float:
    lo = max(0, ms_to_index(lo_ms, sr))
    hi = min(samples.size, ms_to_index(hi_ms, sr))
    if hi <= lo:
        return 0.0
    window: np.ndarray = samples[lo:hi].astype(np.float64)
    return float(np.sqrt(np.mean(np.square(window))))


def reverb_proxy(
    samples: np.ndarray, sr: int, speech: Sequence[SpeechSegment]
) -> float | None:
    """Energy just after speech stops, relative to the speech, in dB.

    A dry room decays inside a few milliseconds and the value is far below
    zero; a live room keeps ringing and the value climbs toward it. Not a
    reverberation time — a proxy, which is all the assembler needs to tell two
    rooms apart. Pass unpadded segments: padding would put the tail being
    measured inside the segment.
    """
    values: list[float] = []
    for segment in speech:
        reference = _window_rms(samples, sr, segment.end_ms - C.REVERB_REF_MS, segment.end_ms)
        tail = _window_rms(samples, sr, segment.end_ms, segment.end_ms + C.REVERB_TAIL_MS)
        if reference <= 0.0:
            continue
        values.append(20.0 * math.log10(max(tail, C.DB_EPSILON) / reference))
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def compute_acoustics(
    samples: np.ndarray,
    sr: int,
    *,
    speech: Sequence[SpeechSegment] | None = None,
    loudness_lufs: float | None = None,
) -> Acoustics:
    """The whole fingerprint. ``loudness_lufs`` comes from ffmpeg, separately."""
    mean, spread = estimate_f0(samples, sr, speech=speech)
    return Acoustics(
        f0_mean=mean,
        f0_std=spread,
        spectral_tilt=spectral_tilt(samples, sr, speech=speech),
        noise_floor_db=noise_floor_db(samples, sr),
        reverb_proxy=reverb_proxy(samples, sr, speech or []),
        loudness_lufs=loudness_lufs,
    )


LoudnessRunner = Callable[[list[str]], str]


def parse_loudnorm(text: str) -> float | None:
    """Pull ``input_i`` out of ffmpeg's loudnorm JSON block on stderr."""
    start = text.rfind("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        return None
    try:
        value = float(payload.get("input_i"))
    except (TypeError, ValueError):
        return None
    return None if math.isinf(value) or math.isnan(value) else value


def _ffmpeg_stderr(command: list[str]) -> str:
    try:
        proc = subprocess.run(  # fixed argv, no shell
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=C.LOUDNESS_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RytpError("ffmpeg is not on PATH; loudness cannot be measured") from exc
    except subprocess.TimeoutExpired as exc:
        raise RytpError(
            f"ffmpeg loudness scan timed out after {C.LOUDNESS_TIMEOUT_S}s"
        ) from exc
    return proc.stderr


def measure_loudness_lufs(wav_path: Path, *, runner: LoudnessRunner | None = None) -> float | None:
    """Integrated loudness in LUFS, measured by ffmpeg.

    ffmpeg is a required binary and implements EBU R128 properly, including
    the gating that a hand-rolled K-weighting would get subtly wrong. The
    runner is injectable so tests never need the binary.
    """
    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-i",
        str(wav_path),
        "-af",
        "loudnorm=print_format=json",
        "-f",
        "null",
        "-",
    ]
    return parse_loudnorm((runner or _ffmpeg_stderr)(command))


def store_acoustics(db: Database, video_id: int, acoustics: Acoustics) -> None:
    """Upsert the video's fingerprint row (contracts §3, ``video_acoustics``)."""
    with db.transaction():
        db.conn.execute(
            "INSERT INTO video_acoustics (video_id, f0_mean, f0_std, spectral_tilt, "
            "noise_floor_db, reverb_proxy, loudness_lufs, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(video_id) DO UPDATE SET "
            "f0_mean = excluded.f0_mean, f0_std = excluded.f0_std, "
            "spectral_tilt = excluded.spectral_tilt, "
            "noise_floor_db = excluded.noise_floor_db, "
            "reverb_proxy = excluded.reverb_proxy, "
            "loudness_lufs = excluded.loudness_lufs, "
            "computed_at = excluded.computed_at",
            (
                video_id,
                acoustics.f0_mean,
                acoustics.f0_std,
                acoustics.spectral_tilt,
                acoustics.noise_floor_db,
                acoustics.reverb_proxy,
                acoustics.loudness_lufs,
                utc_now_iso(),
            ),
        )


def fingerprint_video(
    db: Database,
    video_id: int,
    *,
    wav_path: Path,
    runner: LoudnessRunner | None = None,
) -> Acoustics:
    """Measure one video and store its fingerprint. One pass, one row."""
    samples, sr = read_wav_mono(wav_path)
    # Unpadded segments: a padded end would hide the reverberation tail.
    speech = detect_speech(samples, sr, pad_ms=0)
    acoustics = compute_acoustics(
        samples,
        sr,
        speech=speech,
        loudness_lufs=measure_loudness_lufs(wav_path, runner=runner),
    )
    store_acoustics(db, video_id, acoustics)
    return acoustics
