"""Per-clip spectral features.

DESIGN §4: ``clip_features`` stores MFCC blobs and a spectral centroid
per clip (not per word). This module is the lazy-compute side: given
an audio WAV and a ``[start_ms, end_ms]`` window, compute the features
on demand and cache them in the DB.

The mine stage uses these features for window-to-window similarity
scoring. v1 keeps the dimensionality small
(:data:`rytp.constants.MFCC_COEFFS` coefficients + scalar centroid)
so the BLOB stays light.

Implementation note: this MFCC is a hand-rolled approximation using
a mel filterbank + DCT. It produces stable features on 16 kHz mono
audio without requiring scipy or librosa. For the mine stage's
purposes (similarity scoring between clip windows) the absolute scale
of MFCC values doesn't matter, only the relative shape does — so the
simplified pipeline is acceptable. If you need a textbook MFCC for
phonetic analysis, swap in ``librosa.feature.mfcc`` and rescale.

The :func:`clip_similarity` function is a cosine similarity over the
**mean** MFCC vector across frames; this collapses the time axis
down to a single feature vector per clip. v2 may add DTW or
sequence-level comparison if the mean is too coarse.

Public surface:

* :func:`compute_clip_features` — compute MFCC + spectral centroid.
* :func:`get_or_compute_clip_features` — DB-cached wrapper.
* :func:`clip_similarity` — cosine similarity over mean MFCC.
* :class:`ClipFeatures` — value type.
"""
from __future__ import annotations

import datetime as _dt
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rytp import constants as C
from rytp.db import Database


@dataclass(frozen=True)
class ClipFeatures:
    """Spectral features for one audio clip window.

    Attributes:
        mfcc: ``(n_mfcc, n_frames)`` float32 array.
        spectral_centroid_hz: Scalar centroid (Hz).
    """

    mfcc: np.ndarray
    spectral_centroid_hz: float


def compute_clip_features(audio_path: Path, start_ms: int, end_ms: int) -> ClipFeatures:
    """Compute MFCC + spectral centroid for one clip window.

    Reads PCM bytes from a 16 kHz mono int16 WAV (the only format
    the pipeline produces — see :mod:`rytp.transcribe.extract`).
    Returns zeros if the window is empty so callers don't have to
    special-case it.

    Args:
        audio_path: Path to the 16 kHz mono int16 WAV.
        start_ms: Window start in ms (inclusive).
        end_ms: Window end in ms (exclusive).

    Returns:
        A :class:`ClipFeatures` with the computed values.
    """
    samples = _read_wav_window(audio_path, start_ms, end_ms)
    if samples.size == 0:
        # Empty window — return zeros rather than error.
        return ClipFeatures(
            mfcc=np.zeros((C.MFCC_COEFFS, 1), dtype=np.float32),
            spectral_centroid_hz=0.0,
        )

    # MFCC: small hand-rolled approximation using a mel filterbank +
    # DCT. See module docstring for the tradeoff vs librosa.
    mfcc = _mfcc(samples, sample_rate=C.AUDIO_SAMPLE_RATE_HZ, n_mfcc=C.MFCC_COEFFS)
    centroid = _spectral_centroid(samples, sample_rate=C.AUDIO_SAMPLE_RATE_HZ)
    return ClipFeatures(mfcc=mfcc, spectral_centroid_hz=float(centroid))


def _read_wav_window(path: Path, start_ms: int, end_ms: int) -> np.ndarray:
    """Read PCM frames from a 16 kHz mono int16 WAV into float32 ``[-1, 1]``.

    Multi-channel WAVs are mixed down to mono here. Non-16-bit WAVs
    raise — the pipeline's contract is 16-bit (see
    :mod:`rytp.transcribe.extract`).
    """
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        if sampwidth != 2:
            raise ValueError(
                f"expected 16-bit WAV (sampwidth=2), got sampwidth={sampwidth}"
            )
        start_frame = int(start_ms * rate / C.MS_PER_SECOND)
        end_frame = int(end_ms * rate / C.MS_PER_SECOND)
        n_frames = max(0, end_frame - start_frame)
        if n_frames == 0:
            return np.zeros(0, dtype=np.float32)
        w.setpos(start_frame)
        raw = w.readframes(n_frames)
    # Convert int16 PCM to float32 in [-1, 1] using the canonical
    # max-value :data:`C.INT16_MAX_FLOAT`.
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / C.INT16_MAX_FLOAT
    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return arr


def _mfcc(samples: np.ndarray, *, sample_rate: int, n_mfcc: int) -> np.ndarray:
    """Compute MFCC coefficients using a small mel filterbank + DCT-II.

    Pipeline:

    1. Frame the signal into ``FRAME_DURATION_S`` windows with
       ``HOP_DURATION_S`` overlap, applying a Hann taper to each.
    2. Power spectrum per frame via a real FFT of size ``FFT_SIZE``.
    3. Apply a triangular mel filterbank of ``MEL_FILTER_COUNT``
       filters spanning ``0`` to Nyquist.
    4. Log-magnitude, then DCT-II (orthonormal) keeping the first
       ``n_mfcc`` coefficients.

    This is intentionally minimal — see module docstring for why
    we don't use librosa. The Hann taper (rather than Hamming) gives
    better spectral leakage suppression at the cost of slightly wider
    main lobes; for similarity scoring that's the right tradeoff.
    """
    frame_len = int(C.FRAME_DURATION_S * sample_rate)
    hop_len = int(C.HOP_DURATION_S * sample_rate)
    if samples.size < frame_len:
        # Pad short inputs so we get at least one frame.
        samples = np.pad(samples, (0, frame_len - samples.size))
    n_frames = 1 + (samples.size - frame_len) // hop_len
    if n_frames <= 0:
        return np.zeros((n_mfcc, 1), dtype=np.float32)
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, frame_len),
        strides=(samples.strides[0] * hop_len, samples.strides[0]),
    ).copy()
    window = np.hanning(frame_len).astype(np.float32)
    frames *= window

    # Power spectrum per frame via real FFT.
    spec = np.abs(np.fft.rfft(frames, n=C.FFT_SIZE, axis=1)) ** 2

    # Mel filterbank (triangular filters).
    mel_max = _hz_to_mel(sample_rate / 2)
    mel_pts = np.linspace(0, mel_max, C.MEL_FILTER_COUNT + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = np.floor((C.FFT_SIZE + 1) / sample_rate * hz_pts).astype(int)
    fb = np.zeros((C.MEL_FILTER_COUNT, spec.shape[1]), dtype=np.float32)
    for m in range(1, C.MEL_FILTER_COUNT + 1):
        lo, mid, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if mid > lo:
            fb[m - 1, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[m - 1, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)

    mel_spec = spec @ fb.T
    # Floor the log to avoid log(0); use np.finfo(float).eps, the
    # smallest representable positive double.
    mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)
    log_mel = np.log(mel_spec)

    # DCT-II (orthonormal). Keep the first n_mfcc coefficients.
    n = log_mel.shape[1]
    k = np.arange(n_mfcc)[:, None]
    i = np.arange(n)[None, :]
    dct = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
    mfcc = dct @ log_mel.T  # shape: (n_mfcc, n_frames)
    return mfcc.astype(np.float32)


def _spectral_centroid(samples: np.ndarray, *, sample_rate: int) -> float:
    """Compute scalar spectral centroid (Hz) over the whole window.

    The centroid is the amplitude-weighted mean frequency; higher
    values mean "brighter" audio. Uses the linear FFT magnitude
    (not power), which gives a perceptually-slightly-dark result —
    we use linear because :data:`C.FFT_SIZE` is small and the linear
    form emphasizes the dominant peak.
    """
    if samples.size == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(samples, n=C.FFT_SIZE))
    freqs = np.fft.rfftfreq(C.FFT_SIZE, d=1.0 / sample_rate)
    mag = spec.astype(np.float64)
    total = mag.sum()
    if total <= 0:
        return 0.0
    return float((freqs * mag).sum() / total)


def _hz_to_mel(hz: float) -> float:
    """Slaney's mel scale: ``2595 * log10(1 + hz/700)``.

    Slaney's formula is the standard mel approximation that matches
    human pitch perception up to ~500 Hz and is what librosa uses
    by default.
    """
    return C.MEL_SLOPE * np.log10(1.0 + hz / C.MEL_INTERCEPT_HZ)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_hz_to_mel`."""
    return C.MEL_INTERCEPT_HZ * (10 ** (mel / C.MEL_SLOPE) - 1.0)


def get_or_compute_clip_features(
    db: Database,
    video_id: int,
    audio_path: Path,
    start_ms: int,
    end_ms: int,
) -> ClipFeatures:
    """Return cached clip features, computing + caching if absent.

    Lookup key is ``(video_id, start_ms, end_ms)``. The MFCC is
    stored as a packed float32 BLOB (little-endian native byte order
    — round-trip via ``np.frombuffer(dtype=np.float32)``); the
    centroid as a REAL column.

    Args:
        db: Open Database.
        video_id: FK to the ``videos`` table.
        audio_path: Path to the per-video 16 kHz mono WAV.
        start_ms: Window start (ms).
        end_ms: Window end (ms).

    Returns:
        The :class:`ClipFeatures` for this window.
    """
    row = db.conn.execute(
        """
        SELECT mfcc_blob, spectral_centroid_hz FROM clip_features
        WHERE video_id = ? AND start_ms = ? AND end_ms = ?
        """,
        (video_id, start_ms, end_ms),
    ).fetchone()
    if row is not None:
        mfcc = np.frombuffer(row["mfcc_blob"], dtype=np.float32).reshape(
            C.MFCC_COEFFS, -1
        )
        return ClipFeatures(mfcc=mfcc, spectral_centroid_hz=row["spectral_centroid_hz"])

    feats = compute_clip_features(audio_path, start_ms, end_ms)
    db.conn.execute(
        """
        INSERT INTO clip_features (video_id, start_ms, end_ms, mfcc_blob, spectral_centroid_hz, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            video_id,
            start_ms,
            end_ms,
            feats.mfcc.astype(np.float32).tobytes(),
            feats.spectral_centroid_hz,
            _dt.datetime.utcnow().isoformat(),
        ),
    )
    db.conn.commit()
    return feats


def clip_similarity(a: ClipFeatures, b: ClipFeatures) -> float:
    """Cosine similarity between two clips' MFCC mean vectors.

    Both clips' MFCC arrays are collapsed to a mean vector across the
    time axis, then the cosine similarity is computed. Higher values
    mean more spectrally similar.

    Returns 0.0 if either clip is zero (so callers don't have to
    special-case missing features).

    Args:
        a, b: Two :class:`ClipFeatures` records.

    Returns:
        Cosine similarity in ``[-1, 1]``.
    """
    va = a.mfcc.mean(axis=1)
    vb = b.mfcc.mean(axis=1)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(va @ vb / (na * nb))