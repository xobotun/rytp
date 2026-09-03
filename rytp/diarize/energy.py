"""Simple energy-based speaker diarization.

This module provides :class:`EnergyDiarizer`, a basic speaker diarization
implementation that detects speaker changes based on audio energy patterns.
It doesn't require external models or HF_TOKEN.

The algorithm is intentionally simple and serves as a "better than nothing"
alternative to :class:`NullDiarizer` when you don't have pyannote installed:

1. Load the WAV file and compute short-time energy (RMS) in frames.
2. Detect silence regions (energy below a threshold).
3. Split the audio at silence boundaries into segments.
4. For each segment, compute a simple "voice signature" (energy level,
   zero-crossing rate, spectral centroid).
5. Cluster segments by voice signature similarity.
6. Assign speaker labels to clusters.

The "sensitivity" parameter controls how aggressively the diarizer splits
audio into different speakers:

* ``sensitivity=0.0`` — very conservative, only the most distinct speakers
  are separated (may merge similar voices).
* ``sensitivity=0.5`` — balanced (default).
* ``sensitivity=1.0`` — very aggressive, even similar voices are separated
  (may split a single speaker into multiple labels).

This is not a replacement for proper speaker diarization (pyannote, etc.),
but it can help in cases where you have clear speaker changes with pauses
between them.
"""
from __future__ import annotations

import wave
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from rytp import engines
from rytp.engines import DiarSegment


class EnergyDiarizer:
    """Simple energy-based speaker diarization.

    Args:
        sensitivity: Controls how aggressively the diarizer splits speakers.
            Range: 0.0 (conservative) to 1.0 (aggressive). Default 0.5.
        min_silence_ms: Minimum silence duration to consider as a speaker
            boundary. Default 500ms.
        frame_duration_ms: Analysis frame size in milliseconds. Default 30ms.
        min_segment_ms: Minimum segment duration in milliseconds.
            Segments shorter than this are merged with neighbors. Default 1000ms.
    """

    name = "energy"
    requires_hf_token = False

    def __init__(
        self,
        sensitivity: float = 0.5,
        min_silence_ms: int = 500,
        frame_duration_ms: int = 30,
        min_segment_ms: int = 1000,
    ) -> None:
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError(f"sensitivity must be in [0.0, 1.0], got {sensitivity}")
        self.sensitivity = sensitivity
        self.min_silence_ms = min_silence_ms
        self.frame_duration_ms = frame_duration_ms
        self.min_segment_ms = min_segment_ms

    def diarize(self, audio_path: Path) -> Iterable[DiarSegment]:
        """Detect speaker changes based on audio energy patterns.

        Yields :class:`DiarSegment` records with labels like
        ``SPEAKER_00``, ``SPEAKER_01``, etc.
        """
        # Load WAV
        try:
            samples, sample_rate = _read_wav(audio_path)
        except Exception:
            # If we can't read the file, fall back to a single segment
            from rytp import constants as C

            yield DiarSegment(
                start_ms=0,
                end_ms=C.NULL_DIARIZER_END_MS,
                speaker="SPEAKER_00",
            )
            return

        if samples.size == 0:
            from rytp import constants as C

            yield DiarSegment(
                start_ms=0,
                end_ms=C.NULL_DIARIZER_END_MS,
                speaker="SPEAKER_00",
            )
            return

        # Compute frame-level energy (RMS)
        frame_size = int(self.frame_duration_ms * sample_rate / 1000)
        if frame_size <= 0:
            frame_size = 1
        n_frames = samples.size // frame_size
        if n_frames == 0:
            from rytp import constants as C

            yield DiarSegment(
                start_ms=0,
                end_ms=C.NULL_DIARIZER_END_MS,
                speaker="SPEAKER_00",
            )
            return

        # Reshape into frames
        frames = samples[: n_frames * frame_size].reshape(n_frames, frame_size)
        energy = np.sqrt(np.mean(frames**2, axis=1) + 1e-10)

        # Detect silence: energy below threshold
        # Threshold is adaptive: median energy * factor
        # Higher sensitivity = lower threshold = more speech detected
        energy_median = np.median(energy)
        energy_threshold = energy_median * (1.5 - self.sensitivity)

        is_speech = energy > energy_threshold

        # Convert frame indices to time
        frame_duration_s = frame_size / sample_rate

        # Find continuous speech regions separated by silence
        min_silence_frames = int(self.min_silence_ms / 1000 / frame_duration_s)
        if min_silence_frames < 1:
            min_silence_frames = 1

        segments = _find_speech_segments(
            is_speech, min_silence_frames, frame_duration_s
        )

        if not segments:
            from rytp import constants as C

            yield DiarSegment(
                start_ms=0,
                end_ms=C.NULL_DIARIZER_END_MS,
                speaker="SPEAKER_00",
            )
            return

        # Merge segments that are too short
        min_segment_s = self.min_segment_ms / 1000
        segments = _merge_short_segments(segments, min_segment_s)

        # Compute features for each segment
        features = []
        for start_s, end_s in segments:
            start_sample = int(start_s * sample_rate)
            end_sample = int(end_s * sample_rate)
            seg_samples = samples[start_sample:end_sample]
            if seg_samples.size == 0:
                continue
            feat = _compute_voice_features(seg_samples, sample_rate)
            features.append((start_s, end_s, feat))

        if not features:
            from rytp import constants as C

            yield DiarSegment(
                start_ms=0,
                end_ms=C.NULL_DIARIZER_END_MS,
                speaker="SPEAKER_00",
            )
            return

        # Cluster segments by voice features
        labels = _cluster_segments(features, self.sensitivity)

        # Yield DiarSegment records
        for (start_s, end_s, _), label in zip(features, labels):
            yield DiarSegment(
                start_ms=int(start_s * 1000),
                end_ms=int(end_s * 1000),
                speaker=f"SPEAKER_{label:02d}",
            )


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file and return float32 samples in [-1, 1] and sample rate."""
    with wave.open(str(path), "rb") as w:
        sample_rate = w.getframerate()
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sampwidth}")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, sample_rate


def _find_speech_segments(
    is_speech: np.ndarray, min_silence_frames: int, frame_duration_s: float
) -> list[tuple[float, float]]:
    """Find continuous speech segments separated by silence."""
    segments = []
    in_speech = False
    start_frame = 0
    silence_count = 0

    for i, speech in enumerate(is_speech):
        if speech:
            if not in_speech:
                start_frame = i
                in_speech = True
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    # End of speech segment
                    end_frame = i - silence_count + 1
                    start_s = start_frame * frame_duration_s
                    end_s = end_frame * frame_duration_s
                    segments.append((start_s, end_s))
                    in_speech = False
                    silence_count = 0

    # Handle case where audio ends in speech
    if in_speech:
        end_frame = len(is_speech)
        start_s = start_frame * frame_duration_s
        end_s = end_frame * frame_duration_s
        segments.append((start_s, end_s))

    return segments


def _merge_short_segments(
    segments: list[tuple[float, float]], min_duration_s: float
) -> list[tuple[float, float]]:
    """Merge segments that are shorter than min_duration_s with neighbors."""
    if not segments:
        return segments

    merged = []
    i = 0
    while i < len(segments):
        start, end = segments[i]
        duration = end - start
        if duration < min_duration_s and merged:
            # Merge with previous segment
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, end)
        elif duration < min_duration_s and i + 1 < len(segments):
            # Merge with next segment
            next_start, next_end = segments[i + 1]
            segments[i + 1] = (start, next_end)
        else:
            merged.append((start, end))
        i += 1

    return merged


def _compute_voice_features(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute a simple voice signature for a segment.

    Features:
    - Mean energy (RMS)
    - Zero-crossing rate
    - Spectral centroid (dominant frequency)
    """
    if samples.size == 0:
        return np.zeros(3, dtype=np.float32)

    # Energy (RMS)
    energy = float(np.sqrt(np.mean(samples**2) + 1e-10))

    # Zero-crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(samples))) > 0)
    zcr = float(zero_crossings / samples.size)

    # Spectral centroid
    spec = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    total = spec.sum()
    if total > 0:
        centroid = float((freqs * spec).sum() / total)
    else:
        centroid = 0.0

    return np.array([energy, zcr, centroid], dtype=np.float32)


def _cluster_segments(
    features: list[tuple[float, float, np.ndarray]], sensitivity: float
) -> list[int]:
    """Cluster segments by voice feature similarity.

    Lower sensitivity = more conservative (fewer speakers).
    Higher sensitivity = more aggressive (more speakers).

    Uses a two-pass approach:
    1. Initial clustering with adaptive threshold based on sensitivity.
    2. Merge small clusters (< 10% of total segments) into the nearest
       large cluster to avoid over-segmentation.
    """
    if not features:
        return []

    # Normalize features
    feats = np.array([f for _, _, f in features])
    if feats.size == 0:
        return [0] * len(features)

    feat_mean = feats.mean(axis=0)
    feat_std = feats.std(axis=0) + 1e-10
    feats_norm = (feats - feat_mean) / feat_std

    # Adaptive distance threshold based on sensitivity
    # Lower sensitivity = higher threshold = more merging
    # Higher sensitivity = lower threshold = more splitting
    base_threshold = 1.2
    threshold = base_threshold * (1.5 - sensitivity * 0.8)

    # Greedy clustering
    labels = [-1] * len(features)
    cluster_centroids: list[np.ndarray] = []
    cluster_sizes: list[int] = []
    next_label = 0

    for i, feat in enumerate(feats_norm):
        best_cluster = -1
        best_dist = float("inf")

        for j, centroid in enumerate(cluster_centroids):
            dist = float(np.linalg.norm(feat - centroid))
            if dist < best_dist:
                best_dist = dist
                best_cluster = j

        if best_cluster >= 0 and best_dist < threshold:
            # Assign to existing cluster
            labels[i] = best_cluster
            cluster_sizes[best_cluster] += 1
            # Update centroid (running average)
            n = cluster_sizes[best_cluster]
            cluster_centroids[best_cluster] = (
                cluster_centroids[best_cluster] * (n - 1) + feat
            ) / n
        else:
            # Create new cluster
            labels[i] = next_label
            cluster_centroids.append(feat.copy())
            cluster_sizes.append(1)
            next_label += 1

    # Post-process: merge small clusters into nearest large cluster
    # This prevents over-segmentation when there are many short segments
    total_segments = len(labels)
    min_cluster_size = max(2, total_segments // 20)  # at least 5% of total

    for i in range(next_label):
        if cluster_sizes[i] >= min_cluster_size:
            continue

        # Find the nearest large cluster
        best_target = -1
        best_dist = float("inf")

        for j in range(next_label):
            if j == i or cluster_sizes[j] < min_cluster_size:
                continue
            dist = float(np.linalg.norm(cluster_centroids[i] - cluster_centroids[j]))
            if dist < best_dist:
                best_dist = dist
                best_target = j

        if best_target >= 0:
            # Merge cluster i into cluster best_target
            for k in range(len(labels)):
                if labels[k] == i:
                    labels[k] = best_target
            cluster_sizes[best_target] += cluster_sizes[i]

    # Renumber labels to be contiguous (0, 1, 2, ...)
    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = [label_map[l] for l in labels]

    return labels


# Register so ``engines.resolve_diarizer("energy")`` works
engines.register_diarizer(EnergyDiarizer)
