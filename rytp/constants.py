"""Project-wide constants.

Python has no ``const`` keyword, so the convention is uppercase
module-level names imported where needed. Everything that was a
magical number in the codebase lives here, grouped by subsystem,
with a one-paragraph rationale citing the relevant section of
``DESIGN.md``.

Override defaults in tests via ``monkeypatch.setattr`` against the
public names below (not against ``rytp.constants._SECTION`` private
sentinels — those are intentionally underscored to discourage
test-side mutation).
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Audio format (DESIGN §3 / §7)
# ---------------------------------------------------------------------------
#: Output sample rate for ``data/audio/{video_id}.wav``. Whisper-class
#: STT engines are trained on 16 kHz mono, so the extract stage always
#: resamples to this rate. See DESIGN §3: "single source of audio truth"
#: — every later stage (STT, loudnorm, splice concat) reads from this
#: file, so changing the rate is a project-wide decision.
AUDIO_SAMPLE_RATE_HZ: int = 16_000

#: Number of audio channels in the canonical WAV. Whisper is mono; mixing
#: to mono at extract time keeps the storage and the downstream tooling
#: simple.
AUDIO_CHANNELS: int = 1

#: Maximum value of an int16 sample, used to normalize raw PCM bytes to
#: the float range ``[-1.0, 1.0]``. ``2 ** 15`` because signed 16-bit.
INT16_MAX_FLOAT: float = 32_768.0


# ---------------------------------------------------------------------------
# Time conversions
# ---------------------------------------------------------------------------
#: Milliseconds per second. Used everywhere we convert the DB's ms
#: timestamps to seconds for ffmpeg's ``-ss`` / ``-t`` flags.
MS_PER_SECOND: int = 1000

#: Minutes per millisecond. ``60 * 1000`` is unambiguous but spelled
#: out once here so the chunker reads cleanly.
MS_PER_MINUTE: int = 60 * MS_PER_SECOND

#: Minimum clip duration in seconds. Used by loudnorm/splice to avoid
#: asking ffmpeg for a zero- or negative-length extract that would
#: error out. ``0.001 s`` = 1 ms is the smallest duration ffmpeg's
#: ``-t`` flag accepts reliably across versions.
MIN_CLIP_DURATION_S: float = 0.001


# ---------------------------------------------------------------------------
# FTS5 search (DESIGN §7)
# ---------------------------------------------------------------------------
#: Default row cap on ``search_words`` and ``Database.search_fts``.
#: 200 is generous for a single query; if you need more, paginate.
DEFAULT_SEARCH_LIMIT: int = 200

#: Maximum number of clips returned by :func:`rytp.mine.mine` per
#: invocation. Mirrors the ``--max-clips`` CLI flag default.
DEFAULT_MAX_CLIPS: int = 50


# ---------------------------------------------------------------------------
# Mine stage (DESIGN §7)
# ---------------------------------------------------------------------------
#: Cohesion = LOW. Window is just the matched word; no expansion.
#: Single-word per FTS hit.
COHESION_LOW_WINDOW_MS: int = 0
#: Cohesion = LOW also disables gap-budget expansion; this is the
#: sentinel meaning "don't extend".
COHESION_LOW_MAX_GAP_MS: int = 0

#: Cohesion = MED. 5-second windows, gaps < 1 second. The default.
COHESION_MED_WINDOW_MS: int = 5_000
#: Maximum inter-word gap in milliseconds for the MED window. ``1000``
#: ms ≈ a natural short pause; anything longer is treated as a new
#: utterance.
COHESION_MED_MAX_GAP_MS: int = 1_000

#: Cohesion = HIGH. 15-second windows, gaps < 500 ms. Forces tighter
#: uninterruptedness, useful for catching catchphrases.
COHESION_HIGH_WINDOW_MS: int = 15_000
#: Tighter gap budget at HIGH cohesion. ``500`` ms is roughly the
#: duration of a typical mid-sentence breath.
COHESION_HIGH_MAX_GAP_MS: int = 500

#: Scoring weights (DESIGN §7). The total weights to 1.0; tweak as
#: data accumulates. The current split:
#:  * 0.4 — duration normalized to a 10-second reference window
#:  * 0.3 — lexical density (words per second)
#:  * 0.3 — spectral similarity to K nearest neighbors
SCORE_WEIGHT_LENGTH: float = 0.4
SCORE_WEIGHT_DENSITY: float = 0.3
SCORE_WEIGHT_SPECTRAL: float = 0.3

#: Reference duration in ms used to normalize the length component
#: of the mine score. ``10_000`` ms = 10 s is "a comfortably long
#: clip"; the log curve compresses longer clips so a 5-minute window
#: doesn't dominate a 10-second one.
LENGTH_NORM_REFERENCE_MS: int = 10_000

#: Words-per-second divisor used to normalize density. ``3.0`` is a
#: brisk conversational pace; clips above this get full density
#: score, below scale linearly.
DENSITY_NORM_WPS: float = 3.0

#: Number of nearest-neighbor clips in the same video used for
#: spectral similarity. ``K=3`` balances smoothing (more neighbors)
#: against locality (fewer neighbors).
SPECTRAL_NEIGHBOR_K: int = 3


# ---------------------------------------------------------------------------
# Speaker pause sampling (DESIGN §8)
# ---------------------------------------------------------------------------
#: Floor for sampled pauses. ``50 ms`` is below the threshold of human
#: perception for short gaps — anything shorter sounds like a single
#: continuous word.
MIN_PAUSE_MS: int = 50

#: Ceiling for sampled pauses. ``800 ms`` is a long-but-not-awkward
#: inter-word silence. Longer pauses make the splice feel stilted.
MAX_PAUSE_MS: int = 800

#: Number of samples at which the speaker's own pause distribution
#: fully overrides the global default (DESIGN §8 blend weight
#: ``w = min(n_samples / N, 1.0)``). 200 is "comfortably many" for
#: percentile statistics.
PAUSE_BLEND_SATURATION_SAMPLES: int = 200

#: Fallback mean used when a speaker has no cached stats. ``200 ms``
#: is the rough speech default for English conversations; not exact
#: but a reasonable central value.
GLOBAL_PAUSE_MEAN_MS: float = 200.0

#: Fallback std used when a speaker has no cached stats. ``80 ms``
#: is a guess; tested against the soft blend so the global default
#: doesn't dominate in low-data cases.
GLOBAL_PAUSE_STD_MS: float = 80.0

#: Below this many samples, ``compute_pause_stats`` returns ``None``
#: rather than a low-confidence summary. ``3`` is "we have at least
#: a hint".
MIN_SAMPLES_FOR_PAUSE_STATS: int = 3

#: Minimum word count required for percentile statistics. Below this
#: we still report the mean/std but the percentile clamps in the
#: blend are skipped.
MIN_SAMPLES_FOR_PAUSE_BLEND: int = 3


# ---------------------------------------------------------------------------
# Loudnorm targets (DESIGN §7 / YouTube / podcast norms)
# ---------------------------------------------------------------------------
#: Target integrated loudness, in LUFS. ``-16 LUFS`` is the YouTube
#: default and matches most podcast targets; using this means spliced
#: output sits in the same loudness space as everything else on the
#: platform.
LUFS_TARGET: float = -16.0

#: True-peak ceiling in dBTP. ``-1.0 dBTP`` is the platform-safe
#: limit; exceeding it causes clipping on lossy codecs.
TRUE_PEAK_DBTP: float = -1.0

#: Loudness range target in LU. ``11 LU`` is the EBU R128 default and
#: matches YouTube's reference range.
LOUDNESS_RANGE_LU: float = 11.0


# ---------------------------------------------------------------------------
# Transcripts (DESIGN §7)
# ---------------------------------------------------------------------------
#: Default minimum block duration in seconds for the markdown
#: transcript export (DESIGN §7 "Output formats"). Blocks shorter
#: than this are absorbed into a same-speaker neighbor. ``2.0``
#: seconds is the user-approved default; CLI / API callers can
#: override.
MIN_BLOCK_DURATION_S: float = 2.0


# ---------------------------------------------------------------------------
# STT chunking (DESIGN §11)
# ---------------------------------------------------------------------------
#: Chunk length in minutes for STT. ``25`` is "comfortably below"
#: Whisper's ~30-minute effective context window.
CHUNK_LENGTH_MIN: int = 25

#: Overlap between consecutive chunks in minutes. ``5`` minutes is
#: wide enough that any utterance near a boundary is fully contained
#: in two chunks, enabling confidence-based de-duplication.
OVERLAP_MIN: int = 5

#: Threshold below which the audio is NOT chunked. ``30`` min is the
#: "below Whisper's context window" line.
CHUNK_THRESHOLD_MIN: int = 30

#: Chunk files written under ``out_dir`` are named
#: ``{stem}.chunk{ord:03d}.wav`` — ``3``-digit zero-padded ordinal.
CHUNK_FILENAME_ORD_WIDTH: int = 3


# ---------------------------------------------------------------------------
# Spectrogram / MFCC (DESIGN §4 "clip_features")
# ---------------------------------------------------------------------------
#: Number of MFCC coefficients stored per clip. ``13`` is the
#: classical speech-recognition dimensionality; lower than librosa's
#: default ``20`` because the mine stage only needs rough spectral
#: shape, not phonetic detail.
MFCC_COEFFS: int = 13

#: FFT size for the per-frame spectrogram. ``512`` at 16 kHz gives
#: a frequency resolution of ``16000 / 512 ≈ 31.25 Hz``, fine enough
#: for speech formants.
FFT_SIZE: int = 512

#: Analysis frame duration in seconds. ``25 ms`` is the standard
#: speech frame length.
FRAME_DURATION_S: float = 0.025

#: Frame hop duration in seconds. ``10 ms`` gives ``15 ms`` of
#: overlap between consecutive frames, which smooths the MFCC
#: trajectory.
HOP_DURATION_S: float = 0.010

#: Number of triangular mel filters. ``26`` is the HTK default and
#: matches what Whisper's feature extractor uses.
MEL_FILTER_COUNT: int = 26

#: Slaney's mel-scale constants. ``2595.0`` and ``700.0`` are the
#: canonical coefficients for the log-scale mel formula that matches
#: human pitch perception. See :func:`spectrogram._hz_to_mel`.
MEL_SLOPE: float = 2595.0
MEL_INTERCEPT_HZ: float = 700.0


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------
#: Maximum rows shown on the videos screen before the table becomes
#: unwieldy.
VIDEOS_PAGE_SIZE: int = 500

#: Title column truncation in characters (videos screen).
VIDEOS_TITLE_TRUNCATE_CHARS: int = 60

#: Title column truncation in characters (speakers screen).
SPEAKERS_TITLE_TRUNCATE_CHARS: int = 50


# ---------------------------------------------------------------------------
# yt-dlp / channels
# ---------------------------------------------------------------------------
#: Length of a YouTube video id. Always 11 characters.
YOUTUBE_ID_LENGTH: int = 11


# ---------------------------------------------------------------------------
# Diarizer
# ---------------------------------------------------------------------------
#: Sentinel end-time for the no-op diarizer's single segment. ``2**31-1``
#: ms ≈ 24.85 days; large enough to cover any real video without
#: depending on the actual audio duration (which the diarizer doesn't
#: know). The merger uses interval-overlap, so the exact value doesn't
#: matter as long as it covers the audio.
NULL_DIARIZER_END_MS: int = 2**31 - 1

#: Default label assigned by the no-op diarizer. Matches the
#: convention pyannote uses internally, so users can swap in pyannote
#: without seeing label format change.
NULL_DIARIZER_SPEAKER_LABEL: str = "SPEAKER_00"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
#: Number of trailing characters of stderr preserved when ffmpeg
#: fails. ``2000`` is "enough to see the error message, not so much
#: that we drown the user in noise".
FFMPEG_ERROR_TAIL_CHARS: int = 2000

#: Number of trailing characters of stderr preserved when loudnorm's
#: JSON parse fails. ``400`` — the JSON block is short.
LOUDNORM_PARSE_TAIL_CHARS: int = 400


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------
#: Default page size when listing pending queue items. ``100`` is
#: "more than enough for one inspection cycle".
QUEUE_LIST_LIMIT: int = 100


# ---------------------------------------------------------------------------
# Seconds → ms conversion factor
# ---------------------------------------------------------------------------
#: Single source of truth for the seconds-to-milliseconds
#: multiplication. ``int(x * 1000)`` truncates; downstream code
#: should be tolerant of off-by-one errors at the millisecond scale.
SECONDS_TO_MS: float = MS_PER_SECOND