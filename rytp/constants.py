"""Every tunable value in rytp, each with a comment naming its design section.

No other module may hold a magic number (contracts §1). Sections are
appended, never reordered — later plan parts add their own sections at
the end of this file.

Design: `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md`.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Filesystem layout (design §7 "Search"/"transcripts", contracts §7)
# ---------------------------------------------------------------------------

#: Environment variable that relocates the whole data tree. Contracts §7:
#: "Rooted at RYTP_DATA, default ./data".
DATA_ROOT_ENV_VAR: Final = "RYTP_DATA"

#: Directory used when RYTP_DATA is unset, relative to the working
#: directory (contracts §7).
DEFAULT_DATA_DIRNAME: Final = "data"

#: The one SQLite file. Design §3: "The database is the only source of truth."
DB_FILENAME: Final = "rytp.db"

#: Downloaded assets, one directory per video: media/{video_id}/… (contracts §7).
MEDIA_DIRNAME: Final = "media"

#: Regenerable, prunable working files: cache/wav/{video_id}.wav. Design §4
#: demoted the canonical WAV from a database column to a cache entry.
CACHE_DIRNAME: Final = "cache"

#: Render outputs: output/{render_id}/output.mp4 and report.md (design §9).
OUTPUT_DIRNAME: Final = "output"

#: Regenerable markdown transcripts: transcripts/{video_id}.md. Design §7
#: calls these "regenerable output, never input".
TRANSCRIPTS_DIRNAME: Final = "transcripts"

#: Hand-editable cut lists: cutlists/{name}.toml. Design §8 makes the cut
#: list "the durable representation".
CUTLISTS_DIRNAME: Final = "cutlists"

# ---------------------------------------------------------------------------
# Time (contracts §8)
# ---------------------------------------------------------------------------

#: Contracts §8: "Milliseconds, integers, everywhere." Anything that
#: arrives in seconds — yt-dlp durations, ffmpeg timings — is converted
#: through this, and nothing writes a bare 1000.
MS_PER_SECOND: Final = 1000

# ---------------------------------------------------------------------------
# SQLite connection (design §3, design §5, contracts §8)
# ---------------------------------------------------------------------------

#: Write-ahead logging, so an interactive command can read while the
#: worker writes. Design §5 runs three concurrency pools at once.
SQLITE_JOURNAL_MODE: Final = "WAL"

#: How long a connection waits for a write lock before giving up with
#: "database is locked". Five seconds comfortably covers a worker's
#: longest single write (a video's worth of word rows, design §5).
SQLITE_BUSY_TIMEOUT_MS: Final = 5_000

# ---------------------------------------------------------------------------
# Catalog (design §5 "Acquisition order", design §13 "Catalog size")
# ---------------------------------------------------------------------------

#: The channel listings that must each be enumerated separately. Design
#: §13: "Cataloguing must cover the streams and shorts listings as well
#: as the main one" — live streams are listed apart from videos, and
#: missing them undercounts the corpus. Walked in this order; on an id
#: that appears in two listings, the later listing decides its kind.
CHANNEL_TABS: Final = ("videos", "streams", "shorts")

#: videos.kind implied by the listing an entry came from (design §13).
CHANNEL_TAB_KINDS: Final = {
    "videos": "video",
    "streams": "livestream",
    "shorts": "short",
}

#: Allowed videos.source values. Mirrors the CHECK constraint in
#: contracts §3 so a surface can reject bad input before SQLite does.
VIDEO_SOURCES: Final = ("youtube", "ytdlp", "local")

#: Allowed videos.kind values. Mirrors contracts §3.
VIDEO_KINDS: Final = ("video", "short", "livestream", "other")

#: What anything reached through yt-dlp is recorded as. Design §5 treats
#: acquisition as one yt-dlp path regardless of site.
REMOTE_SOURCE: Final = "ytdlp"

#: What a registered local file is recorded as. Design §5: "Local files
#: register as assets and never get download jobs."
LOCAL_SOURCE: Final = "local"

#: kind assigned when the listing or probe says nothing better.
DEFAULT_VIDEO_KIND: Final = "video"

# ---------------------------------------------------------------------------
# Deletion (contracts §5 "Deletion")
# ---------------------------------------------------------------------------

#: The `JobKind.target_kind` value meaning "this job's target_id is a
#: videos.id". Removal derives the kinds to cancel from part 2's
#: registry rather than listing them (contracts §5): a hardcoded list
#: went stale before any code existed, because part 3 registers
#: `caption_words` on top of the eight that were obvious.
VIDEO_TARGET_KIND: Final = "video"

#: Tables that lose their rows by `ON DELETE CASCADE` when a video goes
#: (contracts §3). Counted before a removal so the report can say what
#: is about to disappear; never deleted by hand.
VIDEO_CASCADE_TABLES: Final = (
    "words",
    "utterances",
    "video_speakers",
    "assets",
    "video_acoustics",
)

# ---------------------------------------------------------------------------
# Health checks (contracts §5 "Health checks")
# ---------------------------------------------------------------------------

#: Oldest interpreter the project supports (contracts §1).
MIN_PYTHON_VERSION: Final = (3, 11)

#: Filename `doctor` writes and deletes to prove the data tree is
#: writable. Distinctive so a leftover is obviously ours.
WRITE_PROBE_FILENAME: Final = ".rytp-write-probe"

#: Status column values in the `doctor` report. Three, not two: a
#: `required=False` check that reports `ok=False` is advisory — a real
#: finding about a real absence that must not fail the command
#: (contracts §5).
CHECK_OK: Final = "ok"
CHECK_FAILED: Final = "FAILED"
CHECK_ADVISORY: Final = "advisory"

# ---------------------------------------------------------------------------
# Surfaces (design §10)
# ---------------------------------------------------------------------------

#: Default row cap on every `list` command, shared by the CLI and the TUI
#: because design §10 requires the two to show the same thing. A 1,600
#: video corpus would otherwise flood a terminal.
DEFAULT_LIST_LIMIT: Final = 50

#: Hard ceiling a caller may ask for, so `--limit 1000000` cannot page
#: the whole corpus into memory.
MAX_LIST_LIMIT: Final = 5_000

#: Title truncation in a rendered result table, in characters (design §10).
TITLE_TRUNCATE_CHARS: Final = 60

#: Printed in place of a NULL cell so columns stay aligned.
NULL_CELL: Final = "-"

# ---------------------------------------------------------------------------
# Engines (design §6)
# ---------------------------------------------------------------------------

#: Environment variables consulted for a Hugging Face token, in order.
#: Only the pyannote diarizer (design §6) needs one.
HF_TOKEN_ENV_VARS: Final = ("HF_TOKEN", "HUGGINGFACE_TOKEN")


# ---------------------------------------------------------------------------
# Jobs and worker pools (design §5)
# ---------------------------------------------------------------------------
#: The three concurrency pools. design §5: the bottlenecks are different
#: resources, so they run simultaneously — one video downloads while
#: another transcribes.
POOLS: tuple[str, ...] = ("network", "gpu", "cpu")

#: Default slots per pool, overridable through the setting
#: ``pool.<name>.size``. ``network`` stays at 1: design §5 makes not
#: getting the owner's IP blocked more important than throughput.
POOL_DEFAULT_SIZE: dict[str, int] = {"network": 1, "gpu": 1, "cpu": 2}

#: Attempts before a job is parked in ``failed``. Throttle responses are
#: refunded and never count against this — they are not the job's fault.
JOB_MAX_ATTEMPTS: int = 5

#: Rows returned by ``rytp jobs list`` when no limit is given.
JOB_LIST_LIMIT: int = 50

#: How many times ``claim`` re-selects after losing a race to another
#: worker thread before giving up for this tick. Three is plenty: a lost
#: race means someone else is making progress.
CLAIM_RETRY_LIMIT: int = 3

#: Seconds a pool loop sleeps when it finds nothing to claim.
WORKER_POLL_INTERVAL_S: float = 2.0

#: Seconds between worker heartbeat writes.
WORKER_HEARTBEAT_INTERVAL_S: float = 15.0

#: A worker lease whose heartbeat is older than this is treated as dead and
#: the next worker to start reclaims its ``running`` jobs. Several heartbeat
#: intervals, so a merely busy worker is never declared dead.
WORKER_LEASE_STALE_S: float = 120.0

#: Backoff after a transient (non-throttle) failure, seconds.
TRANSIENT_BACKOFF_S: int = 60


# ---------------------------------------------------------------------------
# Download policy (design §5)
# ---------------------------------------------------------------------------
#: Randomised pause between two videos on the network pool, seconds.
#: design §5: "randomized 20-60 s delay between videos".
DOWNLOAD_DELAY_MIN_S: float = 20.0
DOWNLOAD_DELAY_MAX_S: float = 60.0

#: Pause after a captions fetch, seconds. Shorter than the inter-video
#: delay because captions are a single tiny request.
DOWNLOAD_FILE_DELAY_S: float = 5.0

#: yt-dlp ``ratelimit``, bytes per second. 1.5 MB/s keeps a home connection
#: usable and keeps the traffic pattern unremarkable. 0 disables the limit.
DOWNLOAD_RATE_LIMIT_BPS: int = 1_500_000

#: yt-dlp ``sleep_interval_requests``: seconds between metadata requests
#: within one download.
DOWNLOAD_SLEEP_REQUESTS_S: float = 1.0

#: Maximum download jobs *started* per UTC day. design §5: "a daily cap
#: defaulting to a couple hundred". Started, not finished — a failed
#: attempt hit the network just as hard as a successful one.
DOWNLOAD_DAILY_CAP: int = 200

#: Exponential backoff ladder for HTTP 429 / 403-forbidden, seconds.
#: design §5: 5 min -> 15 -> 45 -> 2 h. The last entry repeats forever.
#: Applied to the whole network pool, not to one job.
THROTTLE_BACKOFF_LADDER_S: tuple[int, ...] = (300, 900, 2700, 7200)

#: yt-dlp format selectors. The two are joined with a **comma**, which is
#: yt-dlp's "download several formats of the same video" separator (README,
#: FORMAT SELECTION). Unlike ``+`` it does not merge, and design §4 depends
#: on that: keeping the rendition separate is what lets 360p be upgraded to
#: 1080p later without touching the audio a transcript is aligned to.
DOWNLOAD_FORMAT_AUDIO: str = "bestaudio[language=ru]/bestaudio"
DOWNLOAD_FORMAT_VIDEO: str = "bestvideo[height<=?720]/bestvideo"

#: Caption languages, most preferred first. design §6: "ru-orig json3 via
#: yt-dlp" — the ``-orig`` track is the auto-generated original-language
#: one, so ``writeautomaticsub`` must be on to see it.
CAPTION_LANGS: tuple[str, ...] = ("ru-orig", "ru")
CAPTION_FORMAT: str = "json3"


# ---------------------------------------------------------------------------
# Assets (contracts §3, design §4)
# ---------------------------------------------------------------------------
#: Roles a video may hold at most one of. design §4: "at most one canonical
#: audio asset, at most one captions asset, and any number of video
#: renditions". The schema has no unique index for this, so it is enforced
#: in :func:`rytp.db.queries.insert_asset`.
SINGLETON_ASSET_ROLES: frozenset[str] = frozenset({"audio", "captions", "container"})


# ---------------------------------------------------------------------------
# Settings keys (design §5: "every operational value lives in settings")
# ---------------------------------------------------------------------------
SETTING_QUEUE_PAUSED: str = "queue.paused"
SETTING_POOL_SIZE: str = "pool.{pool}.size"
SETTING_DOWNLOAD_DELAY_MIN: str = "download.delay_min_s"
SETTING_DOWNLOAD_DELAY_MAX: str = "download.delay_max_s"
SETTING_DOWNLOAD_FILE_DELAY: str = "download.file_delay_s"
SETTING_DOWNLOAD_RATE_LIMIT: str = "download.rate_limit_bps"
SETTING_DOWNLOAD_SLEEP_REQUESTS: str = "download.sleep_requests_s"
SETTING_DOWNLOAD_DAILY_CAP: str = "download.daily_cap"
SETTING_DOWNLOAD_FORMAT_AUDIO: str = "download.format_audio"
SETTING_DOWNLOAD_FORMAT_VIDEO: str = "download.format_video"
SETTING_CAPTION_LANGS: str = "captions.langs"
SETTING_CAPTION_FORMAT: str = "captions.format"
SETTING_THROTTLE_STREAK: str = "network.throttle_streak"
SETTING_COOLDOWN_UNTIL: str = "network.cooldown_until"
SETTING_WORKER_LEASE: str = "worker.lease"
#: contracts §3: names the aligner `ingest --transcribe` stamps onto the
#: `align` jobs it creates. Empty (the default) means no alignment at all —
#: ingest enqueues no `align` job and the words stay in the `timed` tier.
SETTING_DEFAULT_ALIGNER: str = "default_aligner"
#: contracts §3 "Choosing a transcriber": names the transcriber `ingest
#: --transcribe` stamps onto the `transcribe` jobs it creates, and that
#: `rytp transcribe run` falls back to when no `--transcriber` is given.
#: Unlike the aligner, **empty is not meaningful here** — an empty aligner
#: reads coherently as "no alignment, the words stay `timed`", but an empty
#: transcriber would make `--transcribe` do nothing. So there is no unset
#: path: this setting always names an engine.
SETTING_DEFAULT_TRANSCRIBER: str = "default_transcriber"

#: What SETTING_DEFAULT_TRANSCRIBER reads as when the row was never written.
#: Part 1's migration 11 seeds `default_aligner` and not this key, so this
#: fallback is what makes contracts §3's "defaults to gigaam" true.
#: `gigaam` is the Russian-specific engine and published benchmarks put it at
#: roughly half Whisper's Russian word error rate, which matters because
#: transcribing the corpus costs 35-90 GPU-hours. **A starting point, not a
#: verdict:** the one published test on *noisy YouTube* audio — which is
#: exactly this corpus — reversed the ranking in favour of a Russian-finetuned
#: Whisper. `rytp transcribe compare` exists to revisit it on real material.
DEFAULT_TRANSCRIBER_FALLBACK: str = "gigaam"


# ---------------------------------------------------------------------------
# The ingest chain (design §5, §6)
# ---------------------------------------------------------------------------
#: What `rytp ingest` enqueues for a remote video, in pipeline order.
#: design §5 pulls audio, a rendition and captions together; design §6 turns
#: those captions into tier-1 words for the whole corpus, "searchable within
#: hours of cataloguing it, for no GPU time at all". `fingerprint` fills
#: `video_acoustics`, which design §8's consistency knob reads — without it
#: that knob silently degrades to counting fragments.
#: Kinds owned by later parts (`caption_words`, `fingerprint`) are enqueued
#: only once they are registered, so Part 2 ships and tests standalone and
#: the chain completes itself as each part lands.
INGEST_CHAIN_REMOTE: tuple[str, ...] = (
    "download",
    "captions",
    "caption_words",
    "extract_wav",
    "fingerprint",
)

#: A local file already has its media and has no captions to fetch.
INGEST_CHAIN_LOCAL: tuple[str, ...] = ("extract_wav", "fingerprint")

#: design §6 makes tier 2 opt-in per video — "for videos you actually want to
#: cut from" — and design §13 puts full-corpus transcription at 130-200 GPU
#: hours. So `rytp ingest --transcribe` adds these; nothing else does.
#: `align` is additionally gated on SETTING_DEFAULT_ALIGNER being set — see
#: `_aligner_payload` in rytp/commands/ingest.py. `transcribe` is never
#: gated, but it is always *stamped*: SETTING_DEFAULT_TRANSCRIBER names the
#: engine and `_transcriber_payload` puts it in the payload, so no job is
#: enqueued without one.
INGEST_CHAIN_TRANSCRIBE: tuple[str, ...] = ("transcribe", "align")

#: Videos touched by one bulk `rytp ingest` when no limit is given. design
#: §11 M2 is ~1,600 videos; a default cap keeps an accidental run reviewable
#: and a deliberate one is `--limit 2000`.
INGEST_DEFAULT_LIMIT: int = 500


# ---------------------------------------------------------------------------
# Health checks (contracts §5, "Health checks")
# ---------------------------------------------------------------------------
#: Free space on the data volume below which ``rytp doctor`` complains. The
#: corpus is ~1,600 hours (design §1) and the owner wants to know *before* a
#: batch fills the drive, so this is deliberately generous: 50 GiB is a few
#: dozen more videos, i.e. enough warning to act on.
HEALTH_DISK_FREE_MIN_BYTES: int = 50 * 1024**3

#: Seconds to wait for ``ffmpeg -version``. A binary that cannot answer this
#: in ten seconds is broken in a way worth reporting.
HEALTH_VERSION_TIMEOUT_S: float = 10.0

#: Bytes one ingested video is assumed to cost when the catalogue has no
#: assets yet to measure: roughly an hour of audio plus a 720p rendition.
ESTIMATED_BYTES_PER_VIDEO: int = 500 * 1024**2

#: One gibibyte, for turning byte counts into something a human reads.
BYTES_PER_GIB: int = 1024**3


#: Longest error string stored in ``jobs.last_error``. A yt-dlp traceback can
#: run to kilobytes and the tail is never the informative part.
JOB_ERROR_MAX_CHARS: int = 500

#: How much of an error or note to show in one ``rytp jobs list`` cell.
#: Wider than this and the table stops fitting in a terminal.
JOB_ERROR_PREVIEW_CHARS: int = 80

#: Longest note stored in ``jobs.note``. A note is a warning a human reads in
#: a table cell, not a log line.
JOB_NOTE_MAX_CHARS: int = 300

#: yt-dlp ``retries`` and ``fragment_retries``. Low on purpose: the job
#: queue's own backoff ladder is the real retry mechanism, and hammering
#: inside one invocation is what looks like abuse.
YTDLP_RETRIES: int = 2


# ---------------------------------------------------------------------------
# The WAV cache (design §4, contracts §7)
# ---------------------------------------------------------------------------
#: Sample rate of ``cache/wav/{video_id}.wav``. Whisper-class and GigaAM
#: models are all trained on 16 kHz, and every consumer of the cache — STT,
#: alignment, energy-minimum boundary placement, acoustics — expects it.
AUDIO_SAMPLE_RATE_HZ: int = 16_000

#: Channels in the cached WAV. Mono: nothing downstream uses stereo, and
#: mixing at extract time keeps the storage and the tooling simple.
AUDIO_CHANNELS: int = 1

#: How much of ffmpeg's stderr to keep in an error message. The tail is
#: where ffmpeg puts the actual complaint.
FFMPEG_ERROR_TAIL_CHARS: int = 800

#: How long to wait for a worker thread to notice the stop flag before
#: giving up on it. The threads are daemons, so a stuck ffmpeg cannot keep
#: the process alive past this.
WORKER_JOIN_TIMEOUT_S: float = 30.0
# ---------------------------------------------------------------------------
# Assembly (design §8)
# ---------------------------------------------------------------------------

#: The one knob, from 0.0 "fewest seams" to 1.0 "most consistent sound".
#: Design §8: "defaulting toward fewer seams".
ASSEMBLE_DEFAULT_CONSISTENCY: Final = 0.25

#: The knob's two endpoints, as (seam, switch, acoustic) cost weights.
#: At "fewest seams" a fragment is expensive and changing source is nearly
#: free, so the optimiser buys the longest runs it can from anywhere. The
#: switch weight is not zero even here: given two equally long coverings,
#: the one that stays in one video is still the better video.
ASSEMBLE_WEIGHTS_FEWEST_SEAMS: Final = (1.0, 0.15, 0.0)

#: At "most consistent sound" a fragment is cheap and changing source is
#: expensive, priced by how differently the two videos sound. The seam
#: weight stays at a quarter rather than dropping to zero: free seams
#: would let a long run inside one video be shredded into single words
#: for no gain, which is not what anyone means by "consistent".
ASSEMBLE_WEIGHTS_MOST_CONSISTENT: Final = (0.25, 2.5, 2.0)

#: Cost of a fragment whose first and last words are badly anchored.
#: Only those two boundaries are actually cut, so they dominate — design
#: §4: "align_score — per-word alignment confidence, so the assembler can
#: skip badly-anchored instances."
ASSEMBLE_EDGE_ALIGN_WEIGHT: Final = 0.6

#: Cost of a fragment whose interior words are badly anchored. Interior
#: boundaries are never cut, but a low score there means the transcript
#: itself is doubtful, and design §3 puts precision above recall.
ASSEMBLE_MEAN_ALIGN_WEIGHT: Final = 0.2

#: Stand-in for a NULL align_score on an *aligned* row. With the three
#: tiers (contracts §3) an unaligned row is `timed` and never reaches
#: here, so a NULL can only mean the aligner reported no per-word
#: confidence — MFA does not (design §6). That is "unknown", not "bad":
#: 0.5 keeps such a word usable while preferring a measured one.
ASSEMBLE_DEFAULT_ALIGN_SCORE: Final = 0.5

#: Default floor for --min-align-score: accept everything. Raising it is
#: how the owner trades recall for cut quality on a specific run.
ASSEMBLE_MIN_ALIGN_SCORE: Final = 0.0

#: Cost of leaving one target word uncovered. Large enough that a gap is
#: only ever chosen when the word is genuinely absent, never as a cheaper
#: alternative to an awkward fragment — design §8: "never apply one
#: silently".
ASSEMBLE_GAP_COST: Final = 1000.0

#: Longest run the matcher will build. A fragment longer than this is a
#: quotation, not an assembly, and extending past it wastes queries.
ASSEMBLE_MAX_RUN_WORDS: Final = 40

#: A run stops rather than spanning a silence longer than this. The cut
#: would otherwise carry a pause the target sentence does not have, and
#: design §9 gives pause length to the renderer, not to the matcher.
ASSEMBLE_MAX_INTERNAL_GAP_MS: Final = 800

#: Rows fetched for one target token before extension begins. A Russian
#: function word occurs hundreds of thousands of times in a full corpus
#: (design §4: ~14M word rows); this caps the read, and the ORDER BY
#: makes which rows survive deterministic.
ASSEMBLE_MAX_OCCURRENCES_PER_TOKEN: Final = 2000

#: Occurrences per video carried forward into extension. One video cannot
#: crowd out the rest of the corpus, so the cap preserves source diversity
#: — which is exactly what the consistency knob needs to choose between.
ASSEMBLE_MAX_SEEDS_PER_VIDEO: Final = 4

#: Rows per parameterised IN clause. SQLite's oldest supported limit on
#: host parameters is 999 and each pair costs two, so 400 pairs is safe
#: everywhere without being chatty.
ASSEMBLE_SQL_BATCH: Final = 400

#: Ranked alternatives kept per slot, so design §8's "a choice can be
#: swapped without re-running" costs one hand-edit and no recompute.
ASSEMBLE_MAX_ALTERNATIVES: Final = 3

#: Decimal places costs are rounded to before comparison. Determinism
#: (design §8) needs near-equal candidates to actually tie so the
#: tie-break decides; without rounding, float noise decides instead.
ASSEMBLE_COST_DECIMALS: Final = 6

#: How far a non-zero seed may move a candidate's cost. Design §8: a seed
#: "shakes up choices among near-equal candidates" — small enough that it
#: reorders ties, never overturns a real preference.
ASSEMBLE_SEED_JITTER: Final = 0.05

#: Per-field "one unit of audible difference" for video_acoustics
#: (design §4). A 40 Hz difference in mean pitch, 8 dB of noise floor or
#: 4 LU of loudness each count as one unit; the distance is the mean over
#: the fields both videos actually have.
ASSEMBLE_ACOUSTIC_SCALES: Final = {
    "f0_mean": 40.0,
    "f0_std": 20.0,
    "spectral_tilt": 0.5,
    "noise_floor_db": 8.0,
    "reverb_proxy": 0.3,
    "loudness_lufs": 4.0,
}

#: Distance used when either video has no acoustics row — undiarized and
#: un-fingerprinted videos are the normal case early on. "Middling", so
#: an unmeasured source is neither preferred nor excluded.
ASSEMBLE_UNKNOWN_ACOUSTIC_DISTANCE: Final = 0.5

#: Acoustic distance is clamped here. Beyond one unit per field the two
#: recordings are simply incomparable and further difference changes no
#: decision.
ASSEMBLE_ACOUSTIC_DISTANCE_CAP: Final = 1.0

#: Largest edit distance offered as a substitution. Design §8 calls edit
#: distance "a workable proxy for sound in Russian"; past two edits it
#: stops being one.
ASSEMBLE_SUBSTITUTION_MAX_DISTANCE: Final = 2

#: Only vocabulary within this many characters of the missing word is
#: considered, since a longer or shorter form cannot be within
#: ASSEMBLE_SUBSTITUTION_MAX_DISTANCE anyway.
ASSEMBLE_SUBSTITUTION_LEN_WINDOW: Final = 2

#: Substitutions offered per missing word.
ASSEMBLE_SUBSTITUTION_LIMIT: Final = 5

#: Distinct normalized forms pulled into memory for the edit-distance
#: pass. The length window usually keeps this far smaller; the cap is
#: what stops a one-letter target word from reading the whole vocabulary.
ASSEMBLE_SUBSTITUTION_VOCAB_LIMIT: Final = 50_000

#: Default tail padding added after each fragment's last word (design §8
#: "add padding after a chosen word"). Zero: cut exactly on the measured
#: boundary unless asked otherwise.
ASSEMBLE_DEFAULT_PAD_MS: Final = 0

#: Upper bound on --pad-ms. Past two seconds the padding is a fragment of
#: its own and should be cut as one.
ASSEMBLE_MAX_PAD_MS: Final = 2000

#: Longest target accepted. The DP is quadratic in the target length and
#: a sentence is tens of words; a paste of a whole transcript is a
#: mistake worth naming.
ASSEMBLE_MAX_TARGET_WORDS: Final = 200

#: Characters kept when deriving a cut list name from the target text.
ASSEMBLE_NAME_MAX_CHARS: Final = 40

#: Name used when the target slugifies to nothing at all.
ASSEMBLE_FALLBACK_NAME: Final = "cutlist"

#: Version stamped into every cut list. Bumped only when the on-disk
#: shape changes in a way an old reader would misread; part 6 refuses a
#: version it does not know.
CUTLIST_SCHEMA_VERSION: Final = 1

#: Extension of a cut list file under cutlists/ (contracts §7).
CUTLIST_SUFFIX: Final = ".toml"


# ---------------------------------------------------------------------------
# Part 3 — transcription, alignment and audio analysis (design §6, §11)
# ---------------------------------------------------------------------------

#: Sample rate of the cached WAV every audio stage reads (design §4).
WAV_SAMPLE_RATE_HZ: int = 16000

#: Default spoken language passed to transcribers (design §1: a Russian archive).
DEFAULT_LANGUAGE: str = "ru"

#: Floor for dB conversion, and the epsilon that keeps log10 finite. A 16-bit
#: sample's quietest non-zero step is about -90 dBFS, so -120 dB is safely below
#: anything real while keeping digital silence from becoming -inf (design §6).
DB_FLOOR: float = -120.0
DB_EPSILON: float = 1e-10

# --- Voice activity detection (design §6, "VAD contributes for free") ---

#: Analysis frame and hop for VAD. 20 ms is the usual speech-analysis frame; a
#: 10 ms hop decides a boundary every 10 ms, well under the ~150 ms error that
#: becomes audible in a cut.
VAD_FRAME_MS: int = 20
VAD_HOP_MS: int = 10

#: The noise floor is this percentile of frame energy. 10 assumes at least a
#: tenth of any recording is not speech, which holds for interview audio.
VAD_NOISE_PERCENTILE: float = 10.0

#: Hysteresis band above the noise floor: enter speech at +12 dB, leave at
#: +8 dB. The gap stops a wavering frame shredding one utterance into many.
VAD_ENTER_DB: float = 12.0
VAD_EXIT_DB: float = 8.0

#: Segments shorter than this are discarded as clicks; silences shorter than
#: this are swallowed, because a stop consonant's closure is real silence
#: *inside* a word (design §6).
VAD_MIN_SPEECH_MS: int = 120
VAD_MIN_SILENCE_MS: int = 150

#: Padding either side of a detected segment so a soft onset is not clipped.
#: Acoustics passes pad_ms=0 when it measures a reverberation tail.
VAD_PAD_MS: int = 30

#: GigaAM v3 accepts at most 25 s per call (design §6), so a chunk may never
#: exceed it; the target is lower so chunks usually end at a real silence.
VAD_CHUNK_MAX_MS: int = 25_000
VAD_CHUNK_TARGET_MS: int = 20_000

# --- Boundary refinement (design §6, "boundaries good enough to cut on") ---

#: Short-time energy used to find the quietest point between two words. 5 ms
#: frames resolve a stop closure; a 1 ms hop is the finest the millisecond
#: resolution of the database can use.
ENERGY_FRAME_MS: int = 5
ENERGY_HOP_MS: int = 1

#: RMS values within this of the minimum count as tied, so digital silence
#: (hundreds of exactly equal frames) resolves deterministically to the frame
#: nearest the claimed boundary instead of to the earliest one.
ENERGY_TIE_RMS: float = 1e-6

#: A refined boundary may never move further than this from the aligner's
#: claim. The rail stops a runaway search swallowing a neighbouring word, and
#: 60 ms stays under the ~150 ms at which a mis-cut becomes audible (design §6).
BOUNDARY_SEARCH_MS: int = 60

#: How far the energy minimum may be nudged to land on a zero crossing.
ZERO_CROSSING_SEARCH_MS: int = 5

#: A word shorter than this cannot be real; monotonicity repair widens it.
MIN_WORD_DURATION_MS: int = 20

#: A word edge within this of a VAD segment edge is snapped to it — that edge
#: is measured silence, which beats any search (design §6).
VAD_EDGE_SNAP_MS: int = 40

#: Silence depth at a boundary: reference level is the median frame energy in
#: the 200 ms either side, the minimum is taken within 10 ms of the boundary.
#: The reference window has to reach *past* a pause to find the speech it is
#: comparing against — a window shorter than a typical pause would measure
#: silence against silence and report a depth of zero.
SILENCE_DEPTH_REF_MS: int = 200
SILENCE_DEPTH_WINDOW_MS: int = 10

#: Silence depth (dB) treated as a perfect boundary when measured boundary
#: quality stands in for an aligner that reports no per-word confidence.
SILENCE_DEPTH_FULL_DB: float = 30.0

# --- Acoustic fingerprint (design §4, "Acoustics") ---

#: 40 ms is two pitch periods at the 60 Hz floor — the shortest frame from
#: which autocorrelation can recover a male speaking pitch.
ACOUSTICS_FRAME_MS: int = 40
ACOUSTICS_HOP_MS: int = 20

#: At most this many voiced frames are analysed, evenly strided across the
#: file. The fingerprint is a summary; a full hour adds no information.
ACOUSTICS_MAX_FRAMES: int = 4000

#: Human speaking pitch range. Below 60 Hz is rumble, above 400 Hz is song.
F0_MIN_HZ: int = 60
F0_MAX_HZ: int = 400

#: A frame counts as voiced when its normalised autocorrelation peak clears
#: this and its energy clears the noise floor by this many dB.
F0_VOICED_AUTOCORR: float = 0.3
F0_VOICED_ABOVE_FLOOR_DB: float = 10.0

#: Band over which spectral tilt is fitted, in dB per decade. The top stays
#: below the 8 kHz Nyquist of 16 kHz audio where the anti-alias filter rolls off.
SPECTRAL_TILT_LO_HZ: int = 100
SPECTRAL_TILT_HI_HZ: int = 7800

#: Reverberation proxy: energy in the 150 ms after a speech segment ends,
#: relative to the last 50 ms of that segment. A dry room decays instantly.
REVERB_REF_MS: int = 50
REVERB_TAIL_MS: int = 150

#: ffmpeg loudness scan of an hour of audio finishes well inside this.
LOUDNESS_TIMEOUT_S: int = 600

# --- Engines (contracts §6; design §6, "Engines run out of process") ---

#: Settings keys that point an out-of-process engine at its own interpreter,
#: and an external binary at its path.
SETTINGS_INTERPRETER_PREFIX: str = "engine.interpreter."
SETTINGS_INTERPRETER_DEFAULT: str = "engine.interpreter.default"
SETTINGS_BINARY_PREFIX: str = "engine.binary."

#: One GPU pass over an hour of audio, with headroom for model loading.
ENGINE_SUBPROCESS_TIMEOUT_S: int = 3600

#: Lines of child stderr quoted back when an out-of-process engine fails.
ENGINE_SUBPROCESS_STDERR_TAIL: int = 20

#: nvidia-smi answers in well under a second on a healthy machine; the
#: timeout exists so a wedged driver cannot hang `doctor` (contracts §5).
GPU_PROBE_TIMEOUT_S: int = 10

#: words.engine value for caption-tier rows (design §6, tier 1).
CAPTION_ENGINE: str = "captions:json3"

#: Default model identifiers (design §6, "Recommended stack").
#: **There is deliberately no DEFAULT_TRANSCRIBER here.** contracts §3
#: "Choosing a transcriber" makes the engine a *setting* — Part 2's
#: `SETTING_DEFAULT_TRANSCRIBER`, read through
#: `rytp.transcribe.registry.default_transcriber` — so that a run which used
#: the default can say so. A constant cannot; that is the whole point.
WHISPER_DEFAULT_MODEL: str = "large-v3"
GIGAAM_DEFAULT_MODEL: str = "v3_rnnt"
MFA_ACOUSTIC_MODEL: str = "russian_mfa"
MFA_DICTIONARY: str = "russian_mfa"
WAV2VEC2_MODEL: str = "bond005/wav2vec2-large-ru-golos"

#: wav2vec2 emits one CTC frame per 20 ms at 16 kHz (design §6, "Alignment
#: fallback": 20 ms stride).
WAV2VEC2_STRIDE_MS: float = 20.0

#: Default window and sample size for `transcribe compare` (design §11, M0).
COMPARE_DEFAULT_WINDOW_MS: int = 120_000
COMPARE_SAMPLE_WORDS: int = 40
# ---------------------------------------------------------------------------
# Part 6 — render (design §9)
# ---------------------------------------------------------------------------

#: Seconds before one ffmpeg invocation is abandoned. A whole-programme
#: concat pass over a long cut list decodes minutes of video, and the
#: machine is also transcribing; an hour is generous rather than tight.
RENDER_FFMPEG_TIMEOUT_S: Final = 3600

#: Seconds before one ffprobe invocation is abandoned. It reads a header.
RENDER_FFPROBE_TIMEOUT_S: Final = 60

#: Canvas modes (design §9 "16:9 with pillarboxing by default. Configurable,
#: with 'smallest bounding box across all sources' as an alternative").
RENDER_CANVAS_MODES: Final = ("16:9", "bbox")
RENDER_DEFAULT_CANVAS_MODE: Final = "16:9"

#: The default canvas aspect, as a pair so the arithmetic stays exact.
RENDER_DEFAULT_ASPECT: Final = (16, 9)

#: Bound on the canvas's *longer* edge, whichever dimension that is.
#: A height-only cap is the wrong shape for this: it would downscale a
#: 1080-wide vertical short to fit an 1080-tall box (crushing the one
#: dimension "retain the aspect ratio" is about), and it would clip a
#: 16:10 canvas to a 16:9 one instead of leaving it alone. Bounding
#: whichever edge is longer, and scaling both dimensions down together
#: when it fires, keeps the aspect exact in every orientation; for a
#: 16:9 canvas this reduces to the familiar 1080p height cap, so nothing
#: changes there.
RENDER_MAX_CANVAS_LONG_EDGE: Final = 1920

#: Floor for a derived canvas height, so a mis-probed source cannot
#: produce a 2-pixel output.
RENDER_MIN_CANVAS_HEIGHT: Final = 240

#: Colour of the pillar/letterbox bars (design §9 "pillarboxing").
RENDER_PAD_COLOR: Final = "black"

#: Output frame rate is the commonest source rate, capped here. Above 60
#: the file gets large for no benefit on archive footage.
RENDER_MAX_FPS: Final = 60

#: Frame rate used when no source could be probed for one. 25, not 30:
#: this is a European archive and PAL is the likelier majority.
RENDER_FALLBACK_FPS: Final = 25

#: Final video encode settings. CRF 20 is visually transparent for
#: talking-head content at these resolutions; "medium" is the preset
#: where x264 stops paying for itself on an i7 laptop.
RENDER_VIDEO_CODEC: Final = "libx264"
RENDER_VIDEO_CRF: Final = 20
RENDER_VIDEO_PRESET: Final = "medium"
RENDER_PIXEL_FORMAT: Final = "yuv420p"

#: Per-fragment intermediates. Matroska because it carries PCM, and PCM
#: because AAC's ~1024-sample encoder priming is not trimmed by the
#: concat demuxer and would slide the audio later at every seam.
RENDER_INTERMEDIATE_SUFFIX: Final = ".mkv"
RENDER_INTERMEDIATE_FORMAT: Final = "matroska"
RENDER_INTERMEDIATE_AUDIO_CODEC: Final = "pcm_s16le"

#: Final audio. 48 kHz stereo AAC is what every upload target wants, and
#: every fragment is resampled to it before concatenation so the demuxer
#: never sees a parameter change.
RENDER_AUDIO_CODEC: Final = "aac"
RENDER_AUDIO_BITRATE: Final = "192k"
RENDER_AUDIO_RATE_HZ: Final = 48_000
RENDER_AUDIO_CHANNELS: Final = 2

#: EBU R128 targets for the final programme pass (design §9 "loudness-
#: normalized by default"). -16 LUFS is the speech-content convention and
#: survives every platform's own normalization; -1.5 dBTP leaves room for
#: lossy-codec overshoot. LRA 13 rather than the broadcast 11: a
#: multi-source cut legitimately has a wider range than one recording,
#: and asking for 11 makes loudnorm abandon linear mode and start pumping.
RENDER_TARGET_LUFS: Final = -16.0
RENDER_TRUE_PEAK_DBTP: Final = -1.5
RENDER_LOUDNESS_RANGE_LU: Final = 13.0

#: Per-source gain is clamped to this many dB either way. A source that
#: needs more than 12 dB is broken, not quiet, and boosting it that far
#: would lift its noise floor into the mix.
RENDER_MAX_GAIN_DB: Final = 12.0

#: Sample-peak ceiling for the per-fragment limiter, linear. 0.891 is
#: -1.0 dBFS. It exists only to stop a clamped gain clipping the PCM
#: intermediate before the final true-peak limiter gets a say.
RENDER_LIMITER_CEILING: Final = 0.891

#: How much of a source to measure when `video_acoustics.loudness_lufs`
#: is not already there: two minutes from just before the video's first
#: fragment. Measuring a whole hour to place one gain is not worth the
#: minutes it costs.
RENDER_LOUDNESS_WINDOW_MS: Final = 120_000

#: Hex characters of the plan hash in a derived render id, and the cap on
#: the whole id — it becomes a directory name under output/.
RENDER_ID_HASH_CHARS: Final = 8
RENDER_ID_MAX_CHARS: Final = 64

#: Heading above the source list in the paste-ready description block.
#: The audience for that block reads Russian; the report around it is in
#: English like the rest of the CLI.
RENDER_DESCRIPTION_HEADER: Final = "Источники:"

# --- Pause statistics (design §9 "Gap length") -----------------------------

#: Gaps at or below this are "zero". Aligned boundaries land on measured
#: energy minima, so a 1 ms gap means the two words share a boundary.
PAUSE_ZERO_GAP_MS: Final = 1

#: Gaps longer than this are turn boundaries or edits, not between-word
#: pauses, and would drag the median upward.
PAUSE_MAX_GAP_MS: Final = 2_000

#: Fewer measured gaps than this and the distribution is not worth
#: believing; the constant fallback is used instead.
PAUSE_MIN_SAMPLES: Final = 50

#: Above this share of exactly-zero gaps the distribution is degenerate.
#: Design §6: a Whisper-timed transcript has 78.7% of its gaps at exactly
#: zero because the transcriber sets word[i].end == word[i+1].start, and
#: a naive median over that data is zero. Half is a generous line: real
#: aligned speech puts most between-word gaps above zero.
PAUSE_DEGENERATE_ZERO_FRACTION: Final = 0.5

#: Used whenever measurement is unavailable or degenerate. A short,
#: unobtrusive between-word pause.
PAUSE_FALLBACK_GAP_MS: Final = 180

#: Applied gaps are clamped here regardless of what was measured.
PAUSE_MIN_APPLIED_GAP_MS: Final = 40
PAUSE_MAX_APPLIED_GAP_MS: Final = 600

#: Upper bound on rows pulled for one statistic, so a speaker with a
#: hundred hours of aligned words cannot make the query eat the machine.
PAUSE_SAMPLE_LIMIT: Final = 200_000

#: contracts §3: "Cuttable is defined as source = 'aligned'". Only these
#: rows carry an end_ms, so only these can measure a pause at all.
#: If Part 3 has already added a constant for this literal, use theirs
#: and delete this one rather than keeping two.
ALIGNED_WORD_SOURCE: Final = "aligned"

# ---------------------------------------------------------------------------
# Part 4: index, search and export (design §4, §7)
# ---------------------------------------------------------------------------

#: A pause at or above this splits one utterance from the next when the
#: speaker is unknown (design §4: "split ... on a silence threshold
#: otherwise"). Conversational sentence boundaries sit around 0.5-1.0 s;
#: anything shorter is a within-sentence hesitation and must not split,
#: because a split is what forces phrase search onto the slower
#: cross-boundary walk.
UTTERANCE_SILENCE_GAP_MS: Final = 700

#: Hard cap on words per utterance. Load-bearing for caption tier, whose
#: words inside one caption segment all share a start time and therefore
#: have zero gaps: without this a captioned hour is a single row.
#: Forty words is roughly fifteen seconds of ordinary speech.
UTTERANCE_MAX_WORDS: Final = 40

#: Hard cap on utterance duration, for a slow heavily-paused stretch that
#: stays under the word cap.
UTTERANCE_MAX_DURATION_MS: Final = 20_000

#: How long a caption-tier word is assumed to last when the next word is
#: further away than this, or when it is the last word of the video.
#: Captions carry starts and no ends (design §6), and capping the implied
#: end here is what lets UTTERANCE_SILENCE_GAP_MS fire on caption tier at
#: all — an uncapped implied end always equals the next start, making
#: every caption gap zero.
CAPTION_WORD_FALLBACK_MS: Final = 400

#: words.source values, weakest first. Contracts §3: `caption` has word
#: starts only, `timed` has a transcriber's own timestamps (energy
#: refined, not good enough to cut on — 78.7% of Whisper's word gaps
#: measure exactly zero), `aligned` has forced-alignment boundaries. All
#: three are searchable; the order is what lets a run report the weakest
#: tier it contains.
WORD_SOURCE_RANK: Final = ("caption", "timed", "aligned")

#: words.source that may be cut. Contracts §3: "Cuttable is `source =
#: 'aligned'` and nothing else may be cut." `timed` and `caption` words
#: are searchable and never assembled from (design §8). Part 3 already
#: defines this value as ALIGNED_WORD_SOURCE; reusing that constant
#: rather than duplicating the literal.
CUTTABLE_SOURCE: Final = ALIGNED_WORD_SOURCE

#: Rows a search returns when the caller does not say. Design §7 is a
#: human-facing lookup: a screenful, ranked, not a data dump.
SEARCH_DEFAULT_LIMIT: Final = 20

#: Ceiling on the same, so a typo in --limit cannot pull the corpus into
#: memory.
SEARCH_MAX_LIMIT: Final = 500

#: How many anchor rows the cross-boundary walk will look at before it
#: gives up. The walk only runs when a tier's FTS phrase found nothing,
#: which means the phrase is rare, so this is a guard against a
#: pathological anchor rather than a routine limit (design §7).
SEARCH_WALK_ANCHOR_LIMIT: Final = 2_000

#: The walk anchors on the query's rarest token, and counting occurrences
#: stops here: the ranking only needs to tell "hundreds" from "hundreds
#: of thousands", and at full corpus a common word is ~200k rows.
SEARCH_RARITY_PROBE_LIMIT: Final = 5_000

#: Padding added on each side of an exported or played clip, so a word is
#: not clipped by a boundary that is a few milliseconds optimistic
#: (design §7 "Playback"). Zero disables it.
CLIP_PAD_MS: Final = 150

#: How much of a hit's surrounding sentence a result table shows before
#: it is cut with an ellipsis. Wider than TITLE_TRUNCATE_CHARS because
#: the sentence is the point of the row, but still narrow enough that a
#: screenful of hits stays a table (design §7).
SEARCH_TEXT_TRUNCATE_CHARS: Final = 80

# ---------------------------------------------------------------------------
# Part 7 — speakers, diarization and cross-video linking (design §4, §6, §11)
# ---------------------------------------------------------------------------

#: The local label a diarizer with no speaker model assigns to everything.
#: pyannote's own labels look like this, so one convention covers both.
NULL_DIARIZER_LABEL: str = "SPEAKER_00"

#: Default engine names. The default diarizer is the null one: diarization is
#: opt-in per video (design §6) and the default must never cost GPU time.
DEFAULT_DIARIZER: str = "none"

#: Empty means "do not embed". Embedding is useful only once a second video
#: of the same person exists, so it is off until asked for (design §6).
DEFAULT_EMBEDDER: str = ""

#: Models behind the two real engines (design §6, "Recommended stack").
#: community-1 has roughly half the speaker confusion of pyannote 3.x, which
#: is the host-plus-guest failure mode this corpus is made of.
PYANNOTE_DIARIZATION_MODEL: str = "pyannote/speaker-diarization-community-1"
REDIMNET_DEFAULT_MODEL: str = "b6"

#: Settings keys holding the default engine names, so a machine is configured
#: once rather than on every command line.
SETTINGS_DIARIZER: str = "speakers.diarizer"
SETTINGS_EMBEDDER: str = "speakers.embedder"

# --- Embedding extraction (design §6, "Speaker embeddings") ---

#: A label with less speech than this is not embedded at all. Three seconds
#: is the usual floor below which a speaker embedding is dominated by the
#: phonetic content rather than the voice.
EMBED_MIN_SPEECH_MS: int = 3_000

#: At most this much of one label's speech is fed to the embedder. Thirty
#: seconds saturates every published speaker-verification curve; an hour of
#: the host would cost sixty times as much for no gain.
EMBED_MAX_SPEECH_MS: int = 30_000

#: Segments shorter than this are skipped when choosing embedding windows:
#: a one-second turn is mostly onset and offset.
EMBED_MIN_SEGMENT_MS: int = 1_000

#: Embeddings are stored in video_speakers.embedding as little-endian
#: float32, which is half the size of float64 and further below the noise
#: floor of the model than any downstream comparison can notice. The width
#: is what the codec needs: the format string is built from the vector
#: length at pack time.
EMBEDDING_BYTES_PER_VALUE: int = 4

# --- Cross-video linking (design §6, "Cross-video speaker linking") ---

#: Era bucket width in years. The corpus spans more than a decade and the
#: microphones, rooms and the main speaker's voice all changed; comparing
#: within a bucket is what keeps an enrolment centroid meaningful.
SPEAKER_ERA_YEARS: int = 2

#: Bucket for a video with no published_at — a local file, usually.
SPEAKER_ERA_UNKNOWN: str = "unknown"

#: The two-threshold band (design §6). At or above HI is a confident match,
#: at or below LO a confident non-match, and the space between goes to the
#: human. UNVALIDATED: no measurement of this corpus exists, these are the
#: usual starting points for a cosine speaker-verification score. Re-tune
#: them once real suggestions have been accepted and rejected for a while.
SPEAKER_MATCH_HI: float = 0.70
SPEAKER_MATCH_LO: float = 0.45

#: Across eras, embedding error roughly triples (design §6), so a
#: cross-era comparison needs a much higher score before it may be called
#: confident. Below this it is a suggestion for a human, never a match.
#: UNVALIDATED, same caveat as above.
SPEAKER_MATCH_HI_CROSS_ERA: float = 0.85

#: How far apart two recordings may be acoustically and still be treated as
#: comparable. The distance is the mean of the per-feature differences
#: below, each divided by its scale, so 1.0 means "one typical spread apart
#: on average".
ACOUSTIC_MAX_DISTANCE: float = 1.0

#: Per-feature scales for that distance, in the feature's own units: a
#: plausible spread across recordings of one person (design §4, acoustics).
#: UNVALIDATED — order-of-magnitude guesses until the corpus has been
#: fingerprinted and the real spread can be measured.
ACOUSTIC_FEATURE_SCALES: tuple[tuple[str, float], ...] = (
    ("f0_mean", 20.0),          # Hz
    ("f0_std", 15.0),           # Hz
    ("spectral_tilt", 6.0),     # dB per decade
    ("noise_floor_db", 10.0),   # dB
    ("reverb_proxy", 0.15),     # dimensionless ratio
)

#: How many roster candidates one suggestion call returns per local label.
SPEAKER_SUGGEST_LIMIT: int = 5

#: `speakers remove` unlinks every label naming that person. Below this many
#: links it just does it; at or above, it wants `--yes`. Contracts §5 only
#: *requires* confirmation for commands that delete files, and this deletes
#: none — but undoing an afternoon of mapping by mistyping a name is the
#: kind of loss worth one extra word (design §3: erase and replace, but
#: deliberately).
SPEAKER_REMOVE_CONFIRM_LINKS: int = 5

# --- Surfaces ---

#: Rows the speaker listing commands return before the user must filter.
SPEAKER_LIST_LIMIT: int = 200

#: Titles are truncated harder in the speakers panes than in the video list,
#: because the mapper shows two tables side by side.
SPEAKER_TITLE_TRUNCATE_CHARS: int = 40

# ---------------------------------------------------------------------------
# Part 8 — the TUI shell and its two screens (design §10)
# ---------------------------------------------------------------------------
#: How often the queue screen re-reads the jobs table, in seconds. The worker
#: is a separate process (design §5), so the only way to see progress is to
#: look again. Two seconds is short enough to feel live on a download that
#: takes a minute and long enough that a full-corpus queue is not re-rendered
#: continuously.
TUI_QUEUE_REFRESH_S: float = 2.0

#: Rows the queue screen asks for. design §11 M2 is ~1,600 videos and the
#: chain is five kinds per video, so the table is never scrolled to the end;
#: what matters is what is running and what failed, and the filters get you
#: there faster than scrolling would.
TUI_QUEUE_ROW_LIMIT: int = 200

#: Milliseconds one keystroke moves a cut-list boundary. design §8 prefers
#: hand-edited timings "over clever heuristics". Was 40 ms; QA batch
#: 2026-09-25 (BUGS.md entry 30) lowered it to 10, because 40 ms is a good
#: fraction of a phoneme at conversational speed and was too coarse for
#: word-edge nudging. TUI_CUTLIST_SHIFT_NUDGE_MS below is the new
#: larger step (bound to Shift+arrow); TUI_CUTLIST_COARSE_NUDGE_MS is
#: unchanged — it means gaps, not word edges.
TUI_CUTLIST_NUDGE_MS: int = 10

#: A coarse nudge, for when the boundary is plainly in the wrong place.
TUI_CUTLIST_COARSE_NUDGE_MS: int = 250

# ---------------------------------------------------------------------------
# --- 2026-09-25 QA fix batch ---
# ---------------------------------------------------------------------------
# `docs/superpowers/plans/2026-09-25-qa-fix-batch.md` §1a-§1c and Task 2a/2b.
# Foundation constants for entries 3, 7, 10, 11, 13, 15, 25, 26, 28, 30, 33,
# 34 of `BUGS.md`. Task 2a is the sole owner of this file for the batch;
# Task 2b's migrations 13-15 and the contracts amendment
# (`docs/superpowers/2026-09-25-contracts-amendments.md`) cite these by name.

# --- Score scale (plan §1a, BUGS.md entry 13) ---

#: `words.align_scale` values. An aligner that reports a score writes the
#: scale that produced it; `words.align_score` is never converted between
#: scales — normalising would be lossy and would bake in a conversion
#: nobody can justify (entry 13's recorded decision).
ALIGN_SCALE_ENERGY: Final = "energy"
ALIGN_SCALE_LOGPROB: Final = "logprob"
ALIGN_SCALE_NONE: Final = "none"
ALIGN_SCALE_UNKNOWN: Final = "unknown"

#: All permitted `words.align_scale` values, for validation.
ALIGN_SCALES: Final = (
    ALIGN_SCALE_ENERGY,
    ALIGN_SCALE_LOGPROB,
    ALIGN_SCALE_NONE,
    ALIGN_SCALE_UNKNOWN,
)

#: Per-scale floor for `assemble plan`'s eligibility filter (entries 26, 28).
#: A `None` value means no threshold applies for that scale (MFA writes no
#: score under `none`, and `unknown` is excluded outright by
#: ASSEMBLE_EXCLUDE_UNKNOWN_SCALE below rather than floored). The `logprob`
#: entry is a first guess in the spirit of Part 7's unvalidated thresholds:
#: `exp(-5)` is about 0.7% probability, which admits essentially everything
#: while still excluding a catastrophic mismatch — it exists so the filter
#: is meaningful, not so it is tight.
ASSEMBLE_MIN_ALIGN_BY_SCALE: Final = {
    ALIGN_SCALE_ENERGY: 0.0,
    ALIGN_SCALE_LOGPROB: -5.0,
    ALIGN_SCALE_NONE: None,
    ALIGN_SCALE_UNKNOWN: None,
}

#: `align_scale = 'unknown'` rows are excluded from `assemble plan` by
#: default (entry 26): they are the pre-batch wav2vec2 output that entry 36
#: proved fabricated equally-spaced boundaries, and entry 28's sign-flip
#: hack may additionally have reversed some of their ordering with no way
#: to tell which. `transcribe align` re-running is what clears the flag.
ASSEMBLE_EXCLUDE_UNKNOWN_SCALE: Final = True

# --- Device selection (plan §1b, BUGS.md entry 34) ---

#: `device` parameter accepted by every engine adapter. `"auto"` resolves in
#: the child to `"cuda"` when available, else `"cpu"`; the child moves both
#: the model and every input tensor there and reports the chosen device
#: back so a foreground run's message and a queued run's job note both say
#: what actually ran, rather than silently running 40x slower on the CPU.
ENGINE_DEFAULT_DEVICE: Final = "auto"
ENGINE_DEVICES: Final = ("auto", "cuda", "cpu")

#: Timeout for the child-interpreter probe that answers "does the module
#: import, is CUDA available, which device would be selected" (entries 7,
#: 33). Generous: a probe that loads torch to answer this is not instant.
ENGINE_PROBE_TIMEOUT_S: Final = 60

# --- Hugging Face cache (BUGS.md entry 11) ---

#: Environment variable set in every child and in-process engine's
#: environment before an HF-backed model loads, so the "no symlinks
#: support, enable Windows Developer Mode" warning prints once as a
#: `doctor` advisory rather than per process from a dependency's own code.
HF_SYMLINK_WARNING_ENV: Final = "HF_HUB_DISABLE_SYMLINKS_WARNING"

# --- Progress (plan §1c, BUGS.md entries 3, 10) ---

#: Throttle for the default tty progress sink, milliseconds. A download, a
#: model fetch and a network call were all indistinguishable from a hang;
#: this is how often the one `stderr` line is allowed to repaint.
PROGRESS_TTY_INTERVAL_MS: Final = 250

#: Throttle for the worker's `jobs.progress` write, milliseconds. Coarser
#: than the tty interval: a queued job is read by polling `jobs.list`, not
#: watched continuously.
PROGRESS_DB_INTERVAL_MS: Final = 2_000

# --- TUI cut-list editing (BUGS.md entry 30) ---

#: Milliseconds one Shift+arrow keystroke moves a cut-list boundary — the
#: new larger step now that TUI_CUTLIST_NUDGE_MS (plain arrow) is 10 ms.
TUI_CUTLIST_SHIFT_NUDGE_MS: Final = 100

# --- Transcript rendering (BUGS.md entry 15) ---

#: Default wrap width, in characters, requested for `transcript show` /
#: `transcript build`'s `--line-length` knob, so the transcript wraps to a
#: chosen width instead of whatever the table or the markdown writer decides.
TRANSCRIPT_DEFAULT_LINE_LENGTH: Final = 80

# --- Surfaces (BUGS.md entry 25) ---

#: Glyphs for a boolean cell in `videos list --long` (entry 25: one column
#: per asset role and pipeline stage — audio, captions, videos, transcribed,
#: aligned, indexed, diarized — each cell a tick meaning present/complete or
#: a cross meaning not yet). Named here so any other boolean-cell table can
#: reuse the same vocabulary rather than inlining its own glyph.
CELL_TICK: Final = "✓"
CELL_CROSS: Final = "✗"
