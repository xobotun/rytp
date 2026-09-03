# rytp — Design Overview

A high-level architecture for the tool described in `README.md`: download
YouTube videos (and shorts/livestreams), transcribe them locally with
word-level timestamps and per-speaker labels, let a user mine that corpus
for words/phrases with audio-spectrum-aware selection, and splice the
chosen clips back together into a single output file with a manifest.

This document is intentionally an **architecture overview**. It pins down
the building blocks, data model, and CLI surface so that any single piece
can be implemented independently. It deliberately stops short of
pseudocode for the search/splice stages — those warrant their own design
notes when they're actually built.

---

## 1. Goals & non-goals

**Goals**
- One CLI binary (`rytp`) with subcommands for each stage.
- A single SQLite database that is the source of truth for the channel
  index, the download queue, the per-word transcript, the per-video
  speaker map, and the global speaker roster.
- A pluggable **STT** layer and a pluggable **Diarizer** layer behind
  clean interfaces, with an optional **combined STT+Diarize** interface
  for engines that produce both in one pass, so engines can be swapped
  without touching the rest of the system.
- Support for videos from any source yt-dlp understands, plus local
  files registered into the index, all flowing through the same
  downstream pipeline.
- A local interactive TUI (textual) for browsing the index, queue, and
  transcripts, and for mapping raw diarizer labels to known speaker
  names.
- Two splice modes: `concat` (re-encoded, normalized) and `stream-copy`
  (lossless, faster, but worse cross-clip audio continuity).
- Pause modeling per speaker so that spliced clips of the same speaker
  are joined with a naturalistic inter-word pause rather than a hard
  cut.

**Non-goals (for now)**
- Cloud / hosted STT. Everything runs locally.
- Semantic / embedding-based search. The datamining stage does
  substring + windowing + spectral clustering only.
- Automatic speaker *identification* across videos. Mapping is
  per-video and user-edited, but the **label** a diarizer label maps
  to is drawn from a user-curated global speaker roster.
- Voice-embedding-based speaker suggestion. v1 ships the global
  `speakers` roster and the TUI mapper; a future v2 may add
  per-speaker voice embeddings for auto-suggest. Build the mapper
  first; add embeddings only when the data justifies it.
- A web UI. Mapper lives inside the TUI.

---

## 2. Pipeline at a glance

```
                 ┌─────────────────────┐
   channel url ─▶│  channel.sync       │  yt-dlp flat-playlist
                 │  (list videos)      │──────────────┐
                 └─────────────────────┘              ▼
                                           ┌────────────────────┐
   pause / resume flags ───┐               │  SQLite            │
                           ▼               │  - channels         │
                 ┌─────────────────────┐  │  - videos (kind)   │
                 │  queue.worker       │◀─│  - queue_items     │
                 │  (download)         │  │  - downloads       │
                 └─────────────────────┘  └────────────────────┘
                           │  mp4 / mkv on disk
                           ▼
                  ┌─────────────────────┐
                  │  transcribe.extract │  ffmpeg → 16 kHz mono WAV
                  │  audio              │  data/audio/{video_id}.wav
                  └─────────────────────┘
                            │  one WAV per video, reused by all stages
                            ▼
                  ┌─────────────────────┐
                  │  transcribe.run     │  STT (chunked) +
                  │  (per video)        │  Diarizer (full audio) →
                  │                     │  merged words in one pass
                  └─────────────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │  speakers.map       │  TUI: assign known
                  │  (per video)        │  speaker to each raw label
                  └─────────────────────┘
   query string ──▶┌─────────────────────┐
                   │  mine               │  search + window +
                   │  (datamine)         │  spectrum score
                   └─────────────────────┘
                              │ ordered list of (video, in, out, text)
                              ▼
                   ┌─────────────────────┐
                   │  loudnorm           │  EBU R128 per-clip
                   │  (per clip)         │  → −16 LUFS integrated
                   └─────────────────────┘
                              │ normalized intermediates
                              ▼
                   ┌─────────────────────┐
                   │  splice             │  ffmpeg extract +
                   │  (ffmpeg)           │  concat (re-encode
                   └─────────────────────┘  or stream-copy)
                              │
                              ▼
                   output.mp4 + manifest.json
```

Each box is one subcommand of the `rytp` CLI. Each arrow is data on
disk or in the DB.

---

## 3. Module layout

```
rytp/
├── pyproject.toml
├── rytp/
│   ├── __init__.py
│   ├── cli.py              # typer app, subcommand wiring, HF_TOKEN gate
│   ├── config.py           # paths, env vars, defaults
│   ├── db.py               # SQLite connection, migrations, FTS5 shadow
│   ├── models.py           # dataclasses: Video, Word, Speaker, Clip
│   ├── channels.py         # yt-dlp channel/playlist listing
│   ├── download/
│   │   ├── ytdlp.py        # wrapper around yt-dlp Python API
│   │   └── queue.py        # persistent queue + parallel worker
│   ├── transcribe/
│   │   ├── base.py         # STTEngine ABC
│   │   ├── faster_whisper.py
│   │   ├── combined_base.py # TranscribeDiarizeEngine ABC (§5.3)
│   │   ├── extract.py      # ffmpeg → 16 kHz mono WAV (§7)
│   │   ├── chunking.py     # 25-min / 5-min overlap (§11)
│   │   └── run.py          # one-pass STT + Diarize + merger
│   ├── diarize/
│   │   ├── base.py         # Diarizer ABC
│   │   ├── pyannote.py
│   │   └── none.py         # no-op for single-speaker / user-skip
│   ├── speakers.py         # TUI mapper + global roster helpers
│   ├── spectrogram.py      # per-clip MFCC + spectral feats (mine)
│   ├── mine.py             # FTS5 search + windowing + scoring
│   ├── loudnorm.py         # EBU R128 per-clip intermediates
│   ├── splice.py           # ffmpeg cut + concat, manifest writer
│   └── tui/
│       ├── app.py          # textual app
│       └── screens/
│           ├── videos.py
│           ├── queue.py
│           ├── transcript.py
│           └── speakers.py
└── data/                   # gitignored
    ├── rytp.db
    ├── media/              # downloaded video files (mp4/mkv)
    ├── audio/              # extracted 16 kHz mono WAVs (one per video)
    └── output/             # splice results + manifests
```

Key principles:
- Every stage reads from / writes to the SQLite DB or the `data/`
  tree. No stage talks to another in memory; they all go through
  the DB. This is what makes the queue, the pause/resume, and the
  TUI all possible without re-running anything.
- `data/audio/{video_id}.wav` is the **single source of audio
  truth**. STT, loudnorm, and `mode=concat` splice all read from
  this file. `mode=stream-copy` is the only path that re-reads the
  original media file (because stream-copy can't re-encode). The
  audio is extracted exactly once per video.

---

## 4. Data model (SQLite)

One database file: `data/rytp.db`. Logical groups:

**Channels**
- `channels(id, url, title, last_synced_at)`
  - Channels are optional metadata. A video can exist without one.

**Videos (unified)**
- `videos(id, source, kind, channel_id NULLABLE, youtube_id NULLABLE,
  url NULLABLE, local_path NULLABLE, title, duration, published_at NULLABLE,
  downloaded, downloaded_path NULLABLE, downloaded_audio_path NULLABLE,
  metadata_json)`
  - `source` is `youtube` | `ytdlp` | `local`.
  - `kind` is `video` | `short` | `livestream` | `other`. One table,
    one column — kinds differ only in how yt-dlp lists them, not in
    how the downstream stages treat them.
  - Exactly one of `url` or `local_path` is set, matching `source`.
  - `channel_id` is nullable: a video added with
    `rytp videos add <url>` (off-channel) or `rytp videos add <path>`
    (local) has no channel. The FK to `channels` is therefore loose.
  - `youtube_id` is nullable: only set for `source=youtube` rows
    (and for `source=local` rows registered from a `+`-format
    download where the user supplied the id via `--youtube-id`).
  - `downloaded` and `downloaded_path` only make sense for non-local
    sources; local videos have them set at registration time.
  - `downloaded_audio_path` is set when yt-dlp downloads audio and
    video as separate files (format selector with `+`). The download
    stage merges them into a single container stored in
    `downloaded_path`, but keeps the original audio file path here
    so the audio extraction stage can read it directly. The same
    field is set for `source=local` rows registered with
    `rytp videos add VIDEO --audio AUDIO` (see §6), so the
    local-file-pair path uses the same downstream audio-extract
    code as a fresh download.

**Download queue**
- `queue_items(id, video_id, status, attempts, last_error, enqueued_at,
  started_at, finished_at)`
  - `status` is `pending` | `running` | `done` | `failed` | `paused`.
  - A single global `settings` row holds `queue_paused` (bool). The
    worker loop checks it before pulling the next item.
  - Local videos never enter the queue; the registration step is
    effectively a zero-cost "done" row.

**Transcripts**
- `words(id, video_id, start_ms, end_ms, text, normalized_text,
  confidence, diarizer_speaker NULLABLE, speaker_id NULLABLE)`
  - One row per word. There is **no per-word MFCC blob** in v1;
    the spectral features used by the datamining stage live in a
    separate `clip_features` table (see below) keyed by clip
    boundaries, not by word. This keeps the wide `words` table
    small and avoids committing to an embedding dimensionality
    before we have data.
  - `diarizer_speaker` is the raw label from the diarizer
    (e.g. `SPEAKER_02`), or `NULL` for a combined STT+Diarize engine
    that has already mapped to a known speaker.
  - `speaker_id` is a FK to the global `speakers` table — the canonical
    name. It is `NULL` until the user maps the diarizer label.
- `videos_speaker_map(video_id, diarizer_speaker, speaker_id)`
  — the per-video map from raw label to canonical speaker. Edited
  via the TUI.
- `transcribe_runs(id, video_id, stt_engine, diarizer, combined,
  language, started_at, finished_at NULLABLE)`
  - One row per transcribe invocation. Incremental re-runs are
    possible: changing the engine or diarizer creates a new run;
    the mine stage reads from the *latest* run for each video by
    default.
- `chunks(id, transcribe_run_id, ord, start_ms, end_ms,
  status, words_written)`
  - Records the chunking plan for a transcribe run. `status` is
    `pending` | `done` | `failed`. On retry, only `pending` chunks
    re-run. See §11.

**Global speaker roster**
- `speakers(id, label, aliases_json, notes, created_at)`
  - `label` is the canonical display name (e.g. `Paul Maminov`).
  - `aliases_json` is a list of free-form aliases for fuzzy matching
    when the user is picking a speaker in the TUI mapper
    (e.g. `["Paul", "Павел"]`).
  - Cross-video: a given `speaker_id` appears in many rows of
    `videos_speaker_map`, which is exactly how a recurring speaker
    is shared across a channel.
- `speaker_pause_stats(speaker_id, n_samples, mean_ms, std_ms,
  p10_ms, p25_ms, p50_ms, p75_ms, p90_ms, min_ms, max_ms, updated_at)`
  - Cached distribution of inter-word pause durations for that
    speaker across all videos they've been mapped into. Recomputed
    lazily (see §8).

**Mine / splice**
- `clips(id, video_id, start_ms, end_ms, source_query, created_at)`
  — outputs of the mine stage, before splice.
- `splice_runs(id, output_path, mode, created_at)`
- `splice_clips(splice_run_id, ord, video_id, in_ms, out_ms, text,
  adjusted_in_ms NULLABLE, adjusted_out_ms NULLABLE,
  inserted_pause_ms NULLABLE, video_strategy)`
  — ordered rows for the manifest. `inserted_pause_ms` is the
  synthetic gap the splice stage inserted between this clip and
  the previous one (zero for stream-copy, sampled from
  `speaker_pause_stats` for concat). `video_strategy` is `none` |
  `freeze` | `borrow` — the strategy used to cover the pause on
  the video stream. `adjusted_in_ms` and `adjusted_out_ms` are
  non-null only when `video_strategy = 'borrow'` and record the
  clip's trimmed leading/trailing edge so the original
  `in_ms`/`out_ms` are preserved for audit.

**Settings**
- `settings(key, value)` — singleton key/value bag for things like
  `queue_paused`, `default_stt_engine`, `default_splice_mode`,
  `default_combined_engine`, `worker_concurrency`.

**Mine features (per-clip, not per-word)**
- `clip_features(id, video_id, start_ms, end_ms, mfcc_blob,
  spectral_centroid_hz, updated_at)`
  - One row per audio clip, not per word. The mine stage
    records features for the candidate windows it considers;
    downstream code can recompute from `data/audio/{video_id}.wav`
    on demand. Storing per-clip (rather than per-word) keeps
    the table narrow and avoids committing to per-word
    dimensionality. A future v2 may add per-word features if
    the mine scoring turns out to need them.

**Full-text search**
- `words_fts(words_fts)` — an FTS5 virtual table shadowing
  `words(normalized_text)`. Two triggers (`words_ai`, `words_ad`,
  `words_au`) keep it in sync with the main table. Substring
  search in the mine stage hits this index, not `LIKE` over
  `words`. See §7.

Indexes that matter: `words(video_id, start_ms)`,
`words(speaker_id)`, `queue_items(status)`, `videos(source)`,
`videos(channel_id)`. FTS5 provides its own inverted index
over `normalized_text`; do not also add a `LIKE` index.

---

## 5. Pluggable engines

The transcribe subcommand runs STT and Diarize **in one pass**
over the same `data/audio/{video_id}.wav` and writes `words`
once. The two paths into the merger — separate STT+Diarizer or
a single combined engine — produce identical `words` rows.

### 5.1 STT interface

```python
class STTEngine(Protocol):
    name: str
    requires_hf_token: bool = False
    def transcribe(self, audio_path: Path, *,
                   language: str | None = None) -> Iterable[Word]: ...
```

`Word` is `(start_ms, end_ms, text, confidence)`. Word-level
timestamps are mandatory — the README requires them. The
Diarizer is *not* called from here; STT and diarization are
separate stages whose outputs are joined on `(video_id, time)`
when words are written to the DB.

Concrete implementations:
- `FasterWhisperEngine` (recommended default).
- A `NullEngine` that errors clearly so the user knows STT must
  be configured.

`requires_hf_token` defaults to `False`. STT engines that wrap
gated Hugging Face models set it to `True`; the HF_TOKEN
pre-launch gate in §12 reads this attribute to decide whether
to require the env var.

### 5.2 Diarizer interface

```python
class Diarizer(Protocol):
    name: str
    requires_hf_token: bool = False
    def diarize(self, audio_path: Path) -> Iterable[DiarSegment]: ...

class DiarSegment(NamedTuple):
    start_ms: int
    end_ms: int
    speaker: str  # raw label, e.g. "SPEAKER_02"
```

Concrete implementations:
- `PyannoteDiarizer` (default when `HF_TOKEN` is set;
  `requires_hf_token = True`).
- `NullDiarizer` — assigns `SPEAKER_00` to everything. Lets a
  user skip diarization entirely and still get a working
  pipeline. `requires_hf_token = False`.

The merger that writes `words` to the DB takes both an iterable
of `Word` and an iterable of `DiarSegment` and joins them on
overlapping intervals. If the Diarizer is `NullDiarizer`, every
word gets `diarizer_speaker = "SPEAKER_00"`.

### 5.3 Combined STT + Diarize interface

Some engines (WhisperX, AssemblyAI, whisper.cpp with
`tinydiarize`) produce word-level timestamps **and** speaker
labels in a single pass. Exposing them as a third interface
avoids forcing the user to glue two engines together and
re-implement alignment logic.

```python
class TranscribeDiarizeEngine(Protocol):
    name: str
    requires_hf_token: bool = False
    def transcribe_diarize(self, audio_path: Path, *,
                           language: str | None = None
                           ) -> Iterable[DiarizedWord]: ...

class DiarizedWord(NamedTuple):
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    speaker: str  # raw diarizer label, same semantics as Diarizer
```

The DB writer treats `DiarizedWord` exactly like the output of
`(STTEngine, Diarizer)` after the merger: it goes into the same
`words` table with the same `diarizer_speaker` semantics, and
the user's per-video mapping step is identical. There is no
concrete implementation shipped at v1; the interface exists so
that adding WhisperX later is a single new file with no schema
or pipeline changes.

The merger itself is a small named function —
`merge_words_with_diarization(words, segments)` — called by the
transcribe subcommand. A combined engine short-circuits this
call by handing the writer an already-joined stream.

### 5.4 Pyannote / Hugging Face note

pyannote-audio 3.x pipelines are gated on Hugging Face. There
is no payment — the user clicks **Agree and access** once per
model page in the browser, sets `HF_TOKEN` in their
environment, and the models download and cache under
`~/.cache/huggingface/`. After that first time, the system is
fully offline. The `PyannoteDiarizer` reads `HF_TOKEN` from env;
if it's missing, the CLI prints a clear one-paragraph message
with the exact URLs to visit and a link to the
relevant pyannote model card. This is the only "account" thing the
tool needs.

---

## 6. CLI surface

One binary, subcommand groups:

| Command | What it does |
| --- | --- |
| `rytp channel add <url>` | Register a channel. |
| `rytp channel sync <name>` | yt-dlp flat-playlist, upsert into `videos`. |
| `rytp channel list` | Show channels + counts. |
| `rytp videos add <url-or-path>` | Register a single video: YouTube URL, any yt-dlp URL, or a local file path. Probes metadata, inserts a `videos` row, marks `downloaded=true` for local sources. Accepts `--audio <path>` for a separate audio file (typical of a manual `+`-format yt-dlp run without ffmpeg to merge), in which case `downloaded_audio_path` is also populated. |
| `rytp videos list [--channel ...] [--kind ...] [--source ...]` | Browse the index. |
| `rytp download <video-id-or-url>` | Download a single video, mark `downloaded`. |
| `rytp queue add <video-id>...` | Enqueue videos. |
| `rytp queue worker` | Long-running worker; respects `queue_paused`. Parallel by default (configurable via `settings.worker_concurrency`); one slot per in-flight transcribe so multiple `faster-whisper` workers can run on the same GPU. |
| `rytp queue pause` / `rytp queue resume` | Flip the global flag. |
| `rytp queue list` | Show queue contents. |
| `rytp transcribe <video-id> [--stt faster-whisper] [--diarizer pyannote] [--combined <name>]` | Extract audio WAV, run STT (chunked) and Diarizer (full audio) in one pass, write `words` + the per-video speaker map. `--combined` is mutually exclusive with `--stt`/`--diarizer`. Long videos are chunked automatically for STT (see §11). |
| `rytp speakers add <label> [--alias ...] [--notes ...]` | Add to the global `speakers` roster. |
| `rytp speakers list` | List the roster. |
| `rytp speakers <video-id>` | Open the TUI mapper: assign a known speaker to each raw diarizer label seen in this video. |
| `rytp speakers recompute-pauses` | Recompute `speaker_pause_stats` for every speaker that has changed since the last run. |
| `rytp mine <query> [--cohesion low\|med\|high] [--max-clips N]` | Produce a `clips` set. |
| `rytp splice <clip-set-id> --out out.mp4 [--mode concat\|stream-copy]` | Run ffmpeg, write output + manifest under `data/output/`. |
| `rytp tui` | Launch the interactive TUI. |

Built on **typer** for the subcommands and **textual** for the TUI.
The TUI is not a separate binary — it's just one subcommand that
takes over the terminal. The TUI's speaker screen is a two-pane
mapper: the left pane lists the raw diarizer labels seen in this
video, the right pane is a list of `speakers` rows to pick from.
Aliases from `speakers.aliases_json` are matched loosely against
the raw label for keyboard navigation.

---

## 7. The two interesting stages (at a glance)

Both are deliberately underspecified here; they get their own notes.

**Datamining (`mine.py`)**
- Input: a query string and a cohesion level.
- Step 1: find candidate word matches via the FTS5 virtual
  table `words_fts` (substring match on `normalized_text`).
  Hits are O(log N) even on a million-word corpus.
- Step 2: for each match, expand into windows — single-word at
  low cohesion, multi-word longest-uninterrupted at high
  cohesion.
- Step 3: score windows by (length, lexical density, spectral
  similarity to neighboring chosen windows). Spectral
  similarity uses per-clip features lazily computed from
  `data/audio/{video_id}.wav` and cached in `clip_features`.
- Output: an ordered list of `Clip(video_id, in_ms, out_ms,
  text)` written to the `clips` table.

**Loudness normalization (`loudnorm.py`)**
- Input: a clip set from the mine stage.
- For each clip, extract `[in, out]` from
  `data/audio/{video_id}.wav` with ffmpeg (WAV seeking is
  instant — no re-decode), then run the extracted segment
  through ffmpeg's `loudnorm` filter with target integrated
  loudness **−16 LUFS**, true-peak cap **−1 dBTP**, and EBU
  R128 range target **11 LU**. These are the standard
  YouTube/podcast targets, and matching them means spliced
  output is in the same loudness space as everything else on
  the platform.
- Outputs a per-clip normalized intermediate under
  `data/output/normalized/<clip_id>.m4a`. These intermediates
  are the input to the splice stage.
- This is a **separate stage from splice**, not a step inside
  it. The reason: a `clips` set may be re-spliced in different
  orders, with different pause patterns, or with extra/missing
  clips, without re-running loudness analysis on the same
  source clips.
- `mode=stream-copy` skips loudnorm entirely (stream-copy
  can't re-encode anyway). The `mode=concat` flow always goes
  through loudnorm.
- A future v2 optimization (not v1): run `loudnorm`'s
  measurement-only pass to record per-clip `input_i` /
  `input_tp` / `input_lra` / `target_offset` in the DB, then
  apply the gain in a single concat filtergraph at splice
  time. Saves the per-clip intermediate files and one encode
  pass per clip. Skipped in v1 because ffmpeg's two-pass
  loudnorm in a filtergraph is brittle across clip
  boundaries.

**Splice (`splice.py`)**
- Input: a clip set, a mode flag.
- For `mode=concat`, the splice stage reads the per-clip
  normalized intermediates produced by `loudnorm` (which were
  themselves cut from `data/audio/{video_id}.wav`), concatenates
  them with ffmpeg's concat demuxer, inserts same-speaker
  inter-word pauses (§8), and writes `output.mp4` to
  `data/output/`. The original `data/media/` files are not
  touched in this mode.
- For `mode=stream-copy`, the splice stage extracts raw cuts
  with `-c copy` from the original media files and concatenates
  them with the concat demuxer — faster but with audible
  discontinuities at clip boundaries. This is the only stage
  that re-reads `data/media/`.
- Writes `manifest.json` next to the output containing, in
  order, each clip's source URL/path, in-time, out-time,
  transcript text, originating query, any inserted pause, and
  the strategy used to fill that pause's video (see §8).
- For `mode=concat`, between consecutive clips that share the
  same resolved `speaker_id`, the splice stage samples an
  inter-word pause length from that speaker's
  `speaker_pause_stats` and inserts silence of that length
  between the clips. Cross-speaker transitions skip the pause.

---

## 8. Pause modeling for natural splicing

The goal: when the splice stage glues clips of one speaker's words
back to back, the boundary shouldn't be a hard cut to the previous
word's phoneme tail. It should be a short silence of roughly the
duration the speaker normally leaves between words in their natural
speech.

**What we store.** Per `speaker_id`, the
`speaker_pause_stats` table holds a small summary of inter-word
pause durations for that speaker across every video they've been
mapped into. A pause is the gap between the end of one word and the
start of the next, in milliseconds, taken from the `words` table:

```
pause_ms = words[i+1].start_ms - words[i].end_ms
```

The table stores the count, mean, std, and five percentiles
(10/25/50/75/90) plus min and max. That's enough to do percentile
or truncated-normal sampling without re-scanning the corpus at
splice time.

**How it's built.** Lazily, on demand:
- `rytp speakers recompute-pauses` walks every `speaker_id`,
  scans `words` for rows where `speaker_id` is set, computes the
  gaps, writes the row. Cheap even on a large corpus because it's
  one `GROUP BY` over an indexed `words(speaker_id)`.
- The transcribe subcommand and the speakers TUI also bump a
  `speakers.dirty` flag, and `recompute-pauses` only re-does the
  dirty ones.
- The splice stage reads the cached row and uses a **soft blend**
  with a global default distribution rather than a hard threshold.
  A single weight `w = min(samples / 200, 1.0)` controls the
  blend: `w = 0` means a speaker with no data at all (use the
  global default), `w = 1` means a speaker with 200+ samples (use
  their own distribution exclusively), and values in between
  linearly interpolate. This avoids the cliff that a hard
  threshold would create where 199 samples falls back entirely
  to the global default and 201 samples uses the speaker's own
  stats with no smoothing.

**How it's used.** For two consecutive clips in a `clips` set that
share the same resolved `speaker_id`:
- Sample `pause_ms` from that speaker's distribution. Truncated
  normal with `mu = mean_ms`, `sigma = std_ms`, clamped to
  `[p10_ms, p90_ms]` is the simple default; a more elaborate
  version picks the percentile from a uniform `[0, 1]`.
- Insert `pause_ms` of silence between the two clips. In
  `mode=concat` this is a real ffmpeg `aevalsrc=0` segment
  concatenated with `-c:a aac`. In `mode=stream-copy` the cleanest
  implementation is to also write a tiny silent audio segment to
  disk and concat it via the demuxer, since stream-copy can't
  synthesize silence. The cost is one extra file per inserted
  pause.
- Cross-speaker transitions (the `speaker_id` of clip *i* differs
  from clip *i+1*) use a separate, wider distribution or skip the
  pause entirely. The current pick: **skip the pause** for
  cross-speaker transitions, because a real cross-speaker turn in
  conversational data is often *backchannel* with no gap, and
  forcing a pause would sound stilted. This is the one place where
  a per-speaker distribution isn't the right knob.

**What the video does during the pause.** Audio silence alone is
not enough — if the video stream simply holds the last frame, the
output looks natural; if the splice drops the video track for the
pause window, the viewer sees a black flash between every word
pair, which is unacceptable. The splice stage picks one of two
strategies per pause and records the choice in the manifest:
- `freeze`: ffmpeg's `tpad` filter clones the last frame of the
  preceding clip for `pause_ms` milliseconds. Cheap, deterministic,
  and visually correct for typical talking-head content. This is
  the default.
- `borrow`: trim `pause_ms` off the leading edge of the **next**
  clip and use that as the visual cover. Only viable when the
  next clip is long enough to spare the time; the splice stage
  decides per gap whether the borrow is safe. The borrow budget
  is per-clip, capped so no clip is reduced below some minimum
  length (e.g. 250 ms of usable video on each side of the word).
  When a borrow would push a clip under the cap, fall back to
  `freeze` for that gap and record why.
- The `splice_clips.inserted_pause_ms` field becomes
  `(inserted_pause_ms, video_strategy)` where `video_strategy` is
  `none` (no pause was inserted — e.g. stream-copy mode or
  cross-speaker), `freeze`, or `borrow`. The manifest also records
  the original and trimmed `in_ms` / `out_ms` of any clip that was
  shortened by a borrow, so a re-splice can rebalance without
  losing the audit trail.

**What the manifest records.** Every `splice_clips` row carries
`inserted_pause_ms`, `video_strategy`, and (if `borrow`) the
adjusted in/out times. If a future version changes its mind about
the cross-speaker rule or the borrow budget, the manifest is the
place to look first.

---

## 9. Dependencies (rough)

- **Runtime:** `yt-dlp`, `ffmpeg` (system binary, invoked via
  `subprocess`), `faster-whisper`, `pyannote-audio` (optional, gated by
  `HF_TOKEN`), `numpy`, `scipy` (for the spectral features), `typer`,
  `textual`, `rich`.
- **Dev:** `pytest`, `ruff`, `mypy`.
- **System:** `ffmpeg` in `PATH`; on Windows, the `ffmpeg` release
  zip is fine.

A `pyproject.toml` is generated with the right optional-dependency
extras so users can `pip install rytp[pyannote]` only if they want
diarization.

---

## 10. Open questions for later

These are deliberately left for follow-up design notes, not
this one:
- Exact MFCC dimensionality and storage format for
  `clip_features.mfcc_blob` (per-clip, not per-word in v1).
- The `borrow` video strategy in §8 — per-clip minimum
  length, budget across the chain, debugging UX.
- Whether `inserted_pause_ms` should also gate on intra-word
  prosody features (e.g. final phoneme's voicing) or only on
  the speaker's raw distribution. Probably raw distribution
  is good enough; the data to do better isn't available
  before the first splice.
- Voice-embedding-based speaker suggestion. Deferred to v2
  per §1's non-goals; the TUI mapper is the user-facing
  workflow for v1. When revisited, the schema design from
  the previous version of this document (many embeddings per
  speaker, `recorded_at` and `recording_context` hints,
  per-clip matching instead of per-speaker centroid) is the
  right starting point.

---

## 11. Long-video chunking for STT

Most videos in the corpus are well under an hour, but the
channel includes the occasional 2–4 hour livestream.
Whisper-class STT engines degrade on inputs much longer than
their effective context window (around 30 minutes for stable
quality), so the transcribe subcommand needs a chunking
strategy — but **only for STT**. The Diarizer runs on the
full audio (see below).

**The shape.**
- The transcribe subcommand probes audio length before
  calling the STT engine. Below a threshold (default 30 min)
  it runs the engine directly on the full file.
- Above the threshold it splits the audio into overlapping
  windows: **25-minute chunks with 5-minute overlap** between
  consecutive chunks. The overlap is wide enough that every
  utterance near a chunk boundary is fully contained in two
  chunks, which lets us de-duplicate by confidence.
- Each chunk is transcribed independently by the STT engine;
  its `Word` stream carries absolute timestamps from the
  original audio (chunks are a transport detail, not a schema
  detail).
- A merger step keeps the higher-confidence copy of any word
  whose `[start_ms, end_ms]` appears in two adjacent chunks'
  overlap region. Tie-breaker: keep the copy whose surrounding
  words form a more probable phrase, falling back to the first
  chunk's copy.
- A `chunks` table records what was run for a given
  `(video_id, transcribe_run_id)`, so re-transcription is
  incremental: if the engine changes, only the dirty chunks
  re-run.

**The schema.**
- `transcribe_runs(id, video_id, stt_engine, diarizer, combined,
  started_at, finished_at)`
- `chunks(id, transcribe_run_id, ord, start_ms, end_ms,
  status, words_written)`
  - `status` is `pending` | `done` | `failed`. The transcribe
    subcommand is resumable across crashes: on retry, only
    `pending` chunks run.

**Why this lives inside transcribe, not as a separate
stage.** The mine, splice, and speaker-map stages all consume
`words` and shouldn't need to know that chunking happened.
Keeping it internal to transcribe means the rest of the
pipeline never sees a chunk.

**Diarizer runs on full audio, not chunks.** Pyannote's
diarization is context-dependent: speaker labels are
determined by a model that looks at the whole recording
(or at least a long window of it). If the Diarizer is run
on the same 25-min chunks the STT uses, it can produce
inconsistent labels for the same physical speaker — the
overlap-dedup that works for word confidence doesn't work
for speaker identity. So the transcribe subcommand runs the
Diarizer on the **full audio** in parallel with the STT
chunking, then joins STT words to Diarizer segments on
overlapping intervals in the merger. The Diarizer is cheap
relative to STT anyway.

When a combined STT+Diarize engine is used, the engine
itself decides how to chunk; the transcribe subcommand
treats its output as a single already-joined stream and
skips the local chunking step.

**Manual override.** A flag on the transcribe subcommand
lets the user force chunking on or off, and to adjust the
chunk length, for the rare case where a 4-hour stream
needs finer granularity. The `chunks` table records the
parameters used so the same run is reproducible.

---

## 12. HF_TOKEN pre-launch gate

pyannote-audio 3.x pipelines are gated on Hugging Face. There
is no payment, but the user has to visit the model page in a
browser, click **Agree and access**, and set `HF_TOKEN` in
their environment. After that first time, the system is
fully offline and the token is only consulted for
authentication, not for any network call.

**The requirement.** The user has explicitly asked: don't let
this be a runtime surprise. If the chosen engine needs the
token and it's missing, fail loudly before any work starts.

**The shape.** A single typer **callback** registered on the
top-level app, in `cli.py`, runs before any subcommand body.
It does the following:
1. Reads the configured engine from the subcommand's flags
   (or `settings.default_stt_engine` / `default_diarizer`).
2. For each engine in the selection, checks whether it
   requires `HF_TOKEN` (each engine class declares this with
   a class attribute, e.g. `requires_hf_token: bool = True`).
3. If any required token is missing, print **one** clear
   paragraph to stderr and exit non-zero. The paragraph
   names the model, the URL to visit, the exact env var
   name, and a one-line "after this, it's fully offline".
4. If the token is present, the callback is silent and the
   subcommand proceeds.

The callback runs **before** textual takes over the terminal,
so a `rytp tui` invocation with a missing token never
displays a broken TUI; it prints the help paragraph and
exits. This matches the user's "check before launch" ask.

**What does *not* happen.** The callback does not warn, log,
or nag on every run. Once the token is set, it's set. There
is no TUI screen for "configure your HF token" — the message
is meant to be a single paper cut, not a recurring
notification.

---

## 13. Implementation order

A working v1 pipeline has roughly the stages the diagram
shows, but the right order to build them in is not "in
pipeline order". The early stages have to be solid before
the later ones are useful, and a few pieces (FTS5, parallel
worker, measurement-only loudnorm) are optimizations that
can wait. A reasonable sequence:

1. **`db.py` + migrations + `models.py`.** The schema is the
   backbone. Without the DB, every other stage is just
   in-memory scaffolding. Land the `videos` / `words` /
   `speakers` / `clips` tables and the FTS5 shadow before
   writing any other code.
2. **`config.py` + `cli.py` skeleton.** Typer subcommand
   stubs for every box in the pipeline diagram, so the
   shape of the tool is visible from day one. Empty
   subcommands are fine; the goal is the wiring.
3. **`channels.py` + `download/ytdlp.py` + `download/queue.py`.**
   End-to-end: a YouTube URL produces a file on disk and a
   `videos` row. This validates the DB and CLI loop with
   real-world data before any STT runs.
4. **`transcribe/extract.py` + `transcribe/faster_whisper.py`.**
   STT only, no diarization. Goal: produce a `words` table
   you can browse. Validate the audio WAV path and the
   chunking/merger step.
5. **`diarize/none.py` + `transcribe/run.py` merger.** The
   `NullDiarizer` makes the one-pass flow work end-to-end
   without any real diarization, so the pipeline runs from
   URL to browsable transcript.
6. **`speakers.py` TUI mapper.** Add to the global roster
   (`rytp speakers add`) and the TUI two-pane mapper
   (`rytp speakers <video-id>`). The TUI is the only
   human-in-the-loop step and the one that benefits most
   from being built early — it's how you find schema
   bugs.
7. **`mine.py` against the FTS5 index.** Even a naive
   windowing/spectral scorer is enough to validate the
   search-to-clips path. Keep the scorer simple; iterate
   on the formula later.
8. **`loudnorm.py` + `splice.py` (`mode=concat` first).**
   Wire up the audio WAV → loudnorm → concat → output +
   manifest flow. The pause modeling from §8 lands here
   once the splice stage is real.
9. **`diarize/pyannote.py` + HF_TOKEN gate in `cli.py`.**
   Plug in real diarization. The gate is a small
   pre-launch callback; land it before exposing
   `--diarizer pyannote` so the missing-token case is
   handled.
10. **Long-video chunking in `transcribe/run.py`.** Land
    the `transcribe_runs` and `chunks` tables, the
    25/5-minute split, and the overlap-dedup. The
    Diarizer is full-audio from the start; only the STT
    path chunks.
11. **`splice.py` `mode=stream-copy` + `freeze` video
    strategy.** The second splice mode and the
    freeze-frame default. `borrow` and the loudnorm
    measurement-only optimization are v2.
12. **FTS5 index hardening, parallel worker, the rest of
    the optimizations.** These are improvements to a
    working system, not prerequisites. `worker_concurrency`
    in `settings` controls the semaphore-gated worker
    count; the default is 1 (sequential) until you turn
    it up.

The single biggest risk in this order is the FTS5 index:
if it lands late, the mine stage is unusable at scale
during development and you'll be tempted to add
workarounds that have to be ripped out later. Land it in
step 1, even if it starts empty.
