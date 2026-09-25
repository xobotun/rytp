# rytp — design

**Status:** approved in outline, not yet implemented
**Date:** 2026-09-21
**Supersedes:** `DESIGN.md` (AI-generated, drifted from both the code and the goal)

---

## 1. What this is

A command-line tool that turns a personal video archive into raw material for new videos.

You type a sentence. The tool finds real fragments where those words were actually said, cuts them out of the source videos, and glues them into a single file you can upload — along with a list of every source and timestamp it used.

To do that it has to maintain an index that runs in both directions:

- **Forward** — for a given video, every word, when it was said, and who said it.
- **Backward** — for a given word or phrase, optionally by a given speaker, every place it occurs.

Everything else in this document exists to serve those two lookups and the video that comes out the other end.

### Scope

| | |
|---|---|
| Corpus | ~1,600 videos, ~1 hour each, Russian, spanning more than a decade |
| Machine | One Windows PC: 32 GiB RAM, RTX 3080 Laptop (16 GiB), i7-11800H, 1.3 TiB free |
| Constraints | No Docker. No cloud APIs. No web UI. Local models only. |
| Surfaces | CLI (primary) and TUI (aligned with it) |

### Non-goals

Cloud transcription. A web interface. Automatic speaker identification without human confirmation. Semantic or embedding-based search — this is literal word and phrase matching. Editing transcripts by hand.

---

## 2. Glossary

Plain-language definitions for the terms used below.

| Term | Meaning |
|---|---|
| **ASR** | Automatic speech recognition — software that turns audio into text. |
| **Whisper** | OpenAI's transcriber. Good at many languages, middling at Russian. Currently used. |
| **GigaAM** | A Russian-specific transcriber from Sber, MIT-licensed. Roughly half Whisper's error rate on Russian. |
| **WER** | Word error rate — the percentage of words a transcriber gets wrong. Lower is better. |
| **VAD** | Voice activity detection — finds where speech starts and stops, so long audio can be split at natural silences. |
| **Forced alignment** | Given audio *and* the correct text, work out precisely when each word starts and ends. More accurate than the transcriber's own guess. |
| **MFA** | Montreal Forced Aligner. The best free forced aligner, and it has a Russian model. |
| **Diarization** | Working out who spoke when. |
| **pyannote** | The standard free diarization toolkit. |
| **Speaker embedding** | A numeric fingerprint of a voice, used to judge whether two clips are the same person. |
| **Cut list** | The intermediate artifact between searching and rendering: an ordered list of fragments with sources and timings. |

---

## 3. Principles

**The database is the only source of truth.** Every stage reads and writes SQLite; nothing is handed between stages in memory. This is what makes the queue, pause/resume and crash recovery possible. Markdown transcripts are regenerable output, never input.

**Two indexes, because there are two jobs.** Human search wants forgiveness — word endings, ranking, context. Assembly wants exactness and speed, thousands of lookups per line. Today's code serves both from one index, which is why multi-word search returns zero results.

**Audio and video are separate, upgradeable assets.** A transcript is aligned to the audio. Replacing a 360p rendition with 1080p later must not invalidate anything.

**Precision beats recall for assembly.** A missing word means you search differently. A *falsely* recognised word means the tool confidently cuts audio that doesn't say what the index claims. Word-level trust is recorded and acted on.

**Erase and replace, don't reconcile.** Re-transcribing a video wipes its words and speaker labels. Re-mapping two to five speakers is cheaper than a lifetime of drift bugs.

---

## 4. Data model

Seven groups. SQLite throughout; at full corpus the words table is ~14M rows and roughly 3 GB with indexes, which is comfortable.

### Catalog

`channels` — unchanged from today.

`videos` — pure catalog: source, kind, channel, external id, title, duration, published_at, metadata. **All path and download columns move out.**

### Assets

`assets(id, video_id, role, format_id, path, bytes, width, height, abr, acquired_at)`

`role` is one of `audio`, `video`, `captions`, `container`.

A video has at most one canonical audio asset, at most one captions asset, and any number of video renditions. Upgrading a rendition inserts a row; nothing downstream notices. A local file with embedded audio is a single `container` asset with audio extracted from it.

**Audio and video are downloaded together and kept as separate files** — yt-dlp fetches them separately anyway, and not merging them is what makes later rendition upgrades free.

The decoded 16 kHz WAV moves to `cache/wav/{video_id}.wav` (contracts §7) and is no longer a column anywhere. It is a regenerable cache with an explicit prune command, not a source of truth, and nothing may store a path to it in a table.

### Transcript

`words(id, video_id, ord, start_ms, end_ms, text, normalized_text, stem, confidence, align_score, source, video_speaker_id)`

- `ord` — word ordinal within the video, 0-based. This is what turns longest-run matching into a pointer walk instead of a timestamp search.
- `source` — `caption` or `aligned`. See §6.
- `engine` — which transcriber and aligner produced this row, so a corpus built with more than one stays interpretable.
- `align_score` — per-word alignment confidence, so the assembler can skip badly-anchored instances.
- `video_speaker_id` — nullable FK to `video_speakers`, **not** to the global roster. Null until the video is diarized, which is opt-in.

Indexes: `(video_id, ord)`, `(normalized_text)`, `(stem)`, `(video_speaker_id)`.

### Utterances

`utterances(id, video_id, video_speaker_id NULL, start_ms, end_ms, first_word_ord, last_word_ord, text, normalized_text, stem_text)`

Contiguous runs of words, split on speaker change where speakers are known and on a silence threshold otherwise — so caption-tier and undiarized videos still get sensible units. An FTS5 virtual table shadows `normalized_text` and `stem_text` as two columns. This is what makes phrase search work at all: exact column first, stem column as fallback.

### Settings

`settings(key, value)` — download rates, delays, daily caps, pool sizes, default knob positions. Every operational value in §5 lives here and is overridable per run.

### Speakers

`speakers(id, label, aliases_json, notes, created_at)` — the global roster. One row per real person. Small.

`video_speakers(id, video_id, local_label, speaker_id NULL, embedding BLOB)` — one row per diarized label per video. `speaker_id` is nullable: null means "a distinct voice I never named", non-null means "this is the host".

```
speakers ──1:N── video_speakers ──1:N── words
 (people)         (labels/video)        (ord, timings)
```

Linking a label to a person updates **one** row, not thousands of word rows.

### Acoustics

`video_acoustics(video_id, f0_mean, f0_std, spectral_tilt, noise_floor_db, reverb_proxy, loudness_lufs, computed_at)`

One row per video, one cheap pass over the audio. Serves two purposes: it tells the assembler which sources blend, and it scopes cross-video speaker suggestions to comparable recordings. Replaces the per-clip MFCC machinery, which was both more expensive and aimed at the wrong problem.

### Jobs

`jobs(id, kind, target_id, state, pool, priority, attempts, not_before, last_error, payload_json)`

Kinds: `download`, `captions`, `extract_wav`, `transcribe`, `align`, `diarize`, `fingerprint`, `index`, `render`.

### Dropped

`clips`, `clip_features`, `splice_runs`, `splice_clips`, `chunks`, `queue_items`, `videos_speaker_map`.

---

## 5. Ingest and the job queue

**Readiness is derived from data, not from a dependency graph.** Each job kind has a small predicate over current database state — `transcribe` is runnable when the video has audio, `index` when it has words. There is no edge table to keep consistent, and the pipeline self-heals: prune a cached WAV and the transcribe job becomes blocked while the extract job becomes available. Recovery after a crash is just "ask what's runnable now".

**Three concurrency pools**, because the bottlenecks are different resources:

| Pool | Default | Used by |
|---|---|---|
| `network` | 1 | download, captions |
| `gpu` | 1 | transcribe, align, diarize |
| `cpu` | 2–3 | extract_wav, fingerprint, index, render |

All three run simultaneously, so one video downloads while another transcribes.

**Download policy — conservative, every value tunable:** one download at a time, randomized 20–60 s delay between videos, rate limiting, exponential backoff on 429/403 via `not_before` (5 min → 15 → 45 → 2 h), and a daily cap defaulting to a couple hundred. Settings live in the `settings` table and are overridable per run. Throttling shows up in `rytp jobs stats` rather than as a mystery stall.

**No browser cookies.** yt-dlp's own guidance now warns that cookie-authenticated bulk access risks account and IP bans. Anonymous access is sufficient for everything here, including captions.

**Captions are pulled early and always.** They are tiny, they cost no GPU time, and they are the first thing to disappear when a video is delisted. Fetching them for the whole catalog takes about three hours.

**Acquisition order:** `rytp videos add <url>` catalogs the video *and* enqueues its chain by default — the same chain `rytp ingest <id>` enqueues, pulling audio, a video rendition and captions together. Pass `--register-only` to catalog without enqueueing, the original behaviour. Enqueueing only writes `jobs` rows; a worker (`rytp worker`) still has to drain them before anything is actually fetched. `rytp ingest <id>` remains the way to (re-)enqueue a video's chain later, and the only way to enqueue in bulk (`--channel-id`, `--pending`). `rytp channel add` and `rytp channel sync` are unchanged: they still catalog only, never enqueue, because a channel's chain can be thousands of videos at once and a single video's cannot — that asymmetry is exactly why the video-level default is safe and the channel-level one is not. Local files register as assets and never get download jobs, only `extract_wav`/`fingerprint`.

---

## 6. Transcription

### Two tiers

**Tier 1 — captions.** Pulled for every catalogued video. Words get `source='caption'` and are flagged **not cuttable**. This makes the entire corpus searchable within hours of cataloguing it, for no GPU time at all.

Captions carry per-word start times on a 40 ms grid and **no word end times**. Their alignment is loose. They are a text and search resource only.

**Tier 2 — aligned.** For videos you actually want to cut from. Words get `source='aligned'`, real start and end times, and an alignment score. Promoting a video from tier 1 to tier 2 deletes its caption words. One tier per video at a time; no run history to reason about.

### Getting boundaries good enough to cut on

This is the hard problem, and it is not solved by any single source.

Whisper's word timestamps are unusable for cutting. Measured on the existing sample transcript: **78.7% of word gaps are exactly zero**, median gap 0 ms, and some words have zero duration. The transcriber assigns `word[i].end == word[i+1].start` by construction, absorbing every pause into an adjacent word. Pause modelling built on such data would learn that the median pause is zero.

The approach instead combines everything available:

1. **Text** from the best available transcriber.
2. **Rough timings** from forced alignment against that text.
3. **Final boundary** placed at the *local energy minimum* in the audio between the two words, then snapped to the nearest zero crossing.

Adjacent words sharing one boundary is correct under this scheme, because that shared point is real measured silence rather than a guess.

Voice activity detection contributes for free: words at the edges of a detected speech segment already have a true silence on one side.

### Recommended stack

| Role | Choice | Why |
|---|---|---|
| Transcriber | GigaAM v3, VAD-chunked to ≤25 s | ~8.4 WER on Russian vs ~16.2 for Whisper large-v3, MIT licence, 2–4 GB VRAM |
| Alignment | MFA `russian_mfa` v3.1.0 | 12.5 ms median boundary error; Russian acoustic model, dictionary and G2P all exist |
| Alignment fallback | `bond005/wav2vec2-large-ru-golos` | 20 ms stride, easier install |
| Diarization | pyannote `speaker-diarization-community-1` | ~50% less speaker confusion than 3.x — the host-plus-guest failure mode |
| Speaker embeddings | ReDimNet B6 | Substantially better than pyannote's internal model |
| Captions | `ru-orig` json3 via yt-dlp | Per-word starts, no cookies needed |

**Two caveats carried forward honestly.** GigaAM's advantage was measured on clean benchmarks; the one published test on *noisy YouTube* audio reversed the ranking against a Russian-finetuned Whisper. And no aligner's boundary error has ever been measured on Russian — every figure above is from English. Both are addressed by the calibration milestone in §11.

**Diarization is opt-in per video.** At full corpus it is the single largest GPU cost — larger than transcription — and speakers are only needed for videos being mined.

**Cross-video speaker linking** uses per-era enrolment centroids and a two-threshold band with human confirmation in the middle. A decade of changing microphones, rooms and vocal aging roughly triples embedding error, so automatic linking across eras is not trustworthy and is never applied silently.

**Engines run out of process.** The recommended transcriber and diarizer pin incompatible versions of their shared dependencies, so the engine registry must tolerate an engine living in its own virtual environment behind a subprocess boundary.

### Rejected: multi-speed ensembling

Transcribing each file at 0.9x, 1.0x and 1.1x and combining the results was considered. It is not supported by evidence: inference-time speed perturbation has no published measurement for ASR, and the nearest analogue buys ~1.7% relative, which is noise. It would triple the dominant compute cost, degrade timestamps through resampling, and majority voting would actively discard words found at only one speed.

Two better versions of the same instinct are kept for later: selective re-decoding of low-confidence spans only, and treating cross-pass agreement as a **trust signal** that gates assembly rather than as a transcript improvement.

---

## 7. Search

Two distinct lookups over the same data.

**Human search** — "where did he say this?" Goes through the utterance FTS index. Exact phrase match first; if that returns nothing, retry against the stemmed column so Russian word endings stop mattering. Results show the video, timestamp, speaker, surrounding sentence, and whether the hit is cuttable.

Filters: **by speaker** and **cuttable only**.

Playback: a search hit can be played directly from the TUI, and an export command writes hits out as audio files. Both use ffplay/ffmpeg, which are already required — no audio library to bundle.

**Assembly matching** — exact, no fuzziness, driven by `words(normalized_text)` and `words(video_id, ord)`. Find every occurrence of the first word, then walk forward comparing ordinals. No n-gram table is needed.

**Markdown transcripts** are regenerable output at `data/transcripts/{video_id}.md`, safe to delete. Each block carries a stable anchor — video id plus word ordinal range — so an agent reading the file can point back into the database precisely. This plus existing off-the-shelf SQLite tooling is the whole AI-access story; no custom server.

---

## 8. Assembly

Input: a target sentence. Output: a cut list.

**The core move.** Walk the target text left to right. At each position, find the longest run of words starting there that exists contiguously somewhere in the corpus. Take it, jump forward, repeat. Fewer and longer fragments means fewer seams.

**Scoring** balances run length against acoustic consistency, exposed as **one knob** from "fewest seams" to "most consistent sound", defaulting toward fewer seams. Short mixes tolerate many sources; long ones need consistency, which is why this is a knob rather than a rule. Consistency is judged from `video_acoustics`, and preferring few distinct source videos is a strong and cheap proxy.

**Only cuttable words are eligible.** Caption-tier words are searchable but never assembled from.

**Determinism.** Same input gives the same output. A user-supplied seed shakes up choices among near-equal candidates.

**Controls:** exclude specific videos, restrict to a speaker, set the seed, add padding after a chosen word.

**When a word isn't in the corpus:** report the gap plainly and render everything else. Offer ranked substitutions — same stem first, then orthographically close by edit distance, which is a workable proxy for sound in Russian — but never apply one silently. True phonetic matching is not in scope. Every fragment keeps its ranked alternatives so a choice can be swapped without re-running.

**The cut list is a plain editable file** — fragments, source ids, in/out points in milliseconds. It is the durable representation: hand-edit it and re-render. Timings are editable by hand, which is deliberately preferred over clever heuristics.

**Whole words only for now**, but timings stay precise enough that sub-word cutting can be added later without a schema change.

---

## 9. Render

ffmpeg does the work. Fragments are cut, normalized, concatenated, and written out with a report.

| Aspect | Decision |
|---|---|
| Audio | Loudness-normalized by default (flag to disable), so volume doesn't lurch between sources |
| Video cuts | Hard cuts. No crossfades. |
| Pauses | Freeze-frame the last frame to cover an inserted gap |
| Gap length | Configurable. Defaults to natural between-word pauses measured from aligned transcripts — per speaker where the video is diarized, per video otherwise. Disableable if it doesn't sound right. |
| Canvas | 16:9 with pillarboxing by default. Configurable, with "smallest bounding box across all sources" as an alternative mode. |

Gap defaults depend on aligned transcripts, since caption and Whisper timings cannot measure a pause. This is a soft requirement: if it doesn't sound good in practice, it turns off.

**The report** is a markdown file beside the output containing the assembled text, every fragment with its source and timings, any words that couldn't be found, and a paste-ready block for a video description. The underlying data stays structured so JSON and subtitle output can be added later.

---

## 10. Surfaces

**Each operation is defined once** — name, arguments, result shape — and both the CLI and the TUI are generated from that definition. Alignment between the two then holds by construction rather than by discipline, which is the stated requirement.

**CLI is primary.** Long-running work is a CLI worker process.

**TUI is required for speaker assignment** — the one genuinely interactive task. Everything else appears there too, as far as is practical: browsing videos, reading transcripts, searching, playing hits, editing cut lists, adding to the queue, watching job progress. The worker itself stays a separate process.

**Errors** stay as they are today: one line to stderr, exit non-zero, no traceback.

---

## 11. Milestones

**M0 — Make the choice reversible instead of making it now.** Manual boundary marking is deliberately postponed. Rather than blocking the project on a tedious measurement, build the transcriber and aligner as swappable components plus a comparison command that runs several of them over the same audio and reports where they disagree — on text, on timings, and on how much silence sits at each claimed boundary.

This buys the same safety at lower cost. The open questions from §6 — which transcriber wins on *this* audio, and whether alignment is needed at all — stay open, but answering them later becomes a command rather than a project. Hand-marking a reference set remains the only way to get a true error figure; it can happen whenever there's appetite for it, and nothing downstream waits on it.

The practical consequence for the design: no stage may assume a particular transcriber, and word rows must record which engine produced them so mixed-provenance corpora stay interpretable.

**M1 — One video end to end, from several sources.** Add a handful of URLs, download audio and video, transcribe and align, search, assemble a short sentence drawing on **multiple source videos**, render a real file with a source list. Multiple sources is the point: cross-video seams, volume matching and the consistency knob are the risky parts, and a single-source milestone would prove almost nothing.

**M2 — Corpus.** Catalog the channel, pull captions for everything, make the whole archive searchable without transcribing it.

**M3 — Speakers.** Diarization, the TUI mapper, per-era embedding suggestions with human confirmation.

**M4 — Quality.** The consistency knob tuned against real output, pause modelling, ranked alternatives and swapping, substitution suggestions.

**Later, explicitly deferred:** selective re-decoding of low-confidence spans, cross-pass agreement as a trust gate, sub-word cutting, JSON and subtitle output, a dedicated AI server.

---

## 12. What survives from the current code

**Keep:** the SQLite-through-every-stage architecture, the engine registry pattern (`register_*` plus a class attribute inspected before construction), the CLI error-handling convention, `constants.py` as the single home for tunables, audio extraction, yt-dlp wrapping, loudness normalization, the migration runner.

**Rewrite:** the schema, search and mining, splicing, the TUI, the queue.

**Delete:** per-clip MFCC feature extraction, the merged audio/video download path, `chunks` and multi-run transcript history.

---

## 13. Risks

**Transcript error, not drift, is the failure mode.** Alignment is bounded per chunk and cannot drift far. But at roughly 10% word error rate, one word in ten is mis-anchored and drags its neighbours — and a decade of political commentary is dense with proper names and loanwords that will be missing from any pronunciation dictionary. M0 must measure this.

**Every boundary figure available is from English.** No aligner has published Russian boundary error. M0 is the only way to know.

**Diarization cost.** At full corpus it exceeds transcription. Keeping it opt-in is what makes the project affordable.

**Catalog size is unverified.** Enumeration of the main video listing found fewer videos than expected, most likely because live streams are listed separately. Cataloguing must cover the streams and shorts listings as well as the main one, and live streams are long — they will skew both disk and GPU estimates upward.

**Throughput.** Full-corpus processing is estimated at 130–200 GPU-hours, six to nine days continuous. Nothing in the design depends on doing it all at once.
