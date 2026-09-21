# rytp
A tool to extract speech from videos and splice them into something else

# Goal
To have a console line tools to:
- manipulate videos:
	- list and store metadata (mostly URL + "donwloaded or not" flag) on all videos+shorts+livestreams from a single youtube channel;
	- download a video from the stored metadata list or directly provided url (may be from a different channel or non-youtube source). Utilise ytdlp for that;
	- maintain a queue to download videos from youtube, as well as pause the download process;
- extract audio tracks into text. For a given video, run a local speech-to-text model (or several). The result must:
	- be stored locally
	- have a timestamp for every word retrieved from STT-model
	- have a person mapped for every word (have it editable, since I believe that per-person mapping will be stable within a video – but not across years and different auditoriums/channels)
	- have some spectrogram stored nearby, be it by video or by word. Not sure. The goal is to try to retrieve the closest-sounding words together, rather than the first one
- datamine the extract:
	- take a string as an input
	- depending on level of cohesion the user wants, either find single-word segments, or maximize uninterruptedness of a segment, preferring the longer ones over shorter ones.
	- segments found should be close to each other in audio spectrum, to sound more naturally.
- glue the result together. Given the original videos, the timestamps provided by the datamining stage, and ffmpeg – extract the clips from the originals, and glue them together in order. Also provide a list of original URLs used and timestamps cut out in order they appear in the final result.

# Hardware:
- 32 GiB RAM
- 16 GiB RTX 3080 Laptop GPU
- i7-11800H @ 2.30 GHz

# Architecture

See [DESIGN.md](./DESIGN.md) for the full architecture overview. The
short version:

```
  channel.sync   queue.worker   transcribe.extract   transcribe.run   speakers.map
        │              │               │                   │              │
        ▼              ▼               ▼                   ▼              ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │   SQLite: channels, videos, words, videos_speaker_map, speakers,     │
   │   speaker_pause_stats, transcribe_runs, chunks, clips, splice_*       │
   └──────────────────────────────────────────────────────────────────────┘
        ▲                                                              ▲
        │              data/audio/{video_id}.wav                       │
        └──────────────  (single source of audio truth)  ───────────────┘

                          mine → loudnorm → splice
                            (clips table)   (splice_runs / manifest.json)
```

# CLI surface (DESIGN §6)

```
rytp channel add <url>             # register a channel
rytp channel sync <name>           # yt-dlp flat-playlist, upsert into videos
rytp channel list
rytp videos add <url-or-path>      # register a single video (URL or local)
rytp videos list [--channel ...] [--kind ...] [--source ...]
rytp download <video-id-or-url>    # download one video, mark downloaded
rytp queue add <video-id>...
rytp queue worker                  # long-running; respects queue_paused
rytp queue pause | resume | list
rytp transcribe <video-id> [--stt faster-whisper] [--diarizer pyannote]
                                # extract WAV, STT (chunked), Diarizer (full audio)
rytp speakers add <label> [--alias ...] [--notes ...]
rytp speakers list
rytp speakers <video-id>           # TUI mapper (textual)
rytp speakers recompute-pauses
rytp transcripts export <video-id> [--out ...] [--min-block-s 2.0]
                                # speaker-grouped markdown transcript
rytp mine <query> [--cohesion low|med|high] [--max-clips N]
rytp splice <clip-set-id> --out out.mp4 [--mode concat|stream-copy]
rytp tui                           # launch the textual TUI
```

# Output formats

DESIGN §7 ships two output representations:

* **The ``clips`` table** — the machine-readable output of the mine
  stage. Each row is a window the mine stage picked, with
  ``video_id`` / ``start_ms`` / ``end_ms`` / ``source_query`` /
  ``created_at``. The splice stage consumes this directly.
* **Markdown transcript** (``rytp transcripts export <video-id>``) —
  a human-readable markdown file with one ``### [HH:MM:SS] Speaker``
  header per turn followed by the turn's text. Default path is
  ``data/transcripts/{video_id}.md``. Tweak ``--min-block-s`` to
  change the lower bound on per-speaker block duration; blocks
  shorter than the threshold that have a same-speaker neighbor on
  the immediate left or right are absorbed into that neighbor, so
  dense backchannel doesn't chop the output into one-block-per-word.

  Example output:

  ```markdown
  # My Video Title
  _video_id=42 • duration=0:13 • generated=2024-05-...Z_

  ### [0:00] Alice

  Hello world

  ### [0:02] SPEAKER_01

  hi

  ### [0:03] Alice

  thanks
  ```

# Quickstart

## Option A — install with pip

```bash
# 1. Install
pip install rytp                       # base install (CLI skeleton + DB)
pip install rytp[yt-dlp]               # for `rytp channel sync`
pip install rytp[stt]                  # for faster-whisper
pip install rytp[pyannote]             # for the pyannote diarizer (HF_TOKEN required)
pip install rytp[all]                  # everything above + textual

# 2. Download a channel
rytp channel add https://www.youtube.com/@example
rytp channel sync example

# 3. Queue downloads (optional — pause / resume work here)
rytp queue add 1 2 3 4 5
rytp queue worker

# 4. Transcribe (per video)
rytp transcribe 1 --stt faster-whisper --diarizer none

# 5. Add to the global speaker roster and map per-video
rytp speakers add "Alice" --alias Al --alias Алиса
rytp speakers 1                          # TUI mapper

# 6. Datamine and splice
rytp mine "world" --cohesion med
rytp splice <clip-set-id> --out data/output/result.mp4

# 7. Export a human-readable transcript
rytp transcripts export 1
# → data/transcripts/1.md
```

## Option B — run from a source checkout (no `pip install`)

If you'd rather not install the package, you can run the CLI straight
from the source tree. From the repo root (``D:\Work\rytp`` or
wherever you cloned it):

```bash
python -m rytp --help
python -m rytp channel add https://www.youtube.com/@example
python -m rytp transcripts export 1 --out /tmp/video-1.md
```

The ``__main__`` hook is the same code as the installed ``rytp``
console script, so the two are interchangeable. Optional-dependency
extras (``yt-dlp``, ``faster-whisper``, ``pyannote``, ``textual``)
still need to be installed via pip for the corresponding features
to work — there's no way around that.

## Option C — work entirely offline (no yt-dlp, no YouTube)

The pipeline doesn't require YouTube. If you already have media on
disk — say, files from a manual ``yt-dlp`` run that produced split
streams, or anything ffmpeg can read — register them directly and
skip the network step entirely:

```bash
# Register a single video with a separate audio file (typical of
# `yt-dlp -f "worstvideo+bestaudio"` without ffmpeg to merge):
rytp videos add "/path/to/video.mp4" \
    --audio "/path/to/audio.webm" \
    --youtube-id sample_youtube_id \
    --title "Смерть чиновника"

# Or register a single already-merged local file:
rytp videos add "/path/to/merged.mp4"

# Then run the same downstream pipeline:
rytp transcribe 1 --stt faster-whisper --diarizer none
rytp transcripts export 1
```

See [Separate audio/video files](#separate-audiovideo-files) for
the full pair-registration story and what ``--audio`` does.

# Data directory

By default the SQLite database and the audio / output trees live
under ``./data`` relative to the current working directory. Set
``RYTP_DATA`` to relocate them:

```bash
# Linux / macOS
RYTP_DATA=/var/lib/rytp python -m rytp --help

# PowerShell
$env:RYTP_DATA = "C:\rytp-data"; python -m rytp --help
```

Layout under ``$RYTP_DATA`` (or ``./data`` if unset):

```
rytp.db
media/          # downloaded media (yt-dlp's output)
audio/          # canonical 16 kHz mono WAV per video (extracted)
output/         # splice stage intermediates + final output
  normalized/   # per-clip loudnorm intermediates
transcripts/    # markdown transcripts (transcripts export)
```

# TUI

The textual-based TUI currently exposes the videos table and the
speakers table as read-only screens. Wiring every CLI subcommand
from §6 into the TUI is **v2** — see [DESIGN §6](./DESIGN.md#6-cli-surface)
for the full list. The TUI is launched with:

```bash
rytp tui          # installed
python -m rytp tui # source checkout
```

For anything that doesn't have a TUI screen yet, the CLI is the
canonical interface.

# Subcommand status (v1)

All 21 subcommands from §6 are wired in v1. The CRUD subcommands
(channel / videos / queue / speakers / transcripts) just work; the
heavy-lift subcommands (download / transcribe / mine / splice / tui /
speakers map) require their respective optional extras
(`pip install rytp[yt-dlp]`, `rytp[stt]`, `rytp[pyannote]`,
`rytp[all]`) to actually run, and otherwise print a one-line
install-hint error to stderr.

Quick reference:

* **Channels:** `channel add` / `sync` / `list`
* **Videos:** `videos add` / `list`
* **Download:** `download` (single), `queue add` / `worker` / `pause` / `resume` / `list`
* **Transcribe:** `transcribe` (per video)
* **Speakers:** `speakers add` / `list` / `recompute-pauses` / `map`
* **Mine:** `mine <query>`
* **Splice:** `splice <clip-id> --out out.mp4 [--mode concat|stream-copy]`
* **TUI:** `tui`
* **Transcripts:** `transcripts export <video-id>`

Run `python -m rytp <subcommand> --help` to see the exact options
for any of these.

# Separate audio/video files

yt-dlp can download audio and video streams as separate files when
the format selector contains a ``+`` (e.g.,
``worstvideo[height=720]+bestaudio[language=ru]``). rytp handles
this transparently:

* **Download stage**: detects when audio/video are downloaded
  separately and merges them into a single container using ffmpeg.
  The merged file is stored as ``videos.downloaded_path`` and the
  original audio file path is stored as ``videos.downloaded_audio_path``.
* **Audio extraction**: when a separate audio file is present, the
  ``extract_audio`` stage reads from the audio file instead of
  extracting from the video file. This is faster and avoids
  re-decoding the video stream.
* **Splice (stream-copy mode)**: uses the merged video file for
  video stream extraction.

To use a format selector that downloads separate streams, just
use it as-is with ``rytp download`` — the merging happens
automatically. For example:

```bash
rytp download <video-id> --format "worstvideo[height=720]+bestaudio[language=ru]"
```

The ``--format`` flag (also ``-f``) defaults to the same selector
used in this README's sample yt-dlp run; pass any other selector
to override it.

## Registering an existing audio+video pair

If you already have separate audio and video files on disk
(downloaded manually, from a previous yt-dlp run without ffmpeg,
or pulled out of an existing archive), you can register both at
once with ``rytp videos add``. Pass the video file as the argument
and the audio file with ``--audio``:

```bash
rytp videos add "Смерть чиновника [sample_youtube_id].f311.mp4" \
    --audio "Смерть чиновника [sample_youtube_id].f251-1.webm" \
    --youtube-id sample_youtube_id \
    --title "Смерть чиновника"
```

The row lands in the ``videos`` table with ``source='local'``,
``downloaded=1``, and both paths populated — the same shape as a
freshly-downloaded YouTube row produced by ``rytp download``.
Subsequent ``rytp transcribe`` invocations read from the audio
file directly, skipping the video decode.

The ``--youtube-id`` flag is what makes re-registration idempotent:
passing the same id twice updates the same row instead of
inserting a duplicate.

# Re-running transcribe

Re-running ``rytp transcribe <video-id>`` on the same video will
**replace** the previous transcript with the new one, not append to
it. This prevents duplicate words from appearing in the transcript
export when you re-run the command (e.g., after changing engines
or parameters). The ``transcribe_runs`` table keeps a record of
every run for audit purposes, but only the most recent run's words
are used by downstream stages like mine and transcripts export.

# Speaker diarization

rytp supports three diarization backends:

* **none** (default) — assigns all words to ``SPEAKER_00``. Fast,
  no external dependencies, no HF token required.
* **energy** — simple energy-based speaker diarization. Detects
  speaker changes by analyzing audio energy patterns, zero-crossing
  rate, and spectral characteristics. No external dependencies or
  HF token required. Includes a ``--diarizer-sensitivity`` parameter
  to control how aggressively the diarizer splits speakers.
* **pyannote** — state-of-the-art neural diarization using
  pyannote-audio 3.x. Requires ``pip install rytp[pyannote]`` and
  an ``HF_TOKEN`` environment variable.

## Using the energy diarizer

The energy diarizer is a good middle ground when you want some
speaker separation but don't want to set up pyannote. It works
by:

1. Detecting speech segments using energy thresholding
2. Computing voice features (energy, zero-crossing rate, spectral
   centroid) for each segment
3. Clustering segments by feature similarity

You can adjust the ``sensitivity`` parameter (0.0 to 1.0) to
control how aggressively the diarizer splits speakers:

```bash
# Conservative: fewer speakers, may merge similar voices
rytp transcribe 1 --diarizer energy --diarizer-sensitivity 0.0

# Balanced (default)
rytp transcribe 1 --diarizer energy --diarizer-sensitivity 0.5

# Aggressive: more speakers, may split a single speaker
rytp transcribe 1 --diarizer energy --diarizer-sensitivity 1.0
```

**Limitations**: The energy diarizer is not as accurate as pyannote
or other neural approaches. It works best when speakers have
distinct voice characteristics (different pitch, volume, or
speaking style) and there are clear pauses between speaker turns.
For interview-style content with two distinct speakers, a
sensitivity of 0.0-0.3 usually works well.

# Tests

```bash
pytest -q
```

The test count sits at 215 (3 skipped) and covers the DB / models /
config / CLI gate, channels + download + queue, transcribe
(extract + chunking + faster-whisper), diarize (none + energy +
pyannote), the merger + transcribe pipeline, the speaker roster +
pause-stats, mine + spectrogram, splice + loudnorm + manifest,
the markdown transcript export, the wired v1 CLI subcommands, the
heavy-lift download pipeline (``download_one`` +
``run_queued_item`` with a fake ``YtDlpRunner``), the
audio/video merge inside the download stage, and the
local-file-pair registration path (downloaded-but-not-via-rytp
files registered with ``rytp videos add VIDEO --audio AUDIO``).
Integration tests that exercise the user's bundled
``sample_youtube_id.f311.mp4`` + ``sample_youtube_id.f251-1.webm`` pair
skip cleanly when the files aren't present.

Tests run with no external services (ffmpeg, faster-whisper,
pyannote, yt-dlp, textual are all optional and mocked); the
real-files integration test requires ffmpeg on PATH and the
bundled files.