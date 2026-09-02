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

# Tests

```bash
pytest -q
```

The test count sits at 165+ (3 skipped) and covers the DB / models /
config / CLI gate, channels + download + queue, transcribe
(extract + chunking + faster-whisper), diarize (none + pyannote),
the merger + transcribe pipeline, the speaker roster +
pause-stats, mine + spectrogram, splice + loudnorm + manifest,
and the markdown transcript export. They run with no external
services (ffmpeg, faster-whisper, pyannote, yt-dlp, textual are
all optional and mocked).