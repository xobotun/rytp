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

The goal above is implemented. `CLAUDE.md` is the working guide to the code;
the binding interfaces and the reasoning behind them live in
`docs/superpowers/specs/`.

Stages talk only through SQLite (`data/rytp.db`) plus the `data/` tree, which
is what makes the queue, pause/resume and self-healing work. Every command is
defined once in `rytp/commands/` and generated onto both surfaces — the Typer
CLI and the Textual TUI — from that one definition.

```
  ingest ──► download ──► captions ──► caption_words ─┐
     │           │                                    ├──► index ──► search
     │           └──► extract_wav ──► transcribe ──► align ──► fingerprint
     │                                                 │
     │                                            diarize ──► speakers map
     ▼                                                 │
  job queue (network / gpu / cpu pools)                ▼
                                          assemble ──► render
```

Three things worth knowing before reading the code:

- **`words.source` is `caption | timed | aligned`, and only `aligned` can be
  cut.** Downloaded auto-captions make the whole corpus searchable for no GPU
  time; a transcriber's own word timestamps are good text and unusable
  boundaries; forced alignment plus a snap to a measured energy minimum is what
  produces a cut you can hear.
- **One word row holds exactly one token**, so `кто-то` is two rows with a
  measured boundary between them — because a search query is normalised the
  same way the corpus was.
- **Speaker identity is two-level**: per-video diarizer labels in
  `video_speakers`, a global roster in `speakers`, and linking them is a human
  step (`rytp speakers map`, or `F3` in the TUI).

# Getting started

```bash
scripts/bootstrap.sh          # bootstrap.ps1 on Windows — creates .venv, installs .[dev]
python -m rytp --help
python -m rytp tui
```

Optional extras — `yt-dlp`, `stt`, `pyannote`, `redimnet`, `all` — are
installed separately; anything needing one prints a one-line install hint
rather than a traceback. `RYTP_DATA` relocates the data tree (default
`./data`), and `HF_TOKEN` is needed only by the pyannote diarizer.

`rytp doctor` reports what is missing.

# Tests

```bash
python -m pytest
```

No test touches the network and no model library is required to run the suite.
That also means a green run does not prove the transcriber, aligner, diarizer
or embedder adapters work against the real libraries — they never have been.
