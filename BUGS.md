# BUGS

Running log of problems found while driving rytp by hand. Nothing here is
diagnosed or fixed yet — the point is to accumulate reports and deal with
them in a batch.

Placeholders only: `<url>`, `VIDEO_A`, `https://example.invalid/...`. Never
paste a real video id, channel id or URL into this file.

---

## Fix first

Ordered by what they cost, not by when they were found.

1. **Entry 36 — the wav2vec2 aligner does not actually align.** Every
   `aligned` word it produced has a fabricated, evenly-spaced boundary. This
   invalidates every alignment run so far and is the cause of the bad cuts.
   Nothing downstream can be judged until it is fixed.
2. **Entry 26 — a wav2vec2-aligned corpus cannot be assembled at all.**
   Silent: an empty result reads as "the phrase is not in the corpus". No
   flag works around it.
3. **Entry 28 — the hack for entry 26 inverts the quality ordering** and is
   already applied to the live database, so assembly currently prefers the
   worst-aligned words. Fixing 26 properly is what retires it.
4. **Entry 20 — any status message containing square brackets crashes the
   TUI.** Needs no unusual input; five bare words in the argument field will
   do it.
5. **Entries 3, 10, 22 — nothing reports that it is working.** Not broken,
   but it is why a download, a model fetch and a network call are all
   indistinguishable from a hang. Best solved once, at the seam every engine
   passes through, rather than four times.

The CTC short-chunk failure that was first here has moved to `TODO.md` — it
needs a chunking change rather than a patch, so it is a piece of work rather
than a fix. It still destroys a transcription when it fires; the workaround
until then is to transcribe without `--aligner` and align as a second step.

Everything below is the full log, in the order it was found.

---

## 2026-09-24 — first real run on the Windows box

Environment: fresh `data/` tree (the pre-rewrite one renamed to `data_old`),
`doctor` green, first time anything has actually been downloaded.

### 1. `fetch-video <url>` reports "no video matches `<url>`"

Ran `rytp fetch-video <url>` with a plain video URL and got
`no video matches '<url>'`.

`ingest <url>` behaves the same way.

Registering first worked: `videos add <url>` accepted the URL and registered
it under catalog id `1`, after which the id could be used.

Expected: either accept a URL directly, or say so in the error — the current
message does not hint that the video has to be in the catalog first, or that
`videos add` is the way to put it there.

*Where to look:* `resolve_video_id` in `rytp/commands/__init__.py:394`
matches a row id or an `external_id`, nothing else.

### 2. The `video` parameter description is misleading

Both commands describe the parameter as "Catalog id or external id of the
video". That is accurate but does not tell you a URL is rejected, nor what to
do about it. Needs rewording.

### 3. `ingest` and `fetch` look like they hang

Both sat silently for a couple of minutes with no output. Indistinguishable
from a hang.

Expected: progress reporting — what stage it is on, what it is downloading,
how far along it is.

### 4. `rytp index build` returns "nothing to index" with no next step

Ran `rytp index build` and got `nothing to index` and nothing else.

Expected: say *why* there is nothing (no transcribed videos yet?) and what to
do next.

*Where to look:* `rytp/commands/search.py:58`.

### 5. A missing optional extra gives a traceback, not an install hint

`rytp transcribe run <id>` with the `gigaam` extra not installed printed a
full child-process traceback ending in
`ModuleNotFoundError: No module named 'gigaam'`.

CLAUDE.md states the opposite contract: "Commands needing a missing extra
print a one-line install hint instead of a traceback." The hint does not
survive the out-of-process seam — the failure is raised inside the child in
`rytp/transcribe/subproc.py`, and the parent re-raises it verbatim.

Expected: `install the gigaam extra: pip install -e ".[gigaam]"`, one line,
no traceback.

*Where to look:* `rytp/transcribe/subproc.py:147` and
`rytp/transcribe/engines/gigaam.py:105`.

### 6. `--help` group headings carry no information

A group is introduced as "Commands in ... group". Wanted instead: two or
three words saying what the section is *about*, followed by a comma-separated
list of the commands in it, so the shape of the surface is visible at a
glance without drilling into each group.

### 7. `transcribe engines` reports `interpreter ok` for an engine that cannot run

`transcribe engines` listed `gigaam ... interpreter ok` while
`transcribe run 1` failed on `ModuleNotFoundError: No module named 'gigaam'`.
`mfa` and `wav2vec2` report the same state and are probably in the same
position.

This is currently deliberate — `check_available` in
`rytp/transcribe/registry.py` skips the module check for out-of-process
engines, on the reasoning that `find_spec` in *this* interpreter says nothing
about the child's. The consequence is that the one pre-flight command whose
job is to save you a failed run does not do so.

Expected: probe the configured interpreter (`python -c "import gigaam"`) and
report a state that means "this will actually run". Until then the column
reads as a promise it does not make.

**Confirmed since:** every engine that reported `interpreter ok` fails when
actually invoked — `gigaam` (no module), `mfa` (`FileNotFoundError:
[WinError 2]`, the binary is not installed), `wav2vec2` (`No module named
'torch'`). Three for three. The only engine whose state was truthful is
`whisper`, which is checked in-process. `pyannote` too (`No module named
'pyannote'`), and `redimnet` (`No module named 'redimnet'`). **Five for
five:** every out-of-process engine advertised itself as available and none
of them can run.

### 8. Settings have no CLI surface

`default_transcriber` and `engine.interpreter.<name>` are read from the
`settings` table, and `set_setting` exists in `rytp/db/queries.py:238`, but no
registered command writes one — there is no `settings` group among the 55
commands.

That makes the documented out-of-process workflow unreachable: pyproject
tells you to "point `engine.interpreter.gigaam` at the Python inside its own
virtual environment", and there is no supported way to do it.

### 9. `pip install -e ".[gigaam]"` fails on Python 3.14 / Windows

`sentencepiece` (a gigaam dependency, pinned `<=0.2.0`) has no cp314 Windows
wheel, so pip falls back to building from the sdist. Its build shells out to
a tool that is not present and dies with
`FileNotFoundError: [WinError 2] The system cannot find the file specified`,
then `ERROR: Failed to build 'sentencepiece'`.

Not a rytp defect — an ecosystem gap — but it blocks the recommended
transcriber on the target machine, and entry 8 blocks the designed
workaround. Options: a separate Python 3.12 environment for gigaam (which is
what `out_of_process` exists for), or use `whisper`, which reports `ready`.

### 10. First-run model download reports no progress

`transcribe run 1 --transcriber whisper` printed two warnings and then went
quiet for a long time. It was in fact downloading the `large-v3` weights —
several GiB — in the background, indistinguishable from a hang.

Same family as entry 3, but a distinct cause worth its own fix: this one is a
first-run, one-off, multi-gigabyte fetch. A user who does not know that is
happening will kill it.

Expected: say the model is being downloaded, which one, and how far along.

**Confirmed since, and broader than first written:** the download is only the
first-run case. A normal `transcribe run` is silent for its entire duration —
the owner reports being able to tell it is working *only* from the GPU usage
graph.

The progress unit already exists and costs nothing to surface. The pipeline
splits audio into chunks before transcribing (`plan_chunks`), and the
completion table already reports the count — the run in entry 13 printed
`chunks = 38`. "chunk 12/38" is available at the point work happens; nothing
computes it because nothing asks.

Same for the aligner and diarizer once either can run: both iterate the same
chunk list.

**Confirmed on the aligner too.** Once `engine.interpreter.wav2vec2` pointed
at a Python 3.12 environment, `transcribe align 1 --aligner wav2vec2`
downloaded its model silently in the background, exactly as whisper did. So
this is not one adapter's omission — it is every engine, and it is the first
thing a user meets on each one. Whatever gets built should live at the seam
all engines pass through, not in each adapter.

### 11. Hugging Face symlink warnings reach the user twice

The same `huggingface_hub` `UserWarning` about symlinks not being supported
was printed twice, each with a paragraph of advice about Windows Developer
Mode. It is noise from a dependency, not something rytp is asking the user to
act on.

Expected: suppress it (`HF_HUB_DISABLE_SYMLINKS_WARNING=1` set by rytp), or
surface it once as a single advisory line. Note the underlying condition is
real — without symlinks the HF cache stores duplicates and uses more disk,
which matters on a machine meant to hold a video archive.

### 12. The TUI home screen does not look like it runs anything

Reported as "should we be able to run commands on the first screen, or is it
documentation-only?" — it does run them, which means the screen fails to say
so.

Current behaviour, for the record: the palette runs a short command outright,
queues a long-running one that has an `enqueue`/`queue` flag (then points at
F7), and refuses the eight in `FOREGROUND_ONLY` with a reason and a
copyable command line.

Expected: the home screen should make the live path obvious — that typing a
command and pressing Enter executes or queues it, and that arguments go in
the field below. Related to entry 6: both are about the surface not
describing itself.

Also noticed while reading `navigation.py`: the screen keys are F3, F4, F6,
F7, F8 — F5 is unused. Deliberate, or a gap?

### 13. "median align score 1.00" on a run with no aligner

`transcribe run 1 --transcriber whisper` reported:

    video  words  chunks  tier   engine          median align score
    1      1519   38      timed  whisper+energy  1.00

The same row says `tier = timed` — explicitly *not* aligned, not cuttable —
and `median align score = 1.00`, which reads as a perfect alignment. Both
cannot be true.

The number is real, not a placeholder: with `--refine` on, `_score_boundaries`
in `rytp/transcribe/pipeline.py:166` scores how cleanly each boundary lands in
a measured energy minimum, and those scores populate the column. That is a
*boundary* confidence, not an alignment score, and `engine = whisper+energy`
already says so.

Expected: name the column for what produced it (boundary score / energy
score), or blank it for tiers where no aligner ran. As it stands the one
column a user would check to decide "is this good enough to cut?" says yes
while the tier column says no.

**Partly explained by entry 36:** the uniformity was an artefact. Every word
in a chunk receives the same score by construction, because the aligner
assigns the mean of all frames to all words. The scale problem below is real
and independent, but the suspiciously perfect numbers were this.

**Worse than first written: the column has no unit.** The first real
alignment run reported

    video  words  chunks  tier     engine                   median align score
    1      1519   38      aligned  whisper+wav2vec2+energy  -0.96

against the same column that read `1.00` for `whisper+energy`. The two
numbers come from different producers on different scales:
`_score_boundaries` yields a bounded 0–1 energy measure, while
`torchaudio.functional.forced_align` yields **log-probabilities**
(`rytp/transcribe/align/wav2vec2.py:113`), so −0.96 means roughly 0.38 in
probability. MFA, per contracts §6, reports no score at all.

So `words.align_score` currently holds at least three incommensurable
things, and the report prints them in one column as if they were comparable.
A user cannot read it, and cannot threshold on it.

**Decided (2026-09-24): record the scale alongside the score.** Store what
produced a score next to the score itself, rather than normalising different
producers onto an invented common scale — normalising would be lossy and
would bake in a conversion nobody can justify. Tagging is honest: an energy
boundary score and a wav2vec2 log-probability stay themselves, and the report
can label or group by scale instead of printing them in one column as though
they were comparable.

Needs a `words` column (the schema is append-only, so a new migration tuple)
and therefore a contracts change. MFA writing no score at all stays valid —
contracts §6 already makes `Span.score` optional.

This also removed the blocker on D1, which no longer thresholds on the score
at all.

### 14. The home screen's interaction model is unreachable by keyboard

Three separate reports, one underlying shape: the intended flow exists but
the obvious keys do not reach it.

**14a. Arrow keys do not move through the command list.** `compose` yields
`#filter` (an `Input`) before `#palette` (an `OptionList`), so Textual puts
initial focus in the filter, where Up/Down move the text cursor. The
`OptionList` only receives arrows once it has focus, and nothing forwards
them to it. Expected: Up/Down from the filter should move the highlight,
the way every command palette behaves.

**14b. It is unclear whether to type `videos.add` or `videos add`.** Both
spellings are on screen at once: the palette lists `entry.name`, which is the
dotted `videos.add`, while `usage_line` in `rytp/tui/palette.py:53` renders
`cmd.name.replace(".", " ")` → `videos add`. The CLI takes the spaced form.
Expected: one spelling in the TUI, or a word saying the filter is a filter
and not a place to type a command.

**14c. Enter appears to do nothing.** `on_input_submitted` acts only when
`event.input.id == "arguments"` (`rytp/tui/app.py:166`). Enter in the filter
— where focus starts — is silently discarded: no run, no status line, no
message. The first Enter a new user presses does nothing, invisibly.
Expected: either run the highlighted command, move focus to the arguments
field, or say what Enter is waiting for.

The flow the code intends is: filter → focus the list → select (which moves
focus to the arguments field) → type arguments → Enter. Only the last two
steps are reachable with the keys a user would try first.

### 15. `transcript show` needs a line-length knob

Requested: `--line-length 80` (or similar) so the transcript wraps to a
chosen width instead of whatever the table decides.

`transcript.show` currently takes one parameter, `--video`, and nothing else.
`transcript.build` is in the same position — worth deciding whether the knob
belongs on both, since one writes the markdown file and the other renders a
table.

Confirmed working, for the record: Tab does reach the command list on the TUI
home screen, and the flow works once you know it. The problem is only that
nothing tells you — see entry 14.

### 16. An unknown engine name escapes the error funnel as a traceback

`transcribe align 1 --aligner wav2vec` (a typo for `wav2vec2`) printed a
full rich-rendered traceback, ~60 lines, ending in:

    ValueError: unknown aligner 'wav2vec'; available: mfa, wav2vec2

The message is exactly right. The delivery is not. `rytp/cli.py:169` funnels
errors through `except RytpError`, with the comment "Contracts §8: one line,
exit 1, never a traceback" — but `resolve_aligner` and `resolve_transcriber`
in `rytp/transcribe/registry.py:57,68` raise plain `ValueError`, which the
funnel does not catch.

Expected: `unknown aligner 'wav2vec'; available: mfa, wav2vec2`, one line,
exit 1. The fix is the exception type, not the message.

**A third site, same bug:** `speakers diarize 1 --diarizer pyannot` (another
typo) produced the identical traceback from `resolve_diarizer` in
`rytp/diarize/base.py:72`. So the pattern is at least three functions across
two packages — `resolve_transcriber`, `resolve_aligner`, `resolve_diarizer`
— and is systemic rather than an oversight in one spot. Audit every
non-`RytpError` raise on a command path when fixing this.

Related to entry 5, which is the same contract broken by a different route —
that one crosses the subprocess seam, this one never leaves the parent.

### 17. No aligner can run on the target machine

For the record, since it blocks cutting entirely:

- `mfa` — `FileNotFoundError: [WinError 2]`; the Montreal Forced Aligner
  binary is not installed (it is a conda package, not a pip one).
- `wav2vec2` — `ModuleNotFoundError: No module named 'torch'`.

`torch` has no cp314 Windows wheels yet, the same wall `sentencepiece` hit in
entry 9. So on Python 3.14 there is no path to `aligned` words, which means
no cutting, which means no render.

The same holds beyond aligners: `pyannote` (diarizer) and `redimnet`
(embedder) both fail on missing modules, and both want torch. So speaker
identity — the `speaker + word` half of the project's stated goal — is
blocked by the same wall as cutting. Whisper is the only engine of any kind
that runs on this machine today.

This makes entry 8 the critical one: the designed escape hatch is a separate
Python 3.12 interpreter per engine, and there is no supported way to point
`engine.interpreter.*` at one.

### 18. The arguments field should show the selected command's shape

The `#arguments` input keeps a fixed placeholder, "arguments: value
name=value …", whatever is highlighted. It should show what *this* command
takes.

Cheap to fix: `usage_line` in `rytp/tui/palette.py:51` already renders exactly
that — `videos add <url> [audio=…]` — and `PaletteEntry.usage` already
carries it. Nothing computes it; the placeholder simply never updates.
Set it in `on_option_list_option_selected` (`rytp/tui/app.py:170`), which
already fires on selection.

### 19. The TUI never says what `aligned` and `timed` mean

The tier vocabulary is load-bearing and unexplained. The search screen has a
"Cuttable only" filter on Ctrl+T and prints "· cuttable only" in its status
line (`rytp/tui/screens/search.py:119,141`), with nothing anywhere saying
what makes a word cuttable or why some are not.

Expected: a short glossary in the TUI — `caption` = downloaded subtitles,
searchable, no end times, never cuttable; `timed` = the transcriber's own
word timestamps, good text, boundaries not trustworthy; `aligned` = forced
alignment plus an energy-minimum snap, the only tier that can be cut. F1 is
the obvious home, with a one-line hint wherever a tier or the word
"cuttable" appears.

Becomes more important once D1 lands: with `--allow-timed`, a user has to
understand the difference to make the choice the flag offers.

### 20. The TUI crashes whenever a status message contains square brackets

**This is a crash, and it needs no unusual input.** Reported after typing a
local file path with spaces, but the filename is irrelevant.

`set_status` (`rytp/tui/app.py:158`) hands arbitrary text to
`Static.update`, which parses it as Textual console markup. `usage_line`
spells optional parameters `[title=…]`, and square brackets *are* markup
syntax. So any error carrying a usage line kills the app:

    MarkupError: Expected markup value (found '…] [kind=…] [channel=…]').

Verified locally. `parse_arguments(COMMANDS["videos.add"], "a b c d e")`
raises the identical message that crashed the session — five bare words, no
Cyrillic, no punctuation, no spaces-in-path needed. Any user who over-fills
the arguments field on a command with optional parameters hits this.

Also verified, both fixes work:

- `textual.markup.escape(msg)` before `update` — renders fine.
- constructing `Content(msg)` without markup parsing — renders fine.

Note Rich's own parser tolerates the string; Textual 8 ships a stricter one,
so a test written against `rich.markup` would pass while the app still
crashes. Test against `textual.content.Content.from_markup`.

Wider than errors: `show_result` (`rytp/tui/app.py:154`) routes
`result.message` through the same call, so any command *result* containing
brackets crashes identically. Check `DataTable` cells for the same exposure.

### 21. Unquoted paths in the TUI argument line: split, or silently corrupted

The trigger for entry 20. Typing an unquoted path with spaces produces "too
many values", because each word becomes another positional.

**Quoting does work** — `parse_arguments` (`rytp/tui/palette.py:133`) uses
`shlex.split`, so `"sample_inputs/«имя файла» с пробелами.mp4"` parses to a
single token. Nothing says so: the placeholder reads "arguments: value
name=value …" and never mentions quotes. Fixing entry 18 (show the command's
real usage shape) would go most of the way.

**The serious half is Windows paths.** `shlex.split` runs in POSIX mode,
where backslash is an escape character, so an unquoted native path is
silently mangled rather than rejected:

    D:\Work\rytp\sample_inputs\file.mp4   ->   D:Workrytpsample_inputsfile.mp4

No error. The path is quietly corrupted, and the user gets "… is not a file"
naming a path they never typed. Quoted, it survives intact.

That matters more than the spaces case: it is silent, it is wrong rather
than merely refused, and it happens on the target platform for the most
natural thing a user can type. Options are `posix=False`, a Windows-aware
split, or treating a single trailing positional as the rest of the line
verbatim. A file picker for `videos.add` would sidestep all of it.

### 22. The TUI freezes while a "short" command does network I/O

`videos.add` via the TUI locks the whole interface for seconds before
reporting success.

Cause: `videos.add` is registered `long_running=False`, so `run_selected`
calls its handler inline on Textual's event loop
(`rytp/tui/app.py:141`) — nothing yields, so nothing repaints. But given a
URL the handler calls `_probe_video` (`rytp/commands/catalog.py:125`), which
shells out to yt-dlp for a metadata request. A network round trip on the UI
thread.

The flag is not wrong, it is answering a different question. `long_running`
classifies whether a command has a *queued* form and belongs to the worker;
it says nothing about whether the handler blocks. `videos.add` is correctly
not-queueable — cataloguing is one metadata call, not per-video work — and
still blocks for seconds.

Expected: the TUI should run every handler in a Textual worker rather than
inline, and show that something is in flight. Same family as entries 3 and
10 (no progress anywhere), but this one additionally freezes the interface
rather than merely being quiet.

Worth checking for siblings while fixing: any non-`long_running` command
that touches the network or the disk has the same shape. `channel.add` is
the obvious next candidate — it resolves a channel URL.

### 23. Feature: a "Videos" screen showing each video's pipeline state

Requested: a new TUI screen listing videos, filterable, with shortcuts to
add, download, transcribe, diarize and align — so that the path from
"registered" to "cuttable" is visible and no step is silently skipped.

This is the missing overview. Every other screen answers a question about
one thing (a transcript, a search, a cut list); nothing shows where each
video *is*.

Most of what it needs already exists:

- **Stage state comes free from readiness.** Every job kind declares a
  `readiness(db, target_id) -> READY | BLOCKED | SATISFIED` predicate that is
  a statement about the world, derived from database and filesystem state.
  Nine of the ten kinds target a video: `download`, `captions`,
  `extract_wav`, `caption_words`, `transcribe`, `align`, `fingerprint`,
  `index`, `diarize`. Rendering one column per kind gives the pipeline view
  with no new logic and no risk of a second, disagreeing definition of
  "done".
- **The actions are registered commands.** `ingest`, `transcribe run
  --enqueue`, `transcribe align --enqueue`, `speakers enqueue`,
  `index build`. A shortcut should call the same handler the palette calls,
  not a parallel path.
- **F5 is unbound** (entry 14 notes the screens use F3, F4, F6, F7, F8), so
  the key is free and sits naturally before Search.
- **Adding a screen is one row.** `navigation.SCREENS` holds `ScreenEntry`
  rows and `rytp/tui/app.py:182` notes a screen is "one row in
  `navigation.SCREENS` and no code here".

Worth deciding when it is built:

- `align` and `diarize` are `reopenable=False`, so a satisfied stage will not
  re-fire under `reconcile`. The screen should distinguish "done" from
  "cannot be redone automatically", or re-running a step from here will look
  broken.
- Whether the screen shows tier (`caption` / `timed` / `aligned`) per video.
  It is the single most useful column for "is this cuttable yet", and it is
  where the glossary of entry 19 most needs to appear.
- Whether shortcuts enqueue or run. Enqueue is consistent with the palette's
  behaviour for long-running commands and keeps the UI responsive — which
  entry 22 shows matters.

### 24. The TUI argument line does not accept CLI flag syntax

`fetch-video` with `1 --captions` typed into the arguments field gives "too
many values for fetch-video" — and then crashes via entry 20.

The TUI's grammar is `value` for positionals and `name=value` for named
parameters. `--captions` contains no `=`, so it is taken as a second
positional, and `fetch-video` has only one. The CLI spelling for the same
thing is `--captions`; the TUI spelling is `captions=true`. Nothing marks
the difference, and a user who has spent the day in the CLI will type the
CLI form.

Expected: accept `--name`, `--name=value` and `--no-name` as synonyms for
`name=value` in the argument line. The parser already normalises
`PARAM_ALIASES` by stripping leading dashes
(`rytp/tui/palette.py:129`), so the shape is half there — it just is not
applied to typed input.

Failing that, entry 18's usage-shape placeholder at least shows `[captions=…]`
before the mistake is made.

### 25. Feature: pipeline state and engines in `videos list`

Requested: show per video whether it is downloaded / transcribed / timed /
aligned, and which engines produced that, keeping today's columns available
as `videos list --short`.

The CLI sibling of entry 23, and it should share that entry's data sources
rather than grow a second opinion about what "done" means:

- **Downloaded** — `assets` rows for the video. Note assets are never
  merged, so audio, video renditions, captions and container are separate;
  "downloaded" needs defining as which of those must be present.
- **Tier** — `words.source`, one of `caption | timed | aligned`. This is the
  column that answers "can I cut this yet", so it is the one worth showing
  most.
- **Engine** — `words.engine` already stores a tag per word row, and it
  records the chain rather than one name: the whisper run in entry 13 wrote
  `whisper+energy`. So "what engines were used" is a `DISTINCT engine` away,
  and it already answers the alignment question too, since an aligned run
  tags the aligner into the same string.
- **Everything else** — the readiness predicates, as in entry 23.

Design notes:

- **`--long` rather than `--short`** (owner's revision). Today's columns stay
  the default and the detailed view is opt-in, so nothing existing changes
  shape and no script that parses the listing breaks. It also sidesteps
  checking whether `--short` already means something elsewhere — though
  `--long` still has to clear the flag-vocabulary consistency suite, which
  asserts one spelling with one meaning across every command.

- **Columns named for the thing, ticks in the cells** (owner's revision).
  One column per asset role and per pipeline stage — `audio`, `captions`,
  `videos`, then `transcribed`, `aligned`, `indexed`, `diarized` — each cell
  a tick or a cross. One vocabulary covers both halves: a tick means the file
  is present, or the process completed. Scans down a column as easily as
  across a row, which is what makes "nothing forgotten" work.

- **Can there be more than one audio? No.** `SINGLETON_ASSET_ROLES`
  (`rytp/constants.py:266`) is `{audio, captions, container}`, and design §4
  says at most one canonical audio, at most one captions, any number of video
  renditions. It is enforced in `rytp.db.queries.insert_asset`, not by a
  unique index. The reason matters: `cache/wav/{video_id}.wav` is the single
  audio timeline every word timestamp refers to, so two audio tracks would
  mean two timelines and ambiguous boundaries. A source with several language
  tracks is therefore not representable — **changing that is a contracts
  change**, not a listing change. So the `audio` column is a tick, never a
  count; only `videos` needs one.

- **Cell renderer, decided:** `0` renders as a cross, `1` as a tick, `2` and
  above as the number itself. So a single video rendition reads as a tick
  like every other satisfied thing, and only genuine plurality shows a digit.
  One rule for every column, asset counts and stage flags alike.

- **`--longer` for the detail** (owner's request): video renditions with
  their qualities, the audio track, and the engine chain. Worth weighing
  against a `videos show <id>` detail command instead — a third ever-wider
  table is hard to read at 1.6K rows, and per-video detail is a different
  shape of question from listing. Either way the engine string is already
  a chain (`whisper+energy`), so it answers "what was used" without a join.

- **Show which assets are present, a checkbox per role is enough.** The
  vocabulary is `audio`, `captions`, `container` — at most one each, per
  `SINGLETON_ASSET_ROLES` in `rytp/constants.py:266` — plus any number of
  `video` renditions. So three checkboxes and a count reads naturally:

      audio ✓   captions ✓   container ✗   video ×2

  That also answers the "which of those counts as downloaded" question above
  by not asking it: show what is there and let the reader judge. Worth
  carrying the rendition height as well, since the owner intends to upgrade
  renditions later and the whole point of keeping assets unmerged is that a
  better one can arrive without invalidating the transcript.
- Computing tier and engine per row is a query per video unless it is folded
  into `q.list_videos` as a join. With ~1.6K videos and a default limit this
  is fine either way, but the join is the better shape and keeps the listing
  one statement.
- A video can hold words of more than one tier at once — captions ingested
  and then a transcriber run over the same video. The column has to say which
  it means, or show the best tier present.

### 26. BLOCKING: a wav2vec2-aligned corpus can never be assembled

The unit problem of entry 13, biting for real. With two videos aligned by
wav2vec2, `assemble plan` matches **no words at all**, and no flag can fix it.

The matcher requires (`rytp/assemble/match.py:142`):

    COALESCE(align_score, :default_align) >= :min_align

with `min_align` defaulting to `ASSEMBLE_MIN_ALIGN_SCORE = 0.0`
(`rytp/constants.py:452`). wav2vec2 writes `torchaudio.functional.forced_align`
log-probabilities — negative by definition, median `-0.96` on the first real
run. **Every aligned word fails `>= 0.0`.**

The escape hatch is closed too: `rytp/assemble/__init__.py:104` validates
`0.0 <= min_align_score <= 1.0`, so `--min-align -5` is rejected as invalid
input. There is no value of the flag that admits a negatively-scored word.

The filter and the constant are both coherent on their own — they assume a
0–1 score, which is what the energy scorer produces and what MFA's absent
score coalesces to. Only wav2vec2 violates the assumption, and nothing
anywhere states it.

Note the shape of the failure: not an error, an empty result. A user sees
"no fragments" and reasonably concludes the corpus does not contain the
phrase.

Fixes, in the order they should be considered:

1. Land the scale tag from entry 13, and compare per scale. The principled
   fix, and the other two become unnecessary.
2. Normalise wav2vec2's log-probability into 0–1 on write. Cheap, but it
   invents a conversion and is exactly the lossiness rejected in entry 13.
3. Widen the validation and default. Least work, but leaves `min_align`
   meaning different things on different corpora — the same trap, moved.

Until one lands, wav2vec2 alignment produces cuttable-looking words that
nothing can cut, which makes the aligner effectively useless despite
reporting success.

### 27. `assemble show` cannot list cut lists, and rejects the filename it printed

Two small things met immediately after the first successful plan.

**No way to enumerate cut lists.** `assemble show` requires a name, and
`--list` does not exist. `render list` enumerates renders; nothing
enumerates cut lists, so the only way to find one is to look in
`data/cutlists/`.

**Resolution: add `assemble list`** rather than making `show`'s argument
optional. `list` is the established verb of the vocabulary — `videos list`,
`channel list`, `render list`, `speakers list`, `jobs list` — and the
consistency suite exists to keep one spelling meaning one thing. A `show`
that lists when given nothing and displays when given something is two
meanings on one name. `DELETION_OWNERS` already pairs `assemble.remove` with
the group, so `list` is a missing verb rather than a new concept.

Output must be **copy-pasteable**: bare names, exactly what `assemble show`
and `render run` accept, not paths and not `<name>.toml`.

The missing-argument error should also point at it — "no cut list named; try
`assemble list`". Note that error currently comes from Typer before the
handler runs (`Missing argument 'name'`), so saying anything useful there
needs handling at the CLI-generation level, not in the command.

**The printed path is not accepted back.** `assemble plan` prints
`wrote …/cutlists/<name>.toml`, but `assemble show <name>.toml` appends the
extension again and fails on `<name>.toml.toml`. The command wants the bare
name. Expected: strip a trailing `.toml`, or accept a path as well as a
name. Copy-pasting what the tool just printed should work.

### 28. Negated alignment scores invert the quality ordering

Recorded because a hack for entry 26 was applied to a live database and the
side effect is easy to miss.

`UPDATE words SET align_score = -align_score WHERE align_score < 0` makes
wav2vec2 log-probabilities pass entry 26's `>= 0.0` filter, and assembly then
works. But in log-probability space **closer to zero is better**, so negation
reverses the ranking: a good `-0.1` becomes `0.1` and a poor `-3.0` becomes
`3.0`.

The matcher orders candidates by `COALESCE(align_score, :default_align) DESC`
(`rytp/assemble/match.py:190`), so after the hack it **prefers the
worst-aligned words** among near-equal candidates. It also yields values above
`1.0`, which breaks the 0–1 range the validation and constants assume.

An order-preserving, bounded alternative is `1.0 / (1.0 - align_score)`:
`0 → 1.0`, `-0.96 → 0.51`, `-3.0 → 0.25`. `exp(align_score)` is equivalent in
spirit where SQLite has math functions compiled in.

Affected rows cannot be identified after the fact — the sign is gone, so a
legitimately positive energy score is indistinguishable from a flipped
log-probability. Re-running `transcribe align` rewrites them cleanly. Worth
fixing entry 26 properly rather than leaving a database in this state.

### 29. The cut-list screen's footer is too long to fit

F8 binds twelve keys (`rytp/tui/screens/cutlist.py:104-114`) and the footer
runs off the end of the line. Recoverable only because PowerShell keeps
scrollback.

Expected: the footer should fit at a normal terminal width — group or
abbreviate the bindings, wrap to a second line, or move the full list behind
a key and keep the footer to the handful used constantly. It is the screen
with the most bindings by some margin, so whatever is chosen should be
checked against it rather than against the lighter screens.

### 30. Finer fragment trimming, and a snap-to-clean-boundary key

Requested after using the `[` / `]` and `ctrl+left` / `ctrl+right` nudges:
a modifier for finer adjustment. Proposed 10 ms default with Shift for 1 ms.

Current values: `TUI_CUTLIST_NUDGE_MS = 40`, with
`TUI_CUTLIST_COARSE_NUDGE_MS = 250` used for gaps
(`rytp/constants.py:1081`).

**40 ms is too coarse — that part is agreed.** At conversational speed it is
a good fraction of a phoneme, so trimming a word edge overshoots.

**Suggested instead: plain 10 ms, Shift 100 ms, and a separate snap key.**
Reasoning:

- 1 ms is 16 samples at 16 kHz and inaudible as a timing change. Reaching it
  by keyboard takes ~40 presses to cover one of today's steps.
- What a 1 ms nudge is actually chasing is a clicking seam, and clicks come
  from an amplitude discontinuity, not from timing precision.
- The project already solves that during alignment: boundaries snap to the
  local energy minimum and the nearest zero crossing. Exposing that as a key
  does in one press what 1 ms nudging does in forty, and does it better —
  `rytp/audio/energy.py` already holds the machinery.
- Shift as *coarser* is the more useful direction: when a fragment is wrong
  it is usually wrong by a syllable, not by a millisecond.

If 1 ms is still wanted, a three-level scheme works — plain 10 ms, Shift
100 ms, Ctrl 1 ms — but the snap key is the one that earns its place.

Note this lands on entry 29: the screen already has too many bindings for
its footer, so adding keys needs that resolved first.

### 31. The same duration prints in three different shapes

`assemble plan` reports a cut list as `0:02.340`; `render run` reports the
same kind of value as `0:00:02.700`. Same quantity, different shape, and the
only visual difference is an extra colon.

There are at least four time formatters, each individually justified:

- `format_duration` (`rytp/commands/catalog.py:173`) — `H:MM:SS`, no ms.
- `format_timecode` (`rytp/render/report.py:99`) — `H:MM:SS.mmm`, documented
  as "precise enough to seek to in a player".
- `format_clock` (`rytp/render/report.py:108`) — `M:SS`, or `H:MM:SS` past an
  hour, "the form a description wants".
- Whatever `assemble` uses for `9:54.675` and `0:02.340` — `M:SS.mmm`.

The existence of several is fine: a paste-ready description wants `M:SS`, a
seek target wants milliseconds. The problem is that two commands in one
pipeline print *the same measurement* differently, so a reader comparing a
plan against its render has to notice the colon count.

Expected: one formatter per *purpose*, named for the purpose, and the same
one used wherever that purpose appears. Worth a consistency test, in the
spirit of the flag-vocabulary suite.

Unrelated but noticed alongside: the plan says `2.700` and the container
measures `2.750`. Expected from frame and AAC-frame padding, not a defect —
noted so it is not chased later.

### 32. The out-of-process engine reloads its model once per chunk

Noticed from a CPU graph that spiked and fell repeatedly during alignment
rather than sustaining load.

`realign_video` calls `engine.align(...)` once per chunk
(`rytp/transcribe/pipeline.py:483`), and the wav2vec2 adapter's `align`
calls `run_child` (`rytp/transcribe/align/wav2vec2.py:56`), which is a plain
`subprocess.run` (`rytp/transcribe/subproc.py:83`). So each chunk gets a
**fresh Python process** that imports torch, loads the model, aligns a few
seconds of audio, writes its JSON and exits — discarding the loaded model
every time.

The first real video had 38 chunks, so the model was loaded 38 times. The
sawtooth CPU pattern is process startup and model loading; the alignment
itself is the small part. At archive scale — ~1.6K videos at tens of chunks
each — this is tens of thousands of redundant model loads.

**Confirmed by accident, and it exposes a second problem.** The adapter was
edited on disk while an alignment was running; chunks dispatched after the
edit picked up the new file and moved to the GPU, so one video was aligned
half on the CPU and half on the CUDA device. A persistent worker would have
finished on the CPU.

That proves the per-chunk spawn, but the sharper point is **a long-running
job can silently change behaviour mid-execution when its source changes.**
Harmless here — CPU and CUDA forced alignment compute the same thing — but a
semantic edit would produce one video's words from two code versions with
nothing recording the split. `words.engine` stores the engine tag, not the
code version, so the database cannot show it happened.

Worth deciding, whichever way the granularity goes: load the engine module
once per run rather than per call, so a run is at least internally
consistent. A persistent child (option 2) gets this for free.

The subprocess seam itself is right and load-bearing: design §6 keeps
engines out-of-process because the recommended transcriber and diarizer pin
incompatible dependencies. What is wrong is its granularity — one process
per *call* rather than one per *video* or per engine session.

Options, cheapest first:

1. Send the whole video's chunks in one request and let the child loop
   inside, loading the model once. Smallest change: the request already
   carries `words`, `start_ms` and `end_ms`, so it becomes a list of those.
2. Keep the child alive across calls — a request/response loop over stdin —
   which also helps the transcriber and diarizer. Bigger, and needs a
   lifecycle the current one-shot design does not have.

Note this interacts with the chunking work now in `TODO.md`: whatever
changes how chunks are formed should land alongside a decision about how
many of them cross the process boundary at a time.

### 33. Engines run on the CPU because the default torch wheel has no CUDA

The same CPU graph, plus the traceback in entry 32's sibling report naming
`forced_align\cpu\compute.cpp`, say the alignment ran on the CPU while an
RTX 3080 sat idle.

Cause is environmental rather than a defect: on Windows, `pip install torch`
takes the CPU-only wheel from PyPI. CUDA builds come from
`https://download.pytorch.org/whl/cuXXX`. So an engine venv created the
obvious way is CPU-only, and nothing says so.

rytp cannot install torch for the user, but it can stop the silence:

- `doctor` already reports the GPU (it found the 3080), and
  `transcribe engines` already probes interpreters. Either could report
  whether the engine's interpreter has a CUDA-enabled torch — one line,
  `torch.cuda.is_available()`, run in the child.
- That is the same probe entry 7 needs anyway to stop reporting
  `interpreter ok` for engines that cannot run. Worth doing once, reporting
  both facts.

Practical note for the setup docs: an engine venv wants
`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`
(matching the installed driver), not a bare `pip install torch`.

### 34. Four of the five engines never select a device, so they stay on the CPU

Found while checking whether a CUDA torch install would help entry 33. It
would not, on its own.

Grepping each adapter for `cuda` or `device`:

| Adapter | device selection |
|---|---|
| `rytp/transcribe/engines/whisper.py` | **yes** — `device: str = "auto"`, passed to the model at line 77 |
| `rytp/transcribe/align/wav2vec2.py` | none |
| `rytp/transcribe/engines/gigaam.py` | none |
| `rytp/diarize/pyannote.py` | none |
| `rytp/diarize/embed.py` | none |

Whisper is why transcription showed GPU activity. The other four construct
their models and tensors without ever moving them, so torch leaves
everything on the CPU — an RTX 3080 idles while an i7 runs neural network
inference.

**Confirmed empirically (2026-09-24):** after installing a CUDA build into
the wav2vec2 interpreter, `torch.cuda.is_available()` prints `True` and
alignment still runs on the CPU. Availability is not placement. This closes
the question that entry 33 left open — a CUDA install is necessary and not
sufficient.

This makes the job queue's `gpu` pool partly fictional: `transcribe`,
`align` and `diarize` are all declared `pool=gpu`, and the pool exists to
stop two jobs contending for one card. Three of those four kinds do not
touch the card at all, so the pool is serialising CPU work for no reason
while the GPU is free.

Expected: every adapter takes the same device parameter whisper already has,
defaulting to `auto` — use CUDA when `torch.cuda.is_available()`, else CPU —
and reports which it chose, because "silently ran 40× slower" is exactly the
failure this project keeps producing.

Related: entry 33 (the interpreter may have no CUDA build at all) and entry 7
(`transcribe engines` reports availability it has not checked). All three want
the same probe in the child: does torch import, is CUDA available, which
device was selected.

### 35. Nothing documents how to install an engine so it can use the GPU

The setup instructions do not mention the CUDA wheel, and following them
produces engines that cannot use the card this project was designed around.

Three gaps, in the order a new user meets them:

1. **The extras install a CPU-only torch.** `pip install -e ".[wav2vec2]"`
   resolves `torch>=2.1` from PyPI, and on Windows that is the CPU build.
   Same for `.[redimnet]` and anything else pulling torch. So the documented
   install path yields an engine that silently runs on the CPU — entry 33.
   The GPU build needs an explicit index:

       pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

   with the `cuXXX` matching the installed driver.

2. **The separate-interpreter workflow is undocumented outside a code
   comment.** `pyproject.toml` explains for pyannote and redimnet that they
   should live in their own environments pointed at by
   `engine.interpreter.<name>`, but no user-facing document says how, and
   until today no command could even write that setting. On Python 3.14 this
   is not optional — it is the only way any of these engines run at all
   (entry 9, entry 17).

3. **There is no accurate user-facing setup document to put this in.**
   `README.md` describes the pre-rewrite tree and is stale; `CLAUDE.md` is
   written for an agent, not a person. So the note has nowhere to go until
   one exists.

Worth pairing with a `doctor` check, since documentation is only read once:
`doctor` already reports the GPU and already probes interpreters, so it can
say "engine wav2vec2: torch present, CUDA not available" and name the fix.
That is the same probe entries 7, 33 and 34 all want.

### 36. CRITICAL: the wav2vec2 aligner does not align — it divides the span equally

Found chasing "when I assemble stuff, it produces a weird cut … it feels like
it takes entirely different fragments". It does.

`_word_frames` (`rytp/transcribe/align/wav2vec2.py:123`) is documented as
grouping "the character-level CTC path back into one entry per word … every
non-blank run between space tokens belongs to the next word in order". It
never does that. **Nothing in its loop ever appends to `out`** — the trailing
`if len(out) < len(words) and … : continue` is a no-op at the end of the loop
body. So `out` is always empty when the loop finishes, and the block
commented as a *fallback* — "when the path cannot be split" — executes every
single time:

    width = max(1, (last - first + 1) // max(len(words), 1))

Each word is then handed an equal slice of the chunk's overall speech extent,
and every word receives the same score, the mean of all collected frames.

Demonstrated directly. A path where three words genuinely occupy 2, 2 and 10
frames returns:

    а      frames  0..5   score=-0.1
    бб     frames  6..11  score=-0.1
    ввввв  frames 12..17  score=-0.1

Equal widths, identical scores, no relationship to the CTC path that was
just computed.

**Consequences.**

- Every `aligned` word produced by wav2vec2 has an arbitrary boundary. The
  tier's whole promise — "only `aligned` may be cut", because its boundaries
  are measured — is false for this aligner. Cutting yields fragments that
  start and end mid-word, which is the reported symptom.
- It is arguably **worse than `timed`**: the transcriber at least attempts to
  say where each word is, while this discards that and substitutes equal
  division.
- `AlignmentMismatchError` never fires, because the count of spans is always
  right. The guard checks quantity, not correctness.
- **It explains the uniform scores in entry 13.** A median align score of
  exactly `-0.96` across 1519 words was not a coincidence: every word in a
  chunk gets the same number by construction. That should have been a clue.
- It probably explains the "weird cut" more completely than entry 28 does.
  The inverted ordering is real, but selecting among candidates matters
  little when every candidate's boundaries are fabricated.

**Scope:** specific to the wav2vec2 adapter. MFA is a separate adapter and
was never runnable here, so it is unexamined — worth checking the same way
before trusting it.

The fix is to implement the grouping the docstring already describes: walk
the CTC path, split on the word-delimiter token (wav2vec2 vocabularies use
`|`, not the blank at index 0 that the dead code tests for), and emit one
span per word with that word's own frames and its own mean score. Keep a
real fallback for paths that genuinely cannot be split, and make it
observable rather than silent — a note on the job, since handlers may return
one.

### 37. Split transcript rows at word boundaries instead of wrapping them

Raised by the owner while reviewing entry 15's implementation: "aren't long
rows assembled from the data in the database? Can we not just make them
shorter, rather than wrap?"

Correct, and it is the better design. `transcript_show`
(`rytp/commands/search.py:228`) builds one row per utterance from
`transcript_blocks` (`rytp/index/export.py:246`) and then folds each block's
text with `textwrap.fill`. The wrapping is cosmetic overflow of a row that
already exists.

Building *shorter rows* instead is different in kind, because every word
carries its own `start_ms`:

- **Every displayed line becomes seekable.** Today the second and third
  lines of a wrapped block have no timestamp — they are spill from a row
  that started at one moment. Split at a word boundary and each line has a
  real start and end. For a project whose whole purpose is mapping words to
  timestamps, a transcript line that cannot say when it starts is a wasted
  row.
- **It dissolves the `DataTable` clipping problem** rather than working
  around it. Multi-line cells are what collide with `add_row`'s `height=1`
  default; single-line rows never do.
- **Both surfaces stay identical for free** — same rows, no per-surface
  wrapping.

Two decisions it needs:

- `transcript_blocks` returns joined block text, so splitting needs the
  underlying word rows, or the split has to happen where the words are still
  available.
- `anchor` identifies a hit for `search play`. A block that becomes three
  rows either keeps one anchor across them or gains sub-anchors — pick
  deliberately, because playback resolves through it.

`--line-length` then means "maximum characters per row", a row-building
parameter, rather than "wrap width", a display one. Same flag, sharper
meaning.

### 38. A fifth time formatter the consistency scan cannot see

Follow-up from entry 31's fix. `rytp/timefmt.py` now holds one formatter per
purpose (`format_seek`, `format_spoken`, `format_length`), and
`tests/test_timefmt.py::test_no_module_outside_timefmt_builds_a_private_formatter`
scans `rytp/` for modules rolling their own.

**`rytp/diarize/mapper.py:36 format_ms` is a fifth private formatter that the
scan cannot detect.** It renders `m:ss` with no hour rollover — one
zero-padded field, where the scan looks for two joined by a colon — and is
used by `rytp/commands/speakers.py`. It was named here rather than added to
the scan's `_KNOWN_PRIVATE` allowlist, deliberately: an allowlist entry would
make the net read as complete while leaving a hole in it.

Two things to do when `rytp/diarize/*` is next touched: point it at `timefmt`
(probably `format_spoken`, or its own purpose if the missing hour rollover is
intentional rather than an oversight — a diarizer label past an hour would
currently render as `73:20` rather than `1:13:20`), and widen the scan so a
single-field formatter is caught too.

### 39. `engine.binary.<name>` is a setting nothing reads

Found while building the settings catalogue. `SETTINGS_BINARY_PREFIX =
"engine.binary."` exists in `rytp/constants.py:696` and has **no reader**:
every binary is resolved by `shutil.which` or a hardcoded name
(`render/ffmpeg.py`, `audio/extract.py`, `transcribe/align/mfa.py`,
`transcribe/registry.py`). The catalogue keeps an entry so the drift test
holds, but marks it invisible — there is no read site to verify a default
against and no roster of names to expand.

**It is worth finishing rather than deleting.** The interpreter sibling,
`engine.interpreter.<name>`, is what made the engines usable at all on a
machine where the wheels would not install. A binary override is the same
idea for the one engine that is not a Python module: MFA is a conda binary,
and on Windows a conda environment's `Scripts` directory is frequently not
on `PATH`. Today `MfaAligner.required_binary = "mfa"` is looked up with
`shutil.which` plus a beside-the-interpreter guess, so a user whose MFA
lives somewhere else has no way to say where — exactly the dead end
`engine.interpreter.*` removed for the others.

Either wire it (read it in the same place `required_binary` is resolved, so
a set value wins over the `PATH` search) or delete the constant. Leaving a
named setting that silently does nothing is the worse of the three.

### 40. A job skipped as already-satisfied looked exactly like one that ran — FIXED

Reported as: "I ordered a transcription on a video with a tick in the
transcribed column. It said it'll be enqueued, but nothing happened — no GPU
usage."

Nothing was wrong with the enqueue. `transcribe_readiness`
(`rytp/transcribe/readiness.py:61`) answers `SATISFIED` as soon as a video
has words at *either* tier, and the worker
(`rytp/jobs/worker.py:238`) finishes a `SATISFIED` job immediately without
calling the handler. Correct by design — readiness is a statement about the
world, so "already transcribed" means there is nothing to do.

The defect was that it left **no trace**: `Q.finish()` was called without a
note, so the row read `done` with a blank note — byte-identical to a job
that did the work. The only way to tell them apart was the absence of GPU
activity.

**Fixed:** the skip now writes `JOB_ALREADY_SATISFIED_NOTE` ("already
satisfied; nothing to do"), visible in `jobs list`'s note column and on the
TUI's F7 screen. Folded into the existing
`test_a_job_satisfied_since_enqueue_skips_its_handler` rather than added
alongside it.

**Not changed, because it is the design:** to genuinely re-transcribe, the
words have to go first — `transcribe remove <video>`, then enqueue — or run
the foreground `transcribe run <video>`, which does not consult readiness
and replaces the words outright. Worth considering later whether the Videos
screen's shortcut should say so before enqueueing a no-op, since the screen
knows the column is ticked.

### 41. Nothing says a worker must be running, and nothing notices when one is not

Reported as: "I re-ordered `align` — but it seems to be `pending`?" — after
the same confusion a few minutes earlier with `transcribe`.

`pending` means the row was written and no worker has claimed it. Enqueueing
only writes to `jobs`; a separate `rytp worker` process drains the queue, and
it is `FOREGROUND_ONLY` ("it is the process that drains the queue") so the
TUI deliberately will not start one.

**Nothing anywhere says this.** Not the enqueue message ("queued — F7 to
watch the queue"), not `jobs list`, not `jobs stats`, and there is no
`doctor` check. A user who has not read the design has no way to learn that
queued work needs a second terminal — the queue simply fills up silently and
looks broken. It cost the owner two separate rounds of confusion in one
session, which is the same silent-success family as entries 3, 10, 22 and 40.

**The detection already exists and is unused.** `worker.lease`
(`rytp/jobs/worker.py:78-116`) holds `{"pid", "started_at", "heartbeat"}`
with a heartbeat the running worker refreshes, and `WorkerAlreadyRunning` is
raised off a stale one. So "is a worker alive right now" is a lease read and
a staleness comparison — no new state, no polling.

Three places it should surface, cheapest first:

- **`jobs list` / `jobs stats`** — when anything is `pending` and no lease is
  live, say so: "N pending; no worker is running — start one with
  `rytp worker`". This is the screen a confused user actually looks at.
- **The enqueue message**, in both surfaces — "queued" is only half true if
  nothing will ever pick it up.
- **A `doctor` check**, advisory rather than required: a queue with pending
  work and no live worker is a state worth reporting, but it is not a broken
  installation.

### 42. A detached worker that dies before acquiring its lease leaves no trace

The TUI's F12 "Start worker" spawns a detached `rytp worker` with stdout and
stderr on `DEVNULL`, because contracts §7 enumerates the filesystem layout
exhaustively and has no slot for a log file. That was the right call — a TUI
convenience should not invent a top-level directory — but it leaves a gap.

Once the worker is up, nothing is lost: per-job progress and errors reach
`jobs.progress` and `jobs.error`, which the same F7 screen displays. **The
hole is everything before the lease is acquired.** A bad import, a missing
dependency, an adapter blowing up at module load — those happen before any
job row exists, so a detached worker simply vanishes and the queue stays
`pending` with no explanation.

That is not a hypothetical failure mode here. CLAUDE.md's "What has never
been run" says no transcriber, aligner, diarizer or embedder has executed
against its real library; the adapters were written from documented APIs.
An import-time surprise on first real use is among the *most* likely things
to happen, and F12 is precisely where a user would meet it — with no output
at all.

Options, needing a contracts §7 amendment either way:

- Add a `logs/` slot to the data tree and point the detached child's
  stdout/stderr at `logs/worker-<timestamp>.log`. Simple, and useful beyond
  this case.
- Or have the child write its startup failure to the database before dying,
  which needs no new directory but cannot capture a failure that happens
  before `rytp.db` can be opened — the case most likely to occur.

The first is worth the amendment. Until then, a worker that will not start
should be diagnosed by running `rytp worker` in a terminal, where the
traceback is visible — worth saying in the F12 status line.

### 43. Resident engine workers can now hold three models on the GPU at once

A consequence of removing the per-chunk model reload, not a defect in it —
but it lands squarely on this project's hardware.

Engines now keep one resident child per `(interpreter, module, root)`, and
each child caches its model. Nothing releases them between pipeline stages:
`shutdown_workers()` and dead-worker eviction exist and work, but no caller
invokes them from `pipeline.py`. So a process that transcribes with
`gigaam`, aligns with `wav2vec2` and then diarizes with `pyannote` ends up
**holding all three resident simultaneously**.

The target machine has an RTX 3080 Laptop with 16 GiB. Three models is
plausibly fine and plausibly not, depending on which are loaded — and the
failure, if it comes, is a CUDA out-of-memory partway through a long run,
after the expensive work is done. That is worse than the reload it replaced.

Worth deciding rather than discovering:

- **Release at stage boundaries.** `pipeline.py` knows when it has finished
  transcribing and is about to align. Evicting the previous engine's worker
  there costs one reload per stage — not per chunk — and bounds residency at
  one model.
- **Cap by count or by key.** Keep the most recently used worker and evict
  the rest, the usual cache shape.
- **Leave it and measure first.** `doctor` already reports the GPU and its
  memory, so the honest first step may be to observe an actual three-engine
  run on the real card rather than pre-optimise a problem that may not
  materialise.

Note the worker is per *interpreter* too, and the documented setup puts each
engine in its own virtualenv — so in the owner's configuration these are
separate processes, each holding its own model, which makes the total
footprint more visible but no smaller.

Related, unfixed and unfixable at this seam: **MFA still reloads per call.**
Its `child_main` shells out to the external `mfa align` binary, which has no
server mode, so no amount of Python-side residency helps. Documented in the
adapter.

### 44. No way to un-align a video: `aligned` is a one-way door

Asked for directly: "the aligner fails again, and I want to invalidate the
transcription altogether and downgrade the video back to just `timed`."

There is no command for it. `transcribe remove` deletes a video's words
outright — along with its utterances and speaker labels — so recovering to
`timed` means paying for transcription again. Nothing anywhere sets
`words.source` back from `aligned`.

**Why this matters more than it looks.** The tiers exist precisely to
separate "searchable" from "cuttable". A failed or untrusted alignment
leaves every word still marked `aligned`, which means assembly will happily
cut from it — so the only way to stop a bad alignment poisoning renders is
to destroy the transcript that produced it. The cheap, correct action
(stop trusting these boundaries) is unavailable; only the expensive,
destructive one is.

**What such a command can and cannot do.** `realign_video` overwrites
timings **in place**, so the transcriber's original `timed` boundaries are
gone the moment an aligner runs. A downgrade therefore cannot restore them
— it can only relabel the current boundaries as `timed`, which is the
honest outcome: the timings stay, but they stop being treated as cuttable.
Say so in the command's help, or someone will expect their old timestamps
back.

Shape: `transcribe unalign <video>` — set `source = 'timed'` for that
video's `aligned` rows, clear `align_score` and `align_scale`, leave text,
ordinals and speaker labels untouched. Cheap, reversible by re-running
`transcribe align`, and it makes the tier mean what it says again.

Note it does **not** need to make re-alignment possible: `align_readiness`
never reports `SATISFIED`, so re-aligning already works at any time. The
value is purely in withdrawing trust.

### 45. Re-enqueueing kept the old payload, so a job ran parameters you had withdrawn — FIXED

Reported: "I cancelled an old job on video 1 — it had gigaam as the
transcription engine. I enqueued a new transcription and it reopened the
cancelled job instead, but I have whisper as the default."

Both halves were happening, and the second is the damaging one.

`Q.enqueue` is idempotent by `UNIQUE (kind, target_id)` and its
`ON CONFLICT DO UPDATE` refreshed `priority`, `state` and `not_before` —
but **never `payload_json`**. So the reopened job carried the payload from
the original request. The engine name is stamped into the payload at
enqueue time, so the job would have run `gigaam` after the owner had
switched the default to `whisper` and cancelled the `gigaam` work.

Contracts §5 is explicit that a readiness predicate never sees the payload,
precisely because the payload is the *input* to the work — anything that
changes what the job does lives there. A request carrying different inputs
must not silently inherit the previous ones.

**Fixed:** the payload is refreshed on conflict, guarded by the same
`state = 'running'` check that already protects state and `not_before` —
work in flight cannot have its parameters changed underneath it. Two tests
pin it, and the first fails against the old code with exactly the reported
symptom.

**Considered and not done: making it a new job.** The owner's suggestion was
that a differing payload should create a separate row. That needs the
`UNIQUE (kind, target_id)` constraint relaxed, which is a schema and
contracts change — and the constraint is earning its keep: two `transcribe`
jobs for one video is duplicated GPU work, and comparing engines is what
`transcribe compare` exists for. Refreshing the payload delivers the
intent ("run what I just asked for") without weakening the guarantee that
one video has at most one job of a kind outstanding.

### 46. "never said in the corpus" while search is showing you a hit — FIXED

Reported: "why was I able to find all fragments except the last one? It is
present in the search, though."

    assemble plan "... каннибализм" --allow-timed
      1 word not found: каннибализм; 'каннибализм': never said in the corpus

    search words "каннибализм"
      1 hit(s), stem match, so these are inflected forms
      v3:1722-1722 ... timed ... канниб…

Both were behaving correctly and the pair read as a contradiction.

**Search matched by stem.** Its own header says so. Falling back to stems
when the exact form is absent is a deliberate requirement — the owner asked
for it — because an archive is searched by people who do not know which
inflection was spoken.

**Assembly needs the exact form**, because it cuts real audio. Splicing
`каннибализмом` into a sentence that calls for `каннибализм` puts a word in
the video that was never said in that form. Refusing is right.

The defect was the wording. `AbsenceDiagnosis.total` counts occurrences of
the exact `normalized_text`, so zero means "not in this form" — and it was
reported as "never said in the corpus", which is a much stronger claim and
plainly false while a hit is on screen.

**Fixed:** when the exact form is absent, the diagnosis now looks up forms
sharing its stem (the `words_stem` index already exists, and this runs once
per missing word, off the hot path) and the message names them:

    'каннибализм': not said in this exact form; the corpus has
    каннибализмом (1) — `assemble suggest каннибализм` ranks stand-ins

The original wording survives for a word genuinely absent in every form,
pinned by its own test so the two cases cannot collapse into one.

---

## Open design questions

### D2. Should captions be promotable to cuttable by alignment alone?

Raised by an accidental `transcribe align` on a video that had not been
transcribed. The command refuses it correctly —
`rytp/transcribe/pipeline.py:463`: "video N is caption-tier; run transcribe
run to promote it" — but the refusal is a decision worth revisiting rather
than an obvious truth.

**The opportunity is large at this corpus's scale.** Captions are already
downloaded for the whole archive and cost no GPU. Forced alignment is far
cheaper than transcription — whisper `large-v3` pulled several GiB and runs
the full model; wav2vec2 alignment is a fraction of that. If caption text
could be aligned directly, the entire ~1.6K-video corpus could become
cuttable without ever running a transcriber, which is otherwise the dominant
cost of the project.

**The risk is the one that has come up repeatedly here.** Forced alignment
assumes the transcript is what was actually said. Auto-captions are not
reliably verbatim: they drop words, merge them, and carry no punctuation or
case. Aligning against text that does not match the audio does not fail
loudly — it snaps boundaries confidently to the wrong places, which is
exactly the failure mode that makes entry 13's scores untrustworthy and that
D1 was careful about.

So the question is empirical, not architectural: **how verbatim are the
captions for this archive?** That is measurable — transcribe a handful of
videos with whisper, align both the whisper text and the caption text, and
compare word counts, ordinals and boundary agreement. `transcribe compare`
already exists to run several engines over the same audio and report where
they disagree, so most of the machinery is there.

If captions turn out close to verbatim, this is the cheapest possible route
to a fully cuttable archive. If they do not, the current refusal is right and
should stay. Worth measuring before committing either way.

### D1. Should `timed` words be cuttable when no aligner is available? — DECIDED

**Decision (2026-09-24): yes, behind `--allow-timed`.** The default stays
honest; the degraded mode is opt-in and visible in the output. Still needs
writing up as a change against the contracts.

**Revised the same day — it is an override, not a gate.** `--allow-timed`
means "cut from these anyway, I accept the quality", with no threshold and no
quality logic. The existing rule is untouched: `aligned` remains the only
tier cuttable by default. This is simpler, and it is also the only workable
choice while the score column has no unit (entry 13) — a threshold cannot be
written against a scale that changes per engine.

**The spelling stays `--allow-timed`.** `--force` was considered and is not
available: `assemble plan` already has a `--force` meaning "replace an
existing cut list of this name". Two meanings on one command is exactly what
the flag-vocabulary consistency suite forbids. `--allow-timed` also says what
is permitted rather than how hard to push, which is the more honest label for
an override.

Consequently the threshold bullet below is dropped, and the tier must still
be recorded on each fragment and marked in the render's source list — that
part matters more, not less, once quality is not being checked at all.

Raised after entry 17 left the target machine with no working aligner: if
nothing can align, does it make sense to cut from whisper's own timestamps
in a declared suboptimal mode?

**This is a contracts change**, not a bug. Contracts §schema fixes
`words.source` as `caption | timed | aligned` with only `aligned` cuttable,
and CLAUDE.md says to raise a change rather than vary from it.

Against, and it is measured rather than theoretical: on the pre-rewrite
corpus **78.7% of whisper word gaps were exactly 0 ms**, the median gap was
0, and some words were 0 ms long. Cutting on those boundaries clips
phonemes or swallows neighbours. That measurement is the entire reason the
`timed` tier exists.

For, and this is new: the run in entry 13 was `whisper+energy` — the refine
step ran, snapping boundaries to a measured energy minimum and zero
crossing, reporting a median boundary score of 1.00. That is precisely the
mechanism that makes an `aligned` word safe to cut. The refinement does not
care where the timestamps came from.

The distinction that survives: alignment establishes *which* word sits
where; refinement only tidies boundaries it is handed. Given a timestamp
that is off by a word, refinement snaps to a clean silence in the wrong
place — confidently wrong, and with a high score.

Proposed shape, rather than widening what "cuttable" means:

- Keep `aligned` as the only unconditionally cuttable tier.
- Let `assemble plan` take an opt-in flag (`--allow-timed`, default off).
- ~~Require that refinement ran and the boundary score clears a
  threshold.~~ Dropped: `--allow-timed` is an override, not a gate.
- Record the tier on each fragment and mark it in the render's source list,
  so a degraded cut is visible in the output rather than only at plan time.

Implementation note: the partial index `words_alignable ON
words(normalized_text) WHERE source = 'aligned'` exists precisely to keep
the assembler's hot query off the caption tier. Admitting `timed` needs that
predicate widened or a second index, or the query goes back to scanning.

---

## Fixed already (2026-09-24, not committed)

Three Windows-only test failures, all in `tests/`, none in product code:

- `tests/test_assets.py` — `path.rsplit("/", 1)[-1]` to take a basename
  returns the whole string when the separator is `\`. Now `Path(...).name`.
- `tests/test_render_run.py` — asserted `"/out/o.mp4"`, but
  `str(Path("/out/o.mp4"))` is `\out\o.mp4` on Windows. Now compares against
  `str(Path(...))`.
- `tests/test_cli.py` — `subprocess.run(text=True)` decodes with the locale
  encoding; the CLI deliberately writes UTF-8, so cp1252 choked on rich's
  box-drawing characters, the reader thread died and `proc.stdout` was
  `None`. Now passes `encoding="utf-8"`. This also explains the
  `PytestUnhandledThreadExceptionWarning` in the same run.

Not bugs, recorded so they are not re-investigated:

- A run that reported `ModuleNotFoundError: No module named 'textual'` had not
  activated `.venv`; the package was installed all along.
- `doctor` reporting `HF_TOKEN` unset was a Windows environment-variable
  propagation issue — `setx` and the System Properties dialog only reach
  newly started processes.
- `schema version 17 is newer than this code's 12` was the pre-rewrite
  database still sitting in `data/`. Unrelated migration numbering.
- Logs full of `ΓÇö` are PowerShell 5.1 decoding a native command's UTF-8
  output as the OEM codepage. Capture with
  `cmd /c "... > out.log 2>&1"` to pass the bytes through untouched.
