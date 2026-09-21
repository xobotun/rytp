# rytp — shared contracts

**Companion to:** `2026-09-21-rytp-redesign-design.md`
**Purpose:** Fixed interfaces that every implementation plan must use verbatim, so plans written independently do not contradict each other.

Anything in this file is **binding**. If a plan needs to deviate, that is a design change and must be raised, not silently varied.

---

## 1. Global constraints

- Python >= 3.11. Target 3.11 syntax; `from __future__ import annotations` in every module.
- SQLite only. No server databases, no Docker, no cloud APIs.
- ffmpeg and ffprobe are required system binaries. `ffplay` is used for playback.
- Runs on Windows. No POSIX-only calls, no shell pipelines, no symlinks. Use `pathlib` and `subprocess` with list arguments.
- Line length 100. `ruff` rules `E,F,I,B,UP,N,SIM,RUF`. `mypy` on `rytp/`.
- Heavy dependencies (transcribers, aligners, diarizers) are optional extras, imported lazily inside functions, never at module import time.
- **Base dependencies**, importable at module level and required for the whole test suite to collect: `typer`, `rich`, `textual`, `numpy`, `scipy`, `snowballstemmer`. Audio analysis and stemming are core to the product, not optional add-ons, and a `.[dev]` install must be able to run every test.
- **Test invocation is portable.** Plans write run steps as `python -m pytest <args>`, assuming an activated virtualenv. No absolute interpreter paths, no developer-specific directories, no `/tmp`. The project ships on Windows; a plan that only runs on one machine is a plan that cannot be followed. `RYTP_TEST_TMP` may override the scratch root but must default sensibly without being set.
- No YouTube video ids, channel ids, channel names or URLs in any committed file, including tests and fixtures. Use `VIDEO_A`, `CHANNEL_ONE`, `https://example.invalid/...`.
- Every tunable value lives in `rytp/constants.py` with a comment naming its design section. No inline magic numbers.

## 2. Package layout

Each file has one responsibility. Do not create files outside this tree without saying why.

```
rytp/
  __init__.py  __main__.py  config.py  constants.py  models.py
  db/          __init__.py (Database)  schema.py (MIGRATIONS)  queries.py
  commands/    __init__.py (registry)  catalog.py  ingest.py  transcribe.py
               search.py  assemble.py  render.py  speakers.py
  cli.py                                  # Typer app built from the registry
  tui/         app.py  palette.py  screens/
  jobs/        __init__.py  queue.py  worker.py  readiness.py
  acquire/     ytdlp.py  captions.py  local.py  policy.py
  audio/       extract.py  vad.py  energy.py  acoustics.py
  transcribe/  base.py  registry.py  captions.py  compare.py
               engines/  align/
  diarize/     base.py  none.py  pyannote.py  embed.py
  index/       __init__.py  utterances.py  search.py  export.py
  assemble/    __init__.py  match.py  score.py  cutlist.py
  render/      ffmpeg.py  canvas.py  report.py
```

Ownership by plan part: 1 = `config, constants, models, db/, commands/__init__, cli, tui skeleton, commands/catalog`. 2 = `jobs/, acquire/, audio/extract`. 3 = `transcribe/, audio/{vad,energy,acoustics}`. 4 = `index/`. 5 = `assemble/`. 6 = `render/`. 7 = `diarize/, tui/screens/speakers`.

## 3. Schema

One migration per table, appended to `MIGRATIONS: list[tuple[int, str]]` in `rytp/db/schema.py`. Never edit an existing migration. `schema_version` is bootstrapped by the runner, not by a migration.

**Part 1 creates every table in this section**, including ones only later parts read or write — `renders`, `video_speakers`, `video_acoustics` and the rest. Later parts consume the schema; none of them add DDL. Any part whose tests insert into a table is responsible for supplying every `NOT NULL` column in its fixtures, `video_speakers.engine` included.

```sql
CREATE TABLE channels (
    id              INTEGER PRIMARY KEY,
    url             TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    last_synced_at  TEXT
);

CREATE TABLE videos (
    id            INTEGER PRIMARY KEY,
    source        TEXT NOT NULL CHECK (source IN ('youtube','ytdlp','local')),
    kind          TEXT NOT NULL CHECK (kind IN ('video','short','livestream','other')),
    channel_id    INTEGER REFERENCES channels(id),
    external_id   TEXT,
    url           TEXT,
    title         TEXT NOT NULL,
    duration_ms   INTEGER,
    published_at  TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    UNIQUE (source, external_id)
);
CREATE INDEX videos_channel ON videos(channel_id);
CREATE INDEX videos_source  ON videos(source);

CREATE TABLE assets (
    id          INTEGER PRIMARY KEY,
    video_id    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('audio','video','captions','container')),
    format_id   TEXT,
    path        TEXT NOT NULL,
    bytes       INTEGER,
    width       INTEGER,
    height      INTEGER,
    abr         REAL,
    acquired_at TEXT NOT NULL
);
CREATE INDEX assets_video_role ON assets(video_id, role);
-- A video has at most one audio asset and at most one captions asset.
-- Video renditions are unconstrained: many per video is the point.
CREATE UNIQUE INDEX assets_one_audio    ON assets(video_id) WHERE role = 'audio';
CREATE UNIQUE INDEX assets_one_captions ON assets(video_id) WHERE role = 'captions';

-- speakers precedes video_speakers: the FK points this way.
CREATE TABLE speakers (
    id           INTEGER PRIMARY KEY,
    label        TEXT NOT NULL UNIQUE,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    notes        TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE video_speakers (
    id          INTEGER PRIMARY KEY,
    video_id    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    local_label TEXT NOT NULL,
    speaker_id  INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    embedding   BLOB,
    engine      TEXT NOT NULL,   -- which diarizer produced this label
    UNIQUE (video_id, local_label)
);
CREATE INDEX video_speakers_speaker ON video_speakers(speaker_id);

CREATE TABLE words (
    id               INTEGER PRIMARY KEY,
    video_id         INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    ord              INTEGER NOT NULL,
    start_ms         INTEGER NOT NULL,
    end_ms           INTEGER,
    text             TEXT NOT NULL,
    normalized_text  TEXT NOT NULL,
    stem             TEXT NOT NULL,
    confidence       REAL,
    align_score      REAL,
    source           TEXT NOT NULL CHECK (source IN ('caption','aligned')),
    engine           TEXT NOT NULL,
    video_speaker_id INTEGER REFERENCES video_speakers(id) ON DELETE SET NULL,
    UNIQUE (video_id, ord),
    CHECK (source = 'caption' OR end_ms IS NOT NULL)
);
CREATE INDEX words_video_ord  ON words(video_id, ord);
CREATE INDEX words_normalized ON words(normalized_text);
CREATE INDEX words_stem       ON words(stem);
CREATE INDEX words_speaker    ON words(video_speaker_id);

CREATE TABLE utterances (
    id               INTEGER PRIMARY KEY,
    video_id         INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    video_speaker_id INTEGER REFERENCES video_speakers(id) ON DELETE SET NULL,
    start_ms         INTEGER NOT NULL,
    end_ms           INTEGER NOT NULL,
    first_word_ord   INTEGER NOT NULL,
    last_word_ord    INTEGER NOT NULL,
    text             TEXT NOT NULL,
    normalized_text  TEXT NOT NULL,
    stem_text        TEXT NOT NULL
);
CREATE INDEX utterances_video ON utterances(video_id, start_ms);

CREATE VIRTUAL TABLE utterances_fts USING fts5(
    normalized_text, stem_text,
    content='utterances', content_rowid='id',
    tokenize='unicode61 remove_diacritics 0'
);
CREATE TRIGGER utterances_ai AFTER INSERT ON utterances BEGIN
    INSERT INTO utterances_fts(rowid, normalized_text, stem_text)
    VALUES (new.id, new.normalized_text, new.stem_text);
END;
CREATE TRIGGER utterances_ad AFTER DELETE ON utterances BEGIN
    INSERT INTO utterances_fts(utterances_fts, rowid, normalized_text, stem_text)
    VALUES ('delete', old.id, old.normalized_text, old.stem_text);
END;
CREATE TRIGGER utterances_au AFTER UPDATE ON utterances BEGIN
    INSERT INTO utterances_fts(utterances_fts, rowid, normalized_text, stem_text)
    VALUES ('delete', old.id, old.normalized_text, old.stem_text);
    INSERT INTO utterances_fts(rowid, normalized_text, stem_text)
    VALUES (new.id, new.normalized_text, new.stem_text);
END;

CREATE TABLE video_acoustics (
    video_id             INTEGER PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE,
    f0_mean              REAL,
    f0_std               REAL,
    spectral_tilt        REAL,
    noise_floor_db       REAL,
    reverb_proxy         REAL,
    loudness_lufs        REAL,
    computed_at          TEXT NOT NULL
);

CREATE TABLE jobs (
    id           INTEGER PRIMARY KEY,
    kind         TEXT NOT NULL,
    target_id    INTEGER NOT NULL,
    state        TEXT NOT NULL CHECK (state IN
                   ('pending','running','done','failed','blocked','cancelled')),
    pool         TEXT NOT NULL CHECK (pool IN ('network','gpu','cpu')),
    priority     INTEGER NOT NULL DEFAULT 0,
    attempts     INTEGER NOT NULL DEFAULT 0,
    not_before   TEXT,
    last_error   TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    UNIQUE (kind, target_id)
);
CREATE INDEX jobs_claim ON jobs(state, pool, priority, not_before);

-- A render targets a cut list, which is a named file with no integer
-- identity, and jobs.target_id is an INTEGER. This table gives a render
-- that identity, and gives render history somewhere to live.
CREATE TABLE renders (
    id           INTEGER PRIMARY KEY,
    cutlist_name TEXT NOT NULL,
    output_path  TEXT,
    canvas_mode  TEXT NOT NULL,
    state        TEXT NOT NULL CHECK (state IN ('planned','rendered','failed')),
    created_at   TEXT NOT NULL,
    finished_at  TEXT
);
CREATE INDEX renders_cutlist ON renders(cutlist_name);

CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

`tokenize` is `unicode61`, **never `porter`** — the Porter stemmer is English-only. Russian stemming is done in Python at write time into the `stem` / `stem_text` columns.

`words.end_ms` is null exactly for caption-sourced rows. **Cuttable is defined as `source = 'aligned'`** and nothing else may be cut.

## 4. Core types

`rytp/models.py`:

```python
@dataclass(frozen=True)
class RawWord:
    """What a transcriber emits, before alignment.

    Both timings are optional: a transcriber may emit text only, leaving
    every boundary to the aligner. Timings, when present, are absolute
    against the source audio, never relative to a chunk.
    """
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    confidence: float | None = None

@dataclass(frozen=True)
class Span:
    """A refined time range for one word."""
    start_ms: int
    end_ms: int
    score: float | None

@dataclass(frozen=True)
class DiarSegment:
    start_ms: int
    end_ms: int
    local_label: str

@dataclass(frozen=True)
class Fragment:
    """One contiguous run taken from one video. The unit of a cut list."""
    video_id: int
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    text: str

def normalize_text(text: str) -> str: ...    # casefold; punctuation → space;
                                             # collapse whitespace; fold ё→е
def stem_text(normalized: str) -> str: ...   # snowballstemmer, russian
def utc_now_iso() -> str: ...
```

`normalize_text` folds `ё` to `е`. Russian writers omit the diaeresis inconsistently, so a search for `еще` must find `ещё`. Everything matching against `normalized_text` or `stem` gets this for free; nothing may normalize differently.

**Punctuation becomes a space, not nothing.** `кто-то` normalizes to `кто то`, two tokens. This forces an invariant that every part depends on:

> A stored word row holds exactly one token, and the tokens stored for a piece of text are exactly `normalize_text(text).split()`.

The reason is that a search query runs through the same `normalize_text`. If a hyphenated word were stored as one row containing a space, no single-token lookup could ever reach it — not from Part 4's FTS, not from Part 5's pointer walk — and stripping the hyphen instead would leave the row unreachable from the query side, which normalizes to two tokens. Splitting on both sides is the only choice that makes them agree by construction.

Whoever writes words is responsible for splitting, and for giving the interior boundary a measured time rather than an interpolated one. Part 3 pins this with a test asserting stored tokens equal `normalize_text(text).split()`, so any drift in `normalize_text` fails loudly instead of silently hiding words from search.

**Local videos.** `videos` has no path column — paths live in `assets`. A local file's absolute path is stored in `videos.external_id` with `source = 'local'`, which makes `UNIQUE (source, external_id)` the idempotency key for re-registration. For a local video the filesystem *is* the source system, so this is the column's intended meaning rather than a workaround.

Both text functions live in `rytp/models.py` and are **fully implemented by Part 1** — not stubbed. They are not in `rytp/index/`: `words.stem` is written by Part 3 when words are created, so putting the stemmer under the index package would invert the dependency. There is no `rytp/index/stem.py`.

`stem_text` must be real from the start. Stubbing it and deferring the body to a later part creates a build-order cycle, because Part 3 calls it in production to populate `words.stem` while Part 4 would supply the body. It is a thin wrapper over `snowballstemmer`'s Russian stemmer, which is a base dependency, so there is nothing to defer.

Note that stemming is **not idempotent** (`сказали` → `сказа`, which stems again). An utterance's `normalized_text` and `stem_text` are therefore built by joining the per-word `words.normalized_text` and `words.stem` columns, never by re-running the functions over the joined string.

### Cross-part invariants

**Words are the root of the derived data.** `utterances` and `video_speakers` are both computed from `words` and both copy data that changing words invalidates. Any stage that deletes or replaces a video's words **must** delete that video's `utterances` and `video_speakers` rows in the same transaction. Re-deriving them is the job of Parts 4 and 7 respectively.

## 5. Command registry

`rytp/commands/__init__.py`. Both surfaces are generated from this; neither may define a command of its own.

```python
REQUIRED: Final = object()

@dataclass(frozen=True)
class Param:
    name: str
    type: type                        # str | int | float | bool | Path
    help: str
    default: Any = REQUIRED
    choices: tuple[str, ...] | None = None
    short: str | None = None
    positional: bool = False

@dataclass(frozen=True)
class CommandResult:
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    message: str | None = None

@dataclass(frozen=True)
class Command:
    name: str                         # "videos.add", "search.words"; top-level
                                      # commands are bare and may contain a
                                      # hyphen: "ingest", "fetch-video"
    group: str                        # "videos"; "" for top level
    summary: str
    params: tuple[Param, ...]
    handler: Callable[..., CommandResult]
    long_running: bool = False        # TUI must not run these inline
    cli_only: bool = False            # surface launchers such as `rytp tui`,
                                      # which cannot be invoked from inside
                                      # the surface they launch

COMMANDS: dict[str, Command] = {}
def register(cmd: Command) -> Command: ...
def resolve(name: str) -> Command: ...
```

Handlers take keyword arguments matching `Param.name`, receive an open `Database` as the first positional argument, and return a `CommandResult`. Handlers never print and never call `sys.exit`; they raise, and the surface formats the error.

`Param.type` is a scalar type only. A flag that takes several values is a comma-separated string parsed by the handler.

### Speaker filters — two flags, two identifier spaces

There are two kinds of speaker identity and conflating them is a bug. One shared resolver in `rytp/commands/__init__.py` serves every command that filters by speaker; no part may roll its own.

| Flag | Matches | Example |
|---|---|---|
| `--speaker`, alias `--global-speaker` | `speakers.label`, then `speakers.aliases_json` | `"Иванов Иван Иванович"` |
| `--video-local-speaker` | `video_speakers.local_label` | `SPEAKER_00` |

`--speaker` is the common case and the one the product is for: a named person, tracked across the whole corpus. It never accepts a raw diarizer label. Resolution failure is an error naming the closest roster entries, never a silent empty result.

`--video-local-speaker` **requires a video to be named as well**, via `--video`. A bare `SPEAKER_00` is meaningless across the corpus — it would match the first-detected voice of every diarized video, which is not a person and not a useful answer. Reject the combination rather than returning that.

### Deletion — every entity gets a `remove`

Each group exposes a `<group>.remove`, so there is no entity you can create but not get rid of. Owners:

| Command | Owner | Removes |
|---|---|---|
| `channel.remove` | 1 | The channel. Its videos are **orphaned, not deleted** — `videos.channel_id` is nullable for exactly this. |
| `videos.remove` | 1 | The video, its rows everywhere, and its files on disk. |
| `assets.remove` | 2 | One asset — typically a superseded video rendition, to reclaim disk after an upgrade. |
| `jobs.cancel` | 2 | Queued or failed jobs, by id or by filter. This is "drop stuff from the queue". |
| `transcribe.remove` | 3 | A video's words, and with them its utterances and speaker labels. |
| `index.drop` | 4 | A video's utterances. Derived data; rebuilt by `index.build`. |
| `assemble.remove` | 5 | A cut list file. |
| `render.remove` | 6 | A render's output directory and its `renders` row. |
| `speakers.remove` | 7 | A roster entry. Its `video_speakers` links are set null, not deleted — the local labels survive as unnamed voices. |

Three rules bind all of them.

**Database rows go by cascade; files do not.** The `ON DELETE CASCADE` on `video_id` foreign keys takes care of `words`, `utterances`, `video_speakers`, `assets` and `video_acoustics`. Files under `media/`, `cache/` and `output/` must be removed explicitly, and `jobs` rows have no foreign key. `jobs.target_id` is disambiguated by the row's `kind` column — `JobKind.target_kind` is a Python-level attribute of the registration describing *what* a kind targets, not a column in the table — so removal must look up which kinds target the entity being removed and cancel those jobs in the same transaction, or a worker will later run against something that no longer exists.

**Anything that deletes a file supports `--dry-run` and requires `--yes`.** The dry run prints exactly what would go, counting rows and bytes. Without `--yes`, a command that would delete files refuses. Commands that only touch derived database rows (`index.drop`, `transcribe.remove`) need neither.

**Removal is not a job.** These are synchronous and immediate. A half-deleted entity recovered from a crashed queue is worse than a slow command.

### Health checks — `doctor`

`doctor` is one top-level command owned by Part 1, reporting on everything the tool needs in order to work. Parts contribute their own checks rather than Part 1 knowing about them, using the same shape as job handlers:

```python
@dataclass(frozen=True)
class HealthCheck:
    name: str                       # "ffmpeg", "transcriber:gigaam"
    summary: str
    run: Callable[[Database], "HealthResult"]
    required: bool = True           # False = advisory; absence is not a failure

@dataclass(frozen=True)
class HealthResult:
    ok: bool                        # always the truth about what was found
    detail: str                     # what was found, or what is missing
    remedy: str | None = None       # the exact command that would fix it

HEALTH_CHECKS: dict[str, HealthCheck] = {}
def register_check(check: HealthCheck) -> HealthCheck: ...
```

Part 1 covers the interpreter, the data tree and the database schema version. Part 2 covers `ffmpeg`, `ffprobe`, `yt-dlp` and free disk. Part 3 covers which transcribers and aligners are importable and whether a GPU is visible. Part 4 covers `ffplay` and whether FTS5 is compiled in. Part 7 covers diarizers and `HF_TOKEN`.

A check never raises and never blocks. **`ok` always tells the truth about what was found** — a missing optional engine is `ok=False`, because it is in fact missing. What makes it non-fatal is `required=False` on the check. `doctor` exits non-zero when any check with `required=True` returns `ok=False`, and never otherwise.

Keeping those two ideas apart matters: the alternative is reporting a missing optional tool as `ok=True` to protect the exit code, which makes the output lie about the thing the command exists to tell you. Advisory failures are displayed as such — found or not found, with a remedy — and simply do not affect the exit status.

`required=True` is right for ffmpeg, ffprobe, FTS5, the data tree and the schema version. `required=False` is right for yt-dlp, ffplay, every transcriber, aligner and diarizer, the GPU and `HF_TOKEN`.

### Job handlers

Long-running work is reached from two directions — a command run by hand, and the worker draining the queue — so the mapping from job kind to callable is itself a contract. `rytp/jobs/__init__.py` holds `JOB_HANDLERS: dict[str, Callable[[Database, int, dict], None]]`, keyed by job kind, taking the database, the `target_id`, and the decoded `payload_json`. Each part registers its own kinds:

| Kind | Owner | Handler |
|---|---|---|
| `download`, `captions`, `extract_wav` | Part 2 | `rytp.acquire.*` |
| `transcribe`, `align`, `fingerprint` | Part 3 | `rytp.transcribe.*` |
| `index` | Part 4 | `rytp.index.utterances` |
| `diarize` | Part 7 | `rytp.diarize.*` |
| `render` | Part 6 | `rytp.render.*` |

## 6. Engine protocols

All engines are resolved by name from a registry of **classes**, never instances, and declare `requires_hf_token` as a class attribute so it can be checked before construction. Engines that need an incompatible dependency set set `out_of_process = True` and are invoked via `subprocess` against their own interpreter path from settings.

```python
class Transcriber(Protocol):
    name: str
    requires_hf_token: bool
    out_of_process: bool
    def transcribe(self, audio: Path, *, language: str | None = None,
                   start_ms: int = 0, end_ms: int | None = None) -> Iterable[RawWord]: ...

class Aligner(Protocol):
    name: str
    requires_hf_token: bool
    out_of_process: bool
    def align(self, audio: Path, words: Sequence[str], *,
              start_ms: int, end_ms: int) -> list[Span]: ...

class Diarizer(Protocol):
    name: str
    requires_hf_token: bool
    out_of_process: bool
    def diarize(self, audio: Path) -> Iterable[DiarSegment]: ...
```

Registries live in `rytp/transcribe/registry.py` (`TRANSCRIBERS`, `ALIGNERS`) and `rytp/diarize/base.py` (`DIARIZERS`), with `register_transcriber` / `register_aligner` / `register_diarizer` decorators and `resolve_*` lookups raising `ValueError` naming the available options.

## 7. Filesystem layout

Rooted at `RYTP_DATA`, default `./data`. Created on demand by the command that needs it — **not at module import time**.

```
rytp.db
media/{video_id}/audio.{ext}          # asset role 'audio'
media/{video_id}/video-{format}.{ext} # asset role 'video', one per rendition
media/{video_id}/captions.json3       # asset role 'captions'
cache/wav/{video_id}.wav              # regenerable, prunable
output/{render_id}/output.mp4
output/{render_id}/report.md
transcripts/{video_id}.md
cutlists/{name}.toml
```

## 8. Conventions

**Errors.** Domain errors subclass `RytpError` in `rytp/models.py`. The CLI catches it, prints one line to stderr, exits 1. Never print a traceback for an expected failure. Unexpected exceptions propagate.

**Time.** Milliseconds, integers, everywhere. Timestamps in the database are ISO-8601 UTC strings produced by `datetime.now(UTC).isoformat()`.

**Database access.** Every stage takes an open `Database`; nothing opens its own connection. `Database` exposes `conn` (a `sqlite3.Connection` with `row_factory = sqlite3.Row` and foreign keys on), `migrate()`, `close()`, and `transaction()`. Writes that span more than one statement use `db.transaction()`, which is re-entrant: a nested call joins the outer transaction rather than opening a second one, so a helper that wraps its own writes stays safe to call from inside a larger one.

**Tests.** `pytest`. Fixtures `db`, `data_dir`, `tmp_path` keep their current names and semantics; `tmp_path` stays overridden to a workspace-local root. Every external binary and model is faked. Any test needing ffmpeg is marked `@pytest.mark.skipif(shutil.which("ffmpeg") is None, ...)`. Network access in tests is forbidden outright.

**Commits.** One per task, conventional-commit prefix, present tense.
