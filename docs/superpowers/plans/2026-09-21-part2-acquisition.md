# Part 2 — Assets, Jobs and Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given a catalogued video, download its audio, a video rendition and its captions as separate assets, extract a cached 16 kHz WAV, and drive all of it through a resumable, crash-safe job queue that is polite enough never to get the owner's IP blocked.

**Architecture:** One `jobs` table, no dependency-edge table. Each job kind owns a *readiness predicate* over current database state that answers READY / BLOCKED / SATISFIED, so the pipeline self-heals — prune a cached WAV and `extract_wav` becomes runnable again. A worker process runs three pools (`network`, `gpu`, `cpu`) simultaneously, each pool claiming jobs with a guarded single-row `UPDATE` that is atomic without relying on transaction isolation. Acquisition goes through a `YtDlpRunner` protocol whose only real implementation wraps the yt-dlp Python API; every test uses a fake. Politeness (inter-video delay, rate limit, daily cap, 429/403 backoff) lives in `rytp/acquire/policy.py` and is entirely settings-driven, and throttle backoff is applied to the whole **network pool**, not to one job — deferring a single job would just let the next download hit the same throttled server.

**Tech Stack:** Python 3.11, SQLite (stdlib `sqlite3`), Typer (via Part 1's CLI), yt-dlp (optional extra, lazily imported), ffmpeg/ffprobe as external binaries invoked through `subprocess` with list arguments, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md` (§4 Data model, §5 Ingest and the job queue, §6 captions tier) and the binding `docs/superpowers/specs/2026-09-21-rytp-contracts.md`.

## Global Constraints

Copied from contracts §1 and §8. Every task's requirements implicitly include this section.

- Python >= 3.11. Target 3.11 syntax; `from __future__ import annotations` in **every** module.
- SQLite only. No server databases, no Docker, no cloud APIs.
- ffmpeg and ffprobe are required system binaries. `ffplay` is used for playback.
- Runs on Windows. No POSIX-only calls, no shell pipelines, no symlinks. Use `pathlib` and `subprocess` with list arguments. In the worker, handle `KeyboardInterrupt` only — no `signal.SIGTERM` handler, it does not exist on Windows.
- Line length 100. `ruff` rules `E,F,I,B,UP,N,SIM,RUF`. `mypy` on `rytp/`.
- Heavy dependencies (yt-dlp here) are optional extras, imported lazily **inside functions**, never at module import time.
- No YouTube video ids, channel ids, channel names or URLs in any committed file, including tests and fixtures. Use `VIDEO_A`, `CHANNEL_ONE`, `https://example.invalid/...`.
- Every tunable value lives in `rytp/constants.py` with a comment naming its design section. No inline magic numbers.
- **Errors.** Domain errors subclass `RytpError` from `rytp/models.py`. One line to stderr, exit 1, no traceback. Unexpected exceptions propagate.
- **Time.** Milliseconds, integers, everywhere. Database timestamps are ISO-8601 UTC strings from `datetime.now(UTC).isoformat()`.
- **Database access.** Every stage takes an open `Database`; nothing opens its own connection. The single exception is `rytp/jobs/worker.py`, which is the *process* entry point and opens one `Database` per worker thread — thread-confined, never shared. Multi-statement writes use `db.transaction()`.
- **Tests.** No test may touch the network, ever. Every external binary and model is faked. Tests needing real ffmpeg are marked `@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")`.
- **Commits.** One per task, conventional-commit prefix, present tense.

## How to run the tests

Contracts §1: run steps are portable. Every "run the test" step in this plan is
`python -m pytest <args>` from the repository root **with the project's
virtualenv activated and the package installed in editable mode** — no absolute
interpreter paths, no developer-specific directories, no `/tmp`. The product
ships on Windows.

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q
```

Scratch directories go wherever `tests/conftest.py` puts them; `RYTP_TEST_TMP`
overrides that root but nothing in this plan requires setting it.

## Consumes from Part 1 (confirmed by the Part 1 author)

Part 1 owns `config`, `constants`, `models`, `db/`, `commands/__init__`, `cli` and `commands/catalog`. Part 2 assumes all of it exists, exactly as follows. Nothing in this plan re-implements any of it.

- `rytp/db/__init__.py::Database` — `Database(path: Path)`, `.conn` (`sqlite3.Connection`, `row_factory = sqlite3.Row`, `isolation_level=None` so a bare `execute` autocommits), `.migrate() -> int`, `.schema_version() -> int`, `.close()`, `.transaction()` context manager, `.migrate_to(target)`, `__enter__`/`__exit__`. Connection setup issues `PRAGMA foreign_keys = ON`, `PRAGMA journal_mode = WAL`, `PRAGMA busy_timeout = SQLITE_BUSY_TIMEOUT_MS`. **`transaction()` is re-entrant** — an inner `with db.transaction()` joins the outer one rather than issuing a second `BEGIN`, so Part 2's helpers may wrap freely.
- `rytp/db/schema.py::MIGRATIONS` — Part 1 writes migrations 1..11 (channels, videos, assets, speakers, video_speakers, words, utterances, utterances_fts and its three triggers, video_acoustics, jobs, settings), which create **all** of contracts §3. `LATEST_VERSION` is derived from `MIGRATIONS[-1][0]` and a Part 1 test asserts the versions are contiguous. **Part 2 adds no migration**; anything a later part needs appends from 12, never renumbering.
- `rytp/db/queries.py` — Part 1 defines `clamp_limit`, `insert_channel`, `get_channel`, `find_channel`, `list_channels`, `mark_channel_synced`, `upsert_video`, `get_video`, `list_videos`, `get_setting(db, key, default=None) -> str | None`, `set_setting(db, key, value: str) -> None`. Part 1 defines **no** asset helpers; Task 2 appends them to the same file under its own banner.
- `rytp/config.py` — `data_root() -> Path`, `paths() -> Paths` (a *function*, resolved per call from `RYTP_DATA`; there is no module-level singleton), `ensure_dir(path: Path) -> Path`. `Paths` is frozen with `.root`, `.db`, `.media_dir(video_id: int)`, `.cache_wav(video_id: int)`, `.output_dir(render_id: str)`, `.transcript(video_id: int)`, `.cutlist(name: str)`. **None of these touch the filesystem** — call `config.ensure_dir(p.parent)` immediately before writing.
- `rytp/models.py` — `RytpError`, `NotFoundError(RytpError)`, `InvalidInputError(RytpError)`, and the frozen dataclass `ChannelEntry(external_id: str, title: str, url: str, duration_ms: int | None, kind: str, published_at: str | None)`. Part 2 imports `ChannelEntry`, never redefines it.
- `rytp/constants.py` — additive-only, shared with Part 1. Task 1 appends new sections at the end and touches nothing above them. Part 1's constants, confirmed, are: the filesystem names (`DATA_ROOT_ENV_VAR`, `DEFAULT_DATA_DIRNAME`, `DB_FILENAME`, `MEDIA_DIRNAME`, `CACHE_DIRNAME`, `OUTPUT_DIRNAME`, `TRANSCRIPTS_DIRNAME`, `CUTLISTS_DIRNAME`), `MS_PER_SECOND = 1000`, `SQLITE_JOURNAL_MODE`, `SQLITE_BUSY_TIMEOUT_MS = 5_000`, the catalog set (`CHANNEL_TABS = ("videos", "streams", "shorts")`, `CHANNEL_TAB_KINDS` — a `dict[str, str]` mapping each tab to the `videos.kind` it yields, `VIDEO_SOURCES`, `VIDEO_KINDS`, `REMOTE_SOURCE = "ytdlp"`, `LOCAL_SOURCE = "local"`, `DEFAULT_VIDEO_KIND`), the surface set (`DEFAULT_LIST_LIMIT`, `MAX_LIST_LIMIT`, `TITLE_TRUNCATE_CHARS`, `NULL_CELL`) and `HF_TOKEN_ENV_VARS`. Part 2 uses `MS_PER_SECOND`, `CHANNEL_TABS`, `CHANNEL_TAB_KINDS`, `LOCAL_SOURCE` and `SQLITE_BUSY_TIMEOUT_MS`, and defines the rest of what it needs itself — Part 1 deliberately does **not** define `AUDIO_SAMPLE_RATE_HZ`, `AUDIO_CHANNELS` or `FFMPEG_ERROR_TAIL_CHARS`, so those stay in Part 2's section.
- `rytp/commands/__init__.py` — `REQUIRED`, `Param`, `CommandResult`, `Command`, `COMMANDS`, `register`, `resolve`, plus `HealthCheck`, `HealthResult`, `HEALTH_CHECKS` and `register_check`, exactly as contracts §5. It ends with a block of sibling imports; Task 11 adds one line to that block, which is also what makes Task 14's health checks register. Part 1 owns the `doctor` command that runs them.
- Local files: `videos` has no path column, so Part 1 stores a local file's resolved absolute path in `videos.external_id` with `source='local'`. `rytp/acquire/local.py` reads it from there.
- `tests/conftest.py` — `tmp_path` honours `RYTP_TEST_TMP`, and `data_dir` is just `monkeypatch.setenv("RYTP_DATA", str(tmp_path))`; there is no `config.paths` singleton to patch any more. The `db` fixture gives an opened, migrated `Database` on that tree.
- `videos.source` values in practice: `C.REMOTE_SOURCE` (`'ytdlp'`) for anything catalogued through yt-dlp, `C.LOCAL_SOURCE` (`'local'`) for a local file path. `'youtube'` remains in the CHECK constraint but nothing writes it. Part 2's predicates therefore branch on `source == 'local'`, never on `'youtube'`.

## Produces for Part 1 (two seams Part 1 calls into)

Part 1's `commands/catalog.py` reaches these through lazy indirections it monkeypatches in its own tests, so Part 1 ships before Part 2 exists. Both are defined in Task 5.

- `rytp/acquire/ytdlp.py::enumerate_channel(channel_url: str, *, tabs: Sequence[str] = C.CHANNEL_TABS, runner: object | None = None) -> list[ChannelEntry]` — returns entries in tab order (`videos`, then `streams`, then `shorts`), **without** deduplicating: Part 1 upserts by `(source, external_id)` so the last tab that yields an id wins its `kind`, which is exactly the behaviour wanted for a stream listed in both `/videos` and `/streams`.
- `rytp/acquire/ytdlp.py::probe_video(url: str, *, runner: object | None = None) -> ChannelEntry` — one metadata probe, no download, for `rytp videos add <url>`. Its `kind` is `'livestream'` when yt-dlp's `live_status` is one of is_live / was_live / post_live / is_upcoming and `'video'` otherwise; it can never return `'short'`, because a single probe carries no reliable shorts signal. Part 1's `--kind` override is the way to catalog a short from a bare URL.

## Produces for Parts 3-7: registering a job kind

Contracts §5 makes the job-kind-to-callable mapping a contract, and Part 2 owns
the file it lives in. Every part registers its own kinds through Task 3's
registry, and needs three things:

The names below are deliberately illustrative — `example_kind` belongs to no
part. Substitute your own; do not copy another part's symbols, which are still
settling.

```python
# 1. A LIGHT predicate module in your own package. It may import pathlib,
#    rytp.config, rytp.constants and rytp.db and nothing heavier, because
#    rytp/jobs/__init__.py imports it at module load.
def example_kind_readiness(db: Database, video_id: int) -> Readiness: ...

# 2. A handler thunk at the bottom of rytp/jobs/__init__.py, importing your
#    stage lazily in the body (contracts §1: heavy deps never at import time).
def _run_example_kind(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.example.pipeline import do_the_work

    do_the_work(db, video_id)

# 3. An entry in the registration block at the bottom of the same file.
register_job_kind(
    JobKind(
        name="example_kind",
        pool="gpu",
        readiness=example_kind_readiness,
        handler=_run_example_kind,
        summary="one line about what this kind does",
    )
)
```

`register_job_kind` writes both `JOB_KINDS` and the contracted flat
`JOB_HANDLERS`, so they cannot drift. Two fields exist for kinds that are not
shaped like Part 2's:

- **`target_kind`** (default `"video"`) names the namespace `jobs.target_id`
  lives in. `render` targets a cut list, whose integer id is derived from its
  name, so it sets `target_kind="cutlist"` and the queue will never re-evaluate
  it alongside a video that happens to share the number.
- **`reopenable`** (default `True`) says whether a full `reconcile` may turn a
  `done` job back into work when its output disappears. True for everything
  derived from the corpus. `render` sets `reopenable=False`: it is a one-shot
  the user asked for, and it must not silently re-run on the next worker start.
  Re-running `rytp render` enqueues it again, which does re-render.

**A predicate never sees the payload**, and that is deliberate — a predicate
that changed its mind based on how a job was enqueued would stop being a
statement about the world, and the self-healing property in design §5 would go
with it. A handler does see it, so anything that is an *input* to the work
(a cut-list name, render options) belongs in `payload_json`.

## Note on the existing code

Part 1's first task deletes the whole current `rytp/` and `tests/` tree. `rytp/download/ytdlp.py`, `rytp/download/queue.py` and `rytp/transcribe/extract.py` **will not exist** when this plan runs — do not plan to edit them. They remain readable for reference at `git show 44fc214c:rytp/download/ytdlp.py`, `git show 44fc214c:rytp/download/queue.py`, `git show 44fc214c:rytp/transcribe/extract.py`, and the two patterns worth cribbing from them are named where they are used: the runtime-checkable `YtDlpRunner` protocol plus fake-runner test split (Task 5) and the ffmpeg-subprocess wrapper (Task 9).

## Files

**Created by this plan**

| File | Responsibility |
|---|---|
| `rytp/jobs/__init__.py` | `Readiness`, `JobKind`, the `JOB_KINDS` registry, and registration of Part 2's three built-in kinds |
| `rytp/jobs/readiness.py` | The readiness predicates — the one place that knows what each kind needs |
| `rytp/jobs/queue.py` | Enqueue, atomic claim, terminal transitions, reclaim, reconcile, list, stats, pause |
| `rytp/jobs/worker.py` | Worker lease, per-pool loops, throttle and retry handling, threads |
| `rytp/acquire/__init__.py` | `acquire_media` — the download stage, and package re-exports |
| `rytp/acquire/policy.py` | Download politeness: delays, rate limit, daily cap, error classification, pool cooldown |
| `rytp/acquire/ytdlp.py` | `YtDlpRunner` protocol, `FetchedFile`, `RealYtDlpRunner`, `enumerate_channel` |
| `rytp/acquire/captions.py` | `acquire_captions` — the caption track as its own asset |
| `rytp/acquire/local.py` | `register_local_container` — a local file becomes a `container` asset |
| `rytp/audio/__init__.py` | Empty package marker |
| `rytp/audio/extract.py` | The regenerable WAV cache: `wav_path`, `ensure_wav`, `prune_wav_cache` |
| `rytp/commands/ingest.py` | Part 2's eleven commands and its four `doctor` health checks |
| `tests/fakes.py` | `FakeYtDlpRunner` and `make_video` — shared by six test modules, so not inlined |
| `tests/test_acquire_policy.py` … `tests/test_commands_ingest.py` | One test module per task |

**Modified by this plan** (both files are additive-only and shared with Part 1)

- `rytp/constants.py` — one new section appended at the end (Task 1).
- `rytp/db/queries.py` — asset helpers appended at the end (Task 2).
- `rytp/commands/__init__.py` — one import line added to the existing sibling-import block (Task 11).

---

### Task 1: Constants and the download policy

Design §5 says every operational value is tunable through `settings`. This task puts the defaults in `constants.py` and builds the one object that reads them, plus the error taxonomy the whole part uses. Nothing else can be written until error classification exists, because the worker's retry behaviour hangs off it.

**Files:**
- Modify: `rytp/constants.py` (append a new section at the very end; touch nothing above it)
- Create: `rytp/acquire/__init__.py`
- Create: `rytp/acquire/policy.py`
- Test: `tests/test_acquire_policy.py`

**Interfaces:**
- Consumes: `rytp.db.Database`, `rytp.db.queries.get_setting`, `rytp.db.queries.set_setting`, `rytp.models.RytpError`.
- Produces:
  - `rytp.acquire.policy.AcquireError(RytpError)`, `PermanentAcquireError(AcquireError)`, `VideoUnavailable(PermanentAcquireError)`, `NoCaptions(PermanentAcquireError)`, `RateLimited(AcquireError)`, `DailyCapReached(AcquireError)`, `MissingAssetError(AcquireError)`
  - `ErrorKind` (`THROTTLED`, `UNAVAILABLE`, `TRANSIENT`), `classify_error(message: str) -> ErrorKind`
  - `DownloadPolicy` frozen dataclass and `DownloadPolicy.from_settings(db) -> DownloadPolicy`
  - `next_delay_s(policy, rng) -> float`, `backoff_seconds(policy, streak: int) -> int`
  - `cooldown_until(db) -> str | None`, `is_cooling_down(db, *, now) -> bool`, `record_throttle(db, policy, *, now) -> str`, `clear_throttle(db) -> None`
  - `downloads_started_today(db, *, now) -> int`, `check_daily_cap(db, policy, *, now) -> None`
  - `pool_size(db, pool: str) -> int`

- [ ] **Step 1: Create the `rytp/acquire` package marker**

`rytp/acquire/__init__.py`, for now just the docstring and future import. Task 6 fills it with `acquire_media`.

```python
"""Acquisition: turning a catalogued video into assets on disk.

design §5. The stage entry points live here; the yt-dlp glue is in
:mod:`rytp.acquire.ytdlp` and the politeness rules in
:mod:`rytp.acquire.policy`.
"""

from __future__ import annotations
```

- [ ] **Step 2: Append the constants section**

At the **end** of `rytp/constants.py`. Part 1 owns everything above; this file is additive-only for both plans.

```python
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
```

- [ ] **Step 3: Write the failing tests**

`tests/test_acquire_policy.py`:

```python
"""Tests for the download politeness policy (design §5)."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from rytp import constants as C
from rytp.acquire import policy as P
from rytp.db import Database
from rytp.db.queries import set_setting

NOW = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)


def test_defaults_come_from_constants(db: Database) -> None:
    p = P.DownloadPolicy.from_settings(db)
    assert p.delay_min_s == C.DOWNLOAD_DELAY_MIN_S
    assert p.delay_max_s == C.DOWNLOAD_DELAY_MAX_S
    assert p.daily_cap == C.DOWNLOAD_DAILY_CAP
    assert p.rate_limit_bps == C.DOWNLOAD_RATE_LIMIT_BPS
    assert p.caption_langs == C.CAPTION_LANGS
    assert p.backoff_ladder_s == C.THROTTLE_BACKOFF_LADDER_S


def test_settings_override_every_value(db: Database) -> None:
    set_setting(db, C.SETTING_DOWNLOAD_DELAY_MIN, "1.5")
    set_setting(db, C.SETTING_DOWNLOAD_DELAY_MAX, "2.5")
    set_setting(db, C.SETTING_DOWNLOAD_DAILY_CAP, "7")
    set_setting(db, C.SETTING_DOWNLOAD_RATE_LIMIT, "0")
    set_setting(db, C.SETTING_CAPTION_LANGS, "ru-orig, ru, en")
    p = P.DownloadPolicy.from_settings(db)
    assert (p.delay_min_s, p.delay_max_s) == (1.5, 2.5)
    assert p.daily_cap == 7
    assert p.rate_limit_bps is None  # 0 means "no limit"
    assert p.caption_langs == ("ru-orig", "ru", "en")


def test_next_delay_is_inside_the_configured_band(db: Database) -> None:
    p = P.DownloadPolicy.from_settings(db)
    rng = random.Random(1234)
    for _ in range(50):
        d = P.next_delay_s(p, rng)
        assert C.DOWNLOAD_DELAY_MIN_S <= d <= C.DOWNLOAD_DELAY_MAX_S


def test_backoff_ladder_climbs_then_plateaus(db: Database) -> None:
    p = P.DownloadPolicy.from_settings(db)
    assert [P.backoff_seconds(p, n) for n in (1, 2, 3, 4, 5, 99)] == [
        300, 900, 2700, 7200, 7200, 7200
    ]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("ERROR: HTTP Error 429: Too Many Requests", P.ErrorKind.THROTTLED),
        ("ERROR: HTTP Error 403: Forbidden", P.ErrorKind.THROTTLED),
        ("Sign in to confirm you're not a bot", P.ErrorKind.THROTTLED),
        ("ERROR: Private video. Sign in if you've been granted access", P.ErrorKind.UNAVAILABLE),
        ("ERROR: Video unavailable", P.ErrorKind.UNAVAILABLE),
        ("ERROR: This video has been removed by the uploader", P.ErrorKind.UNAVAILABLE),
        ("ERROR: Requested format is not available", P.ErrorKind.UNAVAILABLE),
        ("ERROR: unable to download video data: timed out", P.ErrorKind.TRANSIENT),
    ],
)
def test_classify_error(message: str, expected: P.ErrorKind) -> None:
    assert P.classify_error(message) is expected


def test_unavailable_wins_over_a_403_in_the_same_message() -> None:
    # A delisted video often answers 403. Treating that as a throttle would
    # freeze the whole network pool for two hours over one dead video.
    msg = "ERROR: HTTP Error 403: Forbidden. Video unavailable"
    assert P.classify_error(msg) is P.ErrorKind.UNAVAILABLE


def test_throttle_sets_a_pool_wide_cooldown_that_climbs(db: Database) -> None:
    p = P.DownloadPolicy.from_settings(db)
    assert P.is_cooling_down(db, now=NOW) is False

    first = P.record_throttle(db, p, now=NOW)
    assert first == (NOW + timedelta(seconds=300)).isoformat()
    assert P.is_cooling_down(db, now=NOW) is True
    assert P.is_cooling_down(db, now=NOW + timedelta(seconds=301)) is False

    second = P.record_throttle(db, p, now=NOW)
    assert second == (NOW + timedelta(seconds=900)).isoformat()


def test_success_clears_the_cooldown_and_the_streak(db: Database) -> None:
    p = P.DownloadPolicy.from_settings(db)
    P.record_throttle(db, p, now=NOW)
    P.record_throttle(db, p, now=NOW)
    P.clear_throttle(db)
    assert P.is_cooling_down(db, now=NOW) is False
    assert P.cooldown_until(db) is None
    # The ladder restarts from the bottom after a clean download.
    assert P.record_throttle(db, p, now=NOW) == (NOW + timedelta(seconds=300)).isoformat()


def test_daily_cap_counts_downloads_started_today(db: Database) -> None:
    yesterday = (NOW - timedelta(days=1)).isoformat()
    today = NOW.isoformat()
    for i, started in enumerate([yesterday, today, today]):
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, payload_json, "
            "created_at, started_at) VALUES "
            "('download', ?, 'done', 'network', '{}', ?, ?)",
            (i + 1, started, started),
        )
    assert P.downloads_started_today(db, now=NOW) == 2


def test_check_daily_cap_raises_when_reached(db: Database) -> None:
    set_setting(db, C.SETTING_DOWNLOAD_DAILY_CAP, "1")
    p = P.DownloadPolicy.from_settings(db)
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, payload_json, "
        "created_at, started_at) VALUES ('download', 1, 'done', 'network', "
        "'{}', ?, ?)",
        (NOW.isoformat(), NOW.isoformat()),
    )
    with pytest.raises(P.DailyCapReached, match="daily cap"):
        P.check_daily_cap(db, p, now=NOW)


def test_zero_daily_cap_means_unlimited(db: Database) -> None:
    set_setting(db, C.SETTING_DOWNLOAD_DAILY_CAP, "0")
    p = P.DownloadPolicy.from_settings(db)
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, payload_json, "
        "created_at, started_at) VALUES ('download', 1, 'done', 'network', "
        "'{}', ?, ?)",
        (NOW.isoformat(), NOW.isoformat()),
    )
    P.check_daily_cap(db, p, now=NOW)  # does not raise


def test_pool_size_defaults_and_overrides(db: Database) -> None:
    assert P.pool_size(db, "network") == 1
    assert P.pool_size(db, "cpu") == C.POOL_DEFAULT_SIZE["cpu"]
    set_setting(db, C.SETTING_POOL_SIZE.format(pool="cpu"), "4")
    assert P.pool_size(db, "cpu") == 4
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `python -m pytest tests/test_acquire_policy.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.acquire'`.

- [ ] **Step 5: Write `rytp/acquire/policy.py`**

```python
"""Download politeness. design §5.

Every value here is a default that ``settings`` can override, and the
throttle backoff deliberately applies to the whole **network pool** rather
than to the job that tripped it: deferring one job would just let the next
download hit the same throttled server a second later.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import get_setting, set_setting
from rytp.models import RytpError


class AcquireError(RytpError):
    """Acquisition failed for a reason worth one line on stderr."""


class PermanentAcquireError(AcquireError):
    """Retrying will not help. The worker parks the job in ``failed``."""


class VideoUnavailable(PermanentAcquireError):
    """Private, removed, geo-blocked, or members-only."""


class NoCaptions(PermanentAcquireError):
    """The video has no caption track in any requested language."""


class RateLimited(AcquireError):
    """HTTP 429 / 403-forbidden. The whole network pool must back off."""


class DailyCapReached(AcquireError):
    """The per-day download cap in settings has been reached."""


class MissingAssetError(AcquireError):
    """A required asset row or its file on disk is absent."""


class ErrorKind(Enum):
    THROTTLED = "throttled"
    UNAVAILABLE = "unavailable"
    TRANSIENT = "transient"


# Checked before the throttle pattern on purpose: a delisted video often
# answers 403, and treating that as a throttle would freeze the pool.
_UNAVAILABLE_RE = re.compile(
    r"private video|video unavailable|removed by the uploader"
    r"|account associated with this video has been terminated"
    r"|members[- ]only|not available in your country|no video formats found"
    # An audio-only source answers this. Retrying it five times against the
    # network is the one thing the whole politeness budget exists to avoid.
    r"|requested format is not available",
    re.IGNORECASE,
)
_THROTTLE_RE = re.compile(
    r"http error 429|http error 403|too many requests|rate[- ]?limit"
    r"|sign in to confirm",
    re.IGNORECASE,
)


def classify_error(message: str) -> ErrorKind:
    """Decide what a failure message means for retry behaviour."""
    if _UNAVAILABLE_RE.search(message):
        return ErrorKind.UNAVAILABLE
    if _THROTTLE_RE.search(message):
        return ErrorKind.THROTTLED
    return ErrorKind.TRANSIENT


def _float(db: Database, key: str, default: float) -> float:
    raw = get_setting(db, key)
    return default if raw is None else float(raw)


def _int(db: Database, key: str, default: int) -> int:
    raw = get_setting(db, key)
    return default if raw is None else int(raw)


def _str(db: Database, key: str, default: str) -> str:
    raw = get_setting(db, key)
    return default if raw is None else raw


def _tuple(db: Database, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = get_setting(db, key)
    if raw is None:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class DownloadPolicy:
    """The tunable half of acquisition, read once per stage invocation."""

    delay_min_s: float
    delay_max_s: float
    file_delay_s: float
    rate_limit_bps: int | None
    sleep_requests_s: float
    daily_cap: int
    backoff_ladder_s: tuple[int, ...]
    format_audio: str
    format_video: str
    caption_langs: tuple[str, ...]
    caption_format: str

    @classmethod
    def from_settings(cls, db: Database) -> DownloadPolicy:
        rate = _int(db, C.SETTING_DOWNLOAD_RATE_LIMIT, C.DOWNLOAD_RATE_LIMIT_BPS)
        return cls(
            delay_min_s=_float(db, C.SETTING_DOWNLOAD_DELAY_MIN, C.DOWNLOAD_DELAY_MIN_S),
            delay_max_s=_float(db, C.SETTING_DOWNLOAD_DELAY_MAX, C.DOWNLOAD_DELAY_MAX_S),
            file_delay_s=_float(db, C.SETTING_DOWNLOAD_FILE_DELAY, C.DOWNLOAD_FILE_DELAY_S),
            rate_limit_bps=rate or None,
            sleep_requests_s=_float(
                db, C.SETTING_DOWNLOAD_SLEEP_REQUESTS, C.DOWNLOAD_SLEEP_REQUESTS_S
            ),
            daily_cap=_int(db, C.SETTING_DOWNLOAD_DAILY_CAP, C.DOWNLOAD_DAILY_CAP),
            backoff_ladder_s=C.THROTTLE_BACKOFF_LADDER_S,
            format_audio=_str(db, C.SETTING_DOWNLOAD_FORMAT_AUDIO, C.DOWNLOAD_FORMAT_AUDIO),
            format_video=_str(db, C.SETTING_DOWNLOAD_FORMAT_VIDEO, C.DOWNLOAD_FORMAT_VIDEO),
            caption_langs=_tuple(db, C.SETTING_CAPTION_LANGS, C.CAPTION_LANGS),
            caption_format=_str(db, C.SETTING_CAPTION_FORMAT, C.CAPTION_FORMAT),
        )


def next_delay_s(policy: DownloadPolicy, rng: random.Random) -> float:
    """Randomised pause before the next video. design §5."""
    return rng.uniform(policy.delay_min_s, policy.delay_max_s)


def backoff_seconds(policy: DownloadPolicy, streak: int) -> int:
    """Ladder position for a throttle streak; the last rung repeats."""
    ladder = policy.backoff_ladder_s
    return ladder[min(max(streak, 1), len(ladder)) - 1]


def cooldown_until(db: Database) -> str | None:
    """ISO timestamp the network pool is frozen until, or None."""
    return get_setting(db, C.SETTING_COOLDOWN_UNTIL) or None


def is_cooling_down(db: Database, *, now: datetime) -> bool:
    until = cooldown_until(db)
    return until is not None and now.isoformat() < until


def record_throttle(db: Database, policy: DownloadPolicy, *, now: datetime) -> str:
    """Advance the throttle streak and freeze the pool. Returns the ISO end."""
    streak = _int(db, C.SETTING_THROTTLE_STREAK, 0) + 1
    until = (now + timedelta(seconds=backoff_seconds(policy, streak))).isoformat()
    # Not wrapped in db.transaction(): set_setting manages its own write and
    # there is nothing here that has to land atomically — a streak without a
    # cooldown, or the reverse, is self-correcting on the next tick.
    set_setting(db, C.SETTING_THROTTLE_STREAK, str(streak))
    set_setting(db, C.SETTING_COOLDOWN_UNTIL, until)
    return until


def clear_throttle(db: Database) -> None:
    """A clean download resets the ladder to the bottom."""
    set_setting(db, C.SETTING_THROTTLE_STREAK, "0")
    set_setting(db, C.SETTING_COOLDOWN_UNTIL, "")


def downloads_started_today(db: Database, *, now: datetime) -> int:
    """Download jobs started since UTC midnight — attempts, not successes."""
    day_start = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    row = db.conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE kind = 'download' AND started_at >= ?",
        (day_start.isoformat(),),
    ).fetchone()
    return int(row["n"])


def check_daily_cap(db: Database, policy: DownloadPolicy, *, now: datetime) -> None:
    """Raise :class:`DailyCapReached` if today's budget is spent."""
    if policy.daily_cap <= 0:
        return
    started = downloads_started_today(db, now=now)
    if started >= policy.daily_cap:
        raise DailyCapReached(
            f"daily cap reached: {started} downloads started today, cap is "
            f"{policy.daily_cap} (setting {C.SETTING_DOWNLOAD_DAILY_CAP})"
        )


def pool_size(db: Database, pool: str) -> int:
    """Slot count for one pool, from settings or the built-in default."""
    return _int(db, C.SETTING_POOL_SIZE.format(pool=pool), C.POOL_DEFAULT_SIZE[pool])
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_acquire_policy.py -q`
Expected: PASS, 12 passed.

- [ ] **Step 7: Lint and type-check**

Run: `python -m ruff check rytp tests && python -m mypy rytp`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add rytp/constants.py rytp/acquire/__init__.py rytp/acquire/policy.py tests/test_acquire_policy.py
git commit -m "feat: add download policy, error taxonomy and Part 2 constants"
```

---

### Task 2: Asset helpers

Contracts §3 gives `assets` no uniqueness constraint, but design §4 says a video has **at most one** canonical audio asset, at most one captions asset, and any number of video renditions. That invariant is enforced here, in the only function that writes the table. Readiness (Task 3) also needs to know whether an asset's file is still on disk, because an asset row pointing at a deleted file must not count as satisfied — that is the self-healing property design §5 asks for.

**Files:**
- Modify: `rytp/db/queries.py` (append at the end; Part 1 owns everything above)
- Test: `tests/test_assets.py`
- Create: `tests/fakes.py`

**Interfaces:**
- Consumes: `rytp.db.Database`, `rytp.constants.SINGLETON_ASSET_ROLES`.
- Produces:
  - `insert_asset(db, *, video_id, role, path, format_id=None, size_bytes=None, width=None, height=None, abr=None, acquired_at=None) -> int`
  - `asset_for(db, video_id, role) -> sqlite3.Row | None` (highest id wins, i.e. the newest rendition)
  - `assets_for(db, video_id, role=None) -> list[sqlite3.Row]`
  - `has_usable_asset(db, video_id, role) -> bool` (row exists **and** its file exists)
  - `prune_missing_assets(db, video_id) -> int`
  - `tests/fakes.py::make_video(db, **overrides) -> int`

- [ ] **Step 1: Create the shared test helper module**

`tests/fakes.py`. It is a plain module, not a conftest, because six test modules import it and only some of them want the yt-dlp fake. Every URL and id in it is a placeholder — contracts §1 forbids real ones.

```python
"""Fakes and factories shared by the Part 2 test modules.

Nothing here touches the network or an external binary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.db import Database

VIDEO_A_URL = "https://example.invalid/watch/VIDEO_A"
CHANNEL_ONE_URL = "https://example.invalid/@CHANNEL_ONE"


def make_video(db: Database, **overrides: Any) -> int:
    """Insert one catalog row and return its id."""
    row: dict[str, Any] = {
        "source": C.REMOTE_SOURCE,
        "kind": "video",
        "channel_id": None,
        "external_id": "VIDEO_A",
        "url": VIDEO_A_URL,
        "title": "Video A",
        "duration_ms": 3_600_000,
        "published_at": "2020-01-01",
        "metadata_json": "{}",
        "created_at": datetime.now(UTC).isoformat(),
    }
    row.update(overrides)
    cur = db.conn.execute(
        "INSERT INTO videos (source, kind, channel_id, external_id, url, title, "
        "duration_ms, published_at, metadata_json, created_at) "
        "VALUES (:source, :kind, :channel_id, :external_id, :url, :title, "
        ":duration_ms, :published_at, :metadata_json, :created_at)",
        row,
    )
    return int(cur.lastrowid)


def touch(path: Path, content: bytes = b"\x00" * 16) -> Path:
    """Create a small real file, parents included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path
```

- [ ] **Step 2: Write the failing tests**

`tests/test_assets.py`:

```python
"""Tests for the asset helpers (design §4, contracts §3)."""

from __future__ import annotations

from pathlib import Path

from rytp.db import Database
from rytp.db.queries import (
    asset_for,
    assets_for,
    has_usable_asset,
    insert_asset,
    prune_missing_assets,
)

from tests.fakes import make_video, touch


def test_insert_and_read_back(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "audio.m4a")
    asset_id = insert_asset(
        db, video_id=vid, role="audio", path=str(p), format_id="140",
        size_bytes=16, abr=128.0,
    )
    row = asset_for(db, vid, "audio")
    assert row is not None
    assert row["id"] == asset_id
    assert row["path"] == str(p)
    assert row["format_id"] == "140"
    assert row["abr"] == 128.0
    assert row["acquired_at"]  # ISO-8601, stamped for us


def test_audio_is_a_singleton_role(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a1.m4a")))
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a2.m4a")))
    rows = assets_for(db, vid, "audio")
    assert len(rows) == 1
    assert rows[0]["path"].endswith("a2.m4a")


def test_video_renditions_accumulate_but_one_per_format(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="video", format_id="360",
                 path=str(touch(tmp_path / "v360.mp4")), height=360)
    insert_asset(db, video_id=vid, role="video", format_id="720",
                 path=str(touch(tmp_path / "v720.mp4")), height=720)
    # Re-fetching the same format replaces that rendition, not the other one.
    insert_asset(db, video_id=vid, role="video", format_id="360",
                 path=str(touch(tmp_path / "v360b.mp4")), height=360)
    rows = assets_for(db, vid, "video")
    assert sorted(r["format_id"] for r in rows) == ["360", "720"]
    assert {r["path"].rsplit("/", 1)[-1] for r in rows} == {"v360b.mp4", "v720.mp4"}


def test_asset_for_returns_the_newest_rendition(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="video", format_id="360",
                 path=str(touch(tmp_path / "v360.mp4")))
    insert_asset(db, video_id=vid, role="video", format_id="1080",
                 path=str(touch(tmp_path / "v1080.mp4")))
    row = asset_for(db, vid, "video")
    assert row is not None and row["format_id"] == "1080"


def test_has_usable_asset_is_false_when_the_file_is_gone(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "audio.m4a")
    insert_asset(db, video_id=vid, role="audio", path=str(p))
    assert has_usable_asset(db, vid, "audio") is True
    p.unlink()
    assert has_usable_asset(db, vid, "audio") is False


def test_prune_missing_assets_drops_only_the_dead_rows(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    alive = touch(tmp_path / "audio.m4a")
    dead = touch(tmp_path / "v720.mp4")
    insert_asset(db, video_id=vid, role="audio", path=str(alive))
    insert_asset(db, video_id=vid, role="video", format_id="720", path=str(dead))
    dead.unlink()
    assert prune_missing_assets(db, vid) == 1
    assert [r["role"] for r in assets_for(db, vid)] == ["audio"]


def test_assets_are_deleted_with_their_video(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    db.conn.execute("DELETE FROM videos WHERE id = ?", (vid,))
    assert assets_for(db, vid) == []
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_assets.py -q`
Expected: collection error, `ImportError: cannot import name 'insert_asset' from 'rytp.db.queries'`.

- [ ] **Step 4: Append the helpers to `rytp/db/queries.py`**

```python
# ---------------------------------------------------------------------------
# Assets (design §4, contracts §3). Owned by Part 2.
# ---------------------------------------------------------------------------


def insert_asset(
    db: Database,
    *,
    video_id: int,
    role: str,
    path: str,
    format_id: str | None = None,
    size_bytes: int | None = None,
    width: int | None = None,
    height: int | None = None,
    abr: float | None = None,
    acquired_at: str | None = None,
) -> int:
    """Record one asset, superseding whatever it replaces.

    The schema has no uniqueness constraint for it, so the design §4 rule —
    at most one audio, one captions and one container per video, any number
    of video renditions but only one per ``format_id`` — is enforced here,
    in the only function that writes the table.
    """
    if role in C.SINGLETON_ASSET_ROLES:
        superseded = ("DELETE FROM assets WHERE video_id = ? AND role = ?", (video_id, role))
    else:
        superseded = (
            "DELETE FROM assets WHERE video_id = ? AND role = ? AND format_id IS ?",
            (video_id, role, format_id),
        )
    stamp = acquired_at or datetime.now(UTC).isoformat()
    with db.transaction():
        db.conn.execute(*superseded)
        cur = db.conn.execute(
            "INSERT INTO assets (video_id, role, format_id, path, bytes, width, "
            "height, abr, acquired_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (video_id, role, format_id, path, size_bytes, width, height, abr, stamp),
        )
        asset_id = int(cur.lastrowid)
    return asset_id


def assets_for(db: Database, video_id: int, role: str | None = None) -> list[sqlite3.Row]:
    """Every asset of a video, oldest first, optionally filtered by role."""
    if role is None:
        return list(
            db.conn.execute(
                "SELECT * FROM assets WHERE video_id = ? ORDER BY id", (video_id,)
            ).fetchall()
        )
    return list(
        db.conn.execute(
            "SELECT * FROM assets WHERE video_id = ? AND role = ? ORDER BY id",
            (video_id, role),
        ).fetchall()
    )


def asset_for(db: Database, video_id: int, role: str) -> sqlite3.Row | None:
    """The current asset of a role — the newest rendition, for ``video``."""
    return db.conn.execute(
        "SELECT * FROM assets WHERE video_id = ? AND role = ? ORDER BY id DESC LIMIT 1",
        (video_id, role),
    ).fetchone()


def has_usable_asset(db: Database, video_id: int, role: str) -> bool:
    """True when a row of this role exists *and* its file is still on disk.

    Readiness depends on this rather than on the row alone: deleting media
    outside the tool must make the download job runnable again (design §5).
    """
    for row in assets_for(db, video_id, role):
        if Path(row["path"]).exists():
            return True
    return False


def prune_missing_assets(db: Database, video_id: int) -> int:
    """Delete asset rows whose file has vanished. Returns the count."""
    doomed = [
        row["id"] for row in assets_for(db, video_id) if not Path(row["path"]).exists()
    ]
    if doomed:
        with db.transaction():
            db.conn.executemany(
                "DELETE FROM assets WHERE id = ?", [(i,) for i in doomed]
            )
    return len(doomed)
```

Add whatever of `import sqlite3`, `from datetime import UTC, datetime`, `from pathlib import Path`, `from rytp import constants as C` is not already at the top of the file.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_assets.py -q`
Expected: PASS, 7 passed. If `test_assets_are_deleted_with_their_video` fails, `PRAGMA foreign_keys` is off on the connection — that is Part 1's `Database`, report it rather than working around it.

- [ ] **Step 6: Commit**

```bash
git add rytp/db/queries.py tests/fakes.py tests/test_assets.py
git commit -m "feat: add asset helpers enforcing the one-audio-per-video rule"
```

---

### Task 3: Job kind registry and readiness predicates

This is the centre of the design. Design §5: *"Readiness is derived from data, not from a dependency graph. Each job kind has a small predicate over current database state."* There is no edge table to keep consistent, so recovery after a crash is just asking what is runnable now.

A predicate returns one of three answers, and the third is what makes the system self-healing:

- `READY` — prerequisites met, the output is missing, run it.
- `BLOCKED` — prerequisites missing, do not run it, but re-check later.
- `SATISFIED` — the output already exists, there is nothing to do.

Prune a cached WAV and `extract_wav` flips from SATISFIED back to READY with nobody editing a graph.

**Files:**
- Create: `rytp/jobs/__init__.py`
- Create: `rytp/jobs/readiness.py`
- Test: `tests/test_jobs_readiness.py`

**Interfaces:**
- Consumes: Task 2's `has_usable_asset`; `rytp.config.paths`; `rytp.constants.POOLS`.
- Produces:
  - `rytp.jobs.Readiness` (`READY`, `BLOCKED`, `SATISFIED`)
  - `rytp.jobs.JobKind(name, pool, readiness, handler, summary, target_kind="video", reopenable=True)` frozen dataclass
  - `rytp.jobs.JOB_KINDS: dict[str, JobKind]` and `rytp.jobs.JOB_HANDLERS: dict[str, Callable[[Database, int, dict], None]]` — the flat view contracts §5 requires, written only by `register_job_kind`
  - `register_job_kind(kind) -> JobKind`, `resolve_job_kind(name) -> JobKind`, `kinds_for_pool(pool) -> tuple[str, ...]`
  - `Predicate = Callable[[Database, int], Readiness]` and `Handler = Callable[[Database, int, dict[str, Any]], None]`
  - `rytp.jobs.readiness.download_readiness / captions_readiness / extract_wav_readiness`
  - Registered kinds: `download` (network), `captions` (network), `extract_wav` (cpu)

**Import-order note for the implementer.** `rytp/jobs/__init__.py` defines `Readiness` and the registry first and imports `rytp.jobs.readiness` **at the bottom** of the file — the same trick the old engine registry used (`git show 44fc214c:rytp/engines.py`). `readiness.py` then does `from rytp.jobs import Readiness` and finds it already defined. Nothing in `readiness.py` may import `rytp.jobs.queue`, and nothing in `__init__.py` may import a handler module at module level; the three handler thunks import lazily inside the function body, which is also what keeps yt-dlp out of the import path (contracts §1).

- [ ] **Step 1: Write the failing tests**

`tests/test_jobs_readiness.py`:

```python
"""Tests for job-kind registration and readiness (design §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import config
from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import insert_asset
from rytp.jobs import (
    JOB_KINDS,
    JobKind,
    Readiness,
    kinds_for_pool,
    register_job_kind,
    resolve_job_kind,
)
from rytp.jobs.readiness import (
    captions_readiness,
    download_readiness,
    extract_wav_readiness,
)

from tests.fakes import make_video, touch


def test_the_three_part_two_kinds_are_registered() -> None:
    assert resolve_job_kind("download").pool == "network"
    assert resolve_job_kind("captions").pool == "network"
    assert resolve_job_kind("extract_wav").pool == "cpu"
    assert set(kinds_for_pool("network")) >= {"download", "captions"}


def test_job_handlers_is_the_contracted_flat_view() -> None:
    # contracts §5 requires rytp/jobs/__init__.py to expose
    # JOB_HANDLERS: dict[str, Callable[[Database, int, dict], None]].
    from rytp.jobs import JOB_HANDLERS

    assert set(JOB_HANDLERS) == set(JOB_KINDS)
    for name, kind in JOB_KINDS.items():
        assert JOB_HANDLERS[name] is kind.handler


def test_kinds_target_videos_by_default_and_are_reopenable() -> None:
    for name in ("download", "captions", "extract_wav"):
        kind = resolve_job_kind(name)
        assert kind.target_kind == "video"
        assert kind.reopenable is True


def test_unknown_kind_names_the_available_ones() -> None:
    with pytest.raises(ValueError, match="download"):
        resolve_job_kind("nope")


def test_registering_a_duplicate_is_refused() -> None:
    kind = JobKind(
        name="download", pool="cpu", readiness=lambda db, t: Readiness.READY,
        handler=lambda db, t, p: None, summary="x",
    )
    with pytest.raises(ValueError, match="already registered"):
        register_job_kind(kind)


def test_registering_an_unknown_pool_is_refused() -> None:
    kind = JobKind(
        name="invented", pool="quantum", readiness=lambda db, t: Readiness.READY,
        handler=lambda db, t, p: None, summary="x",
    )
    with pytest.raises(ValueError, match="quantum"):
        register_job_kind(kind)
    assert "invented" not in JOB_KINDS


def test_download_is_ready_for_a_catalogued_remote_video(db: Database) -> None:
    vid = make_video(db)
    assert download_readiness(db, vid) is Readiness.READY


def test_download_is_satisfied_only_with_both_audio_and_video(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    assert download_readiness(db, vid) is Readiness.READY
    insert_asset(db, video_id=vid, role="video", format_id="720",
                 path=str(touch(tmp_path / "v.mp4")))
    assert download_readiness(db, vid) is Readiness.SATISFIED


def test_deleting_the_media_makes_download_ready_again(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    audio = touch(tmp_path / "a.m4a")
    insert_asset(db, video_id=vid, role="audio", path=str(audio))
    insert_asset(db, video_id=vid, role="video", format_id="720",
                 path=str(touch(tmp_path / "v.mp4")))
    assert download_readiness(db, vid) is Readiness.SATISFIED
    audio.unlink()
    assert download_readiness(db, vid) is Readiness.READY


def test_local_videos_never_get_a_download_job(db: Database) -> None:
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id="C:/archive/clip.mkv", url=None)
    assert download_readiness(db, vid) is Readiness.BLOCKED
    assert captions_readiness(db, vid) is Readiness.BLOCKED


def test_a_remote_video_without_a_url_is_blocked(db: Database) -> None:
    vid = make_video(db, url=None)
    assert download_readiness(db, vid) is Readiness.BLOCKED


def test_a_missing_video_is_blocked_not_an_error(db: Database) -> None:
    assert download_readiness(db, 4242) is Readiness.BLOCKED
    assert extract_wav_readiness(db, 4242) is Readiness.BLOCKED


def test_captions_is_satisfied_once_the_track_is_on_disk(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    assert captions_readiness(db, vid) is Readiness.READY
    insert_asset(db, video_id=vid, role="captions",
                 path=str(touch(tmp_path / "captions.json3")))
    assert captions_readiness(db, vid) is Readiness.SATISFIED


def test_extract_wav_needs_audio_or_a_container(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    assert extract_wav_readiness(db, vid) is Readiness.BLOCKED
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    assert extract_wav_readiness(db, vid) is Readiness.READY


def test_a_container_alone_is_enough_for_extract_wav(db: Database, tmp_path: Path) -> None:
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id=str(tmp_path / "clip.mkv"), url=None)
    insert_asset(db, video_id=vid, role="container", path=str(touch(tmp_path / "clip.mkv")))
    assert extract_wav_readiness(db, vid) is Readiness.READY


def test_pruning_the_wav_makes_extract_ready_again(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    wav = config.paths().cache_wav(vid)
    touch(wav)
    assert extract_wav_readiness(db, vid) is Readiness.SATISFIED
    wav.unlink()  # `rytp cache prune`
    assert extract_wav_readiness(db, vid) is Readiness.READY
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_jobs_readiness.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.jobs'`.

- [ ] **Step 3: Write `rytp/jobs/__init__.py`**

```python
"""The job-kind registry. design §5.

A job kind is four things: a name, the pool it belongs to, a predicate that
looks at the database and says whether running it makes sense right now, and
a handler. There is deliberately no dependency-edge table — the predicate is
the whole dependency story, which is why pruning a cached file is enough to
make its producing job runnable again.

Adding a kind takes three pieces, exactly like the old engine registry:
define the predicate and handler, call :func:`register_job_kind`, and make
sure the module that calls it gets imported.

**For the other plan parts.** Every ``register_job_kind`` call lives at the
bottom of *this* file, so there is one readable table of every kind in the
project, and so that importing ``rytp.jobs`` never drags a transcriber or
ffmpeg into the process. Contribute two things: a *light* predicate module
in your own package — it may import ``pathlib``, ``rytp.config``,
``rytp.constants`` and ``rytp.db`` and nothing heavier — and a handler thunk
here that imports your stage lazily, in the body, exactly like
:func:`_run_download`. Then add your kind to the block at the bottom.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rytp import constants as C
from rytp.db import Database


class Readiness(Enum):
    """What the database says about one job right now."""

    READY = "ready"
    BLOCKED = "blocked"
    SATISFIED = "satisfied"


#: A readiness predicate looks at the database and the filesystem only. It
#: is deliberately **not** given the payload: a predicate that changed its
#: mind based on how a job was enqueued would stop being a statement about
#: the world, and the self-healing property would go with it.
Predicate = Callable[[Database, int], Readiness]

#: The handler signature is fixed by contracts §5: database, ``target_id``,
#: decoded ``payload_json``, returning nothing. Stage functions may return a
#: descriptive string for the CLI; the thunk below throws it away.
Handler = Callable[[Database, int, dict[str, Any]], None]


@dataclass(frozen=True)
class JobKind:
    """One kind of work the queue knows how to schedule."""

    name: str
    pool: str
    readiness: Predicate
    handler: Handler
    summary: str
    #: What namespace ``jobs.target_id`` lives in for this kind. Every Part 2
    #: kind targets a ``videos.id``; ``render`` targets a cut list, whose
    #: integer id is derived from its name. The queue never mixes namespaces
    #: when it re-evaluates jobs by target.
    target_kind: str = "video"
    #: Whether a full :func:`rytp.jobs.queue.reconcile` may turn this kind's
    #: ``done`` jobs back into work when its output disappears. True for
    #: everything derived from the corpus; False for a one-shot the user
    #: asked for, which must not silently re-run on the next worker start.
    reopenable: bool = True


JOB_KINDS: dict[str, JobKind] = {}

#: The flat view contracts §5 requires: job kind -> the callable that does
#: the work. Written only by :func:`register_job_kind`, so it can never
#: drift from :data:`JOB_KINDS`.
JOB_HANDLERS: dict[str, Handler] = {}


def register_job_kind(kind: JobKind) -> JobKind:
    """Add a kind to the registry. Raises on a duplicate or a bad pool."""
    if kind.name in JOB_KINDS:
        raise ValueError(f"job kind {kind.name!r} is already registered")
    if kind.pool not in C.POOLS:
        raise ValueError(
            f"job kind {kind.name!r} wants pool {kind.pool!r}; "
            f"available: {', '.join(C.POOLS)}"
        )
    JOB_KINDS[kind.name] = kind
    JOB_HANDLERS[kind.name] = kind.handler
    return kind


def resolve_job_kind(name: str) -> JobKind:
    """Look a kind up by name, naming the alternatives when it is missing."""
    try:
        return JOB_KINDS[name]
    except KeyError:
        raise ValueError(
            f"unknown job kind {name!r}; available: {', '.join(sorted(JOB_KINDS))}"
        ) from None


def kinds_for_pool(pool: str) -> tuple[str, ...]:
    """Every registered kind that runs on one pool, sorted."""
    return tuple(sorted(n for n, k in JOB_KINDS.items() if k.pool == pool))


def _run_download(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.acquire import acquire_media

    acquire_media(db, video_id)


def _run_captions(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.acquire.captions import acquire_captions

    acquire_captions(db, video_id)


def _run_extract_wav(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.audio.extract import ensure_wav

    ensure_wav(db, video_id)


# Imported last: readiness.py imports Readiness back out of this module, so
# the registry types must already exist when it runs.
from rytp.jobs.readiness import (  # noqa: E402
    captions_readiness,
    download_readiness,
    extract_wav_readiness,
)

register_job_kind(
    JobKind(
        name="download",
        pool="network",
        readiness=download_readiness,
        handler=_run_download,
        summary="fetch the audio and one video rendition as separate files",
    )
)
register_job_kind(
    JobKind(
        name="captions",
        pool="network",
        readiness=captions_readiness,
        handler=_run_captions,
        summary="fetch the caption track",
    )
)
register_job_kind(
    JobKind(
        name="extract_wav",
        pool="cpu",
        readiness=extract_wav_readiness,
        handler=_run_extract_wav,
        summary="decode the cached 16 kHz mono WAV",
    )
)
```

- [ ] **Step 4: Write `rytp/jobs/readiness.py`**

```python
"""Readiness predicates. design §5.

Every predicate answers from the database and the filesystem as they are
*now*. None of them look at the job row, and in particular none of them look
at ``payload_json`` — a predicate that depended on how a job was enqueued
would stop being a statement about the world and the self-healing property
would go with it.
"""

from __future__ import annotations

import sqlite3

from rytp import constants as C
from rytp.config import paths
from rytp.db import Database
from rytp.db.queries import has_usable_asset
from rytp.jobs import Readiness


def _video(db: Database, video_id: int) -> sqlite3.Row | None:
    return db.conn.execute(
        "SELECT id, source, url FROM videos WHERE id = ?", (video_id,)
    ).fetchone()


def _downloadable(row: sqlite3.Row | None) -> bool:
    """A remote, catalogued video we are allowed to fetch from."""
    return row is not None and row["source"] != C.LOCAL_SOURCE and bool(row["url"])


def download_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable when the video is remote and its media is not on disk.

    Satisfied needs *both* the audio and a video rendition: design §4 keeps
    them as separate assets so a rendition can be upgraded later, and a
    half-finished download must not look complete.
    """
    row = _video(db, video_id)
    if row is None:
        return Readiness.BLOCKED
    if has_usable_asset(db, video_id, "audio") and has_usable_asset(db, video_id, "video"):
        return Readiness.SATISFIED
    if not _downloadable(row):
        return Readiness.BLOCKED
    return Readiness.READY


def captions_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable for every remote video. design §6: captions always, early."""
    row = _video(db, video_id)
    if row is None:
        return Readiness.BLOCKED
    if has_usable_asset(db, video_id, "captions"):
        return Readiness.SATISFIED
    if not _downloadable(row):
        return Readiness.BLOCKED
    return Readiness.READY


def extract_wav_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable once there is something to decode from.

    The cached WAV is not recorded in any table (contracts §7), so its
    presence on disk is the whole answer — delete it and this goes straight
    back to READY.
    """
    row = _video(db, video_id)
    if row is None:
        return Readiness.BLOCKED
    if paths().cache_wav(video_id).exists():
        return Readiness.SATISFIED
    if has_usable_asset(db, video_id, "audio") or has_usable_asset(db, video_id, "container"):
        return Readiness.READY
    return Readiness.BLOCKED
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_jobs_readiness.py -q`
Expected: PASS, 16 passed.

- [ ] **Step 6: Prove the lazy handler imports really are lazy**

The handler thunks must not drag `yt_dlp` into the import path. Run:

`python -c "import sys, rytp.jobs; assert 'yt_dlp' not in sys.modules; print('clean')"`
Expected: `clean`.

- [ ] **Step 7: Commit**

```bash
git add rytp/jobs/__init__.py rytp/jobs/readiness.py tests/test_jobs_readiness.py
git commit -m "feat: derive job readiness from database state, not a dependency graph"
```

---

### Task 4: The job queue

Everything that touches the `jobs` table. Two properties matter more than the rest:

**Claiming is atomic without relying on isolation levels.** The claim is a `SELECT` followed by `UPDATE … WHERE id = ? AND state = 'pending'`; a worker that loses the race sees `rowcount == 0` and re-selects. This is correct whatever `db.transaction()` does underneath and whatever SQLite version is installed, which the old `queue.py` also relied on (`git show 44fc214c:rytp/download/queue.py`).

**`attempts` is incremented at claim time, not at failure time.** A job that kills its worker outright still burns an attempt, so a poison job eventually lands in `failed` instead of looping forever after every restart.

`reconcile` is the other half of self-healing. `UNIQUE (kind, target_id)` means a `done` job cannot be re-enqueued, so something has to notice that the world changed underneath it. `reconcile` re-evaluates every `done` and `blocked` job and reopens the ones that are READY again. The worker calls it at startup, and `cache.prune` and `jobs.retry` call it after they change the world.

**Files:**
- Create: `rytp/jobs/queue.py`
- Test: `tests/test_jobs_queue.py`

**Interfaces:**
- Consumes: `rytp.jobs.{Readiness, resolve_job_kind, JOB_KINDS}`, `rytp.db.queries.{get_setting, set_setting}`.
- Produces:
  - `Job` frozen dataclass: `id, kind, target_id, state, pool, priority, attempts, not_before, last_error, payload` and `Job.from_row(row)`
  - `enqueue(db, kind, target_id, *, priority=0, payload=None, now=None) -> int`
  - `claim(db, pool, *, now=None) -> Job | None`
  - `finish(db, job_id, *, now=None) -> None`
  - `block(db, job_id, *, reason, now=None) -> None`
  - `defer(db, job_id, *, not_before, error=None, refund_attempt=False) -> None`
  - `fail(db, job_id, *, error, now=None) -> None`
  - `reclaim_running(db, *, now=None) -> int`
  - `reconcile(db, *, kinds=None, target_id=None, target_kind=None, now=None) -> int` (sweeps `done` and `blocked`; skips kinds with `reopenable=False`)
  - `unblock(db, *, target_id=None, target_kind=None, now=None) -> int` (sweeps `blocked` only — what the worker calls after every finished job)
  - `retry(db, *, job_id=None, kind=None, state="failed", now=None) -> int`
  - `get_job(db, job_id) -> Job` (raises `NotFoundError`)
  - `list_jobs(db, *, state=None, pool=None, kind=None, limit=C.JOB_LIST_LIMIT) -> list[Job]`
  - `JobStats` and `stats(db, *, now=None) -> JobStats`
  - `pause(db)`, `resume(db)`, `is_paused(db) -> bool`

- [ ] **Step 1: Write the failing tests**

`tests/test_jobs_queue.py`:

```python
"""Tests for the job queue (design §5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rytp import config
from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import insert_asset
from rytp.jobs import Readiness
from rytp.jobs import queue as Q
from rytp.models import NotFoundError

from tests.fakes import make_video, temp_job_kind, touch

NOW = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)


def test_enqueue_starts_a_ready_job_pending(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    job = Q.list_jobs(db)[0]
    assert job.id == job_id
    assert (job.kind, job.state, job.pool, job.attempts) == (
        "download", "pending", "network", 0
    )


def test_enqueue_starts_a_blocked_job_blocked(db: Database) -> None:
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id="C:/clip.mkv", url=None)
    Q.enqueue(db, "download", vid, now=NOW)
    assert Q.list_jobs(db)[0].state == "blocked"


def test_enqueue_marks_an_already_satisfied_job_done(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="captions",
                 path=str(touch(tmp_path / "captions.json3")))
    Q.enqueue(db, "captions", vid, now=NOW)
    assert Q.list_jobs(db)[0].state == "done"


def test_enqueue_is_idempotent_and_keeps_one_row(db: Database) -> None:
    vid = make_video(db)
    first = Q.enqueue(db, "download", vid, now=NOW)
    second = Q.enqueue(db, "download", vid, now=NOW)
    assert first == second
    assert len(Q.list_jobs(db)) == 1


def test_enqueue_raises_priority_but_never_lowers_it(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid, priority=5, now=NOW)
    Q.enqueue(db, "download", vid, priority=1, now=NOW)
    assert Q.list_jobs(db)[0].priority == 5


def test_claim_moves_pending_to_running_and_burns_an_attempt(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid, now=NOW)
    job = Q.claim(db, "network", now=NOW)
    assert job is not None
    assert job.state == "running" and job.attempts == 1
    assert Q.claim(db, "network", now=NOW) is None  # nothing left to claim


def test_claim_ignores_other_pools(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    Q.enqueue(db, "extract_wav", vid, now=NOW)
    assert Q.claim(db, "network", now=NOW) is None
    assert Q.claim(db, "cpu", now=NOW) is not None


def test_claim_respects_priority_then_insertion_order(db: Database) -> None:
    low = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    high = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    Q.enqueue(db, "download", low, priority=0, now=NOW)
    Q.enqueue(db, "download", high, priority=9, now=NOW)
    assert Q.claim(db, "network", now=NOW).target_id == high


def test_claim_skips_a_job_whose_not_before_is_in_the_future(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    later = (NOW + timedelta(minutes=5)).isoformat()
    Q.defer(db, job_id, not_before=later, error="429")
    assert Q.claim(db, "network", now=NOW) is None
    assert Q.claim(db, "network", now=NOW + timedelta(minutes=6)) is not None


def test_defer_can_refund_the_attempt(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.defer(db, job_id, not_before=NOW.isoformat(), error="429", refund_attempt=True)
    job = Q.list_jobs(db)[0]
    assert job.state == "pending" and job.attempts == 0


def test_finish_and_fail_are_terminal(db: Database) -> None:
    vid = make_video(db)
    a = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.finish(db, a, now=NOW)
    done = Q.list_jobs(db)[0]
    assert done.state == "done" and done.last_error is None
    Q.fail(db, a, error="boom", now=NOW)
    assert Q.list_jobs(db)[0].last_error == "boom"


def test_reclaim_running_returns_orphans_to_pending(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)      # the worker is now killed
    assert Q.reclaim_running(db, now=NOW) == 1
    job = Q.list_jobs(db)[0]
    assert job.state == "pending"
    assert job.attempts == 1             # the attempt is still spent


def test_reconcile_reopens_a_done_job_whose_output_vanished(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    wav = touch(config.paths().cache_wav(vid))
    job_id = Q.enqueue(db, "extract_wav", vid, now=NOW)
    assert Q.list_jobs(db)[0].state == "done"

    wav.unlink()                          # `rytp cache prune`
    assert Q.reconcile(db, now=NOW) == 1
    job = Q.list_jobs(db)[0]
    assert job.id == job_id and job.state == "pending"


def test_reconcile_unblocks_a_job_whose_prerequisite_arrived(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    Q.enqueue(db, "extract_wav", vid, now=NOW)
    assert Q.list_jobs(db)[0].state == "blocked"
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    assert Q.reconcile(db, now=NOW) == 1
    assert Q.list_jobs(db)[0].state == "pending"


def test_unblock_touches_only_blocked_jobs(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    Q.enqueue(db, "extract_wav", vid, now=NOW)          # blocked: no audio yet
    Q.enqueue(db, "download", vid, now=NOW)             # pending
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    assert Q.unblock(db, target_id=vid, now=NOW) == 1
    states = {j.kind: j.state for j in Q.list_jobs(db)}
    assert states == {"extract_wav": "pending", "download": "pending"}


def test_unblock_never_reopens_a_done_job(db: Database, tmp_path: Path) -> None:
    # A predicate that still answers READY after a successful run must not
    # send the job round again — that is why the worker calls unblock, not
    # reconcile, after every finish.
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    job_id = Q.enqueue(db, "extract_wav", vid, now=NOW)
    Q.finish(db, job_id, now=NOW)
    assert Q.unblock(db, target_id=vid, now=NOW) == 0
    assert Q.get_job(db, job_id).state == "done"


def test_reconcile_can_be_scoped_to_one_video(db: Database, tmp_path: Path) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    Q.enqueue(db, "extract_wav", a, now=NOW)
    Q.enqueue(db, "extract_wav", b, now=NOW)
    for vid, name in ((a, "a.m4a"), (b, "b.m4a")):
        insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / name)))
    assert Q.reconcile(db, target_id=a, now=NOW) == 1
    states = {j.target_id: j.state for j in Q.list_jobs(db)}
    assert states == {a: "pending", b: "blocked"}


def test_unblock_never_crosses_a_target_namespace(db: Database, tmp_path: Path) -> None:
    # A render's target_id is derived from a cut-list name, so it shares no
    # numbering with videos.id. Finishing one must not re-evaluate the other.
    vid = make_video(db)
    Q.enqueue(db, "extract_wav", vid, now=NOW)          # blocked: no audio yet
    with temp_job_kind(
        "t_render", "cpu", lambda db_, t, p: None,
        readiness=lambda db_, t: Readiness.BLOCKED, target_kind="cutlist",
    ):
        Q.enqueue(db, "t_render", vid, now=NOW)         # same integer, other namespace
        insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
        assert Q.unblock(db, target_id=vid, target_kind="cutlist", now=NOW) == 0
        assert Q.unblock(db, target_id=vid, target_kind="video", now=NOW) == 1
    states = {j.kind: j.state for j in Q.list_jobs(db)}
    assert states["extract_wav"] == "pending"
    assert states["t_render"] == "blocked"


def test_reconcile_never_reopens_a_kind_marked_not_reopenable(db: Database) -> None:
    # A render is something the user asked for once. Re-deriving it on every
    # worker start because its output file is gone would be a nasty surprise.
    vid = make_video(db)
    with temp_job_kind(
        "t_render", "cpu", lambda db_, t, p: None, target_kind="cutlist",
        reopenable=False,
    ):
        job_id = Q.enqueue(db, "t_render", vid, now=NOW)
        Q.finish(db, job_id, now=NOW)
        assert Q.reconcile(db, now=NOW) == 0
        assert Q.get_job(db, job_id).state == "done"
        # Asking again explicitly still re-runs it.
        Q.enqueue(db, "t_render", vid, now=NOW)
        assert Q.get_job(db, job_id).state == "pending"


def test_reconcile_leaves_running_and_failed_jobs_alone(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    assert Q.reconcile(db, now=NOW) == 0
    assert Q.list_jobs(db)[0].state == "running"


def test_retry_resets_failed_jobs_and_clears_the_backoff(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.fail(db, job_id, error="boom", now=NOW)
    assert Q.retry(db, state="failed", now=NOW) == 1
    job = Q.list_jobs(db)[0]
    assert (job.state, job.attempts, job.not_before) == ("pending", 0, None)


def test_retry_can_target_a_single_job(db: Database) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    ja = Q.enqueue(db, "download", a, now=NOW)
    jb = Q.enqueue(db, "download", b, now=NOW)
    Q.fail(db, ja, error="x", now=NOW)
    Q.fail(db, jb, error="y", now=NOW)
    assert Q.retry(db, job_id=ja, now=NOW) == 1
    states = {j.id: j.state for j in Q.list_jobs(db)}
    assert states[ja] == "pending" and states[jb] == "failed"


def test_list_jobs_filters(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    Q.enqueue(db, "download", vid, now=NOW)
    Q.enqueue(db, "extract_wav", vid, now=NOW)
    assert [j.kind for j in Q.list_jobs(db, pool="cpu")] == ["extract_wav"]
    assert [j.kind for j in Q.list_jobs(db, kind="download")] == ["download"]
    assert Q.list_jobs(db, state="failed") == []


def test_get_job_round_trips_and_complains_about_a_bad_id(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, payload={"note": "hi"}, now=NOW)
    job = Q.get_job(db, job_id)
    assert job.kind == "download" and job.payload == {"note": "hi"}
    with pytest.raises(NotFoundError, match="4242"):
        Q.get_job(db, 4242)


def test_stats_reports_throttling_rather_than_a_mystery_stall(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    later = (NOW + timedelta(minutes=5)).isoformat()
    Q.defer(db, job_id, not_before=later, error="HTTP Error 429")
    s = Q.stats(db, now=NOW)
    assert s.by_state["pending"] == 1
    assert s.throttled == 1
    assert s.next_not_before == later
    assert s.paused is False


def test_pause_is_a_setting_not_a_state(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid, now=NOW)
    assert Q.is_paused(db) is False
    Q.pause(db)
    assert Q.is_paused(db) is True
    assert Q.list_jobs(db)[0].state == "pending"   # jobs are untouched
    Q.resume(db)
    assert Q.is_paused(db) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_jobs_queue.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.jobs.queue'`.

- [ ] **Step 3: Write `rytp/jobs/queue.py`**

```python
"""The jobs table. design §5.

State lives entirely in SQLite, so killing the process loses nothing: a
worker that dies mid-job leaves a ``running`` row, and the next worker
reclaims it (:func:`reclaim_running`). Pause is a settings flag rather than
a job state, so pausing never rewrites thousands of rows and never has to be
undone correctly.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import get_setting, set_setting
from rytp.jobs import JOB_KINDS, Readiness, resolve_job_kind
from rytp.models import NotFoundError

#: States a full reconcile may reopen from. ``running`` is excluded because
#: a live worker owns it; ``cancelled`` because a human said no.
_RECONCILABLE: tuple[str, ...] = ("done", "blocked")


@dataclass(frozen=True)
class Job:
    """One row of the jobs table, with payload_json already decoded."""

    id: int
    kind: str
    target_id: int
    state: str
    pool: str
    priority: int
    attempts: int
    not_before: str | None
    last_error: str | None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Job:
        return cls(
            id=int(row["id"]),
            kind=row["kind"],
            target_id=int(row["target_id"]),
            state=row["state"],
            pool=row["pool"],
            priority=int(row["priority"]),
            attempts=int(row["attempts"]),
            not_before=row["not_before"],
            last_error=row["last_error"],
            payload=json.loads(row["payload_json"] or "{}"),
        )


@dataclass(frozen=True)
class JobStats:
    """What ``rytp jobs stats`` shows. design §5: throttling must be visible."""

    by_state: dict[str, int]
    pending_by_pool: dict[str, int]
    throttled: int
    next_not_before: str | None
    paused: bool


def _now(now: datetime | None) -> str:
    return (now or datetime.now(UTC)).isoformat()


def _state_for(readiness: Readiness) -> str:
    return {
        Readiness.READY: "pending",
        Readiness.BLOCKED: "blocked",
        Readiness.SATISFIED: "done",
    }[readiness]


def enqueue(
    db: Database,
    kind: str,
    target_id: int,
    *,
    priority: int = 0,
    payload: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> int:
    """Create or refresh one job, and return its id.

    Idempotent by ``UNIQUE (kind, target_id)``. Re-enqueuing recomputes the
    state from the readiness predicate, so running ``rytp ingest`` twice on a
    video whose media was deleted does the right thing instead of nothing. A
    job somebody is currently running is left strictly alone.
    """
    spec = resolve_job_kind(kind)
    state = _state_for(spec.readiness(db, target_id))
    stamp = _now(now)
    with db.transaction():
        db.conn.execute(
            """
            INSERT INTO jobs (kind, target_id, state, pool, priority, attempts,
                              payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT (kind, target_id) DO UPDATE SET
                priority   = MAX(jobs.priority, excluded.priority),
                state      = CASE WHEN jobs.state = 'running'
                                  THEN jobs.state ELSE excluded.state END,
                not_before = CASE WHEN jobs.state = 'running'
                                  THEN jobs.not_before ELSE NULL END
            """,
            (kind, target_id, state, spec.pool, priority,
             json.dumps(payload or {}), stamp),
        )
        row = db.conn.execute(
            "SELECT id FROM jobs WHERE kind = ? AND target_id = ?", (kind, target_id)
        ).fetchone()
    return int(row["id"])


def claim(db: Database, pool: str, *, now: datetime | None = None) -> Job | None:
    """Atomically take the next runnable job of one pool.

    The guarded ``UPDATE`` is what makes this safe across threads and
    processes: a worker that lost the race updates zero rows and tries the
    next candidate. ``attempts`` is spent here, at claim time, so a job that
    kills its worker outright still counts against
    :data:`rytp.constants.JOB_MAX_ATTEMPTS`.
    """
    stamp = _now(now)
    for _ in range(C.CLAIM_RETRY_LIMIT):
        candidate = db.conn.execute(
            """
            SELECT id FROM jobs
            WHERE state = 'pending' AND pool = ?
              AND (not_before IS NULL OR not_before <= ?)
            ORDER BY priority DESC, id ASC
            LIMIT 1
            """,
            (pool, stamp),
        ).fetchone()
        if candidate is None:
            return None
        cur = db.conn.execute(
            """
            UPDATE jobs
            SET state = 'running', started_at = ?, finished_at = NULL,
                attempts = attempts + 1, last_error = NULL
            WHERE id = ? AND state = 'pending'
            """,
            (stamp, candidate["id"]),
        )
        if cur.rowcount == 1:
            row = db.conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (candidate["id"],)
            ).fetchone()
            return Job.from_row(row)
    return None


def finish(db: Database, job_id: int, *, now: datetime | None = None) -> None:
    """Mark a job done."""
    db.conn.execute(
        "UPDATE jobs SET state = 'done', finished_at = ?, last_error = NULL "
        "WHERE id = ?",
        (_now(now), job_id),
    )


def block(db: Database, job_id: int, *, reason: str, now: datetime | None = None) -> None:
    """Park a job whose prerequisites are not there. Reconcile reopens it."""
    db.conn.execute(
        "UPDATE jobs SET state = 'blocked', finished_at = ?, last_error = ? "
        "WHERE id = ?",
        (_now(now), reason, job_id),
    )


def defer(
    db: Database,
    job_id: int,
    *,
    not_before: str,
    error: str | None = None,
    refund_attempt: bool = False,
) -> None:
    """Put a job back in the queue, invisible until ``not_before``.

    ``refund_attempt`` is for throttling: being rate-limited says nothing
    about this job, so it must not push it towards ``failed``.
    """
    # ``started_at`` is deliberately left alone: the attempt reached the
    # network, so it must keep counting against the daily cap.
    db.conn.execute(
        "UPDATE jobs SET state = 'pending', not_before = ?, last_error = ?, "
        "attempts = MAX(attempts - ?, 0) WHERE id = ?",
        (not_before, error, 1 if refund_attempt else 0, job_id),
    )


def fail(db: Database, job_id: int, *, error: str, now: datetime | None = None) -> None:
    """Terminal failure. Only ``rytp jobs retry`` brings it back."""
    db.conn.execute(
        "UPDATE jobs SET state = 'failed', finished_at = ?, last_error = ? WHERE id = ?",
        (_now(now), error[: C.JOB_ERROR_MAX_CHARS], job_id),
    )


def reclaim_running(db: Database, *, now: datetime | None = None) -> int:
    """Return every ``running`` job to ``pending``. Called at worker startup.

    A killed worker leaves rows claiming to be running forever; this is what
    stops them stranding. The spent attempt is deliberately not refunded.
    """
    cur = db.conn.execute(
        "UPDATE jobs SET state = 'pending', started_at = NULL, "
        "last_error = 'reclaimed from a worker that did not finish' "
        "WHERE state = 'running'"
    )
    del now
    return int(cur.rowcount)


def _reevaluate(
    db: Database,
    states: Sequence[str],
    *,
    kinds: Sequence[str] | None,
    target_id: int | None,
    target_kind: str | None,
    now: datetime | None,
) -> int:
    """Re-ask the readiness predicates about settled jobs. design §5.

    ``UNIQUE (kind, target_id)`` means a ``done`` job cannot simply be
    enqueued again, so this is the path by which the world changing — a
    pruned WAV, a deleted rendition, a prerequisite that finally arrived —
    turns back into work. Returns how many jobs changed state.
    """
    wanted = tuple(kinds) if kinds else tuple(JOB_KINDS)
    if target_kind is not None:
        # ``target_id`` means different things to different kinds — a video
        # for everything Part 2 owns, a cut list for ``render`` — so
        # re-evaluating "everything with this target_id" must never cross
        # the namespace boundary.
        wanted = tuple(k for k in wanted if JOB_KINDS[k].target_kind == target_kind)
    if "done" in states:
        wanted = tuple(k for k in wanted if JOB_KINDS[k].reopenable)
    if not wanted:
        return 0
    kind_marks = ", ".join("?" for _ in wanted)
    state_marks = ", ".join("?" for _ in states)
    sql = (
        f"SELECT id, kind, target_id, state FROM jobs "  # noqa: S608 - names are ours
        f"WHERE state IN ({state_marks}) AND kind IN ({kind_marks})"
    )
    params: list[Any] = [*states, *wanted]
    if target_id is not None:
        sql += " AND target_id = ?"
        params.append(target_id)

    changed = 0
    for row in db.conn.execute(sql, params).fetchall():
        spec = JOB_KINDS.get(row["kind"])
        if spec is None:
            continue
        new_state = _state_for(spec.readiness(db, int(row["target_id"])))
        if new_state != row["state"]:
            db.conn.execute(
                "UPDATE jobs SET state = ?, not_before = NULL, finished_at = ? "
                "WHERE id = ? AND state = ?",
                (new_state, None if new_state == "pending" else _now(now),
                 row["id"], row["state"]),
            )
            changed += 1
    return changed


def reconcile(
    db: Database,
    *,
    kinds: Sequence[str] | None = None,
    target_id: int | None = None,
    target_kind: str | None = None,
    now: datetime | None = None,
) -> int:
    """Full sweep of settled jobs: ``done`` and ``blocked`` alike.

    Called where the world may have changed behind the queue's back — worker
    startup, ``cache.prune``, ``jobs.retry``. Not called after every finished
    job: a predicate that still answers READY right after a successful run
    would be flipped straight back to ``pending`` and loop. Kinds marked
    ``reopenable=False`` are never reopened from ``done``, so a render the
    user asked for once does not silently re-run on the next worker start.
    """
    return _reevaluate(db, _RECONCILABLE, kinds=kinds, target_id=target_id,
                       target_kind=target_kind, now=now)


def unblock(
    db: Database,
    *,
    target_id: int | None = None,
    target_kind: str | None = None,
    now: datetime | None = None,
) -> int:
    """Re-check only the ``blocked`` jobs. Safe to call constantly.

    This is what closes the pipeline: ``ingest`` parks ``extract_wav`` as
    blocked because there is no audio yet, and the moment the download job
    finishes this turns it into work. Scoped to one target by default,
    because a full sweep stats every asset of every settled job and the
    corpus is ~1,600 videos.
    """
    return _reevaluate(db, ("blocked",), kinds=None, target_id=target_id,
                       target_kind=target_kind, now=now)


def retry(
    db: Database,
    *,
    job_id: int | None = None,
    kind: str | None = None,
    state: str = "failed",
    now: datetime | None = None,
) -> int:
    """Send settled jobs back to ``pending`` with a clean slate."""
    del now
    where = ["state = ?"]
    params: list[Any] = [state]
    if job_id is not None:
        where.append("id = ?")
        params.append(job_id)
    if kind is not None:
        where.append("kind = ?")
        params.append(kind)
    cur = db.conn.execute(
        "UPDATE jobs SET state = 'pending', attempts = 0, not_before = NULL, "  # noqa: S608
        "last_error = NULL, started_at = NULL, finished_at = NULL "
        f"WHERE {' AND '.join(where)}",
        params,
    )
    return int(cur.rowcount)


def get_job(db: Database, job_id: int) -> Job:
    """One job by id, for the command layer's result rows."""
    row = db.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"job {job_id} does not exist")
    return Job.from_row(row)


def list_jobs(
    db: Database,
    *,
    state: str | None = None,
    pool: str | None = None,
    kind: str | None = None,
    limit: int = C.JOB_LIST_LIMIT,
) -> list[Job]:
    """Jobs, newest priority first, for the CLI and the TUI."""
    where: list[str] = []
    params: list[Any] = []
    for column, value in (("state", state), ("pool", pool), ("kind", kind)):
        if value:
            where.append(f"{column} = ?")
            params.append(value)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)
    rows = db.conn.execute(
        f"SELECT * FROM jobs {clause} ORDER BY priority DESC, id ASC LIMIT ?",  # noqa: S608
        params,
    ).fetchall()
    return [Job.from_row(r) for r in rows]


def stats(db: Database, *, now: datetime | None = None) -> JobStats:
    """Counts by state and pool, plus how much of the queue is throttled."""
    stamp = _now(now)
    by_state = {
        row["state"]: int(row["n"])
        for row in db.conn.execute(
            "SELECT state, COUNT(*) AS n FROM jobs GROUP BY state"
        ).fetchall()
    }
    pending_by_pool = {
        row["pool"]: int(row["n"])
        for row in db.conn.execute(
            "SELECT pool, COUNT(*) AS n FROM jobs WHERE state = 'pending' GROUP BY pool"
        ).fetchall()
    }
    throttled_row = db.conn.execute(
        "SELECT COUNT(*) AS n, MIN(not_before) AS next FROM jobs "
        "WHERE state = 'pending' AND not_before IS NOT NULL AND not_before > ?",
        (stamp,),
    ).fetchone()
    return JobStats(
        by_state={s: by_state.get(s, 0) for s in
                  ("pending", "running", "done", "failed", "blocked", "cancelled")},
        pending_by_pool={p: pending_by_pool.get(p, 0) for p in C.POOLS},
        throttled=int(throttled_row["n"]),
        next_not_before=throttled_row["next"],
        paused=is_paused(db),
    )


def pause(db: Database) -> None:
    """Stop every pool from claiming. Instant, lock-free, loses nothing."""
    set_setting(db, C.SETTING_QUEUE_PAUSED, "1")


def resume(db: Database) -> None:
    set_setting(db, C.SETTING_QUEUE_PAUSED, "0")


def is_paused(db: Database) -> bool:
    return get_setting(db, C.SETTING_QUEUE_PAUSED) == "1"
```

- [ ] **Step 4: Add the one constant `fail` needs**

`fail` truncates, so append to the jobs section of `rytp/constants.py`:

```python
#: Longest error string stored in ``jobs.last_error``. A yt-dlp traceback can
#: run to kilobytes and the tail is never the informative part.
JOB_ERROR_MAX_CHARS: int = 500

#: How much of an error to show in one ``rytp jobs list`` cell. Wider than
#: this and the table stops fitting in a terminal.
JOB_ERROR_PREVIEW_CHARS: int = 80
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_jobs_queue.py -q`
Expected: PASS, 25 passed.

- [ ] **Step 6: Prove the claim really is atomic across connections**

Add this to `tests/test_jobs_queue.py` — it is the one property the whole worker rests on, so it gets a real two-connection test rather than a mocked one:

```python
def test_two_connections_cannot_claim_the_same_job(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid, now=NOW)
    other = Database(config.paths().db)
    try:
        first = Q.claim(db, "network", now=NOW)
        second = Q.claim(other, "network", now=NOW)
    finally:
        other.close()
    assert first is not None
    assert second is None
```

Run: `python -m pytest tests/test_jobs_queue.py -q`
Expected: PASS, 26 passed.

- [ ] **Step 7: Commit**

```bash
git add rytp/jobs/queue.py rytp/constants.py tests/test_jobs_queue.py
git commit -m "feat: add the crash-safe job queue with reconcile"
```

---

### Task 5: The yt-dlp seam, channel enumeration and probing

yt-dlp lives behind a `Protocol` with exactly one real implementation and one fake, which is the pattern the old code already used (`git show 44fc214c:rytp/download/ytdlp.py`). **No test in this repository may reach the network**, so every test in Part 2 passes a `FakeYtDlpRunner`.

Two decisions carry design weight:

**One yt-dlp invocation fetches both files, unmerged.** The format selector joins the audio and video selectors with a **comma**, which is yt-dlp's "download several formats of the same video" separator (README, FORMAT SELECTION: `-f 136/137/mp4/bestvideo,140/m4a/bestaudio`). Unlike `+`, it does not invoke the muxer, so the two files stay separate — which is exactly what design §4 needs for a rendition to be upgradeable later without touching the audio a transcript is aligned to. After the download, `info["requested_downloads"]` holds one entry per format with its own `filepath`, so nothing has to sniff the files with ffprobe the way the old code did.

**Enumeration covers three listings.** Design §13: *"Enumeration of the main video listing found fewer videos than expected, most likely because live streams are listed separately. Cataloguing must cover the streams and shorts listings as well as the main one."* A tab that does not exist is not an error — a channel with no shorts simply yields nothing for that tab.

**Files:**
- Create: `rytp/acquire/ytdlp.py`
- Modify: `tests/fakes.py` (append `FakeYtDlpRunner` and its canned data)
- Test: `tests/test_acquire_ytdlp.py`

**Interfaces:**
- Consumes: Task 1's `classify_error`, `ErrorKind`, `RateLimited`, `VideoUnavailable`, `AcquireError`; `rytp.models.ChannelEntry`; `rytp.constants.CHANNEL_TABS`.
- Produces:
  - `FetchedFile(path, format_id, ext, vcodec, acodec, width, height, abr, size_bytes, language)` frozen dataclass
  - `YtDlpRunner` runtime-checkable protocol: `list_entries`, `probe`, `download`, `download_captions`
  - `RealYtDlpRunner` — the only implementation that touches the network
  - `enumerate_channel(channel_url, *, tabs=C.CHANNEL_TABS, runner=None) -> list[ChannelEntry]`
  - `probe_video(url, *, runner=None) -> ChannelEntry`
  - `tests/fakes.py::FakeYtDlpRunner`, `AUDIO_FILE_SPEC`, `VIDEO_FILE_SPEC`, `CAPTION_FILE_SPEC`

- [ ] **Step 1: Append the fake runner to `tests/fakes.py`**

```python
AUDIO_FILE_SPEC = {
    "name": "140.m4a", "format_id": "140", "ext": "m4a",
    "vcodec": "none", "acodec": "mp4a.40.2", "abr": 129.5, "language": "ru",
}
VIDEO_FILE_SPEC = {
    "name": "136.mp4", "format_id": "136", "ext": "mp4",
    "vcodec": "avc1.4d401f", "acodec": "none", "width": 1280, "height": 720,
}
CAPTION_FILE_SPEC = {
    "name": "captions.ru-orig.json3", "format_id": "ru-orig", "ext": "json3",
    "vcodec": None, "acodec": None, "language": "ru-orig",
}


class FakeYtDlpRunner:
    """A YtDlpRunner that writes small real files and never opens a socket.

    ``entries`` maps a tab URL to the raw yt-dlp entry dicts it should
    return; ``raises`` makes the next download blow up, which is how the
    throttle and unavailable paths are tested.
    """

    def __init__(
        self,
        *,
        entries: dict[str, list[dict[str, Any]]] | None = None,
        info: dict[str, Any] | None = None,
        files: list[dict[str, Any]] | None = None,
        captions: list[dict[str, Any]] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.entries = entries or {}
        self.info = info or {}
        self.files = [AUDIO_FILE_SPEC, VIDEO_FILE_SPEC] if files is None else files
        self.captions = [CAPTION_FILE_SPEC] if captions is None else captions
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def list_entries(self, url: str) -> list[dict[str, Any]]:
        self.calls.append({"op": "list_entries", "url": url})
        if self.raises is not None:
            raise self.raises
        if url not in self.entries:
            raise RuntimeError("ERROR: This channel does not have a videos tab")
        return self.entries[url]

    def probe(self, url: str) -> dict[str, Any]:
        self.calls.append({"op": "probe", "url": url})
        if self.raises is not None:
            raise self.raises
        return self.info

    def _emit(self, out_template: str, specs: list[dict[str, Any]]) -> list[Any]:
        from rytp.acquire.ytdlp import FetchedFile

        parent = Path(out_template).parent
        made = []
        for spec in specs:
            path = touch(parent / spec["name"])
            made.append(
                FetchedFile(
                    path=path,
                    format_id=spec.get("format_id"),
                    ext=spec.get("ext", path.suffix.lstrip(".")),
                    vcodec=spec.get("vcodec"),
                    acodec=spec.get("acodec"),
                    width=spec.get("width"),
                    height=spec.get("height"),
                    abr=spec.get("abr"),
                    size_bytes=path.stat().st_size,
                    language=spec.get("language"),
                )
            )
        return made

    def download(
        self, url: str, *, out_template: str, format_selector: str,
        rate_limit_bps: int | None, sleep_requests_s: float,
    ) -> list[Any]:
        self.calls.append({
            "op": "download", "url": url, "out_template": out_template,
            "format_selector": format_selector, "rate_limit_bps": rate_limit_bps,
            "sleep_requests_s": sleep_requests_s,
        })
        if self.raises is not None:
            raise self.raises
        return self._emit(out_template, self.files)

    def download_captions(
        self, url: str, *, out_template: str, langs: tuple[str, ...],
        sub_format: str, sleep_requests_s: float,
    ) -> list[Any]:
        self.calls.append({
            "op": "download_captions", "url": url, "out_template": out_template,
            "langs": langs, "sub_format": sub_format,
        })
        if self.raises is not None:
            raise self.raises
        return self._emit(out_template, self.captions)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_acquire_ytdlp.py`:

```python
"""Tests for the yt-dlp seam, enumeration and probing. No network, ever."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.acquire.policy import RateLimited, VideoUnavailable
from rytp.acquire.ytdlp import YtDlpRunner, enumerate_channel, probe_video, tab_url

from tests.fakes import CHANNEL_ONE_URL, FakeYtDlpRunner


def _entry(external_id: str, title: str, duration: float | None = 3600.0) -> dict:
    return {
        "id": external_id,
        "title": title,
        "url": f"https://example.invalid/watch/{external_id}",
        "duration": duration,
        "upload_date": "20200102",
    }


def test_the_fake_satisfies_the_protocol() -> None:
    assert isinstance(FakeYtDlpRunner(), YtDlpRunner)


@pytest.mark.parametrize(
    ("given", "tab", "expected"),
    [
        (CHANNEL_ONE_URL, "videos", f"{CHANNEL_ONE_URL}/videos"),
        (CHANNEL_ONE_URL + "/", "streams", f"{CHANNEL_ONE_URL}/streams"),
        (CHANNEL_ONE_URL + "/videos", "shorts", f"{CHANNEL_ONE_URL}/shorts"),
    ],
)
def test_tab_url_normalises_whatever_the_user_pasted(
    given: str, tab: str, expected: str
) -> None:
    assert tab_url(given, tab) == expected


def test_enumeration_covers_videos_streams_and_shorts() -> None:
    runner = FakeYtDlpRunner(entries={
        f"{CHANNEL_ONE_URL}/videos": [_entry("VIDEO_A", "A")],
        f"{CHANNEL_ONE_URL}/streams": [_entry("VIDEO_B", "B", 10800.0)],
        f"{CHANNEL_ONE_URL}/shorts": [_entry("VIDEO_C", "C", 45.0)],
    })
    got = enumerate_channel(CHANNEL_ONE_URL, runner=runner)
    assert [(e.external_id, e.kind) for e in got] == [
        ("VIDEO_A", "video"), ("VIDEO_B", "livestream"), ("VIDEO_C", "short")
    ]
    assert got[0].duration_ms == 3_600_000
    assert got[0].published_at == "2020-01-02"


def test_enumeration_keeps_tab_order_and_does_not_dedupe() -> None:
    # A live stream listed in both /videos and /streams must come back twice,
    # streams last, so the caller's upsert lands on kind='livestream'.
    runner = FakeYtDlpRunner(entries={
        f"{CHANNEL_ONE_URL}/videos": [_entry("VIDEO_B", "B")],
        f"{CHANNEL_ONE_URL}/streams": [_entry("VIDEO_B", "B")],
        f"{CHANNEL_ONE_URL}/shorts": [],
    })
    got = enumerate_channel(CHANNEL_ONE_URL, runner=runner)
    assert [e.kind for e in got] == ["video", "livestream"]


def test_a_channel_without_a_shorts_tab_is_not_an_error() -> None:
    runner = FakeYtDlpRunner(entries={
        f"{CHANNEL_ONE_URL}/videos": [_entry("VIDEO_A", "A")],
        f"{CHANNEL_ONE_URL}/streams": [],
        # no /shorts key at all -> the fake raises "does not have a videos tab"
    })
    got = enumerate_channel(CHANNEL_ONE_URL, runner=runner)
    assert [e.external_id for e in got] == ["VIDEO_A"]


def test_enumeration_stops_on_a_throttle_instead_of_hammering_the_next_tab() -> None:
    runner = FakeYtDlpRunner(raises=RuntimeError("ERROR: HTTP Error 429: Too Many Requests"))
    with pytest.raises(RateLimited):
        enumerate_channel(CHANNEL_ONE_URL, runner=runner)
    assert len(runner.calls) == 1


def test_entries_without_an_id_are_skipped() -> None:
    runner = FakeYtDlpRunner(entries={
        f"{CHANNEL_ONE_URL}/videos": [{"title": "orphan"}, _entry("VIDEO_A", "A")],
        f"{CHANNEL_ONE_URL}/streams": [],
        f"{CHANNEL_ONE_URL}/shorts": [],
    })
    assert [e.external_id for e in enumerate_channel(CHANNEL_ONE_URL, runner=runner)] == [
        "VIDEO_A"
    ]


def test_probe_video_returns_one_channel_entry() -> None:
    runner = FakeYtDlpRunner(info={
        "id": "VIDEO_A", "title": "A", "duration": 90.5,
        "webpage_url": "https://example.invalid/watch/VIDEO_A",
        "upload_date": "20240315", "live_status": "not_live",
    })
    entry = probe_video("https://example.invalid/watch/VIDEO_A", runner=runner)
    assert entry.external_id == "VIDEO_A"
    assert entry.kind == "video"
    assert entry.duration_ms == 90_500
    assert entry.published_at == "2024-03-15"


def test_probe_video_recognises_a_past_live_stream() -> None:
    runner = FakeYtDlpRunner(info={
        "id": "VIDEO_B", "title": "B", "duration": 10800.0,
        "webpage_url": "https://example.invalid/watch/VIDEO_B",
        "live_status": "was_live",
    })
    assert probe_video("https://example.invalid/watch/VIDEO_B", runner=runner).kind == (
        "livestream"
    )


def test_probe_video_maps_a_dead_video_to_video_unavailable() -> None:
    runner = FakeYtDlpRunner(raises=RuntimeError("ERROR: Private video"))
    with pytest.raises(VideoUnavailable):
        probe_video("https://example.invalid/watch/VIDEO_A", runner=runner)


def test_enumeration_asks_for_every_configured_tab() -> None:
    runner = FakeYtDlpRunner(entries={f"{CHANNEL_ONE_URL}/{t}": [] for t in C.CHANNEL_TABS})
    enumerate_channel(CHANNEL_ONE_URL, runner=runner)
    assert [c["url"] for c in runner.calls] == [
        f"{CHANNEL_ONE_URL}/{t}" for t in C.CHANNEL_TABS
    ]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_acquire_ytdlp.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.acquire.ytdlp'`.

- [ ] **Step 4: Write `rytp/acquire/ytdlp.py`**

```python
"""The yt-dlp seam. design §5, §13.

Everything that talks to the network in this project goes through
:class:`YtDlpRunner`, and only :class:`RealYtDlpRunner` implements it for
real — tests always pass a fake. ``yt_dlp`` itself is imported inside
methods, never at module scope, so the CLI stays importable without the
optional extra (contracts §1).

No browser cookies anywhere in this file. design §5: yt-dlp's own guidance
warns that cookie-authenticated bulk access risks account and IP bans, and
anonymous access is sufficient for everything here, captions included.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rytp import constants as C
from rytp.acquire.policy import (
    AcquireError,
    ErrorKind,
    RateLimited,
    VideoUnavailable,
    classify_error,
)
from rytp.models import ChannelEntry

#: yt-dlp ``live_status`` values that mean "this was a stream".
_LIVE_STATUSES: frozenset[str] = frozenset({"is_live", "was_live", "post_live", "is_upcoming"})


@dataclass(frozen=True)
class FetchedFile:
    """One file yt-dlp actually wrote, as described by its format dict."""

    path: Path
    format_id: str | None
    ext: str
    vcodec: str | None
    acodec: str | None
    width: int | None = None
    height: int | None = None
    abr: float | None = None
    size_bytes: int = 0
    language: str | None = None

    @property
    def has_video(self) -> bool:
        return bool(self.vcodec) and self.vcodec != "none"

    @property
    def has_audio(self) -> bool:
        return bool(self.acodec) and self.acodec != "none"


@runtime_checkable
class YtDlpRunner(Protocol):
    """Pluggable runner so tests can substitute a fake."""

    def list_entries(self, url: str) -> list[dict[str, Any]]: ...

    def probe(self, url: str) -> dict[str, Any]: ...

    def download(
        self,
        url: str,
        *,
        out_template: str,
        format_selector: str,
        rate_limit_bps: int | None,
        sleep_requests_s: float,
    ) -> list[FetchedFile]: ...

    def download_captions(
        self,
        url: str,
        *,
        out_template: str,
        langs: tuple[str, ...],
        sub_format: str,
        sleep_requests_s: float,
    ) -> list[FetchedFile]: ...


def translate_error(exc: Exception) -> AcquireError:
    """Turn any runner failure into the exception the worker knows."""
    message = str(exc)
    kind = classify_error(message)
    if kind is ErrorKind.THROTTLED:
        return RateLimited(message)
    if kind is ErrorKind.UNAVAILABLE:
        return VideoUnavailable(message)
    return AcquireError(message)


def tab_url(channel_url: str, tab: str) -> str:
    """Build a channel's per-listing URL, tolerating a pasted tab suffix."""
    base = channel_url.rstrip("/")
    for known in C.CHANNEL_TABS:
        if base.endswith(f"/{known}"):
            base = base[: -(len(known) + 1)]
            break
    return f"{base}/{tab}"


def _published_at(info: dict[str, Any]) -> str | None:
    raw = info.get("upload_date") or info.get("release_date")
    if isinstance(raw, str) and len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return None


def _duration_ms(info: dict[str, Any]) -> int | None:
    raw = info.get("duration")
    if isinstance(raw, (int, float)):
        return int(round(raw * C.MS_PER_SECOND))
    return None


def _to_entry(info: dict[str, Any], *, kind: str, fallback_url: str) -> ChannelEntry | None:
    external_id = info.get("id")
    if not external_id:
        return None
    return ChannelEntry(
        external_id=str(external_id),
        title=str(info.get("title") or external_id),
        url=str(info.get("webpage_url") or info.get("url") or fallback_url),
        duration_ms=_duration_ms(info),
        kind=kind,
        published_at=_published_at(info),
    )


def enumerate_channel(
    channel_url: str,
    *,
    tabs: Sequence[str] = C.CHANNEL_TABS,
    runner: YtDlpRunner | None = None,
) -> list[ChannelEntry]:
    """List a channel's videos, live streams and shorts, in that order.

    Entries are **not** deduplicated: a stream that appears in both
    ``/videos`` and ``/streams`` is returned twice, streams last, so the
    caller's upsert by ``(source, external_id)`` settles on the more
    specific ``kind``. A tab a channel does not have is skipped silently —
    that is not a failure — but a throttle response aborts immediately
    rather than walking into the next tab.
    """
    runner = runner or RealYtDlpRunner()
    out: list[ChannelEntry] = []
    for tab in tabs:
        url = tab_url(channel_url, tab)
        try:
            raw = runner.list_entries(url)
        except Exception as exc:  # noqa: BLE001 - every runner failure is translated
            translated = translate_error(exc)
            if isinstance(translated, RateLimited):
                raise translated from exc
            continue
        # Which listing a video came from is a far better kind signal than
        # guessing from the entry dict. design §13: streams are listed
        # separately, and missing them undercounts the corpus.
        kind = C.CHANNEL_TAB_KINDS.get(tab, "other")
        for info in raw:
            entry = _to_entry(info, kind=kind, fallback_url=url)
            if entry is not None:
                out.append(entry)
    return out


def probe_video(url: str, *, runner: YtDlpRunner | None = None) -> ChannelEntry:
    """One metadata request for ``rytp videos add <url>``. No download."""
    runner = runner or RealYtDlpRunner()
    try:
        info = runner.probe(url)
    except Exception as exc:  # noqa: BLE001
        raise translate_error(exc) from exc
    kind = "livestream" if info.get("live_status") in _LIVE_STATUSES else "video"
    entry = _to_entry(info, kind=kind, fallback_url=url)
    if entry is None:
        raise AcquireError(f"no video id in the metadata for {url}")
    return entry


class RealYtDlpRunner:
    """The only implementation that touches the network."""

    def _ydl(self, opts: dict[str, Any]) -> Any:
        try:
            import yt_dlp
        except ImportError as exc:  # pragma: no cover - exercised by hand
            raise AcquireError(
                "yt-dlp is not installed. Install it with "
                "`pip install -e '.[yt-dlp]'`"
            ) from exc
        base: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "continuedl": True,
            "retries": C.YTDLP_RETRIES,
            "fragment_retries": C.YTDLP_RETRIES,
            "concurrent_fragment_downloads": 1,
        }
        base.update(opts)
        return yt_dlp.YoutubeDL(base)

    def list_entries(self, url: str) -> list[dict[str, Any]]:
        with self._ydl({"extract_flat": "in_playlist", "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        entries = (info or {}).get("entries") or []
        return [e for e in entries if e]

    def probe(self, url: str) -> dict[str, Any]:
        with self._ydl({"skip_download": True}) as ydl:
            return dict(ydl.extract_info(url, download=False) or {})

    def download(
        self,
        url: str,
        *,
        out_template: str,
        format_selector: str,
        rate_limit_bps: int | None,
        sleep_requests_s: float,
    ) -> list[FetchedFile]:
        opts: dict[str, Any] = {
            "format": format_selector,
            "outtmpl": {"default": out_template},
            "sleep_interval_requests": sleep_requests_s,
        }
        if rate_limit_bps:
            opts["ratelimit"] = rate_limit_bps
        with self._ydl(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        return _fetched_files(info or {})

    def download_captions(
        self,
        url: str,
        *,
        out_template: str,
        langs: tuple[str, ...],
        sub_format: str,
        sleep_requests_s: float,
    ) -> list[FetchedFile]:
        opts: dict[str, Any] = {
            "skip_download": True,
            "writesubtitles": True,
            # design §6: the "-orig" track is auto-generated, so this must
            # be on or ru-orig is never offered.
            "writeautomaticsub": True,
            "subtitleslangs": list(langs),
            "subtitlesformat": sub_format,
            "outtmpl": {"default": out_template},
            "sleep_interval_requests": sleep_requests_s,
        }
        with self._ydl(opts) as ydl:
            info = ydl.extract_info(url, download=True) or {}
        out: list[FetchedFile] = []
        for lang, sub in (info.get("requested_subtitles") or {}).items():
            filepath = sub.get("filepath")
            if not filepath:
                continue
            path = Path(filepath)
            out.append(
                FetchedFile(
                    path=path,
                    format_id=lang,
                    ext=sub.get("ext") or path.suffix.lstrip("."),
                    vcodec=None,
                    acodec=None,
                    size_bytes=path.stat().st_size if path.exists() else 0,
                    language=lang,
                )
            )
        if out:
            return out
        # Older yt-dlp builds do not put ``filepath`` on the subtitle dicts.
        # The files are still written next to the template as
        # ``<stem>.<lang>.<ext>``, so find them rather than fail.
        stem = Path(out_template).name.split(".")[0]
        for path in sorted(Path(out_template).parent.glob(f"{stem}.*.{sub_format}")):
            lang = path.name[len(stem) + 1 : -(len(sub_format) + 1)]
            out.append(
                FetchedFile(
                    path=path,
                    format_id=lang,
                    ext=sub_format,
                    vcodec=None,
                    acodec=None,
                    size_bytes=path.stat().st_size,
                    language=lang,
                )
            )
        return out


def _fetched_files(info: dict[str, Any]) -> list[FetchedFile]:
    """Read the per-format results out of a completed download.

    With a comma-joined selector yt-dlp does not merge, so every entry of
    ``requested_downloads`` is one real file with its own ``filepath``.
    """
    out: list[FetchedFile] = []
    for fmt in info.get("requested_downloads") or []:
        filepath = fmt.get("filepath")
        if not filepath:
            continue
        path = Path(filepath)
        out.append(
            FetchedFile(
                path=path,
                format_id=fmt.get("format_id"),
                ext=fmt.get("ext") or path.suffix.lstrip("."),
                vcodec=fmt.get("vcodec"),
                acodec=fmt.get("acodec"),
                width=fmt.get("width"),
                height=fmt.get("height"),
                abr=fmt.get("abr"),
                size_bytes=int(fmt.get("filesize") or fmt.get("filesize_approx") or
                               (path.stat().st_size if path.exists() else 0)),
                language=fmt.get("language"),
            )
        )
    return out
```

- [ ] **Step 5: Add the one constant `RealYtDlpRunner` needs**

Append to the download-policy section of `rytp/constants.py`:

```python
#: yt-dlp ``retries`` and ``fragment_retries``. Low on purpose: the job
#: queue's own backoff ladder is the real retry mechanism, and hammering
#: inside one invocation is what looks like abuse.
YTDLP_RETRIES: int = 2
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_acquire_ytdlp.py -q`
Expected: PASS, 13 passed.

- [ ] **Step 7: Check for forbidden identifiers**

Contracts §1 bans real video ids, channel ids, channel names and URLs from every committed file. Run:

`grep -rnE "youtube\.com|youtu\.be|/@[A-Za-z0-9_]+|[A-Za-z0-9_-]{11}" rytp/acquire tests/fakes.py tests/test_acquire_ytdlp.py | grep -v example.invalid`
Expected: no line that is an actual identifier (matches inside ordinary words are fine).

- [ ] **Step 8: Commit**

```bash
git add rytp/acquire/ytdlp.py rytp/constants.py tests/fakes.py tests/test_acquire_ytdlp.py
git commit -m "feat: add the yt-dlp seam, three-tab channel enumeration and probing"
```

---

### Task 6: The download stage

The handler behind the `download` job kind. One yt-dlp invocation, two files, two asset rows, nothing merged.

**Files:**
- Modify: `rytp/acquire/__init__.py` (replace the stub from Task 1)
- Test: `tests/test_acquire_media.py`

**Interfaces:**
- Consumes: Task 2's `insert_asset` / `prune_missing_assets`, Task 5's `FetchedFile` / `YtDlpRunner` / `RealYtDlpRunner`, Task 1's `DownloadPolicy` and errors, `rytp.config.paths` / `ensure_dir`.
- Produces: `rytp.acquire.acquire_media(db, video_id, *, runner=None, policy=None) -> str` — the one-line message the worker records.

- [ ] **Step 1: Write the failing tests**

`tests/test_acquire_media.py`:

```python
"""Tests for the download stage (design §4, §5). No network, ever."""

from __future__ import annotations

import pytest

from rytp import config
from rytp import constants as C
from rytp.acquire import acquire_media
from rytp.acquire.policy import AcquireError, RateLimited, VideoUnavailable
from rytp.db import Database
from rytp.db.queries import asset_for, assets_for, insert_asset
from rytp.jobs.readiness import download_readiness
from rytp.jobs import Readiness

from tests.fakes import AUDIO_FILE_SPEC, FakeYtDlpRunner, make_video


def test_audio_and_video_land_as_two_separate_assets(db: Database) -> None:
    vid = make_video(db)
    runner = FakeYtDlpRunner()
    message = acquire_media(db, vid, runner=runner)

    audio = asset_for(db, vid, "audio")
    video = asset_for(db, vid, "video")
    assert audio is not None and video is not None
    assert audio["path"].endswith("audio.m4a")
    assert video["path"].endswith("video-136.mp4")
    assert audio["abr"] == 129.5
    assert (video["width"], video["height"]) == (1280, 720)
    assert audio["bytes"] > 0
    assert "audio.m4a" in message and "video-136.mp4" in message


def test_the_two_files_live_under_the_contracted_layout(db: Database) -> None:
    vid = make_video(db)
    acquire_media(db, vid, runner=FakeYtDlpRunner())
    expected_dir = config.paths().media_dir(vid)
    for row in assets_for(db, vid):
        assert row["path"].startswith(str(expected_dir))


def test_nothing_is_merged(db: Database) -> None:
    # design §4: a merged file would make a rendition upgrade impossible
    # without re-fetching the audio a transcript is aligned to.
    vid = make_video(db)
    runner = FakeYtDlpRunner()
    acquire_media(db, vid, runner=runner)
    call = next(c for c in runner.calls if c["op"] == "download")
    assert "," in call["format_selector"]
    assert "+" not in call["format_selector"]
    assert call["format_selector"] == f"{C.DOWNLOAD_FORMAT_AUDIO},{C.DOWNLOAD_FORMAT_VIDEO}"


def test_the_rate_limit_from_settings_reaches_the_runner(db: Database) -> None:
    from rytp.db.queries import set_setting

    set_setting(db, C.SETTING_DOWNLOAD_RATE_LIMIT, "42000")
    vid = make_video(db)
    runner = FakeYtDlpRunner()
    acquire_media(db, vid, runner=runner)
    call = next(c for c in runner.calls if c["op"] == "download")
    assert call["rate_limit_bps"] == 42000


def test_the_job_becomes_satisfied_afterwards(db: Database) -> None:
    vid = make_video(db)
    assert download_readiness(db, vid) is Readiness.READY
    acquire_media(db, vid, runner=FakeYtDlpRunner())
    assert download_readiness(db, vid) is Readiness.SATISFIED


def test_a_rendition_upgrade_leaves_the_audio_alone(db: Database) -> None:
    vid = make_video(db)
    acquire_media(db, vid, runner=FakeYtDlpRunner())
    audio_before = asset_for(db, vid, "audio")

    upgraded = dict(AUDIO_FILE_SPEC)
    hd = {"name": "137.mp4", "format_id": "137", "ext": "mp4",
          "vcodec": "avc1.640028", "acodec": "none", "width": 1920, "height": 1080}
    acquire_media(db, vid, runner=FakeYtDlpRunner(files=[upgraded, hd]))

    assert {r["format_id"] for r in assets_for(db, vid, "video")} == {"136", "137"}
    audio_after = asset_for(db, vid, "audio")
    assert audio_after["path"] == audio_before["path"]


def test_stale_asset_rows_are_pruned_before_a_refetch(db: Database, tmp_path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="video", format_id="999",
                 path=str(tmp_path / "gone.mp4"))
    acquire_media(db, vid, runner=FakeYtDlpRunner())
    assert {r["format_id"] for r in assets_for(db, vid, "video")} == {"136"}


def test_a_local_video_is_refused(db: Database, tmp_path) -> None:
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id=str(tmp_path / "clip.mkv"), url=None)
    with pytest.raises(AcquireError, match="local"):
        acquire_media(db, vid, runner=FakeYtDlpRunner())


def test_a_missing_video_is_refused(db: Database) -> None:
    with pytest.raises(AcquireError, match="4242"):
        acquire_media(db, 4242, runner=FakeYtDlpRunner())


def test_no_audio_format_is_a_plain_failure(db: Database) -> None:
    vid = make_video(db)
    video_only = {"name": "136.mp4", "format_id": "136", "ext": "mp4",
                  "vcodec": "avc1", "acodec": "none"}
    with pytest.raises(AcquireError, match="no audio"):
        acquire_media(db, vid, runner=FakeYtDlpRunner(files=[video_only]))


def test_no_video_rendition_keeps_the_audio_and_fails_permanently(db: Database) -> None:
    vid = make_video(db)
    audio_only = dict(AUDIO_FILE_SPEC)
    with pytest.raises(VideoUnavailable, match="no video"):
        acquire_media(db, vid, runner=FakeYtDlpRunner(files=[audio_only]))
    # The audio we did get is kept — throwing it away would mean fetching it twice.
    assert asset_for(db, vid, "audio") is not None


def test_runner_failures_are_translated(db: Database) -> None:
    vid = make_video(db)
    runner = FakeYtDlpRunner(raises=RuntimeError("ERROR: HTTP Error 429: Too Many Requests"))
    with pytest.raises(RateLimited):
        acquire_media(db, vid, runner=runner)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_acquire_media.py -q`
Expected: collection error, `ImportError: cannot import name 'acquire_media'`.

- [ ] **Step 3: Rewrite `rytp/acquire/__init__.py`**

```python
"""Acquisition: turning a catalogued video into assets on disk. design §5.

The stage entry points live here; the yt-dlp glue is in
:mod:`rytp.acquire.ytdlp` and the politeness rules in
:mod:`rytp.acquire.policy`.

design §4 is the rule this module exists to keep: **audio and video are
downloaded together and kept as separate files.** yt-dlp fetches them
separately anyway, and not merging them is what makes a later 360p -> 1080p
rendition upgrade free — the audio a transcript is aligned to is never
touched.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.acquire.policy import (
    AcquireError,
    DownloadPolicy,
    VideoUnavailable,
)
from rytp.acquire.ytdlp import (
    FetchedFile,
    RealYtDlpRunner,
    YtDlpRunner,
    translate_error,
)
from rytp.db import Database
from rytp.db.queries import insert_asset, prune_missing_assets

__all__ = ["acquire_media", "fetchable_video"]


def fetchable_video(db: Database, video_id: int) -> sqlite3.Row:
    """The catalog row, or a one-line explanation of why we cannot fetch it."""
    row = db.conn.execute(
        "SELECT id, source, url, title FROM videos WHERE id = ?", (video_id,)
    ).fetchone()
    if row is None:
        raise AcquireError(f"video {video_id} is not in the catalog")
    if row["source"] == C.LOCAL_SOURCE:
        raise AcquireError(
            f"video {video_id} is a local file; local files register as a "
            f"container asset and never get download jobs"
        )
    if not row["url"]:
        raise AcquireError(f"video {video_id} has no url to fetch from")
    return row


def _place(src: Path, dst: Path) -> Path:
    """Move a downloaded file to its contracted name. Same directory, so
    ``os.replace`` is atomic on Windows as well as POSIX."""
    if src != dst:
        os.replace(src, dst)
    return dst


def acquire_media(
    db: Database,
    video_id: int,
    *,
    runner: YtDlpRunner | None = None,
    policy: DownloadPolicy | None = None,
) -> str:
    """Fetch the audio and one video rendition as two separate assets."""
    row = fetchable_video(db, video_id)
    policy = policy or DownloadPolicy.from_settings(db)
    runner = runner or RealYtDlpRunner()

    prune_missing_assets(db, video_id)
    out_dir = config.ensure_dir(config.paths().media_dir(video_id))
    selector = f"{policy.format_audio},{policy.format_video}"

    try:
        files = runner.download(
            row["url"],
            out_template=str(out_dir / "%(format_id)s.%(ext)s"),
            format_selector=selector,
            rate_limit_bps=policy.rate_limit_bps,
            sleep_requests_s=policy.sleep_requests_s,
        )
    except Exception as exc:  # noqa: BLE001 - translated into our taxonomy
        raise translate_error(exc) from exc

    audio = next((f for f in files if f.has_audio and not f.has_video), None)
    video = next((f for f in files if f.has_video), None)

    if audio is None:
        raise AcquireError(
            f"video {video_id}: no audio-only format came back from "
            f"selector {policy.format_audio!r}"
        )
    audio_path = _place(audio.path, out_dir / f"audio.{audio.ext}")
    insert_asset(
        db,
        video_id=video_id,
        role="audio",
        path=str(audio_path),
        format_id=audio.format_id,
        size_bytes=audio.size_bytes or audio_path.stat().st_size,
        abr=audio.abr,
    )

    if video is None:
        # The audio is already recorded — re-fetching it later would be a
        # second pointless request. The job fails permanently and shows up
        # in `rytp jobs list --state failed`.
        raise VideoUnavailable(
            f"video {video_id}: audio saved, but no video rendition came back "
            f"from selector {policy.format_video!r}"
        )
    video_path = _place(video.path, out_dir / f"video-{video.format_id}.{video.ext}")
    insert_asset(
        db,
        video_id=video_id,
        role="video",
        path=str(video_path),
        format_id=video.format_id,
        size_bytes=video.size_bytes or video_path.stat().st_size,
        width=video.width,
        height=video.height,
    )
    return f"video {video_id}: {audio_path.name} + {video_path.name}"
```

`FetchedFile` is imported for the type annotations the implementer may add; if `ruff` flags it as unused, drop it from the import list.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_acquire_media.py -q`
Expected: PASS, 12 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/acquire/__init__.py tests/test_acquire_media.py
git commit -m "feat: download audio and a video rendition as separate assets"
```

---

### Task 7: Captions

Design §6: *"Captions are pulled early and always. They are tiny, they cost no GPU time, and they are the first thing to disappear when a video is delisted."* This task only **acquires** the file; parsing json3 into `words` is Part 3's `rytp/transcribe/captions.py` and is deliberately not here.

**Files:**
- Create: `rytp/acquire/captions.py`
- Test: `tests/test_acquire_captions.py`

**Interfaces:**
- Consumes: Task 6's `fetchable_video`, Task 5's runner, Task 2's `insert_asset`.
- Produces: `rytp.acquire.captions.acquire_captions(db, video_id, *, runner=None, policy=None) -> str`

- [ ] **Step 1: Write the failing tests**

`tests/test_acquire_captions.py`:

```python
"""Tests for caption acquisition (design §6). No network, ever."""

from __future__ import annotations

import pytest

from rytp import config
from rytp import constants as C
from rytp.acquire.captions import acquire_captions
from rytp.acquire.policy import NoCaptions, RateLimited
from rytp.db import Database
from rytp.db.queries import asset_for

from tests.fakes import CAPTION_FILE_SPEC, FakeYtDlpRunner, make_video


def test_the_track_becomes_a_captions_asset(db: Database) -> None:
    vid = make_video(db)
    message = acquire_captions(db, vid, runner=FakeYtDlpRunner())
    row = asset_for(db, vid, "captions")
    assert row is not None
    assert row["path"] == str(config.paths().media_dir(vid) / "captions.json3")
    assert row["format_id"] == "ru-orig"
    assert "captions.json3" in message


def test_auto_captions_are_requested_because_ru_orig_is_auto(db: Database) -> None:
    vid = make_video(db)
    runner = FakeYtDlpRunner()
    acquire_captions(db, vid, runner=runner)
    call = next(c for c in runner.calls if c["op"] == "download_captions")
    assert call["langs"] == C.CAPTION_LANGS
    assert call["sub_format"] == C.CAPTION_FORMAT


def test_the_most_preferred_language_wins(db: Database) -> None:
    vid = make_video(db)
    plain_ru = dict(CAPTION_FILE_SPEC, name="captions.ru.json3",
                    format_id="ru", language="ru")
    runner = FakeYtDlpRunner(captions=[plain_ru, dict(CAPTION_FILE_SPEC)])
    acquire_captions(db, vid, runner=runner)
    assert asset_for(db, vid, "captions")["format_id"] == "ru-orig"


def test_a_video_with_no_track_fails_permanently(db: Database) -> None:
    vid = make_video(db)
    with pytest.raises(NoCaptions, match="no caption track"):
        acquire_captions(db, vid, runner=FakeYtDlpRunner(captions=[]))


def test_runner_failures_are_translated(db: Database) -> None:
    vid = make_video(db)
    runner = FakeYtDlpRunner(raises=RuntimeError("ERROR: HTTP Error 429"))
    with pytest.raises(RateLimited):
        acquire_captions(db, vid, runner=runner)


def test_refetching_replaces_rather_than_duplicates(db: Database) -> None:
    from rytp.db.queries import assets_for

    vid = make_video(db)
    acquire_captions(db, vid, runner=FakeYtDlpRunner())
    acquire_captions(db, vid, runner=FakeYtDlpRunner())
    assert len(assets_for(db, vid, "captions")) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_acquire_captions.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.acquire.captions'`.

- [ ] **Step 3: Write `rytp/acquire/captions.py`**

```python
"""Caption acquisition. design §6.

Captions are pulled for every catalogued video, early and always: they are
tiny, they cost no GPU time, and they are the first thing to vanish when a
video is delisted. This module only fetches the file and records the asset —
turning json3 into ``words`` rows is Part 3's job.
"""

from __future__ import annotations

import os

from rytp import config
from rytp.acquire import fetchable_video
from rytp.acquire.policy import DownloadPolicy, NoCaptions
from rytp.acquire.ytdlp import RealYtDlpRunner, YtDlpRunner, translate_error
from rytp.db import Database
from rytp.db.queries import insert_asset


def acquire_captions(
    db: Database,
    video_id: int,
    *,
    runner: YtDlpRunner | None = None,
    policy: DownloadPolicy | None = None,
) -> str:
    """Fetch the caption track and record it as the video's captions asset."""
    row = fetchable_video(db, video_id)
    policy = policy or DownloadPolicy.from_settings(db)
    runner = runner or RealYtDlpRunner()

    out_dir = config.ensure_dir(config.paths().media_dir(video_id))
    try:
        tracks = runner.download_captions(
            row["url"],
            out_template=str(out_dir / "captions.%(ext)s"),
            langs=policy.caption_langs,
            sub_format=policy.caption_format,
            sleep_requests_s=policy.sleep_requests_s,
        )
    except Exception as exc:  # noqa: BLE001 - translated into our taxonomy
        raise translate_error(exc) from exc

    if not tracks:
        raise NoCaptions(
            f"video {video_id}: no caption track in any of "
            f"{', '.join(policy.caption_langs)}"
        )

    order = {lang: i for i, lang in enumerate(policy.caption_langs)}
    chosen = min(tracks, key=lambda t: order.get(t.format_id or "", len(order)))

    # Contracts §7 names the file captions.json3 regardless of which language
    # tag won, so downstream never has to guess at the suffix.
    dest = out_dir / f"captions.{chosen.ext}"
    if chosen.path != dest:
        os.replace(chosen.path, dest)
    insert_asset(
        db,
        video_id=video_id,
        role="captions",
        path=str(dest),
        format_id=chosen.format_id,
        size_bytes=chosen.size_bytes or dest.stat().st_size,
    )
    return f"video {video_id}: {dest.name} ({chosen.format_id})"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_acquire_captions.py -q`
Expected: PASS, 6 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/acquire/captions.py tests/test_acquire_captions.py
git commit -m "feat: fetch captions as their own asset for every catalogued video"
```

---

### Task 8: Local files

Design §5: *"Local files register as assets and never get download jobs."* Design §4: a local file with embedded audio is a single `container` asset. Part 1 stores the resolved absolute path in `videos.external_id` for `source='local'` rows, because contracts §3 dropped every path column from `videos`.

**Files:**
- Create: `rytp/acquire/local.py`
- Test: `tests/test_acquire_local.py`

**Interfaces:**
- Consumes: Task 2's `insert_asset`, Task 1's `AcquireError` / `MissingAssetError`.
- Produces: `rytp.acquire.local.register_local_container(db, video_id) -> str`

- [ ] **Step 1: Write the failing tests**

`tests/test_acquire_local.py`:

```python
"""Tests for local file registration (design §4, §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import constants as C
from rytp.acquire.local import register_local_container
from rytp.acquire.policy import AcquireError, MissingAssetError
from rytp.db import Database
from rytp.db.queries import asset_for, assets_for
from rytp.jobs import Readiness
from rytp.jobs.readiness import download_readiness, extract_wav_readiness

from tests.fakes import make_video, touch


def _local(db: Database, path: Path) -> int:
    return make_video(db, source=C.LOCAL_SOURCE, external_id=str(path), url=None,
                      title=path.name)


def test_a_local_file_becomes_a_container_asset(db: Database, tmp_path: Path) -> None:
    clip = touch(tmp_path / "clip.mkv", b"\x00" * 64)
    vid = _local(db, clip)
    message = register_local_container(db, vid)
    row = asset_for(db, vid, "container")
    assert row is not None
    assert row["path"] == str(clip)
    assert row["bytes"] == 64
    assert "clip.mkv" in message


def test_the_file_is_referenced_in_place_and_not_copied(
    db: Database, tmp_path: Path
) -> None:
    clip = touch(tmp_path / "elsewhere" / "clip.mkv")
    vid = _local(db, clip)
    register_local_container(db, vid)
    assert clip.exists()
    assert asset_for(db, vid, "container")["path"] == str(clip)


def test_registration_unblocks_extract_but_never_download(
    db: Database, tmp_path: Path
) -> None:
    clip = touch(tmp_path / "clip.mkv")
    vid = _local(db, clip)
    register_local_container(db, vid)
    assert extract_wav_readiness(db, vid) is Readiness.READY
    assert download_readiness(db, vid) is Readiness.BLOCKED


def test_registering_twice_keeps_one_row(db: Database, tmp_path: Path) -> None:
    clip = touch(tmp_path / "clip.mkv")
    vid = _local(db, clip)
    register_local_container(db, vid)
    register_local_container(db, vid)
    assert len(assets_for(db, vid, "container")) == 1


def test_a_missing_file_is_reported_in_one_line(db: Database, tmp_path: Path) -> None:
    vid = _local(db, tmp_path / "gone.mkv")
    with pytest.raises(MissingAssetError, match="gone.mkv"):
        register_local_container(db, vid)


def test_a_remote_video_is_refused(db: Database) -> None:
    vid = make_video(db)
    with pytest.raises(AcquireError, match="not a local file"):
        register_local_container(db, vid)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_acquire_local.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.acquire.local'`.

- [ ] **Step 3: Write `rytp/acquire/local.py`**

```python
"""Local files. design §4, §5.

A local file with embedded audio is a single ``container`` asset, referenced
where it already lives — nothing is copied into the data tree. It never gets
a download job; the only work it needs is WAV extraction.

Contracts §3 dropped every path column from ``videos``, so Part 1 stores the
resolved absolute path in ``videos.external_id`` for ``source='local'``
rows, which also makes ``UNIQUE (source, external_id)`` deduplicate
re-registration of the same file.
"""

from __future__ import annotations

from pathlib import Path

from rytp import constants as C
from rytp.acquire.policy import AcquireError, MissingAssetError
from rytp.db import Database
from rytp.db.queries import insert_asset


def register_local_container(db: Database, video_id: int) -> str:
    """Record a catalogued local file as its video's container asset."""
    row = db.conn.execute(
        "SELECT id, source, external_id FROM videos WHERE id = ?", (video_id,)
    ).fetchone()
    if row is None:
        raise AcquireError(f"video {video_id} is not in the catalog")
    if row["source"] != C.LOCAL_SOURCE:
        raise AcquireError(
            f"video {video_id} is not a local file (source={row['source']!r})"
        )
    path = Path(row["external_id"] or "")
    if not path.is_file():
        raise MissingAssetError(f"video {video_id}: {path} is not a file")
    insert_asset(
        db,
        video_id=video_id,
        role="container",
        path=str(path),
        size_bytes=path.stat().st_size,
    )
    return f"video {video_id}: registered container {path.name}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_acquire_local.py -q`
Expected: PASS, 6 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/acquire/local.py tests/test_acquire_local.py
git commit -m "feat: register local files as container assets"
```

---

### Task 9: The WAV cache

Contracts §7: `cache/wav/{video_id}.wav`. Design §4: *"`data/audio/{video_id}.wav` is no longer a column anywhere. It is a regenerable cache with an explicit prune command."* Two rules follow and both are tested:

- **Nothing stores a path to it in a table.** The only way to ask whether it exists is to look at the filesystem, which is exactly what `extract_wav_readiness` does.
- **A killed extraction must not leave a plausible-looking file.** ffmpeg writes to `{video_id}.wav.part` and the result is moved into place with `os.replace`, which is atomic on Windows as well as POSIX.

The design text still says `data/audio/{video_id}.wav`; contracts §7 supersedes it with `cache/wav/`, and Part 1's `paths().cache_wav(video_id)` is the single source of that path.

**Files:**
- Create: `rytp/audio/__init__.py`
- Create: `rytp/audio/extract.py`
- Test: `tests/test_audio_extract.py`

**Interfaces:**
- Consumes: Task 2's `asset_for`, Task 1's `MissingAssetError` / `AcquireError`, `rytp.config.paths` / `ensure_dir`.
- Produces:
  - `rytp.audio.extract.wav_path(video_id) -> Path`
  - `rytp.audio.extract.source_asset(db, video_id) -> sqlite3.Row`
  - `rytp.audio.extract.ensure_wav(db, video_id, *, overwrite=False) -> Path`
  - `rytp.audio.extract.prune_wav_cache(db, *, video_id=None, dry_run=False) -> list[tuple[Path, int]]`
  - `rytp.audio.extract.FfmpegNotFoundError`, `ExtractionError` (both `AcquireError` subclasses)
  - `rytp.audio.extract._run_ffmpeg(cmd) -> subprocess.CompletedProcess[str]` and `rytp.audio.extract._ffmpeg_binary() -> str | None` — the two module-level seams tests monkeypatch, so no test ever needs to touch the global `shutil`

- [ ] **Step 1: Write the failing tests**

`tests/test_audio_extract.py`:

```python
"""Tests for the regenerable WAV cache (design §4, contracts §7)."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from rytp import config
from rytp import constants as C
from rytp.acquire.policy import MissingAssetError
from rytp.audio import extract as E
from rytp.db import Database
from rytp.db.queries import insert_asset

from tests.fakes import make_video, touch

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _fake_ffmpeg(monkeypatch: pytest.MonkeyPatch, *, rc: int = 0, write: bool = True):
    """Replace the ffmpeg call with one that writes the output file itself."""
    seen: list[list[str]] = []

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        if write and rc == 0:
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"RIFF....WAVE")
        return subprocess.CompletedProcess(cmd, rc, "", "boom" if rc else "")

    monkeypatch.setattr(E, "_run_ffmpeg", run)
    return seen


def test_the_cache_path_is_the_contracted_one(db: Database) -> None:
    vid = make_video(db)
    assert E.wav_path(vid) == config.paths().cache_wav(vid)
    assert E.wav_path(vid).parent.name == "wav"


def test_extraction_writes_the_cache_file(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    seen = _fake_ffmpeg(monkeypatch)
    out = E.ensure_wav(db, vid)
    assert out == E.wav_path(vid) and out.exists()
    cmd = seen[0]
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == str(C.AUDIO_CHANNELS)
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == str(C.AUDIO_SAMPLE_RATE_HZ)
    assert cmd[-1].endswith(".wav")


def test_nothing_records_the_wav_path_in_the_database(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    _fake_ffmpeg(monkeypatch)
    E.ensure_wav(db, vid)
    rows = db.conn.execute("SELECT path FROM assets").fetchall()
    assert all("cache" not in r["path"] for r in rows)


def test_a_second_call_is_a_no_op(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    seen = _fake_ffmpeg(monkeypatch)
    E.ensure_wav(db, vid)
    E.ensure_wav(db, vid)
    assert len(seen) == 1
    E.ensure_wav(db, vid, overwrite=True)
    assert len(seen) == 2


def test_a_container_is_used_when_there_is_no_audio_asset(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = touch(tmp_path / "clip.mkv")
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id=str(clip), url=None)
    insert_asset(db, video_id=vid, role="container", path=str(clip))
    seen = _fake_ffmpeg(monkeypatch)
    E.ensure_wav(db, vid)
    assert str(clip) in seen[0]


def test_the_audio_asset_wins_over_a_container(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = touch(tmp_path / "a.m4a")
    clip = touch(tmp_path / "clip.mkv")
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="container", path=str(clip))
    insert_asset(db, video_id=vid, role="audio", path=str(audio))
    seen = _fake_ffmpeg(monkeypatch)
    E.ensure_wav(db, vid)
    assert str(audio) in seen[0]


def test_no_source_asset_is_reported_in_one_line(db: Database) -> None:
    vid = make_video(db)
    with pytest.raises(MissingAssetError, match="no audio or container"):
        E.ensure_wav(db, vid)


def test_a_failed_ffmpeg_leaves_no_partial_file(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    _fake_ffmpeg(monkeypatch, rc=1)
    with pytest.raises(E.ExtractionError, match="boom"):
        E.ensure_wav(db, vid)
    assert not E.wav_path(vid).exists()
    assert list(E.wav_path(vid).parent.glob("*.part")) == []


def test_a_missing_ffmpeg_says_how_to_install_it(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    monkeypatch.setattr(E, "_ffmpeg_binary", lambda: None)
    with pytest.raises(E.FfmpegNotFoundError, match="ffmpeg.org"):
        E.ensure_wav(db, vid)


def test_prune_removes_everything_and_reports_bytes(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    for vid in (a, b):
        touch(E.wav_path(vid), b"\x00" * 100)
    freed = E.prune_wav_cache(db)
    assert sorted(size for _, size in freed) == [100, 100]
    assert not E.wav_path(a).exists() and not E.wav_path(b).exists()


def test_prune_can_target_one_video(db: Database) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    for vid in (a, b):
        touch(E.wav_path(vid))
    E.prune_wav_cache(db, video_id=a)
    assert not E.wav_path(a).exists()
    assert E.wav_path(b).exists()


def test_prune_dry_run_deletes_nothing(db: Database) -> None:
    vid = make_video(db)
    touch(E.wav_path(vid), b"\x00" * 10)
    freed = E.prune_wav_cache(db, dry_run=True)
    assert freed == [(E.wav_path(vid), 10)]
    assert E.wav_path(vid).exists()


def test_prune_on_an_empty_cache_is_fine(db: Database) -> None:
    assert E.prune_wav_cache(db) == []


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_real_ffmpeg_produces_16k_mono_pcm(db: Database, tmp_path: Path) -> None:
    src = tmp_path / "tone.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=1", "-ac", "2", "-ar", "44100", str(src)],
        check=True, capture_output=True,
    )
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(src))
    out = E.ensure_wav(db, vid)
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == C.AUDIO_CHANNELS
        assert w.getframerate() == C.AUDIO_SAMPLE_RATE_HZ
        assert w.getsampwidth() == 2  # int16
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_audio_extract.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.audio'`.

- [ ] **Step 3: Create `rytp/audio/__init__.py`**

```python
"""Audio processing. design §4, §6."""

from __future__ import annotations
```

- [ ] **Step 4: Add the audio constants**

Append to `rytp/constants.py` (its own section; if Part 1 already defines `AUDIO_SAMPLE_RATE_HZ` and `AUDIO_CHANNELS`, use theirs and skip this step):

```python
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
```

- [ ] **Step 5: Write `rytp/audio/extract.py`**

```python
"""The regenerable WAV cache. design §4, contracts §7.

``cache/wav/{video_id}.wav`` is 16 kHz mono int16 and is **not recorded in
any table**. Its presence on disk is the whole truth about it, which is what
lets ``rytp cache prune`` free gigabytes and have the extract jobs quietly
become runnable again (design §5).

ffmpeg writes to a ``.part`` sibling that is moved into place only on
success, so a worker killed mid-decode never leaves a truncated file that
looks like a valid cache entry.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.acquire.policy import AcquireError, MissingAssetError
from rytp.db import Database
from rytp.db.queries import asset_for


class FfmpegNotFoundError(AcquireError):
    """ffmpeg is not on PATH."""


class ExtractionError(AcquireError):
    """ffmpeg returned non-zero, or produced nothing."""


def wav_path(video_id: int) -> Path:
    """Where this video's cached WAV lives. Contracts §7."""
    return config.paths().cache_wav(video_id)


def source_asset(db: Database, video_id: int) -> sqlite3.Row:
    """The best thing to decode from: the audio asset, else the container."""
    for role in ("audio", "container"):
        row = asset_for(db, video_id, role)
        if row is not None and Path(row["path"]).exists():
            return row
    raise MissingAssetError(
        f"video {video_id}: no audio or container asset on disk to extract from"
    )


def _ffmpeg_binary() -> str | None:
    """Where ffmpeg is. A module attribute, so a test can replace just this
    rather than reaching into the global ``shutil``."""
    return shutil.which("ffmpeg")


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """The single subprocess call, isolated so tests can replace it."""
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def ensure_wav(db: Database, video_id: int, *, overwrite: bool = False) -> Path:
    """Decode the cached 16 kHz mono WAV, unless it is already there."""
    out = wav_path(video_id)
    if out.exists() and not overwrite:
        return out

    row = source_asset(db, video_id)
    ffmpeg = _ffmpeg_binary()
    if ffmpeg is None:
        raise FfmpegNotFoundError(
            "ffmpeg not found on PATH. Install it from https://ffmpeg.org/ "
            "(on Windows, unzip the release and add its bin/ directory to PATH)"
        )

    config.ensure_dir(out.parent)
    partial = out.with_suffix(".wav.part")
    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i", row["path"],
        "-vn",                      # audio only, even from a container
        "-map", "0:a:0",            # the first audio stream, explicitly
        "-ac", str(C.AUDIO_CHANNELS),
        "-ar", str(C.AUDIO_SAMPLE_RATE_HZ),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(partial),
    ]
    result = _run_ffmpeg(cmd)
    if result.returncode != 0 or not partial.exists():
        partial.unlink(missing_ok=True)
        raise ExtractionError(
            f"video {video_id}: ffmpeg failed (rc={result.returncode}): "
            f"{result.stderr[-C.FFMPEG_ERROR_TAIL_CHARS:]}"
        )
    os.replace(partial, out)
    return out


def prune_wav_cache(
    db: Database, *, video_id: int | None = None, dry_run: bool = False
) -> list[tuple[Path, int]]:
    """Delete cached WAVs and report what was freed.

    Safe by construction: the cache is regenerable and nothing references it
    by path, so the worst case is that some extract jobs run again.
    """
    del db  # the cache is filesystem-only on purpose
    if video_id is not None:
        candidates = [wav_path(video_id)]
    else:
        cache_dir = config.paths().cache_wav(0).parent
        candidates = sorted(cache_dir.glob("*.wav")) if cache_dir.is_dir() else []

    freed: list[tuple[Path, int]] = []
    for path in candidates:
        if not path.is_file():
            continue
        freed.append((path, path.stat().st_size))
        if not dry_run:
            path.unlink()
    return freed
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_audio_extract.py -q`
Expected: PASS, 14 passed (13 passed + 1 skipped if ffmpeg is absent).

- [ ] **Step 7: Commit**

```bash
git add rytp/audio/__init__.py rytp/audio/extract.py rytp/constants.py tests/test_audio_extract.py
git commit -m "feat: add the regenerable 16 kHz WAV cache with prune"
```

---

### Task 10: The worker

Design §5: three pools run simultaneously, so one video downloads while another transcribes. The worker is the process that makes that happen, and it carries the three properties the rest of the part depends on:

**Throttle backoff freezes the pool, not the job.** If a 429 only deferred the job that hit it, the network loop would immediately claim the next download and hit the same server a second later — precisely the behaviour that gets an IP blocked. A throttle writes a cooldown into `settings`, the network loop checks it *before claiming anything*, and the same timestamp is copied into the job's `not_before` so `rytp jobs stats` shows a number instead of a mystery stall. The throttled job gets its attempt refunded: being rate-limited says nothing about that job.

**A killed worker strands nothing.** The lease in `settings` records pid and a heartbeat. A worker starting up takes over a lease whose heartbeat has gone stale and calls `reclaim_running`, which returns every orphaned `running` row to `pending`. A worker whose lease is still fresh refuses to start a second copy.

**Startup re-asks what is runnable.** `reconcile` runs before the pools do, so a WAV pruned while the worker was down turns back into work without anyone enqueuing anything.

**Files:**
- Create: `rytp/jobs/worker.py`
- Modify: `tests/fakes.py` (append `temp_job_kind`)
- Test: `tests/test_jobs_worker.py`

**Interfaces:**
- Consumes: Tasks 1, 3 and 4 in full; `rytp.config.paths`.
- Produces:
  - `WorkerOptions(pools=C.POOLS, once=False, max_jobs=0, poll_interval_s=C.WORKER_POLL_INTERVAL_S)`
  - `WorkerReport(claimed, done, failed, deferred, blocked)` (mutable dataclass)
  - `WorkerAlreadyRunning(RytpError)`
  - `acquire_lease(db, *, now=None) -> None`, `heartbeat(db, *, now=None) -> None`, `release_lease(db) -> None`
  - `run_pool_once(db, pool, *, policy, rng, report, sleep, now_fn) -> bool`
  - `run_worker(db, options, *, sleep=time.sleep, rng=None, now_fn=utcnow, open_db=Database) -> WorkerReport`

- [ ] **Step 1: Append the job-kind test helper to `tests/fakes.py`**

```python
@contextmanager
def temp_job_kind(
    name: str,
    pool: str,
    handler: Callable[[Any, int, dict[str, Any]], None],
    readiness: Callable[[Any, int], Any] | None = None,
    *,
    target_kind: str = "video",
    reopenable: bool = True,
) -> Iterator[None]:
    """Register a throwaway job kind for one test, then remove it.

    JOB_KINDS and JOB_HANDLERS are module-level registries, so a test that
    added to them permanently would leak into every later test.
    """
    from rytp.jobs import JOB_HANDLERS, JOB_KINDS, JobKind, Readiness, register_job_kind

    register_job_kind(
        JobKind(
            name=name,
            pool=pool,
            readiness=readiness or (lambda db, target: Readiness.READY),
            handler=handler,
            summary="test kind",
            target_kind=target_kind,
            reopenable=reopenable,
        )
    )
    try:
        yield
    finally:
        JOB_KINDS.pop(name, None)
        JOB_HANDLERS.pop(name, None)
```

Add `from contextlib import contextmanager`, `from collections.abc import Callable, Iterator` to the imports at the top of `tests/fakes.py`.

- [ ] **Step 2: Write the failing tests**

`tests/test_jobs_worker.py`:

```python
"""Tests for the worker (design §5). No network, no threads unless asked."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from rytp import constants as C
from rytp.acquire.policy import (
    DownloadPolicy,
    RateLimited,
    VideoUnavailable,
    cooldown_until,
    is_cooling_down,
)
from rytp.db import Database
from rytp.db.queries import insert_asset, set_setting
from rytp.jobs import Readiness, queue as Q, worker as W

from tests.fakes import make_video, temp_job_kind, touch

NOW = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)


def _fixed_now():
    return NOW


def _tick(db: Database, pool: str, *, report=None, sleeps=None, now=NOW):
    """Run exactly one pool iteration with everything deterministic."""
    report = report or W.WorkerReport()
    sleeps = sleeps if sleeps is not None else []
    did = W.run_pool_once(
        db,
        pool,
        policy=DownloadPolicy.from_settings(db),
        rng=random.Random(7),
        report=report,
        sleep=sleeps.append,
        now_fn=lambda: now,
    )
    return did, report, sleeps


def test_a_successful_job_is_marked_done(db: Database) -> None:
    calls: list[int] = []

    def handler(db_, t, p):
        calls.append(t)

    with temp_job_kind("t_ok", "cpu", handler):
        vid = make_video(db)
        Q.enqueue(db, "t_ok", vid, now=NOW)
        did, report, _ = _tick(db, "cpu")
    assert did is True
    assert calls == [vid]
    assert report.done == 1
    assert Q.list_jobs(db)[0].state == "done"


def test_an_idle_pool_reports_nothing_to_do(db: Database) -> None:
    did, report, _ = _tick(db, "cpu")
    assert did is False and report.claimed == 0


def test_a_paused_queue_claims_nothing(db: Database) -> None:
    with temp_job_kind("t_ok", "cpu", lambda db_, t, p: None):
        Q.enqueue(db, "t_ok", make_video(db), now=NOW)
        Q.pause(db)
        did, report, _ = _tick(db, "cpu")
    assert did is False and report.claimed == 0
    assert Q.list_jobs(db)[0].state == "pending"


def test_a_job_satisfied_since_enqueue_skips_its_handler(db: Database) -> None:
    calls: list[int] = []

    def handler(db_, t, p):
        calls.append(t)

    with temp_job_kind(
        "t_sat", "cpu", handler, readiness=lambda db_, t: Readiness.SATISFIED
    ):
        vid = make_video(db)
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, payload_json, created_at) "
            "VALUES ('t_sat', ?, 'pending', 'cpu', '{}', ?)",
            (vid, NOW.isoformat()),
        )
        did, report, _ = _tick(db, "cpu")
    assert did is True and calls == []
    assert report.done == 1 and Q.list_jobs(db)[0].state == "done"


def test_a_job_blocked_since_enqueue_is_parked(db: Database) -> None:
    with temp_job_kind(
        "t_blk", "cpu", lambda db_, t, p: None,
        readiness=lambda db_, t: Readiness.BLOCKED,
    ):
        vid = make_video(db)
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, payload_json, created_at) "
            "VALUES ('t_blk', ?, 'pending', 'cpu', '{}', ?)",
            (vid, NOW.isoformat()),
        )
        did, report, _ = _tick(db, "cpu")
    assert did is True and report.blocked == 1
    assert Q.list_jobs(db)[0].state == "blocked"


def test_a_throttle_freezes_the_whole_network_pool(db: Database) -> None:
    def boom(db_, t, p):
        raise RateLimited("HTTP Error 429: Too Many Requests")

    with temp_job_kind("t_429", "network", boom):
        a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
        b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
        Q.enqueue(db, "t_429", a, now=NOW)
        Q.enqueue(db, "t_429", b, now=NOW)

        did, report, _ = _tick(db, "network")
        assert did is True and report.deferred == 1
        assert is_cooling_down(db, now=NOW) is True

        # The second job must NOT be attempted — that is the whole point.
        did_again, report2, _ = _tick(db, "network")
    assert did_again is False and report2.claimed == 0


def test_the_throttled_job_keeps_its_attempt_and_shows_the_cooldown(
    db: Database,
) -> None:
    def boom(db_, t, p):
        raise RateLimited("HTTP Error 429")

    with temp_job_kind("t_429", "network", boom):
        vid = make_video(db)
        Q.enqueue(db, "t_429", vid, now=NOW)
        _tick(db, "network")
    job = Q.list_jobs(db)[0]
    assert job.state == "pending"
    assert job.attempts == 0                      # claim spent it, throttle refunded it
    assert job.not_before == cooldown_until(db)   # visible in `jobs stats`
    assert Q.stats(db, now=NOW).throttled == 1


def test_the_cooldown_ladder_climbs_across_ticks(db: Database) -> None:
    def boom(db_, t, p):
        raise RateLimited("HTTP Error 429")

    with temp_job_kind("t_429", "network", boom):
        vid = make_video(db)
        Q.enqueue(db, "t_429", vid, now=NOW)
        _tick(db, "network", now=NOW)
        first = cooldown_until(db)
        later = NOW + timedelta(seconds=C.THROTTLE_BACKOFF_LADDER_S[0] + 1)
        _tick(db, "network", now=later)
        second = cooldown_until(db)
    assert first == (NOW + timedelta(seconds=300)).isoformat()
    assert second == (later + timedelta(seconds=900)).isoformat()


def test_a_clean_download_resets_the_ladder(db: Database) -> None:
    with temp_job_kind("t_dl", "network", lambda db_, t, p: None):
        set_setting(db, C.SETTING_THROTTLE_STREAK, "3")
        Q.enqueue(db, "t_dl", make_video(db), now=NOW)
        _tick(db, "network")
    assert is_cooling_down(db, now=NOW) is False
    assert cooldown_until(db) is None


def test_a_dead_video_fails_permanently_without_freezing_the_pool(
    db: Database,
) -> None:
    def gone(db_, t, p):
        raise VideoUnavailable("Private video")

    with temp_job_kind("t_gone", "network", gone):
        vid = make_video(db)
        Q.enqueue(db, "t_gone", vid, now=NOW)
        did, report, _ = _tick(db, "network")
    assert did is True and report.failed == 1
    assert Q.list_jobs(db)[0].state == "failed"
    assert is_cooling_down(db, now=NOW) is False


def test_a_transient_error_is_retried_until_the_attempt_budget_runs_out(
    db: Database,
) -> None:
    def flaky(db_, t, p):
        raise RuntimeError("unable to download video data: timed out")

    with temp_job_kind("t_flaky", "cpu", flaky):
        vid = make_video(db)
        Q.enqueue(db, "t_flaky", vid, now=NOW)
        for i in range(C.JOB_MAX_ATTEMPTS):
            at = NOW + timedelta(seconds=C.TRANSIENT_BACKOFF_S * (i + 1))
            _tick(db, "cpu", now=at)
    job = Q.list_jobs(db)[0]
    assert job.state == "failed"
    assert job.attempts == C.JOB_MAX_ATTEMPTS
    assert "timed out" in (job.last_error or "")


def test_a_download_is_followed_by_the_randomised_delay(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The delay rule keys off the kind name "download", so this test swaps
    # that kind's handler rather than registering a differently named one.
    from rytp.jobs import JOB_KINDS, JobKind

    real = JOB_KINDS["download"]
    monkeypatch.setitem(
        JOB_KINDS,
        "download",
        JobKind(name="download", pool="network", readiness=real.readiness,
                handler=lambda db_, t, p: None, summary=real.summary),
    )
    Q.enqueue(db, "download", make_video(db), now=NOW)
    _, _, sleeps = _tick(db, "network")
    assert len(sleeps) == 1
    assert C.DOWNLOAD_DELAY_MIN_S <= sleeps[0] <= C.DOWNLOAD_DELAY_MAX_S


def test_a_captions_fetch_uses_the_shorter_delay(db: Database) -> None:
    with temp_job_kind("t_caps", "network", lambda db_, t, p: None):
        Q.enqueue(db, "t_caps", make_video(db), now=NOW)
        _, _, sleeps = _tick(db, "network")
    assert sleeps == [C.DOWNLOAD_FILE_DELAY_S]


def test_a_cpu_job_is_not_followed_by_a_delay(db: Database, tmp_path) -> None:
    with temp_job_kind("t_cpu", "cpu", lambda db_, t, p: None):
        Q.enqueue(db, "t_cpu", make_video(db), now=NOW)
        _, _, sleeps = _tick(db, "cpu")
    assert sleeps == []


def test_the_daily_cap_stops_the_network_pool_claiming(db: Database) -> None:
    set_setting(db, C.SETTING_DOWNLOAD_DAILY_CAP, "1")
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, payload_json, created_at, "
        "started_at) VALUES ('download', 999, 'done', 'network', '{}', ?, ?)",
        (NOW.isoformat(), NOW.isoformat()),
    )
    with temp_job_kind("t_dl", "network", lambda db_, t, p: None):
        Q.enqueue(db, "t_dl", make_video(db), now=NOW)
        did, report, _ = _tick(db, "network")
    assert did is False and report.claimed == 0
    assert Q.list_jobs(db, kind="t_dl")[0].state == "pending"


def test_the_lease_refuses_a_second_live_worker(db: Database) -> None:
    W.acquire_lease(db, now=NOW)
    with pytest.raises(W.WorkerAlreadyRunning, match="already running"):
        W.acquire_lease(db, now=NOW + timedelta(seconds=1))


def test_a_stale_lease_is_taken_over_and_orphans_reclaimed(db: Database) -> None:
    with temp_job_kind("t_ok", "cpu", lambda db_, t, p: None):
        Q.enqueue(db, "t_ok", make_video(db), now=NOW)
        Q.claim(db, "cpu", now=NOW)          # a worker claims it, then dies
        W.acquire_lease(db, now=NOW)
        stale = NOW + timedelta(seconds=C.WORKER_LEASE_STALE_S + 1)
        W.acquire_lease(db, now=stale)       # the next worker takes over
        assert Q.reclaim_running(db, now=stale) == 0  # startup already did it
    job = Q.list_jobs(db)[0]
    assert job.state == "pending" and job.attempts == 1


def test_releasing_the_lease_lets_the_next_worker_in(db: Database) -> None:
    W.acquire_lease(db, now=NOW)
    W.release_lease(db)
    W.acquire_lease(db, now=NOW)  # does not raise


def test_run_worker_once_drains_every_pool_and_reconciles(
    db: Database, tmp_path
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    with temp_job_kind("t_net", "network", lambda db_, t, p: None), temp_job_kind(
        "t_cpu", "cpu", lambda db_, t, p: None
    ):
        Q.enqueue(db, "t_net", vid, now=NOW)
        Q.enqueue(db, "t_cpu", vid, now=NOW)
        report = W.run_worker(
            db,
            W.WorkerOptions(once=True),
            sleep=lambda s: None,
            rng=random.Random(7),
            now_fn=_fixed_now,
        )
    assert report.done == 2
    assert {j.state for j in Q.list_jobs(db)} == {"done"}


def test_run_worker_once_honours_max_jobs(db: Database) -> None:
    with temp_job_kind("t_cpu", "cpu", lambda db_, t, p: None):
        a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
        b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
        Q.enqueue(db, "t_cpu", a, now=NOW)
        Q.enqueue(db, "t_cpu", b, now=NOW)
        report = W.run_worker(
            db,
            W.WorkerOptions(pools=("cpu",), once=True, max_jobs=1),
            sleep=lambda s: None,
            rng=random.Random(7),
            now_fn=_fixed_now,
        )
    assert report.done == 1
    assert sorted(j.state for j in Q.list_jobs(db)) == ["done", "pending"]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_jobs_worker.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.jobs.worker'`.

- [ ] **Step 4: Write `rytp/jobs/worker.py`**

```python
"""The worker process. design §5.

Three pools run at once — ``network`` (1 slot, and it stays 1), ``gpu`` (1)
and ``cpu`` (2-3) — so a download and a transcription overlap. Each slot is
a thread with its own :class:`~rytp.db.Database`; SQLite connections are not
shareable across threads and WAL makes several of them cheap.

Two rules that are easy to get wrong and expensive to get wrong:

* A throttle response freezes the **pool**, not the job. Deferring only the
  job that was rate-limited would let the loop claim the next download
  immediately and hit the same server again.
* ``attempts`` is spent at claim time (see :func:`rytp.jobs.queue.claim`),
  so a job that kills its worker still converges on ``failed`` instead of
  being retried forever after every restart.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.acquire import policy as P
from rytp.db import Database
from rytp.db.queries import get_setting, set_setting
from rytp.jobs import Readiness, resolve_job_kind
from rytp.jobs import queue as Q
from rytp.models import RytpError


class WorkerAlreadyRunning(RytpError):
    """Another worker holds a live lease on this database."""


@dataclass(frozen=True)
class WorkerOptions:
    """How this worker run behaves."""

    pools: tuple[str, ...] = C.POOLS
    once: bool = False
    max_jobs: int = 0                        # 0 means "no limit"
    poll_interval_s: float = C.WORKER_POLL_INTERVAL_S


@dataclass
class WorkerReport:
    """What one worker run did. Returned to the CLI for its summary line."""

    claimed: int = 0
    done: int = 0
    failed: int = 0
    deferred: int = 0
    blocked: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------
# The lease. A crashed worker must not strand jobs in ``running`` forever,
# and the jobs table has no column to record who owns a row, so ownership
# lives in settings instead.
# --------------------------------------------------------------------------


def _lease(db: Database) -> dict[str, object] | None:
    raw = get_setting(db, C.SETTING_WORKER_LEASE)
    if not raw:
        return None
    try:
        return dict(json.loads(raw))
    except (ValueError, TypeError):
        return None


def acquire_lease(db: Database, *, now: datetime | None = None) -> None:
    """Claim the right to run, taking over from a worker that died.

    Raises :class:`WorkerAlreadyRunning` when the current lease's heartbeat
    is younger than :data:`rytp.constants.WORKER_LEASE_STALE_S`.
    """
    stamp = now or utcnow()
    held = _lease(db)
    if held is not None:
        beat = str(held.get("heartbeat") or "")
        cutoff = (stamp - timedelta(seconds=C.WORKER_LEASE_STALE_S)).isoformat()
        if beat > cutoff:
            raise WorkerAlreadyRunning(
                f"a worker is already running (pid {held.get('pid')}, last seen "
                f"{beat}); stop it, or wait "
                f"{int(C.WORKER_LEASE_STALE_S)}s for its lease to go stale"
            )
    set_setting(
        db,
        C.SETTING_WORKER_LEASE,
        json.dumps({"pid": os.getpid(), "started_at": stamp.isoformat(),
                    "heartbeat": stamp.isoformat()}),
    )
    Q.reclaim_running(db, now=stamp)


def heartbeat(db: Database, *, now: datetime | None = None) -> None:
    """Keep the lease fresh so nobody else takes it while we are working."""
    held = _lease(db) or {"pid": os.getpid()}
    held["heartbeat"] = (now or utcnow()).isoformat()
    set_setting(db, C.SETTING_WORKER_LEASE, json.dumps(held))


def release_lease(db: Database) -> None:
    """Give the lease up on a clean exit."""
    set_setting(db, C.SETTING_WORKER_LEASE, "")


# --------------------------------------------------------------------------
# One iteration of one pool. Everything above this is bookkeeping; this is
# the part with the policy in it.
# --------------------------------------------------------------------------


def _may_claim(db: Database, pool: str, policy: P.DownloadPolicy, now: datetime) -> bool:
    """Gate the network pool on the cooldown and the daily cap."""
    if Q.is_paused(db):
        return False
    if pool != "network":
        return True
    if P.is_cooling_down(db, now=now):
        return False
    try:
        P.check_daily_cap(db, policy, now=now)
    except P.DailyCapReached:
        return False
    return True


def _handle_failure(
    db: Database,
    job: Q.Job,
    exc: Exception,
    *,
    policy: P.DownloadPolicy,
    report: WorkerReport,
    now: datetime,
) -> None:
    """Decide between freezing the pool, failing, and trying again later."""
    message = str(exc)
    throttled = isinstance(exc, P.RateLimited) or (
        P.classify_error(message) is P.ErrorKind.THROTTLED
    )
    permanent = isinstance(exc, P.PermanentAcquireError) or (
        P.classify_error(message) is P.ErrorKind.UNAVAILABLE
    )

    if throttled:
        until = P.record_throttle(db, policy, now=now)
        # The same timestamp goes on the job so `rytp jobs stats` reports a
        # throttled count instead of an unexplained pause.
        Q.defer(db, job.id, not_before=until, error=message, refund_attempt=True)
        with report.lock:
            report.deferred += 1
        return

    if permanent or job.attempts >= C.JOB_MAX_ATTEMPTS:
        Q.fail(db, job.id, error=message, now=now)
        with report.lock:
            report.failed += 1
        return

    retry_at = (now + timedelta(seconds=C.TRANSIENT_BACKOFF_S)).isoformat()
    Q.defer(db, job.id, not_before=retry_at, error=message)
    with report.lock:
        report.deferred += 1


def run_pool_once(
    db: Database,
    pool: str,
    *,
    policy: P.DownloadPolicy,
    rng: random.Random,
    report: WorkerReport,
    sleep,
    now_fn=utcnow,
) -> bool:
    """Claim and run at most one job. Returns False when there was nothing."""
    now = now_fn()
    if not _may_claim(db, pool, policy, now):
        return False

    job = Q.claim(db, pool, now=now)
    if job is None:
        return False
    with report.lock:
        report.claimed += 1

    spec = resolve_job_kind(job.kind)
    readiness = spec.readiness(db, job.target_id)
    if readiness is Readiness.SATISFIED:
        Q.finish(db, job.id, now=now)
        with report.lock:
            report.done += 1
        return True
    if readiness is Readiness.BLOCKED:
        Q.block(db, job.id, reason="prerequisites are not in place", now=now)
        with report.lock:
            report.blocked += 1
        return True

    try:
        spec.handler(db, job.target_id, job.payload)
    except Exception as exc:  # noqa: BLE001 - the worker classifies, never crashes
        _handle_failure(db, job, exc, policy=policy, report=report, now=now_fn())
    else:
        if pool == "network":
            P.clear_throttle(db)
        Q.finish(db, job.id, now=now_fn())
        # This is what closes the pipeline. `ingest` parks extract_wav as
        # blocked because there is no audio yet; finishing the download has
        # to be what turns it into work, or --once exits with no WAV and the
        # cpu thread polls an empty pending set forever. Scoped to the
        # target's namespace so a render's id can never collide with a video.
        Q.unblock(db, target_id=job.target_id, target_kind=spec.target_kind,
                  now=now_fn())
        with report.lock:
            report.done += 1

    # design §5: a randomised pause between videos, every time the network
    # pool actually reached out — successes and failures alike, because both
    # made a request.
    if pool == "network":
        delay = (
            P.next_delay_s(policy, rng)
            if job.kind == "download"
            else policy.file_delay_s
        )
        sleep(delay)
    return True


# --------------------------------------------------------------------------
# The process.
# --------------------------------------------------------------------------


def _budget_spent(options: WorkerOptions, report: WorkerReport) -> bool:
    if options.max_jobs <= 0:
        return False
    with report.lock:
        return report.claimed >= options.max_jobs


def _drain_inline(db: Database, options: WorkerOptions, report: WorkerReport,
                  *, sleep, rng, now_fn) -> None:
    """``--once``: walk every pool to exhaustion, repeatedly.

    Single-threaded on purpose — it is what the tests and a quick manual run
    want, and determinism is worth more here than overlap. The outer loop
    matters: finishing a download unblocks a cpu job, so one pass in pool
    order would only work by accident.
    """
    policy = P.DownloadPolicy.from_settings(db)
    progressed = True
    while progressed and not _budget_spent(options, report):
        progressed = False
        for pool in options.pools:
            while not _budget_spent(options, report):
                if not run_pool_once(db, pool, policy=policy, rng=rng,
                                     report=report, sleep=sleep, now_fn=now_fn):
                    break
                progressed = True


def _pool_loop(db_path: Path, pool: str, options: WorkerOptions,
               report: WorkerReport, stop: threading.Event, *, open_db) -> None:
    """One slot. Owns its own connection for its whole life."""
    db = open_db(db_path)
    rng = random.Random()
    try:
        while not stop.is_set() and not _budget_spent(options, report):
            policy = P.DownloadPolicy.from_settings(db)
            busy = run_pool_once(
                db, pool, policy=policy, rng=rng, report=report,
                sleep=lambda s: stop.wait(s), now_fn=utcnow,
            )
            if not busy:
                stop.wait(options.poll_interval_s)
    finally:
        db.close()


def run_worker(
    db: Database,
    options: WorkerOptions | None = None,
    *,
    sleep=time.sleep,
    rng: random.Random | None = None,
    now_fn=utcnow,
    open_db=Database,
) -> WorkerReport:
    """Run the worker. ``--once`` drains inline; otherwise, threads."""
    options = options or WorkerOptions()
    report = WorkerReport()
    rng = rng or random.Random()

    acquire_lease(db, now=now_fn())
    Q.reconcile(db, now=now_fn())
    try:
        if options.once:
            _drain_inline(db, options, report, sleep=sleep, rng=rng, now_fn=now_fn)
            return report

        db_path = config.paths().db
        stop = threading.Event()
        threads: list[threading.Thread] = []
        for pool in options.pools:
            for slot in range(P.pool_size(db, pool)):
                thread = threading.Thread(
                    target=_pool_loop,
                    args=(db_path, pool, options, report, stop),
                    kwargs={"open_db": open_db},
                    name=f"rytp-{pool}-{slot}",
                    daemon=True,
                )
                thread.start()
                threads.append(thread)
        try:
            # Windows has no SIGTERM, so Ctrl-C is the only stop signal. The
            # supervisor polls often and heartbeats on its own slower clock,
            # so shutdown is prompt even with a long heartbeat interval.
            last_beat = float("-inf")
            while any(t.is_alive() for t in threads):
                if time.monotonic() - last_beat >= C.WORKER_HEARTBEAT_INTERVAL_S:
                    heartbeat(db, now=now_fn())
                    last_beat = time.monotonic()
                stop.wait(options.poll_interval_s)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=C.WORKER_JOIN_TIMEOUT_S)
        return report
    finally:
        release_lease(db)
```

- [ ] **Step 5: Add the last worker constant**

Append to the jobs section of `rytp/constants.py`:

```python
#: How long to wait for a worker thread to notice the stop flag before
#: giving up on it. The threads are daemons, so a stuck ffmpeg cannot keep
#: the process alive past this.
WORKER_JOIN_TIMEOUT_S: float = 30.0
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_jobs_worker.py -q`
Expected: PASS, 20 passed.

- [ ] **Step 7: Add one real threaded test**

Everything above runs `--once` inline, so add a single test that proves the threaded path starts, does work and shuts down. Append to `tests/test_jobs_worker.py`:

```python
def test_the_threaded_worker_starts_works_and_stops(db: Database) -> None:
    with temp_job_kind("t_cpu", "cpu", lambda db_, t, p: None):
        Q.enqueue(db, "t_cpu", make_video(db), now=NOW)
        report = W.run_worker(
            db,
            W.WorkerOptions(pools=("cpu",), max_jobs=1, poll_interval_s=0.01),
        )
    assert report.done == 1
    assert Q.list_jobs(db)[0].state == "done"
    assert [t for t in threading.enumerate() if t.name.startswith("rytp-")] == []
```

Add `import threading` to the test module's imports.

Run: `python -m pytest tests/test_jobs_worker.py -q`
Expected: PASS, 21 passed, and the run finishes in well under ten seconds. If it hangs, the pool threads are not seeing `max_jobs` — check `_budget_spent`.

- [ ] **Step 8: Commit**

```bash
git add rytp/jobs/worker.py rytp/constants.py tests/fakes.py tests/test_jobs_worker.py
git commit -m "feat: add the three-pool worker with pool-wide throttle backoff"
```

---

### Task 11: The doing commands — `ingest`, `fetch-video`, `worker`

Contracts §5: both surfaces are generated from the registry and neither may define a command of its own. Handlers take keyword arguments matching `Param.name`, receive the open `Database` as the first positional argument, return a `CommandResult`, and **never print and never call `sys.exit`** — they raise, and the surface formats the error.

Design §5 fixes what `ingest` does: *"`rytp videos add <url>` catalogs only. `rytp ingest <id>` enqueues the chain, pulling audio, a video rendition and captions together. Local files register as assets and never get download jobs."*

**The chain runs past Part 2.** `download → captions → caption_words → extract_wav → fingerprint`, plus `transcribe → align` behind `--transcribe`. Three of those belong to Part 3 (names, pools and flags agreed with its author): `caption_words` (cpu) turns the downloaded json3 into tier-1 words, without which design §6's "searchable within hours of cataloguing it" never happens; `fingerprint` (cpu) fills `video_acoustics`, which design §8's consistency knob reads — with the table empty that knob silently degrades to counting fragments, and nothing fails to tell you. `transcribe` and `align` (gpu) stay behind a flag because design §6 makes tier 2 opt-in per video and design §13 prices the corpus at 130-200 GPU hours. `ingest` enqueues only the kinds **registered at run time**, so Part 2 ships and tests on its own and the chain completes itself as each part lands.

**And it takes more than one video.** Design §11's M2 is "catalog the channel, pull captions for everything" — roughly 1,600 videos. `rytp ingest --pending --limit 2000` is that milestone; 1,600 invocations is not a command.

**Files:**
- Create: `rytp/commands/ingest.py`
- Modify: `rytp/commands/__init__.py` (one line in the existing sibling-import block)
- Test: `tests/test_commands_ingest.py`

**Interfaces:**
- Consumes: `rytp.commands.{Command, CommandResult, Param, REQUIRED, register}`; Tasks 4, 6, 7, 8, 9, 10.
- Produces: registered commands `ingest`, `fetch-video`, `worker`.

- [ ] **Step 1: Write the failing tests**

`tests/test_commands_ingest.py`:

```python
"""Tests for the Part 2 commands (contracts §5, design §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import constants as C
from rytp.commands import resolve
from rytp.db import Database
from rytp.db.queries import asset_for, insert_asset
from rytp.jobs import JOB_KINDS, Readiness
from rytp.jobs import queue as Q
from rytp.models import NotFoundError

from tests.fakes import (
    CHANNEL_ONE_URL,
    FakeYtDlpRunner,
    make_video,
    temp_job_kind,
    touch,
)


def _ingest(db: Database, **kwargs: object):
    """Call the ingest handler with the defaults its Params declare."""
    args: dict[str, object] = {
        "video_id": 0, "channel_id": 0, "pending": False, "transcribe": False,
        "limit": C.INGEST_DEFAULT_LIMIT, "priority": 0, "dry_run": False,
    }
    args.update(kwargs)
    return resolve("ingest").handler(db, **args)


def test_ingest_enqueues_every_registered_kind_of_the_chain(db: Database) -> None:
    vid = make_video(db)
    _ingest(db, video_id=vid)
    queued = {j.kind for j in Q.list_jobs(db)}
    # Parts 3, 4 and 6 have not registered their kinds yet, so the chain is
    # exactly Part 2's share of it. Once caption_words and fingerprint exist
    # this set grows with no change here.
    assert queued == {k for k in C.INGEST_CHAIN_REMOTE if k in JOB_KINDS}
    assert queued == {"download", "captions", "extract_wav"}


def test_ingest_names_the_chain_kinds_that_are_not_registered_yet(
    db: Database,
) -> None:
    vid = make_video(db)
    result = _ingest(db, video_id=vid)
    # Silence here is how `fingerprint` never running went unnoticed.
    assert "caption_words" in (result.message or "")
    assert "fingerprint" in (result.message or "")


def test_ingest_enqueues_a_later_parts_kind_once_it_is_registered(
    db: Database, tmp_path: Path
) -> None:
    with temp_job_kind(
        "fingerprint", "cpu", lambda db_, t, p: None,
        readiness=lambda db_, t: Readiness.BLOCKED,
    ):
        vid = make_video(db)
        _ingest(db, video_id=vid)
        assert Q.list_jobs(db, kind="fingerprint") != []


def test_ingest_reports_the_state_each_kind_started_in(db: Database) -> None:
    result = _ingest(db, video_id=make_video(db))
    by_kind = {row[0]: dict(zip(result.columns[1:], row[1:], strict=True))
               for row in result.rows}
    assert by_kind["download"]["pending"] == "1"
    assert by_kind["captions"]["pending"] == "1"
    # No audio yet, so there is nothing to decode.
    assert by_kind["extract_wav"]["blocked"] == "1"


def test_ingest_is_idempotent(db: Database) -> None:
    vid = make_video(db)
    _ingest(db, video_id=vid)
    _ingest(db, video_id=vid)
    assert len(Q.list_jobs(db)) == 3


def test_transcribe_flag_adds_the_tier_two_kinds(db: Database) -> None:
    # design §6: tier 2 is opt-in per video, so this must be off by default.
    with temp_job_kind("transcribe", "gpu", lambda db_, t, p: None), temp_job_kind(
        "align", "gpu", lambda db_, t, p: None, reopenable=False
    ):
        a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
        b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
        _ingest(db, video_id=a)
        assert Q.list_jobs(db, kind="transcribe") == []
        _ingest(db, video_id=b, transcribe=True)
        assert [j.target_id for j in Q.list_jobs(db, kind="transcribe")] == [b]
        assert [j.target_id for j in Q.list_jobs(db, kind="align")] == [b]


def test_bulk_ingest_covers_a_whole_channel(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO channels (id, url, title) VALUES (1, ?, 'CHANNEL_ONE')",
        (CHANNEL_ONE_URL,),
    )
    ours = [
        make_video(db, channel_id=1, external_id=f"VIDEO_{i}",
                   url=f"https://example.invalid/{i}")
        for i in range(3)
    ]
    make_video(db, channel_id=None, external_id="VIDEO_X",
               url="https://example.invalid/x")
    result = _ingest(db, channel_id=1)
    assert "3 video(s)" in (result.message or "")
    assert {j.target_id for j in Q.list_jobs(db, kind="download")} == set(ours)


def test_bulk_ingest_skips_videos_it_has_already_done(db: Database) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    _ingest(db, video_id=a)
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    result = _ingest(db, pending=True)
    assert "1 video(s)" in (result.message or "")
    assert {j.target_id for j in Q.list_jobs(db, kind="download")} == {a, b}


def test_bulk_ingest_honours_the_limit(db: Database) -> None:
    for i in range(5):
        make_video(db, external_id=f"VIDEO_{i}", url=f"https://example.invalid/{i}")
    _ingest(db, limit=2)
    assert len(Q.list_jobs(db, kind="download")) == 2


def test_bulk_ingest_dry_run_enqueues_nothing(db: Database) -> None:
    make_video(db)
    result = _ingest(db, dry_run=True)
    assert result.columns == ("video", "source")
    assert len(result.rows) == 1
    assert Q.list_jobs(db) == []


def test_bulk_ingest_survives_one_unreadable_local_file(
    db: Database, tmp_path: Path
) -> None:
    good = make_video(db, source=C.LOCAL_SOURCE,
                      external_id=str(touch(tmp_path / "ok.mkv")), url=None)
    make_video(db, source=C.LOCAL_SOURCE,
               external_id=str(tmp_path / "gone.mkv"), url=None)
    result = _ingest(db)
    assert "1 skipped" in (result.message or "")
    assert {j.target_id for j in Q.list_jobs(db, kind="extract_wav")} == {good}


def test_ingest_of_a_local_file_registers_it_and_skips_the_downloads(
    db: Database, tmp_path: Path
) -> None:
    clip = touch(tmp_path / "clip.mkv")
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id=str(clip), url=None)
    result = _ingest(db, video_id=vid)
    assert {row[0] for row in result.rows} == {"extract_wav"}
    assert asset_for(db, vid, "container") is not None
    assert Q.list_jobs(db, kind="download") == []


def test_ingest_of_an_unknown_video_raises(db: Database) -> None:
    with pytest.raises(NotFoundError, match="4242"):
        _ingest(db, video_id=4242)


def test_ingest_passes_priority_through(db: Database) -> None:
    _ingest(db, video_id=make_video(db), priority=7)
    assert {j.priority for j in Q.list_jobs(db)} == {7}


def _fake_the_world(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace yt-dlp and ffmpeg where the stage code looks them up.

    ``acquire_media`` and ``acquire_captions`` each did
    ``from ... import RealYtDlpRunner``, so the name to replace lives in
    *their* module, not in ``rytp.acquire.ytdlp``.
    """
    from rytp.audio import extract as E

    monkeypatch.setattr("rytp.acquire.RealYtDlpRunner", FakeYtDlpRunner)
    monkeypatch.setattr("rytp.acquire.captions.RealYtDlpRunner", FakeYtDlpRunner)
    monkeypatch.setattr(E, "_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(E, "_run_ffmpeg", _writing_ffmpeg())


def test_fetch_video_runs_the_whole_chain_inline(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.audio import extract as E

    _fake_the_world(monkeypatch)

    vid = make_video(db)
    result = resolve("fetch-video").handler(db, video_id=vid, captions=True)
    assert asset_for(db, vid, "audio") is not None
    assert asset_for(db, vid, "video") is not None
    assert asset_for(db, vid, "captions") is not None
    assert E.wav_path(vid).exists()
    assert result.message and "audio.m4a" in result.message


def test_fetch_video_can_skip_captions(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_the_world(monkeypatch)

    vid = make_video(db)
    resolve("fetch-video").handler(db, video_id=vid, captions=False)
    assert asset_for(db, vid, "captions") is None


def test_worker_once_reports_what_it_did(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.audio import extract as E

    _fake_the_world(monkeypatch)
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    Q.enqueue(db, "extract_wav", vid)

    result = resolve("worker").handler(db, pool="cpu", once=True, max_jobs=0)
    assert result.message and "1 done" in result.message
    assert E.wav_path(vid).exists()


def test_every_part_two_command_is_registered() -> None:
    for name in (
        "ingest", "fetch-video", "worker",
        "jobs.list", "jobs.stats", "jobs.retry",
        "queue.pause", "queue.resume", "cache.prune",
    ):
        assert resolve(name).summary


def _writing_ffmpeg():
    """An ffmpeg replacement that writes the output file it is asked for."""
    import subprocess

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"RIFF....WAVE")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return run
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_commands_ingest.py -q`
Expected: failures on every test, `ValueError` from `resolve("ingest")` naming the commands that do exist.

- [ ] **Step 3: Write `rytp/commands/ingest.py`**

```python
"""Acquisition and queue commands. design §5, contracts §5.

Handlers never print and never exit — they return a
:class:`~rytp.commands.CommandResult` or raise, and the CLI or the TUI
decides what that looks like.
"""

from __future__ import annotations

from typing import Any

from rytp import constants as C
from rytp.acquire import acquire_media
from rytp.acquire.captions import acquire_captions
from rytp.acquire.local import register_local_container
from rytp.audio.extract import ensure_wav
from rytp.commands import REQUIRED, Command, CommandResult, Param, register
from rytp.db import Database
from rytp.jobs import JOB_KINDS
from rytp.jobs import queue as Q
from rytp.jobs import worker as W
from rytp.models import NotFoundError, RytpError




def _video_source(db: Database, video_id: int) -> str:
    row = db.conn.execute(
        "SELECT source FROM videos WHERE id = ?", (video_id,)
    ).fetchone()
    if row is None:
        raise NotFoundError(f"video {video_id} is not in the catalog")
    return str(row["source"])


def _select_videos(
    db: Database, *, video_id: int, channel_id: int, pending: bool, limit: int
) -> list[tuple[int, str]]:
    """Which videos this invocation ingests: one named, or a filtered set."""
    if video_id:
        row = db.conn.execute(
            "SELECT id, source FROM videos WHERE id = ?", (video_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"video {video_id} is not in the catalog")
        return [(int(row["id"]), str(row["source"]))]

    where: list[str] = []
    params: list[Any] = []
    if channel_id:
        where.append("v.channel_id = ?")
        params.append(channel_id)
    if pending:
        # "Not yet ingested" = no extract_wav job, which every chain contains.
        where.append(
            "NOT EXISTS (SELECT 1 FROM jobs j "
            "WHERE j.target_id = v.id AND j.kind = 'extract_wav')"
        )
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)
    rows = db.conn.execute(
        f"SELECT v.id, v.source FROM videos v {clause} ORDER BY v.id LIMIT ?",  # noqa: S608
        params,
    ).fetchall()
    return [(int(r["id"]), str(r["source"])) for r in rows]


def ingest(
    db: Database,
    *,
    video_id: int = 0,
    channel_id: int = 0,
    pending: bool = False,
    transcribe: bool = False,
    limit: int = C.INGEST_DEFAULT_LIMIT,
    priority: int = 0,
    dry_run: bool = False,
) -> CommandResult:
    """Enqueue the chain for one video, or for a filtered set of them.

    design §11 M2 is "catalog the channel, pull captions for everything" at
    roughly 1,600 videos, so the bulk form is the point of this command, not
    a convenience: `rytp ingest --pending --limit 2000` is M2.
    """
    targets = _select_videos(
        db, video_id=video_id, channel_id=channel_id, pending=pending, limit=limit
    )
    if dry_run:
        return CommandResult(
            columns=("video", "source"),
            rows=tuple((str(v), src) for v, src in targets),
            message=f"would ingest {len(targets)} video(s)",
        )

    states = ("pending", "blocked", "done")
    tally: dict[str, dict[str, int]] = {}
    order: list[str] = []
    missing: set[str] = set()
    skipped: list[str] = []
    ingested = 0

    for vid, source in targets:
        if source == C.LOCAL_SOURCE:
            try:
                register_local_container(db, vid)
            except RytpError as exc:
                # One unreadable file must not abort a 1,600-video run.
                skipped.append(f"{vid}: {exc}")
                continue
            chain = C.INGEST_CHAIN_LOCAL
        else:
            chain = C.INGEST_CHAIN_REMOTE
        if transcribe:
            chain = (*chain, *C.INGEST_CHAIN_TRANSCRIBE)

        ingested += 1
        for kind in chain:
            # Kinds owned by Parts 3, 4 and 6 appear here only once those
            # parts have registered them. Part 2 stays shippable on its own
            # and the chain completes itself as they land.
            if kind not in JOB_KINDS:
                missing.add(kind)
                continue
            if kind not in tally:
                tally[kind] = dict.fromkeys(("queued", *states), 0)
                order.append(kind)
            job_id = Q.enqueue(db, kind, vid, priority=priority)
            state = Q.get_job(db, job_id).state
            tally[kind]["queued"] += 1
            if state in tally[kind]:
                tally[kind][state] += 1

    parts = [f"ingested {ingested} video(s)"]
    if missing:
        parts.append(
            f"not yet available, skipped: {', '.join(sorted(missing))}"
        )
    if skipped:
        parts.append(f"{len(skipped)} skipped ({'; '.join(skipped[:3])})")
    return CommandResult(
        columns=("kind", "queued", *states),
        rows=tuple(
            (kind, *(str(tally[kind][c]) for c in ("queued", *states)))
            for kind in order
        ),
        message="; ".join(parts),
    )


def fetch_video(db: Database, *, video_id: int, captions: bool = True) -> CommandResult:
    """Acquire one video right now, without going through the queue."""
    source = _video_source(db, video_id)
    notes: list[str] = []
    if source == C.LOCAL_SOURCE:
        notes.append(register_local_container(db, video_id))
    else:
        notes.append(acquire_media(db, video_id))
        if captions:
            notes.append(acquire_captions(db, video_id))
    notes.append(f"wav cached at {ensure_wav(db, video_id)}")
    Q.reconcile(db)
    return CommandResult(message="; ".join(notes))


def worker(
    db: Database, *, pool: str = "all", once: bool = False, max_jobs: int = 0
) -> CommandResult:
    """Run the worker until Ctrl-C, or until the queue empties with --once."""
    pools = C.POOLS if pool == "all" else (pool,)
    report = W.run_worker(
        db, W.WorkerOptions(pools=pools, once=once, max_jobs=max_jobs)
    )
    return CommandResult(
        message=(
            f"{report.claimed} claimed, {report.done} done, "
            f"{report.failed} failed, {report.deferred} deferred, "
            f"{report.blocked} blocked"
        )
    )


register(
    Command(
        name="ingest",
        group="",
        summary="Enqueue the acquisition chain for one video or a whole channel.",
        params=(
            Param("video_id", int, "Catalog id of one video. 0 uses the filters.",
                  default=0, positional=True),
            Param("channel_id", int, "Only videos of this channel. 0 means any.",
                  default=0, short="c"),
            Param("pending", bool, "Only videos that have never been ingested.",
                  default=False),
            Param("transcribe", bool,
                  "Also enqueue transcription and alignment (tier 2, GPU).",
                  default=False),
            Param("limit", int, "Maximum videos to touch in the bulk form.",
                  default=C.INGEST_DEFAULT_LIMIT, short="n"),
            Param("priority", int, "Higher runs first.", default=0, short="p"),
            Param("dry_run", bool, "List what would be ingested, enqueue nothing.",
                  default=False),
        ),
        handler=ingest,
    )
)

register(
    Command(
        name="fetch-video",
        group="",
        summary="Download one video's assets and cache its WAV, right now.",
        params=(
            Param("video_id", int, "Catalog id of the video.",
                  default=REQUIRED, positional=True),
            Param("captions", bool, "Also fetch the caption track.", default=True),
        ),
        handler=fetch_video,
        long_running=True,
    )
)

register(
    Command(
        name="worker",
        group="",
        summary="Run the job queue's worker pools.",
        params=(
            Param("pool", str, "Which pool to run.", default="all",
                  choices=("all", *C.POOLS)),
            Param("once", bool, "Drain the queue and exit instead of waiting.",
                  default=False),
            Param("max_jobs", int, "Stop after this many jobs. 0 means no limit.",
                  default=0),
        ),
        handler=worker,
        long_running=True,
    )
)
```

- [ ] **Step 4: Register the module with the command registry**

In `rytp/commands/__init__.py`, in the existing sibling-import block at the bottom of the file, add one line — importing a command module is what registers its commands:

```python
from rytp.commands import ingest as _ingest  # noqa: E402,F401
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_commands_ingest.py -q`
Expected: 17 passed, 1 failed — `test_every_part_two_command_is_registered` still fails on `jobs.list`, which Task 12 adds. Everything else is green.

- [ ] **Step 6: Commit**

```bash
git add rytp/commands/ingest.py rytp/commands/__init__.py tests/test_commands_ingest.py
git commit -m "feat: add the ingest, fetch-video and worker commands"
```

---

### Task 12: The control commands — `jobs.*`, `queue.*`, `cache.prune`

Design §5: *"Throttling shows up in `rytp jobs stats` rather than as a mystery stall."* And the self-healing loop needs its user-facing half: pruning the WAV cache must make the extract jobs runnable again, which means `cache.prune` calls `reconcile` rather than leaving the operator to wonder why nothing happens.

**Files:**
- Modify: `rytp/commands/ingest.py` (append)
- Test: `tests/test_commands_ingest.py` (append)

**Interfaces:**
- Consumes: Task 4's `queue` module in full, Task 9's `prune_wav_cache`.
- Produces: registered commands `jobs.list`, `jobs.stats`, `jobs.retry`, `queue.pause`, `queue.resume`, `cache.prune`.

- [ ] **Step 1: Append the failing tests**

To `tests/test_commands_ingest.py`:

```python
def test_jobs_list_renders_rows_of_strings(db: Database) -> None:
    vid = make_video(db)
    _ingest(db, video_id=vid)
    result = resolve("jobs.list").handler(db, state="", pool="", kind="", limit=50)
    assert result.columns == (
        "id", "kind", "target", "state", "pool", "attempts", "not_before", "error"
    )
    assert len(result.rows) == 3
    assert all(isinstance(cell, str) for row in result.rows for cell in row)


def test_jobs_list_filters_by_state(db: Database) -> None:
    vid = make_video(db)
    _ingest(db, video_id=vid)
    result = resolve("jobs.list").handler(
        db, state="blocked", pool="", kind="", limit=50
    )
    assert [row[1] for row in result.rows] == ["extract_wav"]


def test_jobs_stats_shows_throttling_not_a_mystery_stall(db: Database) -> None:
    from datetime import UTC, datetime, timedelta

    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid)
    soon = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    Q.defer(db, job_id, not_before=soon, error="HTTP Error 429")
    result = resolve("jobs.stats").handler(db)
    flat = {row[0]: row[1] for row in result.rows}
    assert flat["throttled"] == "1"
    assert flat["next attempt"] == soon
    assert flat["paused"] == "no"


def test_jobs_retry_revives_failed_jobs(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid)
    Q.fail(db, job_id, error="boom")
    result = resolve("jobs.retry").handler(db, job_id=0, kind="", state="failed")
    assert "1" in (result.message or "")
    assert Q.get_job(db, job_id).state == "pending"


def test_queue_pause_and_resume_round_trip(db: Database) -> None:
    resolve("queue.pause").handler(db)
    assert Q.is_paused(db) is True
    resolve("queue.resume").handler(db)
    assert Q.is_paused(db) is False


def test_cache_prune_frees_files_and_reopens_the_extract_jobs(
    db: Database, tmp_path: Path
) -> None:
    from rytp.audio import extract as E

    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    touch(E.wav_path(vid), b"\x00" * 200)
    job_id = Q.enqueue(db, "extract_wav", vid)
    assert Q.get_job(db, job_id).state == "done"

    result = resolve("cache.prune").handler(db, video_id=0, dry_run=False)
    assert not E.wav_path(vid).exists()
    assert "200" in (result.message or "")
    # This is the self-healing loop, end to end.
    assert Q.get_job(db, job_id).state == "pending"


def test_cache_prune_dry_run_changes_nothing(db: Database, tmp_path: Path) -> None:
    from rytp.audio import extract as E

    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    touch(E.wav_path(vid))
    job_id = Q.enqueue(db, "extract_wav", vid)
    resolve("cache.prune").handler(db, video_id=0, dry_run=True)
    assert E.wav_path(vid).exists()
    assert Q.get_job(db, job_id).state == "done"


def test_ingest_then_worker_once_acquires_everything(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline claim of Part 2, end to end, with yt-dlp and ffmpeg faked.

    It is the one test that exercises the blocked -> pending transition:
    `ingest` parks extract_wav as blocked because there is no audio yet, and
    only the finished download job can turn it into work.
    """
    from rytp.audio import extract as E

    _fake_the_world(monkeypatch, db)
    vid = make_video(db)

    _ingest(db, video_id=vid)
    assert Q.get_job(db, Q.list_jobs(db, kind="extract_wav")[0].id).state == "blocked"

    result = resolve("worker").handler(db, pool="all", once=True, max_jobs=0)

    assert asset_for(db, vid, "audio") is not None
    assert asset_for(db, vid, "video") is not None
    assert asset_for(db, vid, "captions") is not None
    assert E.wav_path(vid).exists()
    assert {j.state for j in Q.list_jobs(db)} == {"done"}
    assert "3 done" in (result.message or "")
```

This test goes through the real `network` pool, so `_fake_the_world` also has
to neutralise the inter-video delay or it sleeps for up to a minute of real
time. Widen the helper from Task 11 to its final form — it takes the database
now, and turns the delays off through the same settings an operator would:

```python
def _fake_the_world(monkeypatch: pytest.MonkeyPatch, db: Database | None = None) -> None:
    """Replace yt-dlp and ffmpeg where the stage code looks them up.

    ``acquire_media`` and ``acquire_captions`` each did
    ``from ... import RealYtDlpRunner``, so the name to replace lives in
    *their* module, not in ``rytp.acquire.ytdlp``.
    """
    from rytp.audio import extract as E
    from rytp.db.queries import set_setting

    monkeypatch.setattr("rytp.acquire.RealYtDlpRunner", FakeYtDlpRunner)
    monkeypatch.setattr("rytp.acquire.captions.RealYtDlpRunner", FakeYtDlpRunner)
    monkeypatch.setattr(E, "_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(E, "_run_ffmpeg", _writing_ffmpeg())
    if db is not None:
        for key in (C.SETTING_DOWNLOAD_DELAY_MIN, C.SETTING_DOWNLOAD_DELAY_MAX,
                    C.SETTING_DOWNLOAD_FILE_DELAY):
            set_setting(db, key, "0")
```

The three existing callers keep working unchanged; the new test calls it as
`_fake_the_world(monkeypatch, db)`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_commands_ingest.py -q`
Expected: the seven new tests fail with `ValueError: unknown command 'jobs.list'`.

- [ ] **Step 3: Append the handlers and registrations to `rytp/commands/ingest.py`**

First widen the existing audio import at the top of the file:

```python
from rytp.audio.extract import ensure_wav, prune_wav_cache
```

Then append:

```python
_JOB_STATES: tuple[str, ...] = (
    "pending", "running", "done", "failed", "blocked", "cancelled"
)


def jobs_list(
    db: Database, *, state: str = "", pool: str = "", kind: str = "",
    limit: int = C.JOB_LIST_LIMIT,
) -> CommandResult:
    """Show jobs, highest priority first."""
    jobs = Q.list_jobs(
        db, state=state or None, pool=pool or None, kind=kind or None, limit=limit
    )
    rows = tuple(
        (
            str(j.id), j.kind, str(j.target_id), j.state, j.pool, str(j.attempts),
            j.not_before or "",
            next(iter((j.last_error or "").splitlines()), "")[
                : C.JOB_ERROR_PREVIEW_CHARS
            ],
        )
        for j in jobs
    )
    return CommandResult(
        columns=("id", "kind", "target", "state", "pool", "attempts",
                 "not_before", "error"),
        rows=rows,
        message=None if rows else "no jobs match",
    )


def jobs_stats(db: Database) -> CommandResult:
    """Counts by state and pool, plus how much of the queue is backing off."""
    s = Q.stats(db)
    rows: list[tuple[str, str]] = [(name, str(n)) for name, n in s.by_state.items()]
    rows += [(f"pending on {pool}", str(n)) for pool, n in s.pending_by_pool.items()]
    # design §5: throttling must be visible, not a mystery stall.
    rows.append(("throttled", str(s.throttled)))
    rows.append(("next attempt", s.next_not_before or ""))
    rows.append(("paused", "yes" if s.paused else "no"))
    return CommandResult(columns=("metric", "value"), rows=tuple(rows))


def jobs_retry(
    db: Database, *, job_id: int = 0, kind: str = "", state: str = "failed"
) -> CommandResult:
    """Send settled jobs back to pending, then re-ask what is runnable."""
    changed = Q.retry(
        db, job_id=job_id or None, kind=kind or None, state=state
    )
    reopened = Q.reconcile(db)
    return CommandResult(
        message=f"{changed} job(s) retried, {reopened} reopened by reconcile"
    )


def queue_pause(db: Database) -> CommandResult:
    """Stop every pool from claiming. Running jobs finish."""
    Q.pause(db)
    return CommandResult(message="queue paused; running jobs will finish")


def queue_resume(db: Database) -> CommandResult:
    Q.resume(db)
    return CommandResult(message="queue resumed")


def cache_prune(db: Database, *, video_id: int = 0, dry_run: bool = False) -> CommandResult:
    """Delete cached WAVs. They are regenerable, and the jobs reopen.

    design §4 calls the WAV a cache with an explicit prune command; design
    §5 makes pruning it enough to bring the extract jobs back, which is why
    this reconciles afterwards instead of leaving a stale ``done``.
    """
    freed = prune_wav_cache(db, video_id=video_id or None, dry_run=dry_run)
    total = sum(size for _, size in freed)
    reopened = 0 if dry_run else Q.reconcile(db, kinds=("extract_wav",))
    verb = "would free" if dry_run else "freed"
    return CommandResult(
        columns=("file", "bytes"),
        rows=tuple((p.name, str(size)) for p, size in freed),
        message=(
            f"{verb} {total} bytes across {len(freed)} file(s); "
            f"{reopened} extract job(s) reopened"
        ),
    )


register(
    Command(
        name="jobs.list",
        group="jobs",
        summary="List queued, running and finished jobs.",
        params=(
            Param("state", str, "Only this state.", default="",
                  choices=("", *_JOB_STATES)),
            Param("pool", str, "Only this pool.", default="", choices=("", *C.POOLS)),
            Param("kind", str, "Only this job kind.", default=""),
            Param("limit", int, "Maximum rows.", default=C.JOB_LIST_LIMIT, short="n"),
        ),
        handler=jobs_list,
    )
)

register(
    Command(
        name="jobs.stats",
        group="jobs",
        summary="Summarise the queue, including how much of it is backing off.",
        params=(),
        handler=jobs_stats,
    )
)

register(
    Command(
        name="jobs.retry",
        group="jobs",
        summary="Send settled jobs back to pending.",
        params=(
            Param("job_id", int, "One job id. 0 means every match.", default=0),
            Param("kind", str, "Only this job kind.", default=""),
            Param("state", str, "Which state to revive.", default="failed",
                  choices=("failed", "blocked", "cancelled")),
        ),
        handler=jobs_retry,
    )
)

register(
    Command(
        name="queue.pause",
        group="queue",
        summary="Stop the worker pools from claiming new jobs.",
        params=(),
        handler=queue_pause,
    )
)

register(
    Command(
        name="queue.resume",
        group="queue",
        summary="Let the worker pools claim jobs again.",
        params=(),
        handler=queue_resume,
    )
)

register(
    Command(
        name="cache.prune",
        group="cache",
        summary="Delete cached WAVs. They are regenerable.",
        params=(
            Param("video_id", int, "One video's WAV. 0 means all of them.",
                  default=0),
            Param("dry_run", bool, "Report what would go, delete nothing.",
                  default=False),
        ),
        handler=cache_prune,
    )
)
```

- [ ] **Step 4: Run the whole file**

Run: `python -m pytest tests/test_commands_ingest.py -q`
Expected: PASS, 26 passed. `test_ingest_then_worker_once_acquires_everything` is the one to watch: if it reports 2 done instead of 3, `run_pool_once` is not calling `Q.unblock` after finishing the download and the WAV job never leaves `blocked`.

- [ ] **Step 5: Run the whole suite, lint and type-check**

Run:

```bash
python -m pytest -q
python -m ruff check rytp tests
python -m mypy rytp
```

Expected: the whole suite green, ruff clean, mypy clean.

- [ ] **Step 6: Check the CLI is really wired**

Run: `python -m rytp --help`
Expected: `ingest`, `fetch-video`, `worker`, and the `jobs`, `queue` and `cache` groups all appear. Then:

```bash
python -m rytp jobs stats
python -m rytp cache prune --dry-run
```

Expected: both print a table and exit 0 against an empty database.

- [ ] **Step 7: Commit**

```bash
git add rytp/commands/ingest.py tests/test_commands_ingest.py
git commit -m "feat: add the jobs, queue and cache control commands"
```

---

### Task 13: Deletion — `jobs.cancel` and `assets.remove`

Contracts §5's Deletion table: every group exposes a `remove`, so there is no entity you can create but not get rid of. Part 2 owns two of them. Both are synchronous — *"removal is not a job; a half-deleted entity recovered from a crashed queue is worse than a slow command."*

**Decision, stated rather than left ambiguous: `jobs.cancel` refuses to cancel a `running` job.** The alternative — mark it cancelled and let the worker abandon it at a checkpoint — would be a lie here, because Part 2's handlers spend their time inside `yt-dlp` and `ffmpeg` subprocesses that have no checkpoint to abandon at. The worker would finish the download anyway and then write `done` over `cancelled`. So the command refuses, names the offending job ids, and says what to do instead: `rytp queue pause`, stop the worker, and a worker that has already died leaves its rows to be reclaimed on the next start. Cancelling is also not a permanent veto — `rytp ingest` on that video enqueues it again, and `rytp jobs retry --state cancelled` revives it — which is documented in the summary so nobody is surprised.

**`assets.remove` warns, it does not refuse.** The owner may genuinely want the space. Removing an `audio` or `container` asset breaks re-alignment, rendering and WAV extraction for that video; removing `captions` is only safe once caption words have been superseded by an aligned transcript. Both print a plain warning line. Afterwards it reconciles that video's jobs, so the queue reflects what is actually on disk — deleting a rendition makes its `download` job runnable again, which is design §5's self-healing working as intended rather than something to suppress.

**Files:**
- Modify: `rytp/commands/ingest.py` (append)
- Modify: `rytp/constants.py` (one constant)
- Test: `tests/test_commands_remove.py`

**Interfaces:**
- Consumes: Task 4's queue, Task 2's `asset_for` / `assets_for`.
- Produces: registered commands `jobs.cancel` and `assets.remove`; `rytp.jobs.queue.cancel(db, *, job_id=None, kind=None, state=None, target_id=None, now=None) -> int` and `queue.running_matches(db, ...) -> list[Job]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_commands_remove.py`:

```python
"""Tests for Part 2's deletion commands (contracts §5, Deletion)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import constants as C
from rytp.commands import resolve
from rytp.db import Database
from rytp.db.queries import asset_for, assets_for, insert_asset
from rytp.jobs import queue as Q
from rytp.models import InvalidInputError, NotFoundError

from tests.fakes import make_video, touch


def _cancel(db: Database, **kwargs: object):
    args: dict[str, object] = {"job_id": 0, "kind": "", "state": "", "target_id": 0,
                               "dry_run": False}
    args.update(kwargs)
    return resolve("jobs.cancel").handler(db, **args)


def _remove(db: Database, **kwargs: object):
    args: dict[str, object] = {"asset_id": 0, "yes": False, "dry_run": False}
    args.update(kwargs)
    return resolve("assets.remove").handler(db, **args)


def test_cancel_one_job_by_id(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid)
    result = _cancel(db, job_id=job_id)
    assert "1" in (result.message or "")
    assert Q.get_job(db, job_id).state == "cancelled"


def test_cancel_a_whole_bad_batch_by_filter(db: Database) -> None:
    ids = []
    for i in range(3):
        vid = make_video(db, external_id=f"VIDEO_{i}",
                         url=f"https://example.invalid/{i}")
        ids.append(Q.enqueue(db, "download", vid))
        Q.enqueue(db, "captions", vid)
    _cancel(db, kind="download")
    assert {Q.get_job(db, i).state for i in ids} == {"cancelled"}
    assert {j.state for j in Q.list_jobs(db, kind="captions")} == {"pending"}


def test_cancel_can_target_one_video(db: Database) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    Q.enqueue(db, "download", a)
    Q.enqueue(db, "download", b)
    _cancel(db, target_id=a)
    states = {j.target_id: j.state for j in Q.list_jobs(db)}
    assert states == {a: "cancelled", b: "pending"}


def test_cancel_refuses_a_running_job_and_says_what_to_do(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid)
    job = Q.claim(db, "network")
    assert job is not None
    with pytest.raises(InvalidInputError, match="queue pause"):
        _cancel(db, job_id=job.id)
    assert Q.get_job(db, job.id).state == "running"


def test_cancel_by_filter_skips_running_jobs_rather_than_failing(
    db: Database,
) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    Q.enqueue(db, "download", a)
    Q.enqueue(db, "download", b)
    running = Q.claim(db, "network")
    assert running is not None
    result = _cancel(db, kind="download")
    assert "1 still running" in (result.message or "")
    assert Q.get_job(db, running.id).state == "running"
    others = [j for j in Q.list_jobs(db) if j.id != running.id]
    assert {j.state for j in others} == {"cancelled"}


def test_cancel_dry_run_changes_nothing(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid)
    result = _cancel(db, kind="download", dry_run=True)
    assert len(result.rows) == 1
    assert Q.get_job(db, job_id).state == "pending"


def test_cancel_with_no_filter_at_all_is_refused(db: Database) -> None:
    # "rytp jobs cancel" with nothing set would wipe the queue by accident.
    with pytest.raises(InvalidInputError, match="filter"):
        _cancel(db)


def test_a_cancelled_job_is_not_resurrected_by_reconcile(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid)
    _cancel(db, job_id=job_id)
    assert Q.reconcile(db) == 0
    assert Q.get_job(db, job_id).state == "cancelled"


def test_remove_a_superseded_rendition(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    old = touch(tmp_path / "v360.mp4", b"\x00" * 360)
    insert_asset(db, video_id=vid, role="video", format_id="360", path=str(old))
    new = touch(tmp_path / "v1080.mp4")
    insert_asset(db, video_id=vid, role="video", format_id="1080", path=str(new))
    asset_id = [r["id"] for r in assets_for(db, vid, "video")
                if r["format_id"] == "360"][0]

    result = _remove(db, asset_id=asset_id, yes=True)
    assert not old.exists()
    assert new.exists()
    assert "360" in (result.message or "")
    assert [r["format_id"] for r in assets_for(db, vid, "video")] == ["1080"]


def test_remove_without_yes_refuses_because_it_deletes_a_file(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "v360.mp4")
    asset_id = insert_asset(db, video_id=vid, role="video", format_id="360",
                            path=str(p))
    with pytest.raises(InvalidInputError, match="--yes"):
        _remove(db, asset_id=asset_id)
    assert p.exists()


def test_remove_dry_run_reports_bytes_and_deletes_nothing(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "v360.mp4", b"\x00" * 1234)
    asset_id = insert_asset(db, video_id=vid, role="video", format_id="360",
                            path=str(p))
    result = _remove(db, asset_id=asset_id, dry_run=True)
    assert "1234" in (result.message or "")
    assert p.exists()
    assert asset_for(db, vid, "video") is not None


def test_removing_the_audio_warns_loudly_but_still_works(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "audio.m4a")
    asset_id = insert_asset(db, video_id=vid, role="audio", path=str(p))
    result = _remove(db, asset_id=asset_id, yes=True)
    assert "re-align" in (result.message or "")
    assert not p.exists()
    assert asset_for(db, vid, "audio") is None


def test_removing_captions_warns_while_the_words_are_still_caption_tier(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "captions.json3")
    asset_id = insert_asset(db, video_id=vid, role="captions", path=str(p))
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, text, normalized_text, "
        "stem, source, engine) VALUES (?, 0, 0, 'a', 'a', 'a', 'caption', 'x')",
        (vid,),
    )
    result = _remove(db, asset_id=asset_id, yes=True)
    assert "not been superseded" in (result.message or "")


def test_removing_captions_is_quiet_once_an_aligned_transcript_exists(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "captions.json3")
    asset_id = insert_asset(db, video_id=vid, role="captions", path=str(p))
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, "
        "normalized_text, stem, source, engine) "
        "VALUES (?, 0, 0, 10, 'a', 'a', 'a', 'aligned', 'x')",
        (vid,),
    )
    result = _remove(db, asset_id=asset_id, yes=True)
    assert "not been superseded" not in (result.message or "")


def test_removing_a_rendition_makes_its_download_job_runnable_again(
    db: Database, tmp_path: Path
) -> None:
    # design §5's self-healing, from the deletion side.
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio",
                 path=str(touch(tmp_path / "audio.m4a")))
    asset_id = insert_asset(db, video_id=vid, role="video", format_id="720",
                            path=str(touch(tmp_path / "v720.mp4")))
    job_id = Q.enqueue(db, "download", vid)
    assert Q.get_job(db, job_id).state == "done"

    _remove(db, asset_id=asset_id, yes=True)
    assert Q.get_job(db, job_id).state == "pending"


def test_remove_of_an_unknown_asset_raises(db: Database) -> None:
    with pytest.raises(NotFoundError, match="4242"):
        _remove(db, asset_id=4242, yes=True)


def test_remove_tolerates_a_file_already_gone(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "v360.mp4")
    asset_id = insert_asset(db, video_id=vid, role="video", format_id="360",
                            path=str(p))
    p.unlink()
    _remove(db, asset_id=asset_id, yes=True)
    assert assets_for(db, vid, "video") == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_commands_remove.py -q`
Expected: every test fails, `ValueError` from `resolve("jobs.cancel")` naming the commands that do exist.

- [ ] **Step 3: Add `cancel` and `running_matches` to `rytp/jobs/queue.py`**

```python
def _filter_clause(
    *, job_id: int | None, kind: str | None, state: str | None, target_id: int | None
) -> tuple[str, list[Any]]:
    """Shared WHERE for the filter-based job commands."""
    where: list[str] = []
    params: list[Any] = []
    for column, value in (("id", job_id), ("kind", kind),
                          ("state", state), ("target_id", target_id)):
        if value:
            where.append(f"{column} = ?")
            params.append(value)
    return (" AND ".join(where) if where else ""), params


def running_matches(
    db: Database,
    *,
    job_id: int | None = None,
    kind: str | None = None,
    state: str | None = None,
    target_id: int | None = None,
) -> list[Job]:
    """The jobs a cancel would hit that are currently running."""
    clause, params = _filter_clause(job_id=job_id, kind=kind, state=state,
                                    target_id=target_id)
    sql = "SELECT * FROM jobs WHERE state = 'running'"
    if clause:
        sql += f" AND {clause}"
    return [Job.from_row(r) for r in db.conn.execute(sql, params).fetchall()]


def cancellable(
    db: Database,
    *,
    job_id: int | None = None,
    kind: str | None = None,
    state: str | None = None,
    target_id: int | None = None,
) -> list[Job]:
    """The jobs a cancel would actually change: everything but running."""
    clause, params = _filter_clause(job_id=job_id, kind=kind, state=state,
                                    target_id=target_id)
    sql = "SELECT * FROM jobs WHERE state NOT IN ('running', 'cancelled')"
    if clause:
        sql += f" AND {clause}"
    sql += " ORDER BY id"
    return [Job.from_row(r) for r in db.conn.execute(sql, params).fetchall()]


def cancel(db: Database, jobs: Sequence[Job], *, now: datetime | None = None) -> int:
    """Drop jobs out of the queue. Never touches a running job.

    ``cancelled`` is terminal as far as the queue is concerned — ``reconcile``
    only reopens ``done`` and ``blocked`` — but it is not a permanent veto:
    ``rytp jobs retry --state cancelled`` and re-running ``rytp ingest`` both
    bring a job back, deliberately.
    """
    if not jobs:
        return 0
    stamp = _now(now)
    with db.transaction():
        db.conn.executemany(
            "UPDATE jobs SET state = 'cancelled', finished_at = ?, "
            "last_error = NULL WHERE id = ? AND state != 'running'",
            [(stamp, j.id) for j in jobs],
        )
    return len(jobs)
```

- [ ] **Step 4: Append the two commands to `rytp/commands/ingest.py`**

```python
def jobs_cancel(
    db: Database,
    *,
    job_id: int = 0,
    kind: str = "",
    state: str = "",
    target_id: int = 0,
    dry_run: bool = False,
) -> CommandResult:
    """Drop queued or failed jobs, by id or by filter."""
    if not (job_id or kind or state or target_id):
        raise InvalidInputError(
            "jobs cancel needs at least one filter (--job-id, --kind, --state "
            "or --target-id); cancelling the whole queue by accident is worse "
            "than typing one more flag"
        )
    selector = {"job_id": job_id or None, "kind": kind or None,
                "state": state or None, "target_id": target_id or None}
    running = Q.running_matches(db, **selector)
    doomed = Q.cancellable(db, **selector)

    # A worker is mid-download inside yt-dlp or ffmpeg; there is no checkpoint
    # for it to abandon the job at, so marking it cancelled would be a lie the
    # worker overwrites with 'done' a minute later.
    if job_id and running:
        raise InvalidInputError(
            f"job {job_id} is running; stop the worker first "
            f"(`rytp queue pause`, then Ctrl-C it). A worker that has already "
            f"died leaves its jobs to be reclaimed on the next start."
        )

    rows = tuple((str(j.id), j.kind, str(j.target_id), j.state) for j in doomed)
    if dry_run:
        return CommandResult(
            columns=("id", "kind", "target", "state"),
            rows=rows,
            message=f"would cancel {len(doomed)} job(s)",
        )

    Q.cancel(db, doomed)
    parts = [f"cancelled {len(doomed)} job(s)"]
    if running:
        parts.append(
            f"{len(running)} still running and left alone — pause the queue "
            f"and stop the worker to cancel those"
        )
    parts.append("`rytp jobs retry --state cancelled` brings them back")
    return CommandResult(columns=("id", "kind", "target", "state"), rows=rows,
                         message="; ".join(parts))


def assets_remove(
    db: Database, *, asset_id: int, yes: bool = False, dry_run: bool = False
) -> CommandResult:
    """Delete one asset and its file. Typically a superseded rendition."""
    row = db.conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"asset {asset_id} does not exist")

    path = Path(row["path"])
    size = path.stat().st_size if path.is_file() else 0
    label = f"{row['role']}"
    if row["format_id"]:
        label += f" {row['format_id']}"

    warnings: list[str] = []
    if row["role"] in ("audio", "container"):
        warnings.append(
            f"WARNING: this is video {row['video_id']}'s {row['role']}; removing "
            f"it breaks re-align, render and WAV extraction until it is fetched "
            f"again"
        )
    if row["role"] == "captions" and not _has_aligned_words(db, int(row["video_id"])):
        warnings.append(
            "WARNING: this video's caption words have not been superseded by an "
            "aligned transcript, so removing the track loses the only text there is"
        )

    if dry_run:
        return CommandResult(
            columns=("asset", "role", "path", "bytes"),
            rows=((str(asset_id), label, str(path), str(size)),),
            message="; ".join([f"would free {size} bytes", *warnings]),
        )
    if not yes:
        raise InvalidInputError(
            f"assets remove deletes {path} ({size} bytes) from disk; pass --yes "
            f"to confirm, or --dry-run to see what would go"
        )

    if path.is_file():
        path.unlink()
    db.conn.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
    # The queue must reflect what is on disk: dropping a rendition makes its
    # download job runnable again, which is design §5's self-healing seen
    # from the deletion side.
    reopened = Q.reconcile(db, target_id=int(row["video_id"]), target_kind="video")
    return CommandResult(
        message="; ".join([
            f"removed {label} asset {asset_id}, freed {size} bytes",
            f"{reopened} job(s) reopened",
            *warnings,
        ])
    )


def _has_aligned_words(db: Database, video_id: int) -> bool:
    row = db.conn.execute(
        "SELECT 1 FROM words WHERE video_id = ? AND source = 'aligned' LIMIT 1",
        (video_id,),
    ).fetchone()
    return row is not None


register(
    Command(
        name="jobs.cancel",
        group="jobs",
        summary="Drop queued or failed jobs from the queue, by id or filter.",
        params=(
            Param("job_id", int, "One job id. 0 means use the filters.", default=0),
            Param("kind", str, "Only this job kind.", default=""),
            Param("state", str, "Only this state.", default="",
                  choices=("", *_JOB_STATES)),
            Param("target_id", int, "Only jobs for this target. 0 means any.",
                  default=0),
            Param("dry_run", bool, "List what would be cancelled, change nothing.",
                  default=False),
        ),
        handler=jobs_cancel,
    )
)

register(
    Command(
        name="assets.remove",
        group="assets",
        summary="Delete one asset and its file, to reclaim disk after an upgrade.",
        params=(
            Param("asset_id", int, "Id of the asset, from `rytp jobs list` or the "
                  "assets table.", default=REQUIRED, positional=True),
            Param("yes", bool, "Confirm; required because a file is deleted.",
                  default=False),
            Param("dry_run", bool, "Report what would go, delete nothing.",
                  default=False),
        ),
        handler=assets_remove,
    )
)
```

Add `from pathlib import Path` and widen the models import to
`from rytp.models import InvalidInputError, NotFoundError, RytpError` at the top
of the file.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_commands_remove.py -q`
Expected: PASS, 17 passed.

- [ ] **Step 6: Commit**

```bash
git add rytp/jobs/queue.py rytp/commands/ingest.py tests/test_commands_remove.py
git commit -m "feat: add jobs cancel and assets remove"
```

---

### Task 14: Health checks for `doctor`

Contracts §5's Health checks subsection. Part 1 owns the `doctor` command and the `HEALTH_CHECKS` registry; Part 2 registers four checks: `ffmpeg`, `ffprobe`, `yt-dlp` and free disk. The registry mirrors `JOB_HANDLERS`, so the pattern is the one already in this plan.

Two rules from the contract shape the code. **A check never raises and never blocks** — every one of these wraps its probe in a `try` and turns any failure into a `HealthResult`, because a `doctor` that crashes is worse than no `doctor`. And **`ok` always tells the truth about what was found, while `required` decides whether that is fatal**: `doctor` exits non-zero only when a `required=True` check returns `ok=False`. So a missing yt-dlp is honestly `ok=False` on a `required=False` check — local-file workflows genuinely do not need it, and the report says so without failing.

That distinction also settles what the disk checks mean, which is why there are two of them rather than one:

- **`disk`** (`required=True`) asks a yes-or-no question: can the data volume be read at all? `ok=False` here means the tool cannot work, and it should stop the report.
- **`disk-headroom`** (`required=False`) is advisory: `ok=False` when free space is below the threshold. A full drive is a real finding and `ok` says so, but it must not make `doctor` exit non-zero — the owner may be deliberately running the drive close to full, and a `doctor` that fails every day is a `doctor` nobody reads.

The headroom check earns its place. Design §1 puts the corpus at ~1,600 videos of about an hour, and the owner has been explicit about wanting to know before a batch fills the drive — so it does not just print free bytes, it measures the average bytes already spent per video and reports how many more that affords.

**Files:**
- Modify: `rytp/commands/ingest.py` (append)
- Test: `tests/test_health_checks.py`

`HealthCheck` carries `required: bool = True` (contracts §5). `ok` always states what was found; `required` decides whether that finding is fatal.

**Interfaces:**
- Consumes: `rytp.commands.{HealthCheck, HealthResult, register_check, HEALTH_CHECKS}` (Part 1); `rytp.config.data_root`.
- Produces: registered checks `ffmpeg` and `ffprobe` (`required=True`), `yt-dlp` and `disk-headroom` (`required=False`), and `disk` (`required=True`); `rytp.commands.ingest.{_binary_check, check_disk, check_disk_headroom, check_ytdlp}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_health_checks.py`:

```python
"""Tests for Part 2's doctor checks (contracts §5, Health checks)."""

from __future__ import annotations

import subprocess

import pytest

from rytp import constants as C
from rytp.commands import HEALTH_CHECKS
from rytp.commands import ingest as I
from rytp.db import Database
from rytp.db.queries import insert_asset

from tests.fakes import make_video, touch


def test_part_two_registers_its_checks_with_the_right_severity() -> None:
    for name in ("ffmpeg", "ffprobe", "yt-dlp", "disk", "disk-headroom"):
        assert name in HEALTH_CHECKS
        assert HEALTH_CHECKS[name].summary
    # doctor exits non-zero only on a required check, so this split is the
    # difference between "cannot work" and "worth knowing".
    assert HEALTH_CHECKS["ffmpeg"].required is True
    assert HEALTH_CHECKS["ffprobe"].required is True
    assert HEALTH_CHECKS["disk"].required is True
    assert HEALTH_CHECKS["yt-dlp"].required is False
    assert HEALTH_CHECKS["disk-headroom"].required is False


def test_a_present_binary_reports_its_version(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(I.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        I, "_run_version",
        lambda exe: subprocess.CompletedProcess(
            [exe], 0, "ffmpeg version 7.1 Copyright (c)\nbuilt with clang\n", ""
        ),
    )
    result = HEALTH_CHECKS["ffmpeg"].run(db)
    assert result.ok is True
    assert "7.1" in result.detail
    assert "built with clang" not in result.detail   # first line only
    assert result.remedy is None


def test_a_missing_required_binary_fails_with_an_exact_remedy(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(I.shutil, "which", lambda name: None)
    result = HEALTH_CHECKS["ffprobe"].run(db)
    assert result.ok is False
    assert "not on PATH" in result.detail
    assert result.remedy and "ffmpeg.org" in result.remedy


def test_a_binary_that_cannot_answer_is_reported_not_raised(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(I.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def explode(exe: str):
        raise OSError("Exec format error")

    monkeypatch.setattr(I, "_run_version", explode)
    result = HEALTH_CHECKS["ffmpeg"].run(db)   # must not raise
    assert result.ok is False
    assert "Exec format error" in result.detail


def test_a_missing_yt_dlp_is_reported_honestly_but_is_not_fatal(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # contracts §5: ok tells the truth about what was found; required=False on
    # the check is what makes it non-fatal.
    monkeypatch.setattr(I, "_ytdlp_version", lambda: None)
    result = HEALTH_CHECKS["yt-dlp"].run(db)
    assert result.ok is False
    assert "not installed" in result.detail
    assert result.remedy and "pip install" in result.remedy
    assert HEALTH_CHECKS["yt-dlp"].required is False


def test_a_present_yt_dlp_reports_its_version(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(I, "_ytdlp_version", lambda: "2026.9.1")
    result = HEALTH_CHECKS["yt-dlp"].run(db)
    assert result.ok is True and "2026.9.1" in result.detail


def test_a_readable_volume_passes_the_required_disk_check(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A nearly full drive still passes THIS check: the tool can work.
    monkeypatch.setattr(I, "_free_bytes", lambda: 1)
    result = HEALTH_CHECKS["disk"].run(db)
    assert result.ok is True
    assert "GiB free" in result.detail


def test_an_unreadable_data_volume_fails_the_required_disk_check(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode():
        raise OSError("device not ready")

    monkeypatch.setattr(I, "_free_bytes", explode)
    result = HEALTH_CHECKS["disk"].run(db)   # must not raise
    assert result.ok is False
    assert "device not ready" in result.detail


def test_ample_headroom_passes_and_counts_the_videos_that_fit(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(I, "_free_bytes", lambda: C.HEALTH_DISK_FREE_MIN_BYTES * 4)
    result = HEALTH_CHECKS["disk-headroom"].run(db)
    assert result.ok is True
    assert "more video" in result.detail
    assert result.remedy is None


def test_low_headroom_is_an_honest_failure_but_advisory(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(I, "_free_bytes", lambda: C.HEALTH_DISK_FREE_MIN_BYTES // 2)
    result = HEALTH_CHECKS["disk-headroom"].run(db)
    assert result.ok is False
    assert result.remedy and "cache prune" in result.remedy
    # Honest about the finding, but it must not make doctor exit non-zero.
    assert HEALTH_CHECKS["disk-headroom"].required is False


def test_the_headroom_estimate_uses_real_measured_sizes(
    db: Database, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio",
                 path=str(touch(tmp_path / "a.m4a")), size_bytes=C.BYTES_PER_GIB)
    monkeypatch.setattr(I, "_free_bytes", lambda: 10 * C.BYTES_PER_GIB)
    result = HEALTH_CHECKS["disk-headroom"].run(db)
    assert "10 more video" in result.detail


def test_an_unreadable_volume_does_not_crash_the_headroom_check(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode():
        raise OSError("device not ready")

    monkeypatch.setattr(I, "_free_bytes", explode)
    result = HEALTH_CHECKS["disk-headroom"].run(db)   # must not raise
    assert result.ok is False


def test_no_check_raises_whatever_the_machine_looks_like(db: Database) -> None:
    # The real machine, unmocked: whatever is or is not installed, doctor
    # must come back with a result per check and no traceback.
    for name in ("ffmpeg", "ffprobe", "yt-dlp", "disk", "disk-headroom"):
        result = HEALTH_CHECKS[name].run(db)
        assert isinstance(result.ok, bool)
        assert result.detail
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_health_checks.py -q`
Expected: collection error or failures — `HEALTH_CHECKS` has no `ffmpeg` key.

- [ ] **Step 3: Append the checks to `rytp/commands/ingest.py`**

```python
# ---------------------------------------------------------------------------
# doctor checks (contracts §5, Health checks). Part 1 owns the command and the
# registry; these four are Part 2's. None of them may raise, and a missing
# optional dependency is reported rather than failed.
# ---------------------------------------------------------------------------

_FFMPEG_REMEDY = (
    "install ffmpeg from https://ffmpeg.org/ and put its bin/ directory on PATH"
)


def _run_version(exe: str) -> subprocess.CompletedProcess[str]:
    """Isolated so tests can replace it without a real binary."""
    return subprocess.run(
        [exe, "-version"], check=False, capture_output=True, text=True,
        timeout=C.HEALTH_VERSION_TIMEOUT_S,
    )


def _binary_check(name: str) -> HealthResult:
    """Is this binary on PATH, and what does it say its version is?"""
    try:
        exe = shutil.which(name)
        if exe is None:
            return HealthResult(
                ok=False, detail=f"{name} is not on PATH", remedy=_FFMPEG_REMEDY
            )
        proc = _run_version(exe)
        first = next(iter(proc.stdout.splitlines()), "").strip()
        if proc.returncode != 0:
            return HealthResult(
                ok=False,
                detail=f"{exe} exited {proc.returncode} for -version",
                remedy=_FFMPEG_REMEDY,
            )
        return HealthResult(ok=True, detail=f"{exe}: {first or 'version unknown'}")
    except Exception as exc:  # noqa: BLE001 - a check never raises
        return HealthResult(ok=False, detail=f"{name}: {exc}", remedy=_FFMPEG_REMEDY)


def _ytdlp_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("yt-dlp")
    except PackageNotFoundError:
        return None


def check_ytdlp(db: Database) -> HealthResult:
    """Is yt-dlp installed?

    ``ok`` reports honestly that it is not. The check is registered with
    ``required=False``, which is what keeps that from failing ``doctor``:
    local-file workflows never touch the network, and a report that fails
    every day for an extra the owner does not use is a report nobody reads.
    """
    del db
    try:
        found = _ytdlp_version()
    except Exception as exc:  # noqa: BLE001 - a check never raises
        return HealthResult(ok=False, detail=f"yt-dlp: {exc}")
    if found is None:
        return HealthResult(
            ok=False,
            detail="yt-dlp is not installed; downloading and channel sync will "
                   "not work, everything else will",
            remedy="pip install -e '.[yt-dlp]'",
        )
    return HealthResult(ok=True, detail=f"yt-dlp {found}")


def _free_bytes() -> int:
    return shutil.disk_usage(config.data_root()).free


def _headroom(db: Database, free: int) -> tuple[int, int]:
    """How many more videos fit, and what one is measured to cost.

    design §1 puts the corpus at ~1,600 videos of about an hour, so the
    number of bytes left is much less useful than the number of videos left.
    The per-video figure comes from what the catalogue has actually cost,
    falling back to an estimate only while there is nothing to measure.
    """
    row = db.conn.execute(
        "SELECT COUNT(DISTINCT video_id) AS videos, SUM(bytes) AS total FROM assets"
    ).fetchone()
    videos = int(row["videos"] or 0)
    total = int(row["total"] or 0)
    per_video = total // videos if videos and total else C.ESTIMATED_BYTES_PER_VIDEO
    per_video = max(per_video, 1)
    return free // per_video, per_video


def check_disk(db: Database) -> HealthResult:
    """Can the data volume be read at all? Required: nothing works if not."""
    del db
    try:
        free = _free_bytes()
    except Exception as exc:  # noqa: BLE001 - a check never raises
        return HealthResult(
            ok=False,
            detail=f"cannot read free space on {config.data_root()}: {exc}",
            remedy=f"check that {config.data_root()} exists and is readable, or "
                   f"point RYTP_DATA somewhere that is",
        )
    # Deliberately ok even at one byte free: a full drive is a different
    # finding, and disk-headroom is the check that reports it.
    return HealthResult(
        ok=True,
        detail=f"{config.data_root()} readable, {free // C.BYTES_PER_GIB} GiB free",
    )


def check_disk_headroom(db: Database) -> HealthResult:
    """Is there room for more of the corpus? Advisory, not required."""
    try:
        free = _free_bytes()
    except Exception as exc:  # noqa: BLE001 - a check never raises
        return HealthResult(ok=False, detail=f"cannot measure free space: {exc}")

    fits, per_video = _headroom(db, free)
    detail = (
        f"{free // C.BYTES_PER_GIB} GiB free; room for about {fits} more "
        f"video(s) at {per_video // (C.BYTES_PER_GIB // 1024)} MiB each"
    )
    if free < C.HEALTH_DISK_FREE_MIN_BYTES:
        return HealthResult(
            ok=False,
            detail=detail,
            remedy="free space, or reclaim some with `rytp cache prune --yes` "
                   "and `rytp assets remove <id> --yes` on superseded renditions",
        )
    return HealthResult(ok=True, detail=detail)


register_check(HealthCheck(
    name="ffmpeg",
    summary="ffmpeg is on PATH; every audio and video operation needs it.",
    run=lambda db: _binary_check("ffmpeg"),
))
register_check(HealthCheck(
    name="ffprobe",
    summary="ffprobe is on PATH; media is inspected with it.",
    run=lambda db: _binary_check("ffprobe"),
))
register_check(HealthCheck(
    name="yt-dlp",
    summary="yt-dlp is importable; needed only to download and sync channels.",
    run=check_ytdlp,
    required=False,
))
register_check(HealthCheck(
    name="disk",
    summary="The data volume is readable.",
    run=check_disk,
))
register_check(HealthCheck(
    name="disk-headroom",
    summary="Free space on the data volume, counted in videos that still fit.",
    run=check_disk_headroom,
    required=False,
))
```

Widen the imports at the top of `rytp/commands/ingest.py` with `import shutil`,
`import subprocess`, `from rytp import config` and
`from rytp.commands import HealthCheck, HealthResult, register_check`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_health_checks.py -q`
Expected: PASS, 13 passed.

- [ ] **Step 5: See it against the real machine**

Run: `python -m rytp doctor`
Expected: a table with Part 1's checks and Part 2's five. `ffmpeg` and `ffprobe` report a version or name the install remedy; `yt-dlp` reports its version, or reports honestly that it is absent without changing the exit code; `disk` says the volume is readable and `disk-headroom` prints free GiB and a video count. No traceback under any outcome, and the exit code is non-zero only if a required check failed. Then confirm the split does what it claims: `pip uninstall yt-dlp` and re-run — the yt-dlp line reads as failed, and `echo $?` is still 0.

- [ ] **Step 6: Commit**

```bash
git add rytp/commands/ingest.py tests/test_health_checks.py
git commit -m "feat: register Part 2's doctor checks for ffmpeg, ffprobe, yt-dlp and disk"
```

---

## End state a reviewer can check

After Task 12, with only fakes standing in for yt-dlp and ffmpeg:

1. `rytp ingest <id>` on a remote video creates one job per registered chain kind — with only Part 2 present that is `download` and `captions` pending on the `network` pool and `extract_wav` blocked until there is audio, and the result message names `caption_words` and `fingerprint` as not yet available. `rytp ingest --pending --limit 2000` does the same for the whole catalogue, and `--dry-run` lists what it would touch.
2. `rytp worker --once` runs them and ends with an `audio` asset, a `video` asset, a `captions` asset and `cache/wav/<id>.wav` on disk, nothing merged — including the `extract_wav` job, which started `blocked` and became runnable the moment the download finished (`Q.unblock` after every finished job is what does that).
3. Killing the worker mid-job and starting it again returns the orphaned `running` row to `pending` with its attempt already spent; a second worker started while the first is alive refuses with one line.
4. A 429 from the fake runner freezes the whole network pool for five minutes, refunds the job's attempt, and shows up as `throttled 1` with a `next attempt` timestamp in `rytp jobs stats` — and the next download is *not* attempted.
5. `rytp cache prune` deletes the WAV, reports the bytes freed, and flips the `extract_wav` job from `done` back to `pending` without anyone enqueuing anything.
6. `rytp ingest <id>` on a local file creates a `container` asset in place and no download job; one unreadable local file in a bulk run is skipped and counted, not fatal.
7. No test opens a socket, and `grep -rn "yt_dlp" rytp/` shows the import only inside function bodies.
8. `rytp jobs cancel --kind download` empties a bad batch out of the queue and says how many were left alone because they are running; `rytp jobs cancel --job-id <running>` refuses and names the fix. A cancelled job is not resurrected by `reconcile`, but `rytp jobs retry --state cancelled` brings it back.
9. `rytp assets remove <id> --dry-run` prints the path and byte count; with `--yes` it deletes the file, reopens that video's `download` job, and warns in plain words when the asset was the audio or an un-superseded caption track.
10. `rytp doctor` reports `ffmpeg`, `ffprobe`, `yt-dlp`, `disk` and `disk-headroom`. Uninstall yt-dlp and its line reads as failed while the exit code stays 0 — `ok` is honest, `required=False` makes it advisory. A missing ffmpeg fails and exits non-zero, with the install command as its remedy. A nearly full drive fails `disk-headroom` and still exits 0; an unreadable data volume fails `disk` and does not.

## Self-review

**Spec coverage.** Every Part 2 item in design §5 and the brief maps to a task: readiness predicates (Task 3), three pools (Tasks 4, 10), atomic crash-safe claiming (Tasks 4, 10), download policy with every value in `settings` (Task 1), no browser cookies (Task 5 — the option is never set and the module docstring says why), audio + video together and unmerged (Task 6), captions always as their own asset (Task 7), local files as containers with no download job (Task 8), three-listing channel enumeration (Task 5), the WAV cache with a prune command and no path in any table (Task 9), and all nine original commands plus the worker entry point (Tasks 11, 12). Contracts §5's Deletion table gives Part 2 `jobs.cancel` and `assets.remove` (Task 13), and its Health checks subsection gives Part 2 the `ffmpeg`, `ffprobe`, `yt-dlp`, `disk` and `disk-headroom` checks that Part 1's `doctor` runs (Task 14). The chain past Part 2's own kinds — `caption_words`, `fingerprint`, `transcribe`, `align` — is enqueued by Task 11 under a registration guard, with names, pools and `reopenable` flags agreed with the Part 3 author.

**Deliberately out of scope**, and named here so nobody looks for them: parsing json3 captions into `words` (Part 3, `rytp/transcribe/captions.py`), the `transcribe` / `align` / `diarize` / `fingerprint` / `index` / `render` job kinds (each part registers its own through `register_job_kind`), `videos.add` and `channels.sync` (Part 1's `commands/catalog.py`, which calls Task 5's `enumerate_channel` and `probe_video`), and the TUI.

**Contract deviations:** none. This plan tracks the contracts as amended on 2026-09-21, including the "Job handlers" subsection of §5 — `rytp/jobs/__init__.py` exposes `JOB_HANDLERS: dict[str, Callable[[Database, int, dict], None]]`, and every handler takes `(db, target_id, payload)` and returns `None`. `JOB_KINDS` is the richer registry that also carries the pool, the readiness predicate and the summary; `register_job_kind` writes both, so the contracted view can never drift. Four observations for the contract author are in the report rather than changed here: `jobs` has no owner or lease column, so the worker lease lives in `settings`; `assets` has no uniqueness constraint, so design §4's one-audio-one-captions rule is enforced in `insert_asset`; design §4 still names `data/audio/{video_id}.wav` where contracts §7 says `cache/wav/`, and contracts win; and `probe_video` can never yield `kind='short'`.

**Decisions the contracts left to me, stated rather than implied:** `jobs.cancel` **refuses** to cancel a `running` job rather than marking it cancelled for the worker to abandon — Part 2's handlers sit inside `yt-dlp` and `ffmpeg` subprocesses with no checkpoint, so the worker would finish the work and overwrite `cancelled` with `done`; the command names the running ids and points at `rytp queue pause`. `assets.remove` **warns and proceeds** on the audio, container and un-superseded-captions cases, because the owner may genuinely want the space. For `doctor`, the split between "what was found" and "does it matter" is carried by `HealthCheck.required`: a missing yt-dlp is honestly `ok=False` on a `required=False` check. The disk question is two checks for the same reason — `disk` (`required=True`) fails only when the data volume cannot be read, while `disk-headroom` (`required=False`) fails honestly when space is low without making `doctor` exit non-zero, because the owner may be deliberately running the drive close to full.

**Ambiguities resolved here, not in the contracts:** throttle backoff applies to the whole network pool rather than to the job that tripped it; `enumerate_channel` does not deduplicate across tabs (Part 1 upserts, so the last tab wins the `kind`); a finished job triggers `unblock` (blocked jobs only) rather than a full `reconcile`, because a predicate that still answers READY after a successful run would otherwise loop forever; readiness predicates are payload-free while handlers are not; and `jobs.target_id` is namespaced by `JobKind.target_kind` so a non-video target (a cut list) cannot collide with a `videos.id`.
