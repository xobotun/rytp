"""The schema, the fixtures that write into it, and every declared dependency.

Contracts §3 makes each part responsible for the NOT NULL columns of the tables
it inserts into, which means no part is responsible for checking that they all
did. Contracts §5 pins signatures precisely because naming a responsibility
without naming a shape is how three parts invented three speaker resolvers.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

from rytp.db import Database
from rytp.db.schema import MIGRATIONS
from tests.consistency import read_sources, repo_root, source_files

# --- the schema is exactly the contract ------------------------------

CONTRACT_TABLES = {
    "channels",
    "videos",
    "assets",
    "speakers",
    "video_speakers",
    "words",
    "utterances",
    "utterances_fts",
    "video_acoustics",
    "jobs",
    "renders",
    "settings",
    "schema_version",
}

CONTRACT_INDEXES = {
    "videos_channel",
    "videos_source",
    "assets_video_role",
    "assets_one_audio",
    "assets_one_captions",
    "video_speakers_speaker",
    "words_video_ord",
    "words_normalized",
    "words_alignable",
    "words_timed",
    "words_stem",
    "words_speaker",
    "utterances_video",
    "jobs_claim",
    "renders_cutlist",
}

CONTRACT_TRIGGERS = {"utterances_ai", "utterances_ad", "utterances_au"}


def _names(db: Database, kind: str) -> set[str]:
    rows = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = ?", (kind,)
    ).fetchall()
    return {
        str(row["name"])
        for row in rows
        if not str(row["name"]).startswith("sqlite_")
        and not re.fullmatch(r"utterances_fts_\w+", str(row["name"]))
    }


def test_the_tables_are_exactly_the_ones_the_contract_lists(db: Database) -> None:
    """contracts §3: "Part 1 creates every table in this section… Later parts
    consume the schema; none of them add DDL"."""
    assert _names(db, "table") == CONTRACT_TABLES


def test_the_indexes_are_exactly_the_ones_the_contract_lists(db: Database) -> None:
    """`words_alignable` is here because finding 4.1 was an index that did not
    exist: without it the assembler's hot lookup scales with the caption tier."""
    assert _names(db, "index") == CONTRACT_INDEXES


def test_the_fts_triggers_all_exist(db: Database) -> None:
    assert _names(db, "trigger") == CONTRACT_TRIGGERS


def test_migrations_are_contiguous_and_nobody_renumbered(db: Database) -> None:
    versions = [version for version, _sql in MIGRATIONS]
    assert versions == list(range(1, len(versions) + 1)), versions
    assert db.schema_version() == versions[-1]


def _columns(db: Database, table: str) -> set[str]:
    return {row["name"] for row in db.conn.execute(f"PRAGMA table_info({table})")}


def test_words_declares_align_scale(db: Database) -> None:
    """Contracts §3 (amended): `words.align_scale`, gained by migration 13."""
    assert "align_scale" in _columns(db, "words")


def test_jobs_declares_progress(db: Database) -> None:
    """Contracts §3 (amended): `jobs.progress`, gained by migration 14 —
    distinct from `jobs.note` (contracts §5 "Job handlers")."""
    assert "progress" in _columns(db, "jobs")


def test_align_scales_constant_matches_the_contracts_comment(db: Database) -> None:
    """contracts §3's `words` DDL block names the permitted `align_scale`
    values in a comment: "The scale that produced align_score: 'energy' |
    'logprob' | 'none' | 'unknown'." Parse the comment itself rather than
    asserting a hardcoded set twice, so the two can only agree by actually
    matching."""
    from rytp import constants as C

    contracts_path = (
        repo_root() / "docs" / "superpowers" / "specs" / "2026-09-21-rytp-contracts.md"
    )
    text = contracts_path.read_text(encoding="utf-8")
    match = re.search(
        r"The scale that produced align_score:(.*?)\.", text, re.DOTALL
    )
    assert match, "contracts §3 no longer names the align_scale values in this comment"
    named = set(re.findall(r"'(\w+)'", match.group(1)))
    assert named == set(C.ALIGN_SCALES)


def test_the_three_transcript_tiers_are_enforced_by_the_database(db: Database) -> None:
    """contracts §3: "Cuttable is `source = 'aligned'` and nothing else may be
    cut". `timed` exists because a transcriber's own word timings are not good
    enough to cut on, and calling them `aligned` is the mistake the tier was
    invented to make impossible."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'words'"
    ).fetchone()["sql"]
    for tier in ("caption", "timed", "aligned"):
        assert f"'{tier}'" in sql, tier


def test_a_caption_row_is_the_only_one_allowed_to_have_no_end(db: Database) -> None:
    """contracts §3's CHECK. Part 4's `implied_end_ms` exists because of it."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'words'"
    ).fetchone()["sql"]
    assert "source = 'caption' OR end_ms IS NOT NULL" in sql.replace("\n", " ")


def test_the_fts_tokenizer_is_never_the_english_one(db: Database) -> None:
    """Porter is English-only; on a Russian corpus it stems nothing at all."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'utterances_fts'"
    ).fetchone()["sql"]
    assert "unicode61" in sql
    assert "porter" not in sql


# --- NOT NULL columns and the fixtures that must supply them ---------


def _required_columns(db: Database, table: str) -> set[str]:
    """NOT NULL, no default, not the rowid alias — i.e. every insert's job."""
    return {
        str(row["name"])
        for row in db.conn.execute(f"PRAGMA table_info({table})")
        if row["notnull"] and row["dflt_value"] is None and not row["pk"]
    }


_INSERT_RE = re.compile(
    r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+(\w+)\s*\(([^)]*)\)", re.IGNORECASE | re.DOTALL
)


def _literal_inserts() -> list[tuple[Path, str, set[str]]]:
    """(file, table, columns) for every literal column-list INSERT under tests/.

    Column lists are frequently wrapped across two adjacent string literals to
    stay under the line-length limit, which puts a stray quote character in
    the raw source text between two column names (e.g. ``"…, "\\n    "col)"``).
    A naive split on ``,`` treats that quote as part of a column name, so this
    extracts identifiers directly instead of splitting on commas — the joins
    and whitespace are not word characters and never survive the match.
    """
    found: list[tuple[Path, str, set[str]]] = []
    for path, text in read_sources("tests").items():
        for table, columns in _INSERT_RE.findall(text):
            names = set(re.findall(r"\w+", columns))
            found.append((path, table, names))
    return found


def test_every_fixture_supplies_the_not_null_columns_of_the_table_it_writes(
    db: Database,
) -> None:
    """The `video_speakers.engine` finding, generalised.

    An omitted NOT NULL column fails at insert time, in whichever test happens
    to run first, with a message about a constraint rather than about a
    fixture. This names the file and the column instead.
    """
    tables = _names(db, "table")
    missing: list[str] = []
    for path, table, columns in _literal_inserts():
        if table not in tables or table.endswith("_fts"):
            continue
        absent = _required_columns(db, table) - columns
        if absent:
            missing.append(f"{path.name}: INSERT INTO {table} omits {sorted(absent)}")
    assert not missing, missing


def test_the_shared_video_fixture_writes_a_complete_row(db: Database) -> None:
    """`tests/fakes.py::make_video` is used by six modules; if it ever stops
    filling a required column, everything downstream fails at once."""
    from tests.fakes import make_video

    video_id = make_video(db)
    row = db.conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    for column in _required_columns(db, "videos"):
        assert row[column] is not None, column


# --- import hygiene --------------------------------------------------


def _child(code: str) -> subprocess.CompletedProcess[str]:
    """Run one line in a fresh interpreter, from a directory with no data tree."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )


def test_models_imports_snowballstemmer_lazily() -> None:
    """Explicitly called out by the review as load-bearing and unpinned.

    `rytp/transcribe/base.py` is imported inside a foreign interpreter — an
    engine's own virtualenv, which has rytp's dependencies but not necessarily
    rytp's extras. It reaches `rytp.models` for `RawWord` and `Span`. A
    module-level `import snowballstemmer` there turns a missing package in
    someone else's environment into an out-of-process engine that cannot start.
    """
    source = (repo_root() / "rytp" / "models.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in getattr(node, "names", [])
    }
    module_names = {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "snowballstemmer" not in top_level | module_names, (
        "rytp/models.py imports snowballstemmer at module level; move it inside "
        "stem_text()"
    )
    assert "snowballstemmer" in source, "the stemmer must still be used, just lazily"

    proc = _child(
        "import sys, rytp.models; "
        "print('leaked' if 'snowballstemmer' in sys.modules else 'clean')"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "clean"


def test_importing_the_registries_pulls_in_no_optional_extra() -> None:
    """contracts §1: heavy dependencies are "imported lazily inside functions,
    never at module import time". The registries are what every command and the
    worker import, so this is the import path that matters."""
    proc = _child(
        "import sys, rytp.commands, rytp.jobs, rytp.tui.app; "
        "heavy = {'torch', 'transformers', 'faster_whisper', 'gigaam', "
        "'pyannote', 'yt_dlp', 'torchaudio'} & set(sys.modules); "
        "print(sorted(heavy))"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


def test_importing_rytp_creates_no_directories(tmp_path: Path) -> None:
    """The gotcha the old codebase had: a module-level `paths` singleton that
    mkdir'd on import. contracts §7: created "by the command that needs it —
    **not at module import time**"."""
    proc = subprocess.run(
        [sys.executable, "-c", "import rytp, rytp.config, rytp.commands"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        env={**__import__("os").environ, "PYTHONPATH": str(repo_root())},
    )
    assert proc.returncode == 0, proc.stderr
    assert list(tmp_path.iterdir()) == []


def test_every_base_dependency_is_importable() -> None:
    """contracts §1 names six, and a `.[dev]` install must run every test."""
    for name in ("typer", "rich", "textual", "numpy", "scipy", "snowballstemmer"):
        importlib.import_module(name)


def test_every_module_uses_postponed_annotations() -> None:
    """contracts §1: "`from __future__ import annotations` in **every** module"."""
    missing = [
        str(path.relative_to(repo_root()))
        for path in source_files("rytp")
        if path.name != "__init__.py" or path.read_text(encoding="utf-8").strip()
        if "from __future__ import annotations" not in path.read_text(encoding="utf-8")
    ]
    assert not missing, missing


def test_no_real_identifier_is_committed() -> None:
    """contracts §1, and the owner's standing instruction: no video ids, channel
    ids, channel names or URLs anywhere, tests included.

    A channel-handle path segment is exempt exactly when it follows the
    sanctioned `example.invalid` placeholder domain (contracts §1's own
    example is `https://example.invalid/...`), so `tests/fakes.py`'s
    `CHANNEL_ONE_URL` is not flagged as a real channel handle.
    """
    pattern = re.compile(r"youtube\.com|youtu\.be|(?<!example\.invalid)/@[A-Za-z0-9_]{3,}")
    offenders = [
        str(path)
        for package in ("rytp", "tests")
        for path, text in read_sources(package).items()
        if pattern.search(text)
    ]
    assert not offenders, offenders


# --- what every part declared it consumes ----------------------------

#: (module, attribute, required parameter names or None). Assembled from the
#: `Interfaces / Consumes` blocks of Parts 1-7. contracts §5 fixes the ones
#: with parameter lists; the rest need only exist, because a part that renamed
#: an export would otherwise fail at whatever hour the consumer first runs.
CONSUMED: tuple[tuple[str, str, tuple[str, ...] | None], ...] = (
    ("rytp.models", "normalize_text", ("text",)),
    ("rytp.models", "stem_text", ("normalized",)),
    ("rytp.models", "utc_now_iso", ()),
    ("rytp.models", "RawWord", None),
    ("rytp.models", "Span", None),
    ("rytp.models", "DiarSegment", None),
    ("rytp.models", "Fragment", None),
    ("rytp.models", "RytpError", None),
    ("rytp.models", "NotFoundError", None),
    ("rytp.models", "InvalidInputError", None),
    ("rytp.commands", "SpeakerFilter", None),
    (
        "rytp.commands",
        "resolve_speaker_filter",
        ("db", "speaker", "video_local_speaker", "video_id"),
    ),
    ("rytp.commands", "resolve_video_id", ("db", "ref")),
    ("rytp.commands", "register", ("cmd",)),
    ("rytp.commands", "resolve", ("name",)),
    ("rytp.commands", "register_check", ("check",)),
    ("rytp.commands", "HEALTH_CHECKS", None),
    ("rytp.commands", "SPEAKER_PARAMS", None),
    ("rytp.commands", "PARAM_ALIASES", None),
    ("rytp.cli", "build_app", None),
    ("rytp.config", "paths", ()),
    ("rytp.config", "ensure_dir", ("path",)),
    ("rytp.db", "Database", None),
    ("rytp.db.queries", "get_setting", ("db", "key", "default")),
    ("rytp.db.queries", "set_setting", ("db", "key", "value")),
    ("rytp.jobs", "register_job_kind", ("kind",)),
    ("rytp.jobs", "resolve_job_kind", ("name",)),
    ("rytp.jobs", "kinds_for_pool", ("pool",)),
    ("rytp.jobs.queue", "enqueue", ("db", "kind", "target_id", "priority", "payload")),
    ("rytp.jobs.queue", "list_jobs", ("db", "state", "pool", "kind", "limit")),
    ("rytp.jobs.queue", "stats", ("db",)),
    ("rytp.jobs.queue", "retry", ("db", "job_id", "kind", "state")),
    ("rytp.jobs.queue", "get_job", ("db", "job_id")),
    ("rytp.jobs.queue", "is_paused", ("db",)),
    ("rytp.jobs.worker", "run_worker", None),
    ("rytp.audio.extract", "wav_path", ("video_id",)),
    ("rytp.transcribe.registry", "TRANSCRIBERS", None),
    ("rytp.transcribe.registry", "ALIGNERS", None),
    ("rytp.diarize.base", "DIARIZERS", None),
    ("rytp.assemble.cutlist", "load_cutlist", ("path",)),
    ("rytp.assemble.cutlist", "write_cutlist", ("cutlist", "path")),
    ("rytp.assemble.cutlist", "cutlist_path", ("name",)),
    ("rytp.assemble.cutlist", "validate_name", ("name",)),
    ("rytp.assemble.cutlist", "CutList", None),
    ("rytp.assemble.cutlist", "Slot", None),
    ("rytp.assemble.cutlist", "Alternative", None),
    ("rytp.assemble.cutlist", "Substitution", None),
    ("rytp.assemble.cutlist", "CutlistError", None),
    ("rytp.render.ffmpeg", "Tools", None),
    ("rytp.tui.palette", "palette_entries", ("commands",)),
    ("rytp.tui.palette", "parse_arguments", ("cmd", "text")),
    ("rytp.tui.palette", "cli_invocation", ("cmd", "values")),
    ("rytp.tui.screens.search", "SearchScreen", None),
    ("rytp.tui.screens.transcript", "TranscriptVideosScreen", None),
    ("rytp.tui.screens.speakers", "SpeakerVideosScreen", None),
)


@pytest.mark.parametrize(("module", "attribute", "params"), CONSUMED)
def test_a_declared_dependency_exists_with_the_declared_shape(
    module: str, attribute: str, params: tuple[str, ...] | None
) -> None:
    """Every plan's `Interfaces / Consumes` block is a promise from another
    part. This is the one place all of them are checked at once."""
    obj = getattr(importlib.import_module(module), attribute)
    if params is None:
        return
    signature = inspect.signature(obj)
    declared = set(signature.parameters)
    assert set(params) <= declared, (
        f"{module}.{attribute} takes {sorted(declared)}; consumers expect "
        f"{sorted(set(params) - declared)} as well"
    )


def test_the_speaker_filter_keeps_the_two_fields_every_consumer_reads() -> None:
    """contracts §5 rule 3: the expansion to `video_speakers.id` happens inside
    the resolver, so consumers filter on `video_speaker_ids` directly."""
    import dataclasses

    from rytp.commands import SpeakerFilter

    fields = {f.name for f in dataclasses.fields(SpeakerFilter)}
    assert fields == {"video_speaker_ids", "description"}
