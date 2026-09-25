"""`settings.catalog`: the discoverability half of the settings group.

`settings.list` can only show what has already been set — it cannot teach
anyone that `engine.interpreter.<name>` exists, which is exactly the gap
that once cost the owner an evening. This module covers the catalogue
itself: that every well-known `SETTING(S)_*` constant in `rytp.constants`
has an entry (so the two cannot drift apart, `test_every_setting_constant_
is_catalogued` below), that a key already set never also appears here, and
that the two dynamic families — `pool.{pool}.size` and
`engine.interpreter.<name>` — are expanded against real pools and real
registered engines rather than printed as a template.
"""

from __future__ import annotations

import re
import sys

import rytp.commands.settings  # noqa: F401 - importing registers the commands
from rytp import constants as C
from rytp.commands import resolve
from rytp.commands.settings import _LITERALS, _TEMPLATES
from rytp.db import Database
from rytp.db import queries as q

_SETTING_CONSTANT_RE = re.compile(r"^SETTINGS?_[A-Z0-9_]+$")


def _all_setting_constants() -> set[str]:
    """Every `rytp.constants` attribute that names a settings key.

    Filtered by name (`SETTING_` / `SETTINGS_` prefix) and by being a
    string value — `DEFAULT_TRANSCRIBER_FALLBACK`, `DEFAULT_DIARIZER` and
    `DEFAULT_EMBEDDER` hold *default values*, not keys, and none of them
    match the prefix, so this never confuses the two.
    """
    return {
        name
        for name, value in vars(C).items()
        if _SETTING_CONSTANT_RE.match(name) and isinstance(value, str)
    }


# -- the drift test ----------------------------------------------------


def test_every_setting_constant_is_catalogued() -> None:
    """Fails if a `SETTING_*`/`SETTINGS_*` constant has no catalogue entry,
    or if the catalogue names a constant that no longer exists — the two
    must never drift apart.
    """
    catalogued = {entry.constant for entry in _LITERALS} | {
        entry.constant for entry in _TEMPLATES
    }
    actual = _all_setting_constants()
    missing = actual - catalogued
    stale = catalogued - actual
    assert not missing, f"constants with no catalogue entry: {sorted(missing)}"
    assert not stale, f"catalogue entries naming a nonexistent constant: {sorted(stale)}"


def test_catalogue_has_no_duplicate_keys() -> None:
    literal_keys = [entry.key for entry in _LITERALS]
    assert len(literal_keys) == len(set(literal_keys))


# -- registration --------------------------------------------------------


def test_settings_catalog_is_registered() -> None:
    assert resolve("settings.catalog").name == "settings.catalog"


# -- shape -----------------------------------------------------------------


def test_catalog_on_a_fresh_database_has_the_expected_columns(db: Database) -> None:
    result = resolve("settings.catalog").handler(db)
    assert result.columns == ("key", "default", "about")
    assert result.rows


def test_catalog_rows_are_sorted_by_key(db: Database) -> None:
    result = resolve("settings.catalog").handler(db)
    keys = [row[0] for row in result.rows]
    assert keys == sorted(keys)


def test_catalog_never_lists_a_key_that_is_set(db: Database) -> None:
    """A key that is set appears in `settings.list` only, never in both."""
    q.set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "gigaam")
    q.set_setting(db, C.SETTING_QUEUE_PAUSED, "1")
    result = resolve("settings.catalog").handler(db)
    keys = {row[0] for row in result.rows}
    assert C.SETTING_DEFAULT_TRANSCRIBER not in keys
    assert C.SETTING_QUEUE_PAUSED not in keys


def test_catalog_excludes_default_aligner_seeded_by_migration(db: Database) -> None:
    """`default_aligner` is seeded (empty) by the migration, so on a real
    database it is always a *set* key, never a catalogue entry.
    """
    result = resolve("settings.catalog").handler(db)
    keys = {row[0] for row in result.rows}
    assert C.SETTING_DEFAULT_ALIGNER not in keys


def test_catalog_empty_message_when_everything_is_set(db: Database) -> None:
    for entry in _LITERALS:
        q.set_setting(db, entry.key, "x")
    for tmpl in _TEMPLATES:
        for key, _default in tmpl.expand(db):
            q.set_setting(db, key, "x")
    result = resolve("settings.catalog").handler(db)
    assert result.rows == ()
    assert result.message == "every known setting is already set"


# -- specific defaults, verified against the read site --------------------


def test_default_transcriber_falls_back_to_gigaam(db: Database) -> None:
    result = resolve("settings.catalog").handler(db)
    row = next(r for r in result.rows if r[0] == C.SETTING_DEFAULT_TRANSCRIBER)
    assert row[1] == C.DEFAULT_TRANSCRIBER_FALLBACK == "gigaam"


def test_default_aligner_states_the_behaviour_not_a_value(db: Database) -> None:
    # Force the seeded row out of the way so the entry is visible.
    q.set_setting(db, C.SETTING_DEFAULT_ALIGNER, "mfa")
    resolve("settings.unset").handler(db, key=C.SETTING_DEFAULT_ALIGNER)
    result = resolve("settings.catalog").handler(db)
    row = next(r for r in result.rows if r[0] == C.SETTING_DEFAULT_ALIGNER)
    assert "align" in row[1] and "timed" in row[1]


def test_download_rate_limit_default_notes_unlimited(db: Database) -> None:
    result = resolve("settings.catalog").handler(db)
    row = next(r for r in result.rows if r[0] == C.SETTING_DOWNLOAD_RATE_LIMIT)
    assert str(C.DOWNLOAD_RATE_LIMIT_BPS) in row[1]
    assert "unlimited" in row[1]


def test_download_daily_cap_default_notes_no_cap(db: Database) -> None:
    result = resolve("settings.catalog").handler(db)
    row = next(r for r in result.rows if r[0] == C.SETTING_DOWNLOAD_DAILY_CAP)
    assert str(C.DOWNLOAD_DAILY_CAP) in row[1]
    assert "no cap" in row[1]


def test_caption_langs_default_is_the_comma_form(db: Database) -> None:
    """`_tuple` in `rytp/acquire/policy.py` parses a comma-separated string,
    so the catalogue must show that form, not the Python tuple repr.
    """
    result = resolve("settings.catalog").handler(db)
    row = next(r for r in result.rows if r[0] == C.SETTING_CAPTION_LANGS)
    assert row[1] == ",".join(C.CAPTION_LANGS)


def test_worker_lease_default_says_unheld(db: Database) -> None:
    result = resolve("settings.catalog").handler(db)
    row = next(r for r in result.rows if r[0] == C.SETTING_WORKER_LEASE)
    assert "no lease" in row[1] or "unset" in row[1]


def test_speakers_diarizer_default_is_none(db: Database) -> None:
    result = resolve("settings.catalog").handler(db)
    row = next(r for r in result.rows if r[0] == C.SETTINGS_DIARIZER)
    assert row[1] == C.DEFAULT_DIARIZER == "none"


def test_speakers_embedder_default_states_no_embedding_step(db: Database) -> None:
    result = resolve("settings.catalog").handler(db)
    row = next(r for r in result.rows if r[0] == C.SETTINGS_EMBEDDER)
    assert "no embedding" in row[1]


def test_interpreter_default_is_sys_executable_when_wholly_unset(db: Database) -> None:
    result = resolve("settings.catalog").handler(db)
    row = next(r for r in result.rows if r[0] == C.SETTINGS_INTERPRETER_DEFAULT)
    assert row[1] == sys.executable


def test_engine_binary_prefix_is_catalogued_but_not_printed(db: Database) -> None:
    """Nothing reads `engine.binary.<name>` anywhere in the codebase, so a
    printed default would be a guess. The constant is still catalogued (the
    drift test requires it) but its family is marked not-visible and
    contributes no rows.
    """
    tmpl = next(t for t in _TEMPLATES if t.constant == "SETTINGS_BINARY_PREFIX")
    assert tmpl.visible is False
    result = resolve("settings.catalog").handler(db)
    assert not any(row[0].startswith(C.SETTINGS_BINARY_PREFIX) for row in result.rows)


# -- the two dynamic families ------------------------------------------


def test_pool_size_expands_to_one_row_per_real_pool(db: Database) -> None:
    result = resolve("settings.catalog").handler(db)
    expected = {
        C.SETTING_POOL_SIZE.format(pool=pool): str(C.POOL_DEFAULT_SIZE[pool])
        for pool in C.POOLS
    }
    got = {
        row[0]: row[1]
        for row in result.rows
        if row[0].startswith("pool.") and row[0].endswith(".size")
    }
    assert got == expected


def test_pool_size_row_disappears_once_that_pool_is_set(db: Database) -> None:
    key = C.SETTING_POOL_SIZE.format(pool="gpu")
    q.set_setting(db, key, "4")
    result = resolve("settings.catalog").handler(db)
    keys = {row[0] for row in result.rows}
    assert key not in keys
    # The other pools are still unset and still listed.
    assert C.SETTING_POOL_SIZE.format(pool="network") in keys
    assert C.SETTING_POOL_SIZE.format(pool="cpu") in keys


def test_interpreter_expands_to_one_row_per_out_of_process_engine(db: Database) -> None:
    """Only out-of-process engines ever consult `engine.interpreter.<name>`
    (`rytp/transcribe/registry.py:interpreter_for` is never called for an
    in-process one) — `whisper` and `none` must not be listed.
    """
    result = resolve("settings.catalog").handler(db)
    keys = {
        row[0][len(C.SETTINGS_INTERPRETER_PREFIX) :]
        for row in result.rows
        if row[0].startswith(C.SETTINGS_INTERPRETER_PREFIX)
        and row[0] != C.SETTINGS_INTERPRETER_DEFAULT
    }
    assert {"gigaam", "mfa", "wav2vec2", "pyannote", "redimnet"} <= keys
    assert "whisper" not in keys
    assert "none" not in keys


def test_interpreter_row_default_cascades_through_the_shared_default(
    db: Database,
) -> None:
    """When `engine.interpreter.default` is set, every still-unset per-engine
    row must show *that* path, not `sys.executable` — it is the effective
    default `interpreter_for` would actually return.
    """
    shared = "/opt/py312/bin/python"
    q.set_setting(db, C.SETTINGS_INTERPRETER_DEFAULT, shared)
    result = resolve("settings.catalog").handler(db)
    row = next(
        r for r in result.rows if r[0] == f"{C.SETTINGS_INTERPRETER_PREFIX}gigaam"
    )
    assert row[1] == shared


def test_interpreter_row_disappears_once_that_engine_is_set(db: Database) -> None:
    key = f"{C.SETTINGS_INTERPRETER_PREFIX}gigaam"
    q.set_setting(db, key, "/opt/py312/bin/python")
    result = resolve("settings.catalog").handler(db)
    keys = {row[0] for row in result.rows}
    assert key not in keys
    assert f"{C.SETTINGS_INTERPRETER_PREFIX}mfa" in keys


# -- import weight, per the task's stated constraint -----------------------


def test_catalog_handler_never_imports_torch(db: Database) -> None:
    """Resolving the engine list must not import a heavy library — engines
    are registered classes and their dependencies are imported lazily inside
    functions precisely so the registry stays cheap. `torch` is not
    installed in the dev venv, so this would raise ModuleNotFoundError if
    the catalogue ever imported an adapter's heavy dependency at collection
    time instead of leaving it behind the out-of-process seam.
    """
    assert "torch" not in sys.modules
    resolve("settings.catalog").handler(db)
    assert "torch" not in sys.modules
