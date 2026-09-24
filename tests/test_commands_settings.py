"""The settings command group: full CRUD over the `settings` table.

`rytp/db/queries.py` has always had `get_setting` / `set_setting`, but
before this module nothing registered a command over them, so the only
way to point an out-of-process engine at a working interpreter — the
`engine.interpreter.<name>` setting `rytp/transcribe/registry.py`'s
`interpreter_for` reads — was to open a Python shell. That is the case
this module exists to close; see `test_settings_set_is_what_interpreter_for_reads`.
"""

from __future__ import annotations

import pytest

import rytp.commands.settings  # noqa: F401 - importing registers the commands
from rytp.commands import COMMANDS, resolve
from rytp.db import Database
from rytp.db import queries as q
from rytp.models import NotFoundError
from rytp.transcribe.registry import interpreter_for

GROUP = ("settings.list", "settings.get", "settings.set", "settings.unset")


# -- registration ------------------------------------------------------


def test_group_is_registered() -> None:
    for name in GROUP:
        assert name in COMMANDS


def test_resolve_finds_every_command() -> None:
    for name in GROUP:
        assert resolve(name).name == name


# -- list ----------------------------------------------------------------


def test_list_on_a_freshly_migrated_database(db: Database) -> None:
    """`default_aligner` is seeded empty by the migration (contracts §3); a
    fresh database is not literally empty, but nothing else has been set.
    """
    result = resolve("settings.list").handler(db)
    assert result.rows == (("default_aligner", ""),)


def test_list_shows_every_key_alphabetically(db: Database) -> None:
    q.set_setting(db, "default_transcriber", "gigaam")
    q.set_setting(db, "default_aligner", "mfa")
    result = resolve("settings.list").handler(db)
    assert result.columns == ("key", "value")
    assert result.rows == (
        ("default_aligner", "mfa"),
        ("default_transcriber", "gigaam"),
    )


# -- get -------------------------------------------------------------------


def test_get_reads_back_a_set_value(db: Database) -> None:
    q.set_setting(db, "default_transcriber", "gigaam")
    result = resolve("settings.get").handler(db, key="default_transcriber")
    assert result.rows == (("default_transcriber", "gigaam"),)


def test_get_missing_key_raises_not_found(db: Database) -> None:
    with pytest.raises(NotFoundError):
        resolve("settings.get").handler(db, key="never-set")


# -- set -------------------------------------------------------------------


def test_set_creates_a_new_key(db: Database) -> None:
    resolve("settings.set").handler(db, key="default_aligner", value="mfa")
    assert q.get_setting(db, "default_aligner") == "mfa"


def test_set_overwrites_an_existing_key(db: Database) -> None:
    q.set_setting(db, "default_aligner", "mfa")
    resolve("settings.set").handler(db, key="default_aligner", value="")
    assert q.get_setting(db, "default_aligner") == ""


def test_set_rejects_an_empty_key(db: Database) -> None:
    from rytp.models import InvalidInputError

    with pytest.raises(InvalidInputError):
        resolve("settings.set").handler(db, key="  ", value="anything")


def test_set_accepts_a_dotted_engine_key(db: Database) -> None:
    resolve("settings.set").handler(
        db, key="engine.interpreter.wav2vec2", value="/opt/py312/bin/python"
    )
    assert q.get_setting(db, "engine.interpreter.wav2vec2") == "/opt/py312/bin/python"


# -- unset -----------------------------------------------------------------


def test_unset_deletes_a_key(db: Database) -> None:
    q.set_setting(db, "default_aligner", "mfa")
    resolve("settings.unset").handler(db, key="default_aligner")
    assert q.get_setting(db, "default_aligner") is None


def test_unset_missing_key_raises_not_found(db: Database) -> None:
    with pytest.raises(NotFoundError):
        resolve("settings.unset").handler(db, key="never-set")


def test_unset_then_get_raises(db: Database) -> None:
    q.set_setting(db, "default_aligner", "mfa")
    resolve("settings.unset").handler(db, key="default_aligner")
    with pytest.raises(NotFoundError):
        resolve("settings.get").handler(db, key="default_aligner")


# -- the motivating workflow (task description) -----------------------


def test_settings_set_is_what_interpreter_for_reads(db: Database) -> None:
    """The whole point: `settings set engine.interpreter.wav2vec2 <path>` is
    the only supported way to make `interpreter_for` return that path, since
    torch/sentencepiece have no wheels under the owner's system interpreter
    and every one of these engines must run under a separate Python.
    """
    path = "/opt/py312/bin/python"
    resolve("settings.set").handler(db, key="engine.interpreter.wav2vec2", value=path)
    assert interpreter_for(db, "wav2vec2") == path
