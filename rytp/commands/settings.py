"""The settings command group: full CRUD over the `settings` table.

`rytp/db/queries.py` has always had `get_setting` / `set_setting`
(contracts §3), but until this module nothing registered a command over
them, so a value written there was reachable only from Python. That
blocked real work: every out-of-process engine (`gigaam`, `mfa`,
`wav2vec2`, `pyannote`, `redimnet`) is invoked under a separate Python
interpreter chosen by `engine.interpreter.<name>`
(`rytp/transcribe/registry.py:interpreter_for`), and on the owner's
Windows box the system interpreter has no wheels for torch or
sentencepiece — pointing an engine at a working Python 3.12 virtualenv
is the *only* way to run it. `rytp/transcribe/health.py` and
`rytp/diarize/health.py` already print
``rytp settings set engine.interpreter.<name> <path>`` as the fix for a
missing interpreter; this module is what makes that line true.

Keys are deliberately open-ended: `default_transcriber`, `default_aligner`
and `engine.interpreter.<name>` are the well-known ones, but nothing here
validates against a fixed set of them, and no such list is wanted — a new
part can introduce a setting without touching this file.
"""

from __future__ import annotations

from rytp.commands import Command, CommandResult, Param, register
from rytp.db import Database
from rytp.db import queries as q
from rytp.models import InvalidInputError, NotFoundError

__all__ = ["settings_get", "settings_list", "settings_set", "settings_unset"]


def settings_list(db: Database) -> CommandResult:
    """Every key currently set, alphabetically."""
    rows = db.conn.execute("SELECT key, value FROM settings ORDER BY key").fetchall()
    return CommandResult(
        columns=("key", "value"),
        rows=tuple((row["key"], row["value"]) for row in rows),
        message=None if rows else "no settings are set",
    )


def settings_get(db: Database, *, key: str) -> CommandResult:
    """Read one value.

    Raises `NotFoundError` when the key was never set — the same "not
    found" answer `resolve_video_id` gives for an unknown video, rather
    than a silent empty result that could as easily mean "set to empty".
    """
    value = q.get_setting(db, key)
    if value is None:
        raise NotFoundError(f"no setting {key!r} is set")
    return CommandResult(columns=("key", "value"), rows=((key, value),))


def settings_set(db: Database, *, key: str, value: str) -> CommandResult:
    """Create or update one value."""
    key = key.strip()
    if not key:
        raise InvalidInputError("settings set needs a key")
    q.set_setting(db, key, value)
    return CommandResult(
        columns=("key", "value"), rows=((key, value),), message=f"set {key}"
    )


def settings_unset(db: Database, *, key: str) -> CommandResult:
    """Delete one value.

    Raises `NotFoundError` when the key was never set, matching `get`:
    unsetting nothing is not a silent no-op.
    """
    deleted = db.conn.execute("DELETE FROM settings WHERE key = ?", (key,)).rowcount
    if not deleted:
        raise NotFoundError(f"no setting {key!r} is set")
    return CommandResult(message=f"unset {key}")


register(
    Command(
        name="settings.list",
        group="settings",
        summary="List every setting currently set.",
        params=(),
        handler=settings_list,
    )
)

register(
    Command(
        name="settings.get",
        group="settings",
        summary="Read one setting.",
        params=(
            Param(
                "key",
                str,
                "Setting key, e.g. default_transcriber, default_aligner, or "
                "engine.interpreter.<name>.",
                positional=True,
            ),
        ),
        handler=settings_get,
    )
)

register(
    Command(
        name="settings.set",
        group="settings",
        summary="Create or update one setting.",
        params=(
            Param(
                "key",
                str,
                "Setting key, e.g. default_transcriber, default_aligner, or "
                "engine.interpreter.<name>.",
                positional=True,
            ),
            Param("value", str, "Value to store.", positional=True),
        ),
        handler=settings_set,
    )
)

register(
    Command(
        name="settings.unset",
        group="settings",
        summary="Delete one setting.",
        params=(
            Param("key", str, "Setting key to remove.", positional=True),
        ),
        handler=settings_unset,
    )
)
