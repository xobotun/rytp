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

`settings.catalog` is the discoverability half of the group: the owner
spent an evening unable to run any model engine because nothing told them
`engine.interpreter.<name>` existed, and `settings list` — which shows
only what is *already* set — cannot teach anyone what they could set.
`_LITERALS` and `_TEMPLATES` below are that catalogue: every well-known key
that is not currently set, with the value it effectively reads as. Two
shapes are templates rather than literal keys (`pool.{pool}.size` and
`engine.interpreter.<name>`) and are expanded against real pools and
registered out-of-process engines rather than printed as a pattern — that
expansion is the point, not a cosmetic nicety.

`CommandResult` (contracts §5) holds exactly one table and one message, so
"a second table" cannot be a second `columns`/`rows` pair on `settings.list`
without varying from the contract. `settings.catalog` is a sibling command
instead — its own table, its own place in `--help` and the palette —
and `settings.list` only grows a one-line pointer to it when something is
missing. `tests/test_commands_settings.py` and the drift test in
`tests/test_settings_catalog.py` cover both.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from rytp import constants as C
from rytp.commands import Command, CommandResult, Param, register
from rytp.db import Database
from rytp.db import queries as q
from rytp.models import InvalidInputError, NotFoundError

__all__ = [
    "settings_catalog",
    "settings_get",
    "settings_list",
    "settings_set",
    "settings_unset",
]


# ---------------------------------------------------------------------------
# The catalogue.
#
# Every entry's `default` is the value the *read site* falls back to when
# the key is unset — never a guess. Where unset has a behaviour rather than
# a value (`default_aligner` empty, `speakers.embedder` empty) the default
# text says the behaviour instead of inventing a literal one.
# ---------------------------------------------------------------------------


class _Literal:
    """One well-known settings key that is not a template.

    `constant` is the name of the `rytp.constants` attribute holding this
    key — never the key text itself — so the drift test can check every
    `SETTING(S)_*` constant has an entry without hand-copying key strings.
    """

    __slots__ = ("_default", "about", "constant", "key")

    def __init__(
        self, constant: str, key: str, about: str, default: str | DefaultResolver
    ) -> None:
        self.constant = constant
        self.key = key
        self.about = about
        self._default = default

    def default(self, db: Database) -> str:
        return self._default(db) if callable(self._default) else self._default


class _Template:
    """A family of keys generated from one shape, e.g. ``pool.{pool}.size``.

    `expand` returns the real, currently-unset-able keys in the family as
    `(key, default)` pairs — never the `{pool}`/`<name>` pattern itself.
    `visible=False` keeps an entry in the catalogue (so the drift test is
    satisfied) without printing it, for a shape nothing reads yet.
    """

    __slots__ = ("about", "constant", "expand", "pattern", "visible")

    def __init__(
        self,
        constant: str,
        pattern: str,
        about: str,
        expand: Expander,
        *,
        visible: bool = True,
    ) -> None:
        self.constant = constant
        self.pattern = pattern
        self.about = about
        self.expand = expand
        self.visible = visible


DefaultResolver = Callable[[Database], str]
Expander = Callable[[Database], tuple[tuple[str, str], ...]]


def _static(value: str) -> str:
    return value


def _out_of_process_engine_names() -> tuple[str, ...]:
    """Registered engine names whose class runs under a separate interpreter.

    Only these ever consult `engine.interpreter.<name>` —
    `rytp.transcribe.registry.interpreter_for` is only called when
    `out_of_process` is true, so an in-process engine (`whisper`, `none`)
    would list a setting that does nothing.

    Import-light on purpose: `rytp.transcribe.engines`, `rytp.transcribe.align`
    and `rytp.diarize` each only register classes at import time (their
    adapters import heavy libraries lazily inside functions, never at module
    load), so this never touches torch or any other model dependency.
    """
    import rytp.diarize
    import rytp.transcribe.align
    import rytp.transcribe.engines  # noqa: F401
    from rytp.diarize.base import DIARIZERS
    from rytp.diarize.embed import EMBEDDERS
    from rytp.transcribe.registry import ALIGNERS, TRANSCRIBERS

    names: set[str] = set()
    for registry in (TRANSCRIBERS, ALIGNERS, DIARIZERS, EMBEDDERS):
        for name, cls in registry.items():
            if getattr(cls, "out_of_process", False):
                names.add(name)
    return tuple(sorted(names))


def _expand_pool_size(db: Database) -> tuple[tuple[str, str], ...]:
    return tuple(
        (C.SETTING_POOL_SIZE.format(pool=pool), str(C.POOL_DEFAULT_SIZE[pool]))
        for pool in C.POOLS
    )


def _expand_interpreter(db: Database) -> tuple[tuple[str, str], ...]:
    from rytp.transcribe.registry import interpreter_for

    return tuple(
        (f"{C.SETTINGS_INTERPRETER_PREFIX}{name}", interpreter_for(db, name))
        for name in _out_of_process_engine_names()
    )


_LITERALS: tuple[_Literal, ...] = (
    _Literal(
        "SETTING_QUEUE_PAUSED",
        C.SETTING_QUEUE_PAUSED,
        "Freezes every job pool from claiming work. Written by `jobs pause` / "
        "`jobs resume` — not for hand-editing. Unset means the queue runs.",
        _static("0 (not paused)"),
    ),
    _Literal(
        "SETTING_DOWNLOAD_DELAY_MIN",
        C.SETTING_DOWNLOAD_DELAY_MIN,
        "Minimum randomised pause before the next download, seconds.",
        _static(str(C.DOWNLOAD_DELAY_MIN_S)),
    ),
    _Literal(
        "SETTING_DOWNLOAD_DELAY_MAX",
        C.SETTING_DOWNLOAD_DELAY_MAX,
        "Maximum randomised pause before the next download, seconds.",
        _static(str(C.DOWNLOAD_DELAY_MAX_S)),
    ),
    _Literal(
        "SETTING_DOWNLOAD_FILE_DELAY",
        C.SETTING_DOWNLOAD_FILE_DELAY,
        "Extra pause between files within one video's download, seconds.",
        _static(str(C.DOWNLOAD_FILE_DELAY_S)),
    ),
    _Literal(
        "SETTING_DOWNLOAD_RATE_LIMIT",
        C.SETTING_DOWNLOAD_RATE_LIMIT,
        "Download speed cap, bytes/second. 0 or unset means unlimited.",
        _static(f"{C.DOWNLOAD_RATE_LIMIT_BPS} (0 or unset means unlimited)"),
    ),
    _Literal(
        "SETTING_DOWNLOAD_SLEEP_REQUESTS",
        C.SETTING_DOWNLOAD_SLEEP_REQUESTS,
        "Pause between individual HTTP requests within a download, seconds.",
        _static(str(C.DOWNLOAD_SLEEP_REQUESTS_S)),
    ),
    _Literal(
        "SETTING_DOWNLOAD_DAILY_CAP",
        C.SETTING_DOWNLOAD_DAILY_CAP,
        "Downloads started per day before `DailyCapReached` stops new ones. "
        "0 or negative means no cap.",
        _static(f"{C.DOWNLOAD_DAILY_CAP} (<= 0 means no cap)"),
    ),
    _Literal(
        "SETTING_DOWNLOAD_FORMAT_AUDIO",
        C.SETTING_DOWNLOAD_FORMAT_AUDIO,
        "yt-dlp format selector for the audio-only download.",
        _static(C.DOWNLOAD_FORMAT_AUDIO),
    ),
    _Literal(
        "SETTING_DOWNLOAD_FORMAT_VIDEO",
        C.SETTING_DOWNLOAD_FORMAT_VIDEO,
        "yt-dlp format selector for the video rendition download.",
        _static(C.DOWNLOAD_FORMAT_VIDEO),
    ),
    _Literal(
        "SETTING_CAPTION_LANGS",
        C.SETTING_CAPTION_LANGS,
        "Caption languages to request, most preferred first, comma-separated.",
        _static(",".join(C.CAPTION_LANGS)),
    ),
    _Literal(
        "SETTING_CAPTION_FORMAT",
        C.SETTING_CAPTION_FORMAT,
        "Caption track format requested from yt-dlp.",
        _static(C.CAPTION_FORMAT),
    ),
    _Literal(
        "SETTING_THROTTLE_STREAK",
        C.SETTING_THROTTLE_STREAK,
        "Consecutive throttle count; written by the download backoff after a "
        "429/403 — not for hand-editing. Unset means no active streak.",
        _static("0 (no active streak)"),
    ),
    _Literal(
        "SETTING_COOLDOWN_UNTIL",
        C.SETTING_COOLDOWN_UNTIL,
        "ISO timestamp the network pool is frozen until; written by the "
        "download backoff — not for hand-editing. Unset means not cooling down.",
        _static("(unset) not cooling down"),
    ),
    _Literal(
        "SETTING_WORKER_LEASE",
        C.SETTING_WORKER_LEASE,
        "The running worker's pid/heartbeat, as JSON; written by the worker "
        "process to claim exclusive ownership of the queue — not for "
        "hand-editing. Unset means no worker currently holds the lease.",
        _static("(unset) no lease held"),
    ),
    _Literal(
        "SETTING_DEFAULT_ALIGNER",
        C.SETTING_DEFAULT_ALIGNER,
        "Aligner `ingest --transcribe` stamps onto the `align` jobs it "
        "creates. Empty/unset means no `align` job is enqueued at all and "
        "words stay in the `timed` tier.",
        _static("(unset) no `align` job is enqueued; words stay `timed`"),
    ),
    _Literal(
        "SETTING_DEFAULT_TRANSCRIBER",
        C.SETTING_DEFAULT_TRANSCRIBER,
        "Transcriber `ingest --transcribe` stamps onto `transcribe` jobs, and "
        "what `transcribe run` falls back to without `--transcriber`. Unlike "
        "the aligner, empty is not meaningful here, so this always resolves "
        "to a name.",
        _static(C.DEFAULT_TRANSCRIBER_FALLBACK),
    ),
    _Literal(
        "SETTINGS_INTERPRETER_DEFAULT",
        C.SETTINGS_INTERPRETER_DEFAULT,
        "Shared fallback interpreter for any out-of-process engine with no "
        "engine.interpreter.<name> of its own.",
        lambda db: sys.executable,
    ),
    _Literal(
        "SETTINGS_DIARIZER",
        C.SETTINGS_DIARIZER,
        "Diarizer `speakers diarize` uses when none is named on the command "
        "line. Unset means the null diarizer runs — one label for the whole "
        "file, the correct answer for a single-speaker video.",
        _static(C.DEFAULT_DIARIZER),
    ),
    _Literal(
        "SETTINGS_EMBEDDER",
        C.SETTINGS_EMBEDDER,
        "Embedder run right after diarization when none is named. Unset "
        "means no embedding step runs at all — labels are assigned but no "
        "voice fingerprint is extracted.",
        _static("(unset) no embedding step runs"),
    ),
)

_TEMPLATES: tuple[_Template, ...] = (
    _Template(
        "SETTING_POOL_SIZE",
        C.SETTING_POOL_SIZE,
        "Slot count for one job pool.",
        _expand_pool_size,
    ),
    _Template(
        "SETTINGS_INTERPRETER_PREFIX",
        f"{C.SETTINGS_INTERPRETER_PREFIX}<name>",
        "Python interpreter an out-of-process engine (a registered "
        "transcriber, aligner, diarizer or embedder) runs under — the fix "
        "for a system interpreter with no wheels for torch or sentencepiece.",
        _expand_interpreter,
    ),
    _Template(
        "SETTINGS_BINARY_PREFIX",
        f"{C.SETTINGS_BINARY_PREFIX}<name>",
        "Declared for an external binary's path. No code currently reads "
        "engine.binary.<name> (checked: nothing in rytp/render, rytp/audio "
        "or rytp/transcribe consults it), so there is no real default to "
        "report and no roster of names to expand against. Kept out of the "
        "printed catalogue for that reason; entered here only so this dead "
        "constant cannot silently drift from the drift test.",
        lambda db: (),
        visible=False,
    ),
)


def _set_keys(db: Database) -> set[str]:
    return {row["key"] for row in db.conn.execute("SELECT key FROM settings").fetchall()}


def _unset_catalogue_rows(db: Database, set_keys: set[str]) -> tuple[tuple[str, str, str], ...]:
    """Known settings not in `set_keys`, as `(key, default, about)` rows."""
    rows: list[tuple[str, str, str]] = []
    for lit in _LITERALS:
        if lit.key not in set_keys:
            rows.append((lit.key, lit.default(db), lit.about))
    for tmpl in _TEMPLATES:
        if not tmpl.visible:
            continue
        for key, default in tmpl.expand(db):
            if key not in set_keys:
                rows.append((key, default, tmpl.about))
    rows.sort(key=lambda row: row[0])
    return tuple(rows)


def settings_list(db: Database) -> CommandResult:
    """Every key currently set, alphabetically.

    A key that is set never also appears in `settings.catalog` — see
    `_unset_catalogue_rows`. When something known is missing, the message
    points at the sibling command rather than trying to cram a second table
    into this one: `CommandResult` (contracts §5) holds exactly one.
    """
    set_rows = db.conn.execute("SELECT key, value FROM settings ORDER BY key").fetchall()
    set_keys = {row["key"] for row in set_rows}
    n_unset = len(_unset_catalogue_rows(db, set_keys))
    if not set_rows:
        message = "no settings are set" if not n_unset else (
            f"no settings are set; {n_unset} known setting(s) are not — run "
            "`settings catalog` to see them with their effective defaults"
        )
    elif n_unset:
        message = (
            f"{n_unset} known setting(s) are not set — run `settings catalog` "
            "to see them with their effective defaults"
        )
    else:
        message = None
    return CommandResult(
        columns=("key", "value"),
        rows=tuple((row["key"], row["value"]) for row in set_rows),
        message=message,
    )


def settings_catalog(db: Database) -> CommandResult:
    """Known settings that are not currently set, with their effective defaults.

    This is the discoverability half of the group: `settings.list` can only
    show what has already been set, which cannot teach anyone what else is
    there to set. `pool.{pool}.size` and `engine.interpreter.<name>` are
    templates, expanded here against real pools and registered
    out-of-process engines rather than printed as a pattern.
    """
    rows = _unset_catalogue_rows(db, _set_keys(db))
    return CommandResult(
        columns=("key", "default", "about"),
        rows=rows,
        message=None if rows else "every known setting is already set",
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
        name="settings.catalog",
        group="settings",
        summary="List known settings that are not set, with effective defaults.",
        params=(),
        handler=settings_catalog,
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
