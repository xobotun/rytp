"""Every rytp operation, defined exactly once (contracts §5).

Design §10: "Each operation is defined once — name, arguments, result
shape — and both the CLI and the TUI are generated from that definition.
Alignment between the two then holds by construction rather than by
discipline."

A handler takes an open :class:`rytp.db.Database` as its first positional
argument and the command's parameters as keyword arguments, and returns a
:class:`CommandResult`. Handlers never print and never call ``sys.exit``:
they raise :class:`rytp.models.RytpError` and the surface decides how a
failure looks.

Two flags name a speaker and they mean different things, so the
resolution lives here once rather than in every command that filters
(contracts §5). See :func:`resolve_speaker_filter`.

A command marked ``cli_only`` is still a registry command — the CLI
generates it like any other — but the TUI palette leaves it out, because
it launches a surface and cannot sensibly be invoked from inside the
surface it launches. ``tui`` is the one part 1 registers; part 7 marks
``speakers.map`` the same way.

Registration happens as a side effect of importing a command module, so
the imports at the bottom of this file are load-bearing — a command
module that nobody imports is a command that does not exist. Later plan
parts add their own line to that block.
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rytp import constants as C
from rytp.config import paths
from rytp.db import LATEST_VERSION, Database
from rytp.models import InvalidInputError, NotFoundError, RytpError

__all__ = [
    "COMMANDS",
    "GROUP_SUMMARIES",
    "HEALTH_CHECKS",
    "PARAM_ALIASES",
    "PARAM_TYPES",
    "REQUIRED",
    "SPEAKER_PARAMS",
    "Command",
    "CommandResult",
    "HealthCheck",
    "HealthCheckError",
    "HealthResult",
    "Param",
    "SpeakerFilter",
    "cli_path",
    "doctor",
    "health_status",
    "leaf_name",
    "register",
    "register_check",
    "resolve",
    "resolve_speaker_filter",
    "resolve_video_id",
    "run_health_checks",
]

#: Sentinel for "the user must supply this". Distinct from ``None``,
#: which is a perfectly good default meaning "left empty".
REQUIRED: Final = object()

#: Parameter types both surfaces know how to prompt for and parse.
PARAM_TYPES: Final = (str, int, float, bool, Path)

#: Extra long spellings a parameter also answers to, keyed by parameter
#: name. Contracts §5 names ``--global-speaker`` as an alias of
#: ``--speaker``; ``Param`` is fixed by that contract and has no alias
#: field, so the spellings live here and both surfaces read them.
PARAM_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "speaker": ("--global-speaker",),
}

#: How many roster entries a failed ``--speaker`` lookup suggests.
_SPEAKER_SUGGESTIONS: Final = 3

_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*(\.[a-z][a-z0-9-]*)?$")


@dataclass(frozen=True)
class Param:
    """One argument, described once for both surfaces."""

    name: str
    type: type
    help: str
    default: Any = REQUIRED
    choices: tuple[str, ...] | None = None
    short: str | None = None
    positional: bool = False


@dataclass(frozen=True)
class CommandResult:
    """What a handler returns: an optional table and an optional line.

    Strings only. Formatting a value for display is the handler's job,
    so the CLI and the TUI cannot render the same result differently.
    """

    columns: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    message: str | None = None


@dataclass(frozen=True)
class Command:
    """One operation."""

    name: str  # dotted: "videos.add", "search.words"
    group: str  # "videos"; "" for top level
    summary: str
    params: tuple[Param, ...]
    handler: Callable[..., CommandResult]
    long_running: bool = False  # TUI must not run these inline
    cli_only: bool = False  # surface launchers; absent from the TUI palette


COMMANDS: dict[str, Command] = {}

#: One short phrase per non-empty `Command.group`, read by both surfaces —
#: the CLI's group help (`rytp.cli`) and the TUI palette's group heading —
#: contracts §5 "GROUP_SUMMARIES". It says what the group is *about*
#: ("videos" -> "the local video catalog"), never the list of commands in
#: it: the command list is generated from `COMMANDS`, not written out a
#: second time here. `_validate` refuses to register a command whose group
#: has no entry, so a new group cannot ship undescribed (BUGS.md entry 6).
GROUP_SUMMARIES: dict[str, str] = {
    "videos": "the local video catalog",
    "channel": "the channels videos are catalogued from",
    "index": "the search index built from transcripts",
    "search": "finding and exporting what was said",
    "transcript": "reading one video's transcript",
    "assemble": "planning cuts across the corpus",
    "transcribe": "turning audio into words and word boundaries",
    "render": "cutting and rendering a planned video",
    "speakers": "the speaker roster and per-video mapping",
    "settings": "defaults and per-engine configuration",
    "jobs": "the background job queue",
    "queue": "pausing and resuming the worker",
    "cache": "cached, regenerable derived files",
    "assets": "downloaded and rendered files on disk",
}


def leaf_name(cmd: Command) -> str:
    """The last segment of the dotted name — what the CLI calls the subcommand."""
    return cmd.name.rsplit(".", 1)[-1]


def cli_path(cmd: Command) -> tuple[str, ...]:
    """The argv path to this command: ``("videos", "add")`` or ``("ingest",)``."""
    return tuple(cmd.name.split("."))


def _validate(cmd: Command) -> None:
    """Reject a registration that either surface could not render.

    These are programming errors, not user errors, so they raise
    ``ValueError`` rather than a ``RytpError``: they fire at import time
    and must not be swallowed by the CLI's one-line error handler.
    """
    if not _NAME_RE.match(cmd.name):
        raise ValueError(
            f"bad command name {cmd.name!r}: expected 'verb' or 'group.verb' in "
            "lowercase, digits and hyphens"
        )
    expected_group = cmd.name.split(".")[0] if "." in cmd.name else ""
    if cmd.group != expected_group:
        raise ValueError(
            f"command {cmd.name!r} declares group {cmd.group!r}; the dotted name "
            f"implies {expected_group!r}"
        )
    if not cmd.summary.strip():
        raise ValueError(f"command {cmd.name!r} needs a summary: both surfaces show it")
    if cmd.group and cmd.group not in GROUP_SUMMARIES:
        raise ValueError(
            f"command {cmd.name!r} is in group {cmd.group!r}, which has no entry in "
            "GROUP_SUMMARIES; add one so the group is described, not just named "
            "(contracts §5, BUGS.md entry 6)"
        )

    seen: set[str] = set()
    positional_finished = False
    for param in cmd.params:
        if param.name in seen:
            raise ValueError(f"command {cmd.name!r} has a duplicate parameter {param.name!r}")
        seen.add(param.name)
        if param.type not in PARAM_TYPES:
            names = ", ".join(t.__name__ for t in PARAM_TYPES)
            raise ValueError(
                f"parameter {cmd.name}.{param.name} has type {param.type!r}; "
                f"the surfaces can only render: {names}"
            )
        if param.choices is not None and param.type is not str:
            raise ValueError(
                f"parameter {cmd.name}.{param.name} has choices but is not a str"
            )
        if param.short is not None:
            if param.positional:
                raise ValueError(
                    f"parameter {cmd.name}.{param.name} is positional and cannot have "
                    "a short flag"
                )
            if not re.fullmatch(r"-[A-Za-z]", param.short):
                raise ValueError(
                    f"parameter {cmd.name}.{param.name} has short flag {param.short!r}; "
                    "expected a single dash and one letter, e.g. '-n'"
                )
        if param.positional:
            if positional_finished:
                raise ValueError(
                    f"command {cmd.name!r} lists positional parameter {param.name!r} "
                    "after an option; positionals must come first"
                )
        else:
            positional_finished = True


def register(cmd: Command) -> Command:
    """Validate and add a command. Returns it, so it can be assigned."""
    _validate(cmd)
    if cmd.name in COMMANDS:
        raise ValueError(f"command {cmd.name!r} is already registered")
    COMMANDS[cmd.name] = cmd
    return cmd


def resolve(name: str) -> Command:
    """Look a command up by dotted name."""
    try:
        return COMMANDS[name]
    except KeyError:
        available = ", ".join(sorted(COMMANDS)) or "(none registered)"
        raise NotFoundError(f"unknown command {name!r}; available: {available}") from None


# ---------------------------------------------------------------------
# Speaker filters — two flags, two identifier spaces (contracts §5)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SpeakerFilter:
    """A speaker selection, already expanded to the ids consumers filter on.

    Contracts §5 pins this shape. ``video_speaker_ids`` is **empty when
    the filter resolved but matches nothing** — a real person nobody has
    mapped to a video yet. An empty set must return no rows, never every
    row: that is the classic silent-wrong-answer bug in a SQL ``IN``
    clause, and it is what "show everything" (a ``None`` filter) is for.
    """

    video_speaker_ids: frozenset[int]  # already expanded, ready to filter on
    description: str  # for display, e.g. 'speaker "Иванов Иван Иванович"'


#: The three flags every speaker-filtering command shares. Splice these
#: into a command's ``params`` rather than restating them, so ``--speaker``
#: means the same thing everywhere. A handler converts its `video` string
#: with `resolve_video_id` before calling the resolver, whose signature is
#: fixed by contracts §5 and takes an int.
SPEAKER_PARAMS: Final[tuple[Param, ...]] = (
    Param(
        "speaker",
        str,
        "A named person from the roster: their label or one of their aliases.",
        default=None,
    ),
    Param(
        "video_local_speaker",
        str,
        "A raw per-video diarizer label such as SPEAKER_00. Requires --video.",
        default=None,
    ),
    Param(
        "video",
        str,
        "Video row id or external id. Required with --video-local-speaker.",
        default=None,
    ),
)


def resolve_speaker_filter(
    db: Database,
    *,
    speaker: str | None = None,
    video_local_speaker: str | None = None,
    video_id: int | None = None,
) -> SpeakerFilter | None:
    """Turn the speaker flags into the ``video_speakers.id`` set to filter on.

    Contracts §5 fixes this signature. There are two identifier spaces
    and conflating them is a bug:

    * ``--speaker`` (alias ``--global-speaker``, handled by
      :data:`PARAM_ALIASES`) names a person in the global roster, matched
      against ``speakers.label`` and then against ``speakers.aliases_json``.
      It never accepts a raw diarizer label.
    * ``--video-local-speaker`` names one diarized voice inside one video
      and **requires** a video. A bare ``SPEAKER_00`` would otherwise
      match the first-detected voice of every diarized video, which is
      not a person and not a useful answer.

    The expansion to ``video_speakers.id`` happens here, not in the
    caller: every consumer filters ``words.video_speaker_id`` or
    ``utterances.video_speaker_id``, and a resolver that stopped at
    ``speakers.id`` would push the same join into all of them.

    Returns:
        ``None`` when no speaker was named at all — the "show everything"
        case, which is distinct from a filter that matched nothing.

    Raises:
        InvalidInputError: if both identifier spaces are used at once, or
            a local label is given without a video.
        NotFoundError: if a name does not resolve. "No such person" and
            "that person said nothing" are different answers, so this
            never degrades to an empty result.
    """
    if speaker is not None and video_local_speaker is not None:
        raise InvalidInputError(
            "--speaker names a person and --video-local-speaker names one "
            "video's diarizer label; filter by one or the other"
        )

    if video_local_speaker is not None:
        if video_id is None:
            raise InvalidInputError(
                "--video-local-speaker requires --video: a label like "
                f"{video_local_speaker!r} is only meaningful inside one video, "
                "and alone it would match the first voice of every diarized one"
            )
        row = db.conn.execute(
            "SELECT id FROM video_speakers WHERE video_id = ? AND local_label = ?",
            (video_id, video_local_speaker),
        ).fetchone()
        if row is None:
            known = [
                str(other["local_label"])
                for other in db.conn.execute(
                    "SELECT local_label FROM video_speakers WHERE video_id = ?"
                    " ORDER BY local_label",
                    (video_id,),
                )
            ]
            listed = ", ".join(known) if known else "(this video is not diarized)"
            raise NotFoundError(
                f"video {video_id} has no speaker label {video_local_speaker!r}; "
                f"it has: {listed}"
            )
        return SpeakerFilter(
            video_speaker_ids=frozenset({int(row["id"])}),
            description=f"local label {video_local_speaker} in video {video_id}",
        )

    if speaker is None:
        return None

    speaker_id, label = _resolve_roster_speaker(db, speaker)
    sql = "SELECT id FROM video_speakers WHERE speaker_id = ?"
    params: list[Any] = [speaker_id]
    description = f'speaker "{label}"'
    if video_id is not None:
        sql += " AND video_id = ?"
        params.append(video_id)
        description += f" in video {video_id}"
    # May legitimately come back empty: a roster entry nobody has mapped
    # to a video yet. Rule 2 — the caller must then return nothing.
    return SpeakerFilter(
        video_speaker_ids=frozenset(
            int(row["id"]) for row in db.conn.execute(sql, params)
        ),
        description=description,
    )


def _resolve_roster_speaker(db: Database, wanted: str) -> tuple[int, str]:
    """Match a roster entry by label, then by alias. Returns ``(id, label)``.

    The label comes back so a filter found through an alias still
    describes itself by the person's canonical name.
    """
    row = db.conn.execute(
        "SELECT id, label FROM speakers WHERE label = ?", (wanted,)
    ).fetchone()
    if row is not None:
        return int(row["id"]), str(row["label"])

    labels: list[str] = []
    for candidate in db.conn.execute("SELECT id, label, aliases_json FROM speakers"):
        labels.append(str(candidate["label"]))
        try:
            aliases = json.loads(candidate["aliases_json"])
        except json.JSONDecodeError:
            aliases = []
        if isinstance(aliases, list) and wanted in [str(alias) for alias in aliases]:
            return int(candidate["id"]), str(candidate["label"])

    near = difflib.get_close_matches(wanted, labels, n=_SPEAKER_SUGGESTIONS)
    if near:
        suggestion = f"did you mean: {', '.join(near)}?"
    elif labels:
        suggestion = f"the roster has: {', '.join(sorted(labels))}"
    else:
        suggestion = "the speaker roster is empty"
    raise NotFoundError(f"no speaker matches {wanted!r}; {suggestion}")


def _looks_like_a_url(ref: str) -> bool:
    """Same test `videos add` uses to tell a URL from a bare id (catalog.py)."""
    return "://" in ref


def resolve_video_id(db: Database, ref: str) -> int:
    """Find a video by row id, by external id, or by its catalogued URL.

    Public because `videos.remove` resolves the same way the speaker
    filter does, and two spellings of "which video?" would be one too
    many.

    Raises:
        NotFoundError: naming `videos add` when `ref` looks like a URL —
            the owner's first command was `fetch-video <url>` on a URL
            that had never been catalogued, and "no video matches" alone
            does not say a video must be registered first, or how
            (BUGS.md entries 1 and 2).
    """
    row = (
        db.conn.execute("SELECT id FROM videos WHERE id = ?", (int(ref),)).fetchone()
        if ref.isdigit()
        else db.conn.execute(
            "SELECT id FROM videos WHERE external_id = ? OR url = ?", (ref, ref)
        ).fetchone()
    )
    if row is None:
        if _looks_like_a_url(ref):
            raise NotFoundError(
                f"no video matches {ref!r}; register it first with: "
                f"rytp videos add {ref}"
            )
        raise NotFoundError(f"no video matches {ref!r}")
    return int(row["id"])


# ---------------------------------------------------------------------
# Health checks (contracts §5 "Health checks")
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class HealthResult:
    """What one check found.

    ``ok`` always tells the truth about what was found. A missing
    optional engine is ``ok=False``, because it is in fact missing — the
    alternative, reporting it as ``ok=True`` to protect the exit code,
    makes the output lie about the thing `doctor` exists to tell you.
    What makes an absence non-fatal is ``required=False`` on the check
    (contracts §5).
    """

    ok: bool
    detail: str
    remedy: str | None = None


@dataclass(frozen=True)
class HealthCheck:
    """One thing `doctor` looks at.

    ``required=True`` for anything the base install genuinely needs —
    ffmpeg, ffprobe, FTS5, the data tree, the schema version.
    ``required=False`` for yt-dlp, ffplay, every transcriber, aligner and
    diarizer, the GPU and ``HF_TOKEN``: their absence is reported, and
    does not affect the exit status.
    """

    name: str  # "ffmpeg", "transcriber:gigaam"
    summary: str
    run: Callable[[Database], HealthResult]
    required: bool = True  # False = advisory; absence is not a failure


HEALTH_CHECKS: dict[str, HealthCheck] = {}


class HealthCheckError(RytpError):
    """`doctor` found something broken. The message is the whole report."""


def register_check(check: HealthCheck) -> HealthCheck:
    """Add a check. Parts 2, 3, 4 and 7 call this from their own modules."""
    if not check.summary.strip():
        raise ValueError(f"health check {check.name!r} needs a summary")
    if check.name in HEALTH_CHECKS:
        raise ValueError(f"health check {check.name!r} is already registered")
    HEALTH_CHECKS[check.name] = check
    return check


def run_health_checks(db: Database) -> list[tuple[HealthCheck, HealthResult]]:
    """Run every registered check in name order. Never raises.

    A check that raises anyway becomes an ``ok=False`` result rather than
    taking `doctor` down with it: a broken check is itself a finding, and
    whether it is fatal is still decided by the check's ``required``.
    """
    results: list[tuple[HealthCheck, HealthResult]] = []
    for name in sorted(HEALTH_CHECKS):
        check = HEALTH_CHECKS[name]
        try:
            result = check.run(db)
        # Broad on purpose: a check must never take `doctor` down.
        except Exception as exc:
            result = HealthResult(
                ok=False,
                detail=f"the check itself raised: {exc}",
                remedy=f"fix or unregister the {name!r} health check",
            )
        results.append((check, result))
    return results


def _check_python(db: Database) -> HealthResult:
    major, minor = C.MIN_PYTHON_VERSION
    running = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= C.MIN_PYTHON_VERSION:
        return HealthResult(True, f"Python {running}")
    return HealthResult(
        False,
        f"Python {running}, need {major}.{minor} or newer",
        remedy=f"install Python {major}.{minor}+ and re-run scripts/bootstrap.sh",
    )


def _check_data_tree(db: Database) -> HealthResult:
    root = paths().root
    if not root.is_dir():
        return HealthResult(
            False,
            f"{root} does not exist",
            remedy="run any rytp command, for example `rytp channel list`, to create it",
        )
    probe = root / C.WRITE_PROBE_FILENAME
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        return HealthResult(
            False,
            f"{root} is not writable ({exc.strerror or exc})",
            remedy=f"grant write access to {root}, or point RYTP_DATA somewhere writable",
        )
    return HealthResult(True, f"{root} exists and is writable")


def _check_schema(db: Database) -> HealthResult:
    found = db.schema_version()
    if found == LATEST_VERSION:
        return HealthResult(True, f"schema version {found}")
    if found < LATEST_VERSION:
        return HealthResult(
            False,
            f"schema version {found}, code expects {LATEST_VERSION}",
            remedy="run any rytp command, for example `rytp channel list`, to migrate",
        )
    return HealthResult(
        False,
        f"schema version {found} is newer than this code's {LATEST_VERSION}",
        remedy="update rytp: this database was written by a newer version",
    )


register_check(
    HealthCheck(
        name="python",
        summary="The interpreter is new enough.",
        run=_check_python,
    )
)
register_check(
    HealthCheck(
        name="data-tree",
        summary="The data tree exists and can be written to.",
        run=_check_data_tree,
    )
)
register_check(
    HealthCheck(
        name="schema",
        summary="The database is migrated to the version this code expects.",
        run=_check_schema,
    )
)


def health_status(check: HealthCheck, result: HealthResult) -> str:
    """One of the three status words. Only ``FAILED`` affects the exit code."""
    if result.ok:
        return C.CHECK_OK
    return C.CHECK_FAILED if check.required else C.CHECK_ADVISORY


def _health_rows(
    results: list[tuple[HealthCheck, HealthResult]],
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (
            check.name,
            health_status(check, result),
            result.detail,
            result.remedy or C.NULL_CELL,
        )
        for check, result in results
    )


def doctor(db: Database, **_: object) -> CommandResult:
    """Report on everything rytp needs in order to work.

    An advisory check that reports an absence is shown as ``advisory``,
    with its remedy, and does not affect the exit status. Only a
    ``required`` check reporting ``ok=False`` is fatal (contracts §5).

    Raises:
        HealthCheckError: if a required check failed. Its message is the
            full report, so the surface's one-line error handling still
            shows every check and every remedy.
    """
    results = run_health_checks(db)
    rows = _health_rows(results)
    broken = [check.name for check, result in results if not result.ok and check.required]
    advisory = [
        check.name for check, result in results if not result.ok and not check.required
    ]
    columns = ("check", "status", "detail", "remedy")

    def summary() -> str:
        tail = f" ({len(advisory)} advisory)" if advisory else ""
        return f"{len(results)} checks, all required ok{tail}"

    if not broken:
        return CommandResult(columns=columns, rows=rows, message=summary())

    lines = [
        f"{len(broken)} of {len(results)} required checks failed: {', '.join(broken)}"
    ]
    for check, result in results:
        lines.append(
            f"  {check.name}: {health_status(check, result)} — {result.detail}"
        )
        if result.remedy and not result.ok:
            lines.append(f"      fix: {result.remedy}")
    raise HealthCheckError("\n".join(lines))


register(
    Command(
        name="doctor",
        group="",
        summary="Check that everything rytp needs is installed and working.",
        params=(),
        handler=doctor,
    )
)


def _open_tui(db: Database, **_: object) -> CommandResult:
    """Hand the open database to the Textual app and block until it exits.

    Imported lazily so a plain `rytp videos list` never pays for Textual.
    """
    from rytp.tui.app import run_tui

    run_tui(db)
    return CommandResult(message="tui closed")


register(
    Command(
        name="tui",
        group="",
        summary="Open the interactive surface.",
        params=(),
        handler=_open_tui,
        cli_only=True,
    )
)

# Importing a command module is what registers its commands. Keep this
# block last: the modules below import names defined above.
from rytp.commands import catalog as _catalog  # noqa: E402,F401,I001
from rytp.commands import ingest as _ingest  # noqa: E402,F401
from rytp.commands import assemble as _assemble  # noqa: E402,F401
from rytp.commands import transcribe as _transcribe  # noqa: E402,F401
from rytp.commands import render as _render  # noqa: E402,F401
from rytp.commands import search as _search  # noqa: E402,F401
from rytp.commands import speakers as _speakers  # noqa: E402,F401
from rytp.commands import settings as _settings  # noqa: E402,F401
