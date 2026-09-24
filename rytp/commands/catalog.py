"""Catalog commands: register channels and videos, and list them (design §5).

Cataloguing is deliberately separate from acquisition. These commands
write rows and nothing else — no downloads, no jobs, no files on disk.
Design §5: "`rytp videos add <url>` catalogs only. `rytp ingest <id>`
enqueues the chain."
"""

from __future__ import annotations

import shutil
import tomllib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.commands import Command, CommandResult, Param, register, resolve_video_id
from rytp.db import Database
from rytp.db import queries as q
from rytp.models import (
    ChannelEntry,
    InvalidInputError,
    NotFoundError,
    RytpError,
    utc_now_iso,
)

__all__ = [
    "channel_add",
    "channel_list",
    "channel_remove",
    "channel_sync",
    "cutlists_naming_video",
    "directory_bytes",
    "format_duration",
    "looks_local",
    "parse_tabs",
    "resolve_channel",
    "truncate",
    "video_files",
    "videos_add",
    "videos_list",
    "videos_remove",
]


def truncate(text: str, limit: int = C.TITLE_TRUNCATE_CHARS) -> str:
    """Shorten a title for a table cell, marking that it was cut."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """`1 channel` / `3 channels`, so the message line reads like English."""
    word = singular if count == 1 else (plural_form or f"{singular}s")
    return f"{count} {word}"


# -- channels ---------------------------------------------------------


def channel_add(db: Database, *, url: str, title: str | None = None) -> CommandResult:
    """Register a channel by URL. Nothing is fetched."""
    url = url.strip()
    if not url:
        raise InvalidInputError("channel add needs a URL")
    resolved_title = (title or url).strip()
    channel_id, created = q.insert_channel(db, url=url, title=resolved_title)
    return CommandResult(
        columns=("id", "url", "title"),
        rows=((str(channel_id), url, resolved_title),),
        message=f"{'added' if created else 'updated'} channel {channel_id}",
    )


def channel_list(db: Database, *, limit: int = C.DEFAULT_LIST_LIMIT) -> CommandResult:
    """List registered channels, newest first, with their video counts."""
    rows = q.list_channels(db, limit=limit)
    return CommandResult(
        columns=("id", "title", "videos", "last synced", "url"),
        rows=tuple(
            (
                str(row["id"]),
                truncate(row["title"]),
                str(row["n_videos"]),
                row["last_synced_at"] or C.NULL_CELL,
                row["url"],
            )
            for row in rows
        ),
        message=plural(len(rows), "channel"),
    )


# -- seams part 2 fills in --------------------------------------------


def _video_job_kinds() -> tuple[str, ...]:
    """Job kinds whose `jobs.target_id` is a `videos.id`.

    Derived from part 2's registry, never listed here: `target_kind` is a
    Python attribute of the registration (contracts §5), and the obvious
    eight kinds already miss part 3's `caption_words`.

    Part 1 ships before `rytp.jobs` exists. With no registry there are no
    job kinds and nothing creates job rows, so an empty tuple is the
    correct answer rather than a guess.
    """
    try:
        from rytp.jobs import JOB_KINDS
    except ImportError:
        return ()
    return tuple(
        sorted(
            name
            for name, kind in JOB_KINDS.items()
            if kind.target_kind == C.VIDEO_TARGET_KIND
        )
    )


def _probe_video(url: str) -> ChannelEntry:
    """Ask yt-dlp what a URL is. One metadata request, no download.

    Implemented by `rytp/acquire/ytdlp.py` (plan part 2). Kept behind a
    module-level function so part 1 can be tested without it and without
    a network.
    """
    try:
        from rytp.acquire.ytdlp import probe_video
    except ImportError as exc:
        raise RytpError(
            "registering a URL needs yt-dlp metadata; install the extra: "
            'pip install -e ".[yt-dlp]"'
        ) from exc
    return probe_video(url)


def _enumerate_channel(channel_url: str, tabs: Sequence[str]) -> list[ChannelEntry]:
    """List a channel's videos across the given tabs.

    Implemented by `rytp/acquire/ytdlp.py` (plan part 2). It must return
    entries in tab order and must not deduplicate: `channel_sync` relies
    on a later tab overwriting an earlier one's `kind`.
    """
    try:
        from rytp.acquire.ytdlp import enumerate_channel
    except ImportError as exc:
        raise RytpError(
            "channel sync needs yt-dlp to enumerate the channel; install the "
            'extra: pip install -e ".[yt-dlp]"'
        ) from exc
    return enumerate_channel(channel_url, tabs=tuple(tabs))


# -- videos -----------------------------------------------------------


def looks_local(target: str) -> bool:
    """True when `target` names a file on disk rather than a URL.

    A Windows path like `C:\\videos\\a.mp4` contains a colon, so the test
    is for a scheme separator, not for a colon.
    """
    if "://" in target:
        return False
    return Path(target).expanduser().exists()


def format_duration(ms: int | None) -> str:
    """Milliseconds as H:MM:SS, or the null placeholder."""
    if ms is None:
        return C.NULL_CELL
    seconds, _ = divmod(int(ms), C.MS_PER_SECOND)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def resolve_channel(db: Database, ref: str) -> int:
    """Find a channel by row id, URL or exact title."""
    row = q.get_channel(db, int(ref)) if ref.isdigit() else q.find_channel(db, ref)
    if row is None:
        raise NotFoundError(
            f"no channel matches {ref!r}; register it with: rytp channel add <url>"
        )
    return int(row["id"])


def check_choice(name: str, value: str | None, allowed: tuple[str, ...]) -> None:
    """Guard a handler called directly, not through a surface."""
    if value is not None and value not in allowed:
        raise InvalidInputError(f"{name} must be one of: {', '.join(allowed)}")


def videos_add(
    db: Database,
    *,
    target: str,
    title: str | None = None,
    kind: str | None = None,
    channel: str | None = None,
) -> CommandResult:
    """Catalog one video from a URL or a local file. Nothing is downloaded."""
    target = target.strip()
    if not target:
        raise InvalidInputError("videos add needs a URL or a file path")
    check_choice("kind", kind, C.VIDEO_KINDS)
    channel_id = resolve_channel(db, channel) if channel else None

    if looks_local(target):
        path = Path(target).expanduser().resolve()
        if not path.is_file():
            raise NotFoundError(f"{path} is not a file")
        source = C.LOCAL_SOURCE
        # `videos` has no path column, so the resolved path is the
        # natural key that makes re-registration idempotent. Part 2's
        # acquire/local.py turns it into a 'container' asset.
        external_id = str(path)
        url: str | None = None
        resolved_title = title or path.stem
        resolved_kind = kind or C.DEFAULT_VIDEO_KIND
        duration_ms: int | None = None
        published_at: str | None = None
    elif "://" in target:
        entry = _probe_video(target)
        source = C.REMOTE_SOURCE
        external_id = entry.external_id
        url = entry.url
        resolved_title = title or entry.title
        resolved_kind = kind or entry.kind or C.DEFAULT_VIDEO_KIND
        duration_ms = entry.duration_ms
        published_at = entry.published_at
    else:
        raise NotFoundError(
            f"{target!r} is neither a URL nor an existing file"
        )

    video_id, created = q.upsert_video(
        db,
        source=source,
        kind=resolved_kind,
        channel_id=channel_id,
        external_id=external_id,
        url=url,
        title=resolved_title,
        duration_ms=duration_ms,
        published_at=published_at,
    )
    return CommandResult(
        columns=("id", "source", "kind", "title"),
        rows=((str(video_id), source, resolved_kind, truncate(resolved_title)),),
        message=f"{'added' if created else 'updated'} video {video_id}",
    )


def videos_list(
    db: Database,
    *,
    channel: str | None = None,
    kind: str | None = None,
    source: str | None = None,
    search: str | None = None,
    limit: int = C.DEFAULT_LIST_LIMIT,
) -> CommandResult:
    """List catalogued videos, newest first, narrowed by the given filters."""
    check_choice("kind", kind, C.VIDEO_KINDS)
    check_choice("source", source, C.VIDEO_SOURCES)
    channel_id = resolve_channel(db, channel) if channel else None
    rows = q.list_videos(
        db,
        channel_id=channel_id,
        kind=kind,
        source=source,
        search=search,
        limit=limit,
    )
    return CommandResult(
        columns=("id", "source", "kind", "duration", "channel", "title", "external id"),
        rows=tuple(
            (
                str(row["id"]),
                row["source"],
                row["kind"],
                format_duration(row["duration_ms"]),
                row["channel_title"] or C.NULL_CELL,
                truncate(row["title"]),
                row["external_id"] or C.NULL_CELL,
            )
            for row in rows
        ),
        message=plural(len(rows), "video"),
    )


# -- channel sync -------------------------------------------------------


def parse_tabs(text: str) -> tuple[str, ...]:
    """Parse `--tabs`, always returning them in the canonical order.

    Order matters: a live stream is listed under both /videos and
    /streams, and the last listing to mention a video decides its kind.
    Sorting the user's selection into `constants.CHANNEL_TABS` order
    makes that outcome independent of how the flag was typed.
    """
    requested = {part.strip().lower() for part in text.split(",") if part.strip()}
    unknown = requested - set(C.CHANNEL_TABS)
    if unknown:
        raise InvalidInputError(
            f"unknown channel tab(s): {', '.join(sorted(unknown))}; "
            f"choose from {', '.join(C.CHANNEL_TABS)}"
        )
    if not requested:
        raise InvalidInputError(
            f"channel sync needs at least one tab: {', '.join(C.CHANNEL_TABS)}"
        )
    return tuple(tab for tab in C.CHANNEL_TABS if tab in requested)


def channel_sync(
    db: Database, *, channel: str, tabs: str = ",".join(C.CHANNEL_TABS)
) -> CommandResult:
    """Re-enumerate a channel's listings and catalog everything found.

    Design §13: the main video listing alone undercounts the corpus,
    because live streams are listed separately. All three listings are
    walked by default.
    """
    channel_id = resolve_channel(db, channel)
    row = q.get_channel(db, channel_id)
    assert row is not None  # resolve_channel raised if it were missing
    selected = parse_tabs(tabs)

    entries = _enumerate_channel(row["url"], selected)
    added = 0
    refreshed = 0
    with db.transaction():
        for found in entries:
            _video_id, created = q.upsert_video(
                db,
                source=C.REMOTE_SOURCE,
                kind=found.kind or C.DEFAULT_VIDEO_KIND,
                channel_id=channel_id,
                external_id=found.external_id,
                url=found.url,
                title=found.title,
                duration_ms=found.duration_ms,
                published_at=found.published_at,
            )
            if created:
                added += 1
            else:
                refreshed += 1
        q.mark_channel_synced(db, channel_id, utc_now_iso())

    by_kind = Counter(
        kind
        for (kind,) in db.conn.execute(
            "SELECT kind FROM videos WHERE channel_id = ?", (channel_id,)
        )
    )
    return CommandResult(
        columns=("kind", "catalogued"),
        rows=tuple(
            (kind, str(by_kind[kind]))
            for kind in C.VIDEO_KINDS
            if by_kind.get(kind)
        ),
        message=(
            f"{added} new, {refreshed} refreshed from "
            f"{len(entries)} listing entries across {', '.join(selected)}"
        ),
    )


# -- removal (contracts §5 "Deletion") --------------------------------


def channel_remove(db: Database, *, channel: str) -> CommandResult:
    """Remove a channel registration. Its videos are orphaned, not deleted.

    `videos.channel_id` is nullable for exactly this case (contracts §5):
    dropping a channel is not a request to lose its videos. The foreign
    key has no `ON DELETE` clause, so the orphaning is explicit — without
    it SQLite refuses the delete. No files are touched, so this needs
    neither `--dry-run` nor `--yes`.
    """
    channel_id = resolve_channel(db, channel)
    with db.transaction():
        orphaned = db.conn.execute(
            "UPDATE videos SET channel_id = NULL WHERE channel_id = ?", (channel_id,)
        ).rowcount
        db.conn.execute("DELETE FROM channels WHERE id = ?", (channel_id,))
    return CommandResult(
        message=(
            f"removed channel {channel_id}; "
            f"{plural(orphaned, 'video')} orphaned, none deleted"
        )
    )


def directory_bytes(path: Path) -> int:
    """Total size of a file or of everything under a directory. 0 if absent."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def video_files(video_id: int) -> list[Path]:
    """Every path on disk that belongs to one video and exists right now.

    Cascades do not reach the filesystem (contracts §5), so these are
    removed by hand. `transcripts/{video_id}.md` is included with the
    other two: it is named after the video and would otherwise be a file
    about something that no longer exists.
    """
    layout = config.paths()
    candidates = [
        layout.media_dir(video_id),
        layout.cache_wav(video_id),
        layout.transcript(video_id),
    ]
    return [path for path in candidates if path.exists()]


def cutlists_naming_video(video_id: int) -> list[str]:
    """Names of cut lists that mention this video id.

    A warning, never a block. Parts 5 and 6 treat a dangling `video_id`
    as corrupt input and refuse to render; the owner should hear about
    it at removal time instead of at render time. A cut list that will
    not parse is skipped rather than guessed at.
    """

    def mentions(node: object) -> bool:
        if isinstance(node, dict):
            if node.get("video_id") == video_id:
                return True
            return any(mentions(value) for value in node.values())
        if isinstance(node, list):
            return any(mentions(item) for item in node)
        return False

    directory = config.paths().root / C.CUTLISTS_DIRNAME
    if not directory.is_dir():
        return []
    found: list[str] = []
    for path in sorted(directory.glob("*.toml")):
        try:
            with path.open("rb") as handle:
                document = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if mentions(document):
            found.append(path.stem)
    return found


def videos_remove(
    db: Database, *, video: str, dry_run: bool = False, yes: bool = False
) -> CommandResult:
    """Remove a video: its row, everything that cascades from it, and its files.

    Synchronous and immediate — contracts §5 is explicit that removal is
    never a job, because a half-deleted entity recovered from a crashed
    queue is worse than a slow command.
    """
    video_id = resolve_video_id(db, video)

    counts = {
        table: int(
            db.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE video_id = ?", (video_id,)
            ).fetchone()[0]
        )
        for table in C.VIDEO_CASCADE_TABLES
    }
    job_kinds = _video_job_kinds()
    placeholders = ", ".join("?" for _ in job_kinds)
    counts["jobs"] = (
        int(
            db.conn.execute(
                f"SELECT COUNT(*) FROM jobs WHERE target_id = ? AND kind IN"
                f" ({placeholders})",
                (video_id, *job_kinds),
            ).fetchone()[0]
        )
        if job_kinds
        else 0
    )
    files = video_files(video_id)
    total_bytes = sum(directory_bytes(path) for path in files)
    counts["files"] = len(files)
    affected_cutlists = cutlists_naming_video(video_id)

    rows = tuple((name, str(value)) for name, value in counts.items())
    warning = ""
    if affected_cutlists:
        warning = (
            f"; warning: cut list(s) {', '.join(affected_cutlists)} reference this "
            "video and will not render until you edit them"
        )

    if dry_run:
        return CommandResult(
            columns=("what", "count"),
            rows=rows,
            message=(
                f"dry run: video {video_id} would be removed with "
                f"{total_bytes} bytes across {plural(len(files), 'path')}{warning}"
            ),
        )
    if not yes:
        raise InvalidInputError(
            f"videos remove deletes {total_bytes} bytes from disk and cannot be "
            "undone; re-run with --yes, or with --dry-run to see what would go"
        )

    # The database first, in one transaction: if a file then refuses to
    # go, the catalog is still consistent and the leftover is named.
    with db.transaction():
        if job_kinds:
            db.conn.execute(
                f"DELETE FROM jobs WHERE target_id = ? AND kind IN ({placeholders})",
                (video_id, *job_kinds),
            )
        db.conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))

    stubborn: list[str] = []
    for path in files:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            stubborn.append(f"{path} ({exc.strerror or exc})")
    if stubborn:
        warning += f"; could not delete: {', '.join(stubborn)}"

    return CommandResult(
        columns=("what", "count"),
        rows=rows,
        message=(
            f"removed video {video_id} and {total_bytes} bytes across "
            f"{plural(len(files), 'path')}{warning}"
        ),
    )


register(
    Command(
        name="channel.add",
        group="channel",
        summary="Register a channel by URL. Catalogs only; nothing is fetched.",
        params=(
            Param("url", str, "Channel URL.", positional=True),
            Param("title", str, "Title to store. Defaults to the URL.", default=None),
        ),
        handler=channel_add,
    )
)

register(
    Command(
        name="channel.list",
        group="channel",
        summary="List registered channels and how many videos each has.",
        params=(
            Param("limit", int, "Maximum rows.", default=C.DEFAULT_LIST_LIMIT, short="-n"),
        ),
        handler=channel_list,
    )
)

register(
    Command(
        name="videos.add",
        group="videos",
        summary="Catalog one video from a URL or a local file. Nothing is downloaded.",
        params=(
            Param("target", str, "A URL, or a path to a local media file.", positional=True),
            Param(
                "title",
                str,
                "Title to store. Defaults to what the source reports.",
                default=None,
            ),
            Param("kind", str, "Video kind.", default=None, choices=C.VIDEO_KINDS),
            Param(
                "channel", str, "Channel id, URL or title to file it under.", default=None
            ),
        ),
        handler=videos_add,
    )
)

register(
    Command(
        name="videos.list",
        group="videos",
        summary="List catalogued videos.",
        params=(
            Param("channel", str, "Only this channel (id, URL or title).", default=None),
            Param("kind", str, "Only this kind.", default=None, choices=C.VIDEO_KINDS),
            Param("source", str, "Only this source.", default=None, choices=C.VIDEO_SOURCES),
            Param("search", str, "Substring of the title.", default=None),
            Param("limit", int, "Maximum rows.", default=C.DEFAULT_LIST_LIMIT, short="-n"),
        ),
        handler=videos_list,
    )
)

register(
    Command(
        name="channel.sync",
        group="channel",
        summary="Re-enumerate a channel's videos, live streams and shorts listings.",
        params=(
            Param("channel", str, "Channel id, URL or title.", positional=True),
            Param(
                "tabs",
                str,
                "Comma-separated listings to walk.",
                default=",".join(C.CHANNEL_TABS),
            ),
        ),
        handler=channel_sync,
        long_running=True,
    )
)

register(
    Command(
        name="channel.remove",
        group="channel",
        summary="Remove a channel registration. Its videos are orphaned, not deleted.",
        params=(Param("channel", str, "Channel id, URL or title.", positional=True),),
        handler=channel_remove,
    )
)

register(
    Command(
        name="videos.remove",
        group="videos",
        summary="Remove a video, everything derived from it, and its files on disk.",
        params=(
            Param("video", str, "Video id or external id.", positional=True),
            Param(
                "dry_run",
                bool,
                "Print what would be removed and change nothing.",
                default=False,
            ),
            Param(
                "yes",
                bool,
                "Confirm. Required, because this deletes files.",
                default=False,
            ),
        ),
        handler=videos_remove,
    )
)
