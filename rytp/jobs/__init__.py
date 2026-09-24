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
#: decoded ``payload_json``, returning an optional short note.
#:
#: ``None`` is the normal case and must stay cheap. A note is for something
#: the operator needs to know about work that nonetheless *succeeded* — Part
#: 3 returns one when re-transcribing discards a video's speaker mapping.
#: The worker stores it on the job row, which is the only way such a warning
#: survives a batch of hundreds of videos: on the bulk path there is nobody
#: reading stdout.
Handler = Callable[[Database, int, dict[str, Any]], str | None]


@dataclass(frozen=True)
class JobKind:
    """One kind of work the queue knows how to schedule."""

    name: str
    pool: str
    readiness: Predicate
    handler: Handler
    summary: str
    #: What namespace ``jobs.target_id`` lives in for this kind. Every Part 2
    #: kind targets a ``videos.id``; ``render`` targets a ``renders`` row and
    #: sets ``target_kind="render"``. The queue never mixes namespaces when it
    #: re-evaluates jobs by target.
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


# Part 2's stages return None on a routine success. A note on every job
# would be noise, and noise is precisely what defeats the mechanism: the
# point is that forty videos quietly losing their speaker labels stands out.
def _run_download(db: Database, video_id: int, payload: dict[str, Any]) -> str | None:
    from rytp.acquire import acquire_media

    acquire_media(db, video_id)
    return None


def _run_captions(db: Database, video_id: int, payload: dict[str, Any]) -> str | None:
    from rytp.acquire.captions import acquire_captions

    # The one Part 2 case worth a note: the preferred caption language was
    # not available and a fallback was taken.
    return acquire_captions(db, video_id).fallback_note


def _run_extract_wav(
    db: Database, video_id: int, payload: dict[str, Any]
) -> str | None:
    from rytp.audio.extract import ensure_wav

    ensure_wav(db, video_id)
    return None


def _run_render(db: Database, target_id: int, payload: dict[str, Any]) -> str | None:
    from rytp.render.run import run_render_job

    run_render_job(db, target_id, payload)
    return None


# Imported last: readiness.py imports Readiness back out of this module, so
# the registry types must already exist when it runs.
from rytp.jobs.readiness import (  # noqa: E402
    captions_readiness,
    download_readiness,
    extract_wav_readiness,
)
from rytp.render.readiness import render_readiness  # noqa: E402

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


# ---------------------------------------------------------------------------
# Part 3: transcription, alignment and the acoustic fingerprint
# ---------------------------------------------------------------------------
from rytp.transcribe.readiness import (  # noqa: E402 - registration order
    align_readiness,
    caption_words_readiness,
    fingerprint_readiness,
    transcribe_readiness,
)


def _load_transcribe_engines() -> None:
    """Import the adapter packages, which is what makes engine names resolve.

    Contracts §6: an engine registers itself at import time, so a name that is
    never imported is a name that does not exist. Deferred to call time so the
    queue module stays light.
    """
    import rytp.transcribe.align
    import rytp.transcribe.engines  # noqa: F401


def _run_caption_words(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.transcribe.captions import ingest_captions

    ingest_captions(db, video_id)
    # Caption words are searchable words, so the index is stale too.
    from rytp.transcribe.pipeline import enqueue_index

    enqueue_index(db, video_id)


def _run_transcribe(db: Database, video_id: int, payload: dict[str, Any]) -> str | None:
    from rytp import constants as C
    from rytp.audio.extract import wav_path
    from rytp.transcribe.pipeline import enqueue_index, speaker_loss_warning, transcribe_video
    from rytp.transcribe.registry import default_transcriber

    _load_transcribe_engines()
    # `ingest --transcribe` and `transcribe run --enqueue` both stamp the
    # engine at enqueue time, so this branch is for a job made by hand.
    # contracts §3 still applies: resolve the setting, never a constant, and
    # say which engine it picked — which for a job is the note the worker
    # stores on the row. If it is not installed, `transcribe_video` raises
    # with the install hint and the job fails; it must not run another one.
    named = str(payload.get("transcriber") or "").strip()
    transcriber = named or default_transcriber(db)
    outcome = transcribe_video(
        db,
        video_id,
        wav_path=wav_path(video_id),
        transcriber=transcriber,
        aligner=payload.get("aligner") or None,
        language=payload.get("language") or C.DEFAULT_LANGUAGE,
        refine=bool(payload.get("refine", True)),
    )
    # New words mean the video's utterances are gone: ask Part 4 to rebuild.
    enqueue_index(db, video_id)
    # `_run_handler` (the hand-run CLI path) surfaces this warning in its
    # CommandResult message; a job that discarded `outcome` would silently
    # drop it for the worker, which is the bulk path and the one that
    # matters (contracts §5, jobs.note).
    notes = []
    if not named:
        notes.append(
            f"no transcriber in the payload; used {transcriber!r} from setting "
            f"{C.SETTING_DEFAULT_TRANSCRIBER}"
        )
    warning = speaker_loss_warning(video_id, outcome.speakers_lost)
    if warning:
        notes.append(warning)
    return ". ".join(notes) if notes else None


def _run_align(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.audio.extract import wav_path
    from rytp.models import RytpError
    from rytp.transcribe.pipeline import enqueue_index, realign_video

    aligner = payload.get("aligner")
    if not aligner:
        raise RytpError(
            f"align job for video {video_id} has no 'aligner' in its payload"
        )
    _load_transcribe_engines()
    realign_video(
        db,
        video_id,
        wav_path=wav_path(video_id),
        aligner=str(aligner),
        refine=bool(payload.get("refine", True)),
    )
    # Re-timing rewrote every boundary, so the utterances are gone too.
    enqueue_index(db, video_id)


def _run_fingerprint(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.audio.acoustics import fingerprint_video
    from rytp.audio.extract import wav_path

    fingerprint_video(db, video_id, wav_path=wav_path(video_id))


def _run_index(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.index.utterances import index_video

    del payload  # the video id is the whole input; nothing is configurable
    index_video(db, video_id)


def _run_diarize(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.diarize.pipeline import run_diarize_job

    run_diarize_job(db, video_id, payload)


register_job_kind(
    JobKind(
        name="caption_words",
        pool="cpu",
        readiness=caption_words_readiness,
        handler=_run_caption_words,
        summary="turn one video's downloaded captions into caption-tier words",
    )
)

register_job_kind(
    JobKind(
        name="transcribe",
        pool="gpu",
        readiness=transcribe_readiness,
        handler=_run_transcribe,
        summary="transcribe, align and refine one video into cuttable words",
    )
)

register_job_kind(
    JobKind(
        name="align",
        pool="gpu",
        readiness=align_readiness,
        handler=_run_align,
        summary="re-time one video's words with a different aligner",
        # A one-shot the user asked for: a reconcile must not re-run it.
        reopenable=False,
    )
)

register_job_kind(
    JobKind(
        name="fingerprint",
        pool="cpu",
        readiness=fingerprint_readiness,
        handler=_run_fingerprint,
        summary="measure one video's acoustic fingerprint",
    )
)

register_job_kind(
    JobKind(
        name="render",
        pool="cpu",
        readiness=render_readiness,
        handler=_run_render,
        summary="cut, normalise and join a cut list into one file",
        target_kind="render",
    )
)

from rytp.index.utterances import index_readiness  # noqa: E402

register_job_kind(
    JobKind(
        name="index",
        pool="cpu",
        readiness=index_readiness,
        handler=_run_index,
        summary="rebuild one video's utterances and its search index",
    )
)

from rytp.diarize.readiness import diarize_readiness  # noqa: E402

register_job_kind(
    JobKind(
        name="diarize",
        pool="gpu",
        readiness=diarize_readiness,
        handler=_run_diarize,
        summary="work out who spoke when, and label this video's words",
        # design §6: diarization is opt-in per video and the largest GPU
        # cost in the project. A reconcile sweep must never re-derive it.
        reopenable=False,
    )
)
