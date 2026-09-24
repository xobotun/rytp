"""Diarize one video: labels in the database, labels on the words (design §6).

Two things happen here and both are deliberate.

**`words.video_speaker_id` is written.** Once, at diarize time, from
overlap. Everything downstream — the mapper's left pane, the search
filter, Part 6's per-speaker pause measurement, utterance splitting — reads
that column. This is the *only* stage that writes it, and it is not the same
thing as linking a label to a person: that happens later and touches one
row of `video_speakers`, never the words.

**The utterances are rebuilt.** Design §4 splits utterances on speaker
change, and design §7's speaker filter searches utterances rather than
words. Stamping the words without re-deriving the utterances would leave
the video searchable by word and unsearchable by speaker, which is exactly
the gap this part exists to close. So the video's utterances are deleted
and its `index` job enqueued; Part 4 rebuilds them from the labelled words.
The deletion is unconditional (design §3: "erase and replace, don't
reconcile"); the enqueue happens only when the `index` kind is registered,
so Part 7 can be built and tested before Part 4 exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rytp import constants as C
from rytp import progress
from rytp.config import paths
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.base import DiarizeError, Diarizer, load_diarizer
from rytp.diarize.segments import label_words, local_labels
from rytp.models import DiarSegment, RytpError

if TYPE_CHECKING:
    from rytp.diarize.embed import SpeakerEmbedder


@dataclass(frozen=True)
class DiarizeOutcome:
    """What one diarization run did, for the command line and the job log."""

    video_id: int
    diarizer: str
    n_labels: int
    n_segments: int
    n_words_labelled: int
    n_words_unlabelled: int
    reindexed: bool

    def message(self) -> str:
        tail = "" if self.reindexed else " (no index job kind registered)"
        return (
            f"video {self.video_id}: {self.n_labels} speakers from "
            f"{self.n_segments} segments via {self.diarizer}; "
            f"{self.n_words_labelled} words labelled, "
            f"{self.n_words_unlabelled} left unlabelled{tail}"
        )


def _validated(segments: list[DiarSegment]) -> list[DiarSegment]:
    for segment in segments:
        if segment.end_ms <= segment.start_ms:
            raise DiarizeError(
                f"diarizer returned a segment whose end ({segment.end_ms} ms) is not "
                f"after its start ({segment.start_ms} ms)"
            )
        if not segment.local_label:
            raise DiarizeError("diarizer returned a segment with an empty label")
    return segments


def request_reindex(db: Database, video_id: int) -> bool:
    """Drop the video's utterances and ask for them to be rebuilt.

    Returns whether an `index` job was enqueued. Part 4 owns that kind; the
    imports are inside the function so Part 7 depends on neither Part 4's
    import order nor the queue being in the diarize package's import path.
    """
    db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
    from rytp.jobs import JOB_KINDS

    if "index" not in JOB_KINDS:
        return False
    from rytp.jobs.queue import enqueue

    enqueue(db, "index", video_id)
    return True


def diarize_video(
    db: Database, video_id: int, *, wav_path: Path, diarizer: Diarizer
) -> DiarizeOutcome:
    """Run a diarizer over one video and record who spoke when."""
    words = store.word_spans(db, video_id)
    if not words:
        raise DiarizeError(f"video {video_id} has no words to label; transcribe it first")

    progress.report("diarize", detail=f"video {video_id}: {diarizer.name} segmenting")
    segments = _validated(list(diarizer.diarize(wav_path)))
    progress.report(
        "diarize", done=1, total=1, detail=f"video {video_id}: {len(segments)} segments"
    )
    if not segments:
        raise DiarizeError(f"diarizer {diarizer.name!r} found no speech in {wav_path}")

    labels = local_labels(segments)
    by_word_label = label_words(words, segments)

    with db.transaction():
        # Labels first: every id the stamp needs must already exist.
        label_ids = {
            label: store.upsert_video_speaker(db, video_id, label, engine=diarizer.name)
            for label in labels
        }
        # A label from a previous run that this run did not produce goes,
        # and its words are released by ON DELETE SET NULL.
        for existing in store.label_rows(db, video_id):
            if existing.local_label not in label_ids:
                db.conn.execute(
                    "DELETE FROM video_speakers WHERE id = ?",
                    (existing.video_speaker_id,),
                )
        store.stamp_words(
            db,
            video_id,
            {word_id: label_ids[label] for word_id, label in by_word_label.items()},
        )
        reindexed = request_reindex(db, video_id)

    return DiarizeOutcome(
        video_id=video_id,
        diarizer=diarizer.name,
        n_labels=len(labels),
        n_segments=len(segments),
        n_words_labelled=len(by_word_label),
        n_words_unlabelled=len(words) - len(by_word_label),
        reindexed=reindexed,
    )


def wav_for(db: Database, video_id: int) -> Path:
    """The cached WAV this video is diarized from (contracts §7).

    The cache is regenerable and prunable, so a missing file is a normal
    state rather than corruption — say which command puts it back.
    """
    path = paths().cache_wav(video_id)
    if not path.exists():
        raise RytpError(
            f"no cached audio for video {video_id} at {path}; "
            f"run the extract_wav job, or `rytp ingest {video_id}` and let the "
            f"worker decode it"
        )
    return path


def diarizer_name(db: Database, requested: str | None) -> str:
    """What to diarize with: the request, then the setting, then the default."""
    if requested:
        return requested
    from rytp.transcribe.registry import setting

    return setting(db, C.SETTINGS_DIARIZER) or C.DEFAULT_DIARIZER


def run_diarize_job(db: Database, video_id: int, payload: dict[str, object]) -> None:
    """The `diarize` job kind's body (contracts §5).

    Reached from two directions — `rytp speakers diarize` by hand and the
    worker draining the queue — which is why the orchestration lives here
    rather than in either caller. Embeds in the same pass when an embedder
    is named: reconstructing speech from the just-labelled words is cheap
    (design §6), so there is no reason to make it a separate job.
    """
    wav = wav_for(db, video_id)
    name = diarizer_name(db, str(payload.get("diarizer") or ""))
    diarize_video(db, video_id, wav_path=wav, diarizer=load_diarizer(db, name))

    embedder = embedder_name(db, str(payload.get("embedder") or ""))
    if embedder:
        from rytp.diarize.embed import load_embedder

        embed_video_speakers(
            db, video_id, wav_path=wav, embedder=load_embedder(db, embedder)
        )


@dataclass(frozen=True)
class EmbedOutcome:
    """What one embedding pass did."""

    video_id: int
    embedder: str
    n_embedded: int
    n_skipped: int

    def message(self) -> str:
        return (
            f"video {self.video_id}: {self.n_embedded} voices embedded with "
            f"{self.embedder}, {self.n_skipped} skipped for too little speech"
        )


def embedder_name(db: Database, requested: str | None) -> str:
    """What to embed with: the request, then the setting, then the default."""
    if requested:
        return requested
    from rytp.transcribe.registry import setting

    return setting(db, C.SETTINGS_EMBEDDER) or C.DEFAULT_EMBEDDER


def embed_video_speakers(
    db: Database, video_id: int, *, wav_path: Path, embedder: SpeakerEmbedder
) -> EmbedOutcome:
    """Give every sufficiently talkative label of one video a voice vector.

    The speech is reconstructed from the labelled words, not from a stored
    segment list, so this can be re-run on its own at any time — and it
    should be, whenever a better embedder arrives. A label with less than
    `C.EMBED_MIN_SPEECH_MS` of speech is skipped rather than embedded
    badly: a vector from two seconds of audio is a fingerprint of the
    phonemes, not of the voice, and a wrong suggestion is worse than none.
    """
    from rytp.diarize.embed import l2_normalize, pack_embedding
    from rytp.diarize.segments import speech_ms_by_label, windows_for_label

    segments = store.label_segments(db, video_id)
    totals = speech_ms_by_label(segments)
    rows = store.label_rows(db, video_id)

    # Reported before and after the transaction, not inside it: a progress
    # write inside `db.transaction()` would join it and stay invisible on
    # the shared connection until the whole pass commits, which defeats the
    # point for the worker's database sink (plan Task 8a).
    progress.report("embed", done=0, total=len(rows), detail=f"video {video_id}: {embedder.name}")
    embedded = skipped = 0
    with db.transaction():
        for row in rows:
            if totals.get(row.local_label, 0) < C.EMBED_MIN_SPEECH_MS:
                store.set_embedding(db, row.video_speaker_id, None)
                skipped += 1
                continue
            windows = windows_for_label(segments, row.local_label)
            vector = list(embedder.embed(wav_path, windows))
            store.set_embedding(
                db, row.video_speaker_id, pack_embedding(l2_normalize(vector))
            )
            embedded += 1
    progress.report(
        "embed", done=len(rows), total=len(rows),
        detail=f"video {video_id}: {embedded} embedded, {skipped} skipped",
    )

    return EmbedOutcome(
        video_id=video_id,
        embedder=embedder.name,
        n_embedded=embedded,
        n_skipped=skipped,
    )
