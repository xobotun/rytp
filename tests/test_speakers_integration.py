"""One video, transcript to named speakers, with nothing heavy installed."""

from __future__ import annotations

import wave
from pathlib import Path

import rytp.commands.speakers  # noqa: F401 - importing registers the commands
from rytp.commands import resolve
from rytp.config import paths
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.link import MATCH
from rytp.jobs import JOB_HANDLERS, JOB_KINDS
from rytp.jobs import queue as Q
from tests.fake_speaker_engines import ConstantEmbedder, FakeDiarizer, registered
from tests.fakes import make_video
from tests.test_speakers_labels import add_words


class LongDiarizer(FakeDiarizer):
    """Two voices, each with more speech than the embedding floor.

    The default fake script gives each label under a second, which is below
    `C.EMBED_MIN_SPEECH_MS` — correct behaviour, and useless for testing
    suggestions, because nothing would be embedded and the assertion would
    pass on an empty result.
    """

    name = "fake-long"
    script = ((0, 5_000, "SPEAKER_00"), (5_000, 8_000, "SPEAKER_01"))


def cache_a_wav(video_id: int, ms: int = 20_000) -> Path:
    path = paths().cache_wav(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x01\x00" * (16 * ms))
    return path


def a_transcribed_video(db: Database, external_id: str, published_at: str) -> int:
    video_id = make_video(
        db,
        external_id=external_id,
        url=f"https://example.invalid/{external_id}",
        published_at=published_at,
    )
    add_words(db, video_id, [(0, 900), (1_000, 1_900), (2_000, 2_900)])
    cache_a_wav(video_id)
    return video_id


def a_long_video(db: Database, external_id: str, published_at: str) -> int:
    """Same, with words long enough that :class:`LongDiarizer` gives each
    label more than `C.EMBED_MIN_SPEECH_MS` of speech to embed."""
    video_id = make_video(
        db,
        external_id=external_id,
        url=f"https://example.invalid/{external_id}",
        published_at=published_at,
    )
    add_words(db, video_id, [(0, 2_500), (2_500, 5_000), (5_000, 8_000)])
    cache_a_wav(video_id)
    return video_id


def test_from_a_transcript_to_named_speakers_and_back(db: Database, data_dir: Path) -> None:
    first = a_transcribed_video(db, "VIDEO_A", "2020-01-01")

    # Nothing has been diarized, so nothing is queued for the GPU.
    assert Q.list_jobs(db) == []

    # The owner opts one video in, by hand.
    with registered(FakeDiarizer, ConstantEmbedder):
        resolve("speakers.diarize").handler(
            db,
            video=str(first),
            diarizer="fake-diarizer",
            embedder="fake-constant-embedder",
        )

    # Two citable voices, both nameless, both with the words to prove it,
    # and both recording which engine drew them (contracts §3).
    labels = store.label_rows(db, first)
    assert [row.local_label for row in labels] == ["SPEAKER_00", "SPEAKER_01"]
    assert all(row.speaker_id is None for row in labels)
    assert all(row.engine == "fake-diarizer" for row in labels)
    assert sum(row.n_words for row in labels) == 3

    # Naming one is one row, and the words are untouched.
    resolve("speakers.add").handler(db, label="Host One", aliases="h1", notes=None)
    words_before = db.conn.execute(
        "SELECT group_concat(id || ':' || COALESCE(video_speaker_id, 'x')) FROM words"
    ).fetchone()[0]
    resolve("speakers.link").handler(
        db, video=str(first), label="SPEAKER_00", speaker="h1"
    )
    assert (
        db.conn.execute(
            "SELECT group_concat(id || ':' || COALESCE(video_speaker_id, 'x')) FROM words"
        ).fetchone()[0]
        == words_before
    )

    # The forward index now answers "who said this word".
    row = db.conn.execute(
        """
        SELECT s.label FROM words w
        JOIN video_speakers vs ON vs.id = w.video_speaker_id
        JOIN speakers s        ON s.id = vs.speaker_id
        WHERE w.video_id = ? ORDER BY w.ord LIMIT 1
        """,
        (first,),
    ).fetchone()
    assert row["label"] == "Host One"

    # And the backward index answers "where did this person say it".
    count = db.conn.execute(
        """
        SELECT COUNT(*) FROM words w
        JOIN video_speakers vs ON vs.id = w.video_speaker_id
        WHERE vs.speaker_id = (SELECT id FROM speakers WHERE label = 'Host One')
          AND w.normalized_text = 'w0'
        """
    ).fetchone()[0]
    assert count == 1


def test_a_second_video_is_suggested_never_linked(db: Database, data_dir: Path) -> None:
    first = a_long_video(db, "VIDEO_A", "2020-01-01")
    second = a_long_video(db, "VIDEO_B", "2020-06-01")
    with registered(LongDiarizer, ConstantEmbedder):
        for video_id in (first, second):
            resolve("speakers.diarize").handler(
                db,
                video=str(video_id),
                diarizer="fake-long",
                embedder="fake-constant-embedder",
            )
    # Both videos really did get vectors — otherwise the assertions below
    # would pass on an empty result and prove nothing.
    assert all(row.has_embedding for row in store.label_rows(db, second))

    resolve("speakers.add").handler(db, label="Host One", aliases="", notes=None)
    resolve("speakers.link").handler(
        db, video=str(first), label="SPEAKER_00", speaker="Host One"
    )

    before = db.conn.total_changes
    result = resolve("speakers.suggest").handler(db, video=str(second), label="", limit=5)
    assert db.conn.total_changes == before, "suggestions must never write"

    # Identical vectors in the same era: as confident as this ever gets.
    assert result.rows
    assert {row[3] for row in result.rows} == {MATCH}
    # And still nobody is linked. That is the whole point.
    assert store.find_label(db, second, "SPEAKER_00").speaker_id is None
    assert store.find_label(db, second, "SPEAKER_01").speaker_id is None


def test_the_queue_path_reaches_the_same_place(db: Database, data_dir: Path) -> None:
    video_id = a_transcribed_video(db, "VIDEO_A", "2020-01-01")
    resolve("speakers.enqueue").handler(db, video=str(video_id), priority=0)
    job = Q.list_jobs(db)[0]
    assert (job.kind, job.pool, job.state) == ("diarize", "gpu", "pending")
    with registered(FakeDiarizer):
        JOB_HANDLERS["diarize"](db, job.target_id, {"diarizer": "fake-diarizer"})
    assert len(store.label_rows(db, video_id)) == 2
    # And the finished job is never re-derived behind the owner's back.
    assert JOB_KINDS["diarize"].reopenable is False


def test_re_transcribing_wipes_the_mapping_and_that_is_fine(
    db: Database, data_dir: Path
) -> None:
    # Contracts §4: whatever replaces a video's words deletes its
    # video_speakers rows in the same transaction. Part 7 assumes that and
    # adds no preservation mechanism (design §3, "erase and replace").
    video_id = a_transcribed_video(db, "VIDEO_A", "2020-01-01")
    with registered(FakeDiarizer):
        resolve("speakers.diarize").handler(
            db, video=str(video_id), diarizer="fake-diarizer", embedder=""
        )
    resolve("speakers.add").handler(db, label="Host One", aliases="", notes=None)
    resolve("speakers.link").handler(
        db, video=str(video_id), label="SPEAKER_00", speaker="Host One"
    )

    with db.transaction():
        db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
        db.conn.execute("DELETE FROM words WHERE video_id = ?", (video_id,))
        db.conn.execute("DELETE FROM video_speakers WHERE video_id = ?", (video_id,))

    assert store.label_rows(db, video_id) == []
    # The roster survives — re-mapping two to five labels is the whole price.
    assert store.find_speaker(db, "Host One") is not None


def test_every_speakers_command_reaches_both_surfaces(db: Database) -> None:
    from rytp.cli import build_app, command_paths
    from rytp.commands import COMMANDS
    from rytp.tui.palette import palette_entries

    registered_names = {name for name in COMMANDS if name.startswith("speakers.")}
    cli = command_paths(build_app())
    assert registered_names <= cli

    palette = {entry.name for entry in palette_entries()}
    # Every one but the launcher, which cannot be launched from inside the
    # surface it launches (contracts §5, `cli_only`).
    assert registered_names - {"speakers.map"} <= palette
    assert "speakers.map" not in palette


def test_the_group_can_undo_everything_it_creates(db: Database) -> None:
    # contracts §5: no entity you can create but not get rid of.
    from rytp.commands import COMMANDS

    assert "speakers.remove" in COMMANDS


def test_doctor_reports_on_speakers_without_raising(db: Database, data_dir: Path) -> None:
    # contracts §5: a check never raises and a missing optional engine is
    # reported, not fatal.
    from rytp.commands import HEALTH_CHECKS

    for name in ("diarizers", "hf-token"):
        result = HEALTH_CHECKS[name].run(db)
        assert isinstance(result.ok, bool)
        assert result.detail
