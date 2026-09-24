"""The speakers command group: registered once, used by both surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest

import rytp.commands.speakers  # noqa: F401 - importing registers the commands
from rytp import constants as C
from rytp.commands import COMMANDS, REQUIRED, resolve
from rytp.config import paths
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.embed import pack_embedding
from rytp.models import NotFoundError, RytpError
from tests.fake_speaker_engines import FakeDiarizer, FakeEmbedder, registered
from tests.fakes import make_video
from tests.test_speakers_labels import DIARIZER, add_label, add_words

GROUP = (
    "speakers.add",
    "speakers.alias",
    "speakers.diarize",
    "speakers.remove",
    "speakers.embed",
    "speakers.engines",
    "speakers.enqueue",
    "speakers.labels",
    "speakers.link",
    "speakers.list",
    "speakers.map",
    "speakers.suggest",
    "speakers.unlink",
)


def cache_a_wav(video_id: int, ms: int = 20_000) -> Path:
    import wave

    path = paths().cache_wav(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x01\x00" * (16 * ms))
    return path


def a_diarized_video(db: Database) -> int:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 4_000), (4_000, 8_000)])
    host = add_label(db, video_id, "SPEAKER_00")
    guest = add_label(db, video_id, "SPEAKER_01")
    store.stamp_words(db, video_id, {word_ids[0]: host, word_ids[1]: guest})
    return video_id


# -- registration ----------------------------------------------------------


def test_every_command_is_registered_in_the_speakers_group() -> None:
    for name in GROUP:
        assert COMMANDS[name].group == "speakers"


def test_the_long_ones_are_marked_long_running() -> None:
    assert COMMANDS["speakers.diarize"].long_running is True
    assert COMMANDS["speakers.embed"].long_running is True
    assert COMMANDS["speakers.list"].long_running is False


def test_the_mapper_launcher_is_cli_only() -> None:
    # It opens the interactive surface, so it cannot be run from inside it.
    assert COMMANDS["speakers.map"].cli_only is True
    assert COMMANDS["speakers.list"].cli_only is False


def test_every_parameter_has_help_and_a_scalar_type() -> None:
    for name in GROUP:
        for param in COMMANDS[name].params:
            assert param.help
            assert param.type in (str, int, float, bool, Path)


def test_required_parameters_are_positional() -> None:
    link = {p.name: p for p in COMMANDS["speakers.link"].params}
    assert link["video"].default is REQUIRED
    assert link["video"].positional is True


# -- the roster ------------------------------------------------------------


def test_add_creates_a_speaker_and_reports_the_id(db: Database) -> None:
    result = resolve("speakers.add").handler(
        db, label="Host One", aliases="host, h1", notes=None
    )
    assert "Host One" in (result.message or "")
    assert store.find_speaker(db, "h1") is not None


def test_add_says_when_the_speaker_was_already_there(db: Database) -> None:
    resolve("speakers.add").handler(db, label="Host One", aliases="", notes=None)
    result = resolve("speakers.add").handler(db, label="Host One", aliases="", notes=None)
    assert "already" in (result.message or "").lower()


def test_add_warns_about_an_alias_another_speaker_owns(db: Database) -> None:
    resolve("speakers.add").handler(db, label="Host One", aliases="boss", notes=None)
    result = resolve("speakers.add").handler(db, label="Guest Two", aliases="boss", notes=None)
    assert "boss" in (result.message or "")


def test_list_returns_the_roster_as_rows_of_strings(db: Database) -> None:
    store.add_speaker(db, "Host One", aliases=("h1",))
    result = resolve("speakers.list").handler(db)
    assert result.columns[0] == "id"
    assert all(isinstance(cell, str) for row in result.rows for cell in row)
    assert result.rows[0][1] == "Host One"


def test_alias_appends_to_an_existing_speaker(db: Database) -> None:
    store.add_speaker(db, "Host One", aliases=("h1",))
    resolve("speakers.alias").handler(db, speaker="Host One", aliases="boss")
    assert store.find_speaker(db, "boss").label == "Host One"


def test_alias_on_an_unknown_speaker_raises(db: Database) -> None:
    with pytest.raises(NotFoundError):
        resolve("speakers.alias").handler(db, speaker="nobody", aliases="x")


def test_remove_deletes_the_person_and_keeps_their_labels(db: Database) -> None:
    video_id = a_diarized_video(db)
    store.add_speaker(db, "Host One", aliases=("h1",))
    resolve("speakers.link").handler(
        db, video=str(video_id), label="SPEAKER_00", speaker="h1"
    )
    result = resolve("speakers.remove").handler(db, speaker="h1", yes=False)
    assert "Host One" in (result.message or "")
    assert "1 local label" in (result.message or "")
    assert store.find_speaker(db, "Host One") is None
    # Both voices are still there, both unnamed.
    rows = store.label_rows(db, video_id)
    assert len(rows) == 2
    assert all(row.speaker_id is None for row in rows)


def test_remove_asks_before_undoing_a_lot_of_mapping(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    for index in range(C.SPEAKER_REMOVE_CONFIRM_LINKS):
        other = make_video(
            db, external_id=f"VIDEO_{index}", url=f"https://example.invalid/{index}"
        )
        label_id = add_label(db, other, "SPEAKER_00")
        store.link_video_speaker(db, label_id, speaker_id)

    with pytest.raises(RytpError, match="--yes"):
        resolve("speakers.remove").handler(db, speaker="Host One", yes=False)
    assert store.find_speaker(db, "Host One") is not None

    result = resolve("speakers.remove").handler(db, speaker="Host One", yes=True)
    assert f"{C.SPEAKER_REMOVE_CONFIRM_LINKS} local labels" in (result.message or "")
    assert store.find_speaker(db, "Host One") is None


def test_remove_and_unlink_describe_different_things(db: Database) -> None:
    # The names are one letter of intent apart; the summaries must not be.
    remove = COMMANDS["speakers.remove"].summary.lower()
    unlink = COMMANDS["speakers.unlink"].summary.lower()
    assert "roster" in remove and "label" in unlink
    assert remove != unlink


def test_remove_an_unknown_speaker_raises(db: Database) -> None:
    with pytest.raises(NotFoundError):
        resolve("speakers.remove").handler(db, speaker="nobody", yes=True)


# -- labels and linking ----------------------------------------------------


def test_labels_lists_a_videos_voices(db: Database) -> None:
    video_id = a_diarized_video(db)
    result = resolve("speakers.labels").handler(db, video=str(video_id))
    assert [row[0] for row in result.rows] == ["SPEAKER_00", "SPEAKER_01"]


def test_labels_shows_which_diarizer_produced_each_one(db: Database) -> None:
    video_id = a_diarized_video(db)
    result = resolve("speakers.labels").handler(db, video=str(video_id))
    assert result.columns[-1] == "diarizer"
    assert {row[-1] for row in result.rows} == {DIARIZER}


def test_labels_on_an_undiarized_video_says_so_rather_than_returning_nothing(
    db: Database,
) -> None:
    video_id = make_video(db)
    result = resolve("speakers.labels").handler(db, video=str(video_id))
    assert result.rows == ()
    assert "diariz" in (result.message or "").lower()


def test_link_names_a_local_label(db: Database) -> None:
    video_id = a_diarized_video(db)
    store.add_speaker(db, "Host One", aliases=("h1",))
    result = resolve("speakers.link").handler(
        db, video=str(video_id), label="SPEAKER_00", speaker="h1"
    )
    assert "Host One" in (result.message or "")
    assert store.find_label(db, video_id, "SPEAKER_00").speaker_label == "Host One"


def test_link_to_an_unknown_speaker_raises(db: Database) -> None:
    video_id = a_diarized_video(db)
    with pytest.raises(NotFoundError):
        resolve("speakers.link").handler(
            db, video=str(video_id), label="SPEAKER_00", speaker="nobody"
        )


def test_link_to_an_unknown_label_raises(db: Database) -> None:
    video_id = a_diarized_video(db)
    store.add_speaker(db, "Host One")
    with pytest.raises(NotFoundError):
        resolve("speakers.link").handler(
            db, video=str(video_id), label="SPEAKER_99", speaker="Host One"
        )


def test_unlink_puts_a_label_back_to_nobody(db: Database) -> None:
    video_id = a_diarized_video(db)
    store.add_speaker(db, "Host One")
    resolve("speakers.link").handler(
        db, video=str(video_id), label="SPEAKER_00", speaker="Host One"
    )
    resolve("speakers.unlink").handler(db, video=str(video_id), label="SPEAKER_00")
    assert store.find_label(db, video_id, "SPEAKER_00").speaker_id is None


# -- the two long ones -----------------------------------------------------


def test_diarize_runs_the_named_engine(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 900), (1_000, 1_900), (2_000, 2_900)])
    cache_a_wav(video_id)
    with registered(FakeDiarizer):
        result = resolve("speakers.diarize").handler(
            db, video=str(video_id), diarizer="fake-diarizer", embedder=""
        )
    assert "2 speakers" in (result.message or "")
    assert len(store.label_rows(db, video_id)) == 2


def test_diarize_can_embed_in_the_same_pass(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 4_000), (4_000, 8_000)])
    cache_a_wav(video_id)
    with registered(FakeDiarizer, FakeEmbedder):
        resolve("speakers.diarize").handler(
            db, video=str(video_id), diarizer="fake-diarizer", embedder="fake-embedder"
        )
    assert any(row.has_embedding for row in store.label_rows(db, video_id))


def test_diarize_without_a_cached_wav_says_which_step_makes_it(
    db: Database, data_dir: Path
) -> None:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 400)])
    with pytest.raises(RytpError, match="extract"):
        resolve("speakers.diarize").handler(db, video=str(video_id), diarizer="none", embedder="")


def test_embed_is_runnable_on_its_own(db: Database, data_dir: Path) -> None:
    video_id = a_diarized_video(db)
    cache_a_wav(video_id)
    with registered(FakeEmbedder):
        result = resolve("speakers.embed").handler(
            db, video=str(video_id), embedder="fake-embedder"
        )
    assert "embedded" in (result.message or "")
    assert all(row.has_embedding for row in store.label_rows(db, video_id))


def test_embed_with_no_embedder_configured_refuses_rather_than_doing_nothing(
    db: Database, data_dir: Path
) -> None:
    video_id = a_diarized_video(db)
    cache_a_wav(video_id)
    with pytest.raises(RytpError, match="embedder"):
        resolve("speakers.embed").handler(db, video=str(video_id), embedder="")


# -- the queue -------------------------------------------------------------


def test_enqueue_adds_exactly_one_diarize_job(db: Database, data_dir: Path) -> None:
    from rytp.jobs import queue as Q

    video_id = make_video(db)
    add_words(db, video_id, [(0, 400)])
    cache_a_wav(video_id)
    resolve("speakers.enqueue").handler(db, video=str(video_id), priority=5)
    jobs = Q.list_jobs(db)
    assert [(job.kind, job.state, job.priority) for job in jobs] == [
        ("diarize", "pending", 5)
    ]


# -- suggestions -----------------------------------------------------------


def test_suggest_reports_the_verdict_and_writes_nothing(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    other = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    other_words = add_words(db, other, [(0, 4_000)])
    other_label = add_label(db, other, "SPEAKER_00")
    store.stamp_words(db, other, dict.fromkeys(other_words, other_label))
    store.set_embedding(db, other_label, pack_embedding([1.0, 0.0, 0.0, 0.0]))
    store.link_video_speaker(db, other_label, speaker_id)

    video_id = a_diarized_video(db)
    label = store.find_label(db, video_id, "SPEAKER_00")
    store.set_embedding(db, label.video_speaker_id, pack_embedding([1.0, 0.0, 0.0, 0.0]))

    before = db.conn.total_changes
    result = resolve("speakers.suggest").handler(
        db, video=str(video_id), label="", limit=C.SPEAKER_SUGGEST_LIMIT
    )
    assert db.conn.total_changes == before
    assert "Host One" in {row[1] for row in result.rows}
    assert "match" in {row[3] for row in result.rows}


def test_suggest_says_plainly_when_there_is_nothing_to_go_on(db: Database) -> None:
    video_id = a_diarized_video(db)
    result = resolve("speakers.suggest").handler(db, video=str(video_id), label="", limit=5)
    assert result.rows == ()
    assert "embed" in (result.message or "").lower()


def test_suggest_names_a_diarizer_mismatch_rather_than_going_quiet(
    db: Database,
) -> None:
    # An empty list because two engines are in play must not look like an
    # empty list because nothing was embedded.
    speaker_id, _ = store.add_speaker(db, "Host One")
    other = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    other_words = add_words(db, other, [(0, 4_000)])
    other_label = add_label(db, other, "SPEAKER_00", engine="pyannote")
    store.stamp_words(db, other, dict.fromkeys(other_words, other_label))
    store.set_embedding(db, other_label, pack_embedding([1.0, 0.0, 0.0, 0.0]))
    store.link_video_speaker(db, other_label, speaker_id)

    video_id = a_diarized_video(db)          # labels come from DIARIZER
    label = store.find_label(db, video_id, "SPEAKER_00")
    store.set_embedding(db, label.video_speaker_id, pack_embedding([1.0, 0.0, 0.0, 0.0]))

    result = resolve("speakers.suggest").handler(db, video=str(video_id), label="", limit=5)
    assert result.rows == ()
    assert "diarizer" in (result.message or "")
    assert "pyannote" in (result.message or "")


# -- engines and the mapper launcher --------------------------------------


def test_engines_lists_both_registries(db: Database) -> None:
    result = resolve("speakers.engines").handler(db)
    names = {row[0] for row in result.rows}
    assert {"none", "pyannote", "redimnet"} <= names
    kinds = {row[1] for row in result.rows}
    assert kinds == {"diarizer", "embedder"}


def test_map_launches_the_mapper_for_that_video(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.commands import speakers as commands_speakers

    seen: list[int] = []
    monkeypatch.setattr(
        commands_speakers, "_run_mapper", lambda db_, video_id: seen.append(video_id)
    )
    video_id = a_diarized_video(db)
    resolve("speakers.map").handler(db, video=str(video_id))
    assert seen == [video_id]


def test_map_refuses_a_video_with_no_labels(db: Database) -> None:
    video_id = make_video(db)
    with pytest.raises(RytpError, match="diariz"):
        resolve("speakers.map").handler(db, video=str(video_id))
