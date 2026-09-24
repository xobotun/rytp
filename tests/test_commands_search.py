"""The Part 4 commands (contracts §5, design §7, §10)."""

from __future__ import annotations

from pathlib import Path

import pytest

import rytp.commands.search  # noqa: F401 - importing is what registers them
from rytp import constants as C
from rytp.commands import COMMANDS, CommandResult, resolve
from rytp.db import Database
from rytp.index import export
from rytp.index.utterances import index_video
from rytp.models import NotFoundError, RytpError
from tests.test_index_search import add_words, corpus, name_speaker
from tests.test_index_utterances import make_speaker, make_video

NAMES = (
    "index.build",
    "index.drop",
    "search.words",
    "search.play",
    "search.export",
    "transcript.build",
)


@pytest.fixture(autouse=True)
def never_spawn_a_player(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Same guard as the export tests: nothing here reaches a real binary."""
    recorded: list[list[str]] = []

    def fake_run(cmd: list[str]) -> object:
        import subprocess

        recorded.append(list(cmd))
        if cmd[-1].endswith(".wav"):
            Path(cmd[-1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(export, "_run", fake_run)
    monkeypatch.setattr(export, "_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(export, "_ffplay_binary", lambda: "/usr/bin/ffplay")
    return recorded


def cached_wav(video_id: int) -> Path:
    import struct
    import wave

    from rytp.config import paths

    path = paths().cache_wav(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(C.AUDIO_CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(C.AUDIO_SAMPLE_RATE_HZ)
        handle.writeframes(struct.pack("<h", 0) * C.AUDIO_SAMPLE_RATE_HZ)
    return path


# --- registration ----------------------------------------------------


def test_every_part_four_command_is_registered() -> None:
    for name in NAMES:
        assert name in COMMANDS, name
        assert COMMANDS[name].group == name.split(".")[0]
        assert COMMANDS[name].summary


def test_the_blocking_commands_are_flagged_long_running() -> None:
    """Design §10: the TUI must not run these inline."""
    assert resolve("index.build").long_running is True
    assert resolve("search.play").long_running is True
    assert resolve("search.export").long_running is True
    assert resolve("search.words").long_running is False
    assert resolve("transcript.build").long_running is False


def test_every_handler_returns_a_command_result(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    assert isinstance(resolve("search.words").handler(db, query="добрый вечер"), CommandResult)
    assert isinstance(resolve("index.build").handler(db), CommandResult)
    assert isinstance(
        resolve("transcript.build").handler(db, video=str(video_id)), CommandResult
    )


# --- index.build -----------------------------------------------------


def test_index_build_indexes_one_video(db: Database) -> None:
    video_id = make_video(db)
    add_words(db, video_id, "Добрый вечер")
    result = resolve("index.build").handler(db, video=str(video_id))
    assert result.rows == ((str(video_id), "1"),)
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 1


def test_index_build_with_no_video_indexes_everything_that_needs_it(
    db: Database,
) -> None:
    first = make_video(db, "First")
    second = make_video(db, "Second")
    add_words(db, first, "Добрый вечер")
    add_words(db, second, "Здравствуйте друзья")
    index_video(db, first)
    result = resolve("index.build").handler(db)
    assert result.rows == ((str(second), "1"),)


def test_index_build_says_so_when_there_is_nothing_to_do(db: Database) -> None:
    result = resolve("index.build").handler(db)
    assert "nothing" in (result.message or "")


def test_index_build_can_enqueue_instead_of_working(db: Database) -> None:
    """Part 2's ingest chain stops at extract_wav; this is what creates the
    index jobs once words exist."""
    video_id = make_video(db)
    add_words(db, video_id, "Добрый вечер")
    result = resolve("index.build").handler(db, enqueue=True)
    assert db.conn.execute(
        "SELECT state FROM jobs WHERE kind = 'index' AND target_id = ?", (video_id,)
    ).fetchone()["state"] == "pending"
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 0
    assert "queued" in (result.message or "")


def test_index_build_rejects_an_unknown_video(db: Database) -> None:
    with pytest.raises(NotFoundError):
        resolve("index.build").handler(db, video=str(404))


# --- search.words ----------------------------------------------------


def test_search_words_prints_an_anchor_per_hit(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер дорогие друзья")
    result = resolve("search.words").handler(db, query="дорогие друзья")
    assert result.columns == ("anchor", "video", "time", "speaker", "tier", "text")
    assert result.rows[0][0] == f"v{video_id}:2-3"
    assert result.rows[0][2] == "00:00:00.600"


def test_search_words_names_the_tier_that_answered(db: Database) -> None:
    """The owner wants to know when a result is an inflection."""
    corpus(db, "Сегодня было странное ощущение")
    exact = resolve("search.words").handler(db, query="ощущение")
    stemmed = resolve("search.words").handler(db, query="ощущения")
    assert "exact" in (exact.message or "")
    assert "stem" in (stemmed.message or "")


def test_search_words_flags_a_hit_that_spans_a_boundary(db: Database) -> None:
    from tests.test_index_search import split_corpus

    split_corpus(db)
    result = resolve("search.words").handler(db, query="добрый вечер")
    assert "boundary" in (result.message or "")


def test_search_words_reports_no_hits_without_raising(db: Database) -> None:
    corpus(db, "Добрый вечер")
    result = resolve("search.words").handler(db, query="совершенно другое")
    assert result.rows == ()
    assert "no hits" in (result.message or "")


def test_search_words_names_the_transcript_tier_of_each_hit(db: Database) -> None:
    """Contracts §3 has three, and `timed` is not `caption`.

    A `timed` hit is real text that cannot be cut *yet* — the fix is
    `rytp transcribe align`. A caption hit needs transcribing. Printing
    `no` for both would hide the difference that tells the user what to do.
    """
    corpus(db, "Добрый вечер", source="caption", title="Captioned")
    corpus(db, "Добрый вечер", source="timed", title="Timed")
    corpus(db, "Добрый вечер", source="aligned", title="Aligned")
    rows = resolve("search.words").handler(db, query="добрый вечер").rows
    assert sorted(row[4] for row in rows) == ["aligned", "caption", "timed"]


def test_search_words_says_how_many_hits_are_not_cuttable_yet(db: Database) -> None:
    corpus(db, "Добрый вечер", source="timed")
    result = resolve("search.words").handler(db, query="добрый вечер")
    assert "not cuttable yet" in (result.message or "")


def test_search_words_names_the_speaker_filter_it_applied(db: Database) -> None:
    """`SpeakerFilter.description` exists so this module does not re-derive
    the wording (contracts §5)."""
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    name_speaker(db, host, "Ведущий")
    add_words(db, video_id, "Добрый вечер", speaker_id=host)
    index_video(db, video_id)
    result = resolve("search.words").handler(
        db, query="добрый вечер", speaker="Ведущий"
    )
    assert "Ведущий" in (result.message or "")


def test_search_words_passes_the_filters_through(db: Database) -> None:
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    name_speaker(db, host, "Ведущий")
    add_words(db, video_id, "Добрый вечер", speaker_id=host)
    index_video(db, video_id)
    corpus(db, "Добрый вечер", title="Anonymous")
    words = resolve("search.words").handler
    assert len(words(db, query="добрый вечер").rows) == 2
    assert len(words(db, query="добрый вечер", speaker="Ведущий").rows) == 1
    assert len(words(db, query="добрый вечер", limit=1).rows) == 1


def test_speaker_means_a_person_and_never_a_diarizer_label(db: Database) -> None:
    """Contracts §5: `--speaker` matches the roster, never `SPEAKER_00`.

    The same flag had two meanings across two commands before this rule
    existed, on the query the whole product is for.
    """
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    name_speaker(db, host, "Ведущий")
    add_words(db, video_id, "Добрый вечер", speaker_id=host)
    index_video(db, video_id)
    with pytest.raises(RytpError):
        resolve("search.words").handler(db, query="добрый вечер", speaker="SPEAKER_00")


def test_an_unknown_person_is_an_error_not_an_empty_table(db: Database) -> None:
    """A silent empty result reads as "he never said it", which is a lie."""
    corpus(db, "Добрый вечер")
    with pytest.raises(RytpError):
        resolve("search.words").handler(db, query="добрый вечер", speaker="Никто")


def test_a_local_speaker_label_requires_a_video(db: Database) -> None:
    """A bare SPEAKER_00 would match the first-detected voice of every
    diarized video — not a person, and not an answer (contracts §5)."""
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    add_words(db, video_id, "Добрый вечер", speaker_id=host)
    index_video(db, video_id)
    with pytest.raises(RytpError):
        resolve("search.words").handler(
            db, query="добрый вечер", video_local_speaker="SPEAKER_00"
        )
    scoped = resolve("search.words").handler(
        db, query="добрый вечер", video_local_speaker="SPEAKER_00", video=str(video_id)
    )
    assert len(scoped.rows) == 1


def test_both_speaker_flags_reach_the_pinned_shared_resolver(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Part 4 adds no resolution logic; it calls the one contracts §5 pins,
    with the keywords that signature fixes."""
    import rytp.commands.search as search_module
    from rytp.commands import resolve_speaker_filter

    calls: list[dict[str, object]] = []

    def spy(db_: Database, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return resolve_speaker_filter(db_, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(search_module, "resolve_speaker_filter", spy)
    corpus(db, "Добрый вечер")
    resolve("search.words").handler(db, query="добрый вечер")
    assert calls == [{"speaker": None, "video_local_speaker": None, "video_id": None}]


def test_search_words_truncates_a_long_sentence(db: Database) -> None:
    corpus(db, "слово " * 30 + "конец")
    result = resolve("search.words").handler(db, query="конец")
    assert len(result.rows[0][5]) <= C.SEARCH_TEXT_TRUNCATE_CHARS + 1


def test_search_words_rejects_an_empty_query(db: Database) -> None:
    with pytest.raises(RytpError):
        resolve("search.words").handler(db, query="   ")


# --- search.play and search.export -----------------------------------


def test_search_play_plays_the_anchor(
    db: Database, never_spawn_a_player: list[list[str]]
) -> None:
    video_id = corpus(db, "Добрый вечер")
    cached_wav(video_id)
    result = resolve("search.play").handler(db, anchor=f"v{video_id}:0-1")
    assert never_spawn_a_player[0][0] == "/usr/bin/ffplay"
    assert "Добрый вечер" in (result.message or "")


def test_search_play_rejects_an_anchor_that_is_not_one(db: Database) -> None:
    with pytest.raises(RytpError):
        resolve("search.play").handler(db, anchor="добрый вечер")


def test_search_export_writes_one_file_per_hit(db: Database, data_dir: Path) -> None:
    first = corpus(db, "Добрый вечер", title="First")
    second = corpus(db, "Добрый вечер", title="Second")
    cached_wav(first)
    cached_wav(second)
    out_dir = data_dir / "clips"
    result = resolve("search.export").handler(
        db, query="добрый вечер", out_dir=out_dir
    )
    assert len(result.rows) == 2
    assert sorted(p.name for p in out_dir.iterdir()) == [
        f"v{first}_0-1.wav",
        f"v{second}_0-1.wav",
    ]


def test_exported_filenames_have_no_colon(db: Database, data_dir: Path) -> None:
    """Windows path components cannot contain ':'."""
    video_id = corpus(db, "Добрый вечер")
    cached_wav(video_id)
    resolve("search.export").handler(db, query="добрый вечер", out_dir=data_dir / "clips")
    assert all(":" not in p.name for p in (data_dir / "clips").iterdir())


def test_search_export_raises_when_there_is_nothing_to_export(
    db: Database, data_dir: Path
) -> None:
    corpus(db, "Добрый вечер")
    with pytest.raises(NotFoundError):
        resolve("search.export").handler(
            db, query="совершенно другое", out_dir=data_dir / "clips"
        )


# --- index.drop (contracts §5, Deletion) -----------------------------


def test_index_drop_removes_the_utterances_and_says_how_many(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    result = resolve("index.drop").handler(db, video=str(video_id))
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 0
    assert "1" in (result.message or "")
    assert "index build" in (result.message or "")


def test_index_drop_makes_the_video_unsearchable_and_build_brings_it_back(
    db: Database,
) -> None:
    video_id = corpus(db, "Добрый вечер")
    resolve("index.drop").handler(db, video=str(video_id))
    assert resolve("search.words").handler(db, query="добрый вечер").rows == ()
    resolve("index.build").handler(db, video=str(video_id))
    assert len(resolve("search.words").handler(db, query="добрый вечер").rows) == 1


def test_index_drop_leaves_no_stale_fts_row(db: Database) -> None:
    """Through the surface, because this is the quietest way to break."""
    video_id = corpus(db, "Добрый вечер")
    resolve("index.drop").handler(db, video=str(video_id))
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "добрый вечер"',),
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "INSERT INTO utterances_fts(utterances_fts) VALUES('integrity-check')"
    ).fetchall() == []


def test_index_drop_needs_no_confirmation_because_it_deletes_no_files() -> None:
    """Contracts §5: `--dry-run` and `--yes` are required only of commands
    that delete files. Utterances are derived rows."""
    names = {param.name for param in resolve("index.drop").params}
    assert names == {"video"}


def test_index_drop_rejects_an_unknown_video(db: Database) -> None:
    with pytest.raises(NotFoundError):
        resolve("index.drop").handler(db, video=str(404))


# --- health checks (contracts §5) ------------------------------------


def test_both_part_four_checks_are_registered() -> None:
    from rytp.commands import HEALTH_CHECKS

    assert {"ffplay", "fts5"} <= set(HEALTH_CHECKS)
    for name in ("ffplay", "fts5"):
        assert HEALTH_CHECKS[name].summary


def test_ffplay_is_advisory_and_fts5_is_required() -> None:
    """Contracts §5: `required`, not `ok`, decides whether doctor fails.

    Search is inert without FTS5, so that one should take the exit code
    down. Export works without ffplay, so that one should not.
    """
    from rytp.commands import HEALTH_CHECKS

    assert HEALTH_CHECKS["ffplay"].required is False
    assert HEALTH_CHECKS["fts5"].required is True


def test_the_fts5_check_passes_on_a_working_sqlite(db: Database) -> None:
    from rytp.commands import HEALTH_CHECKS

    result = HEALTH_CHECKS["fts5"].run(db)
    assert result.ok is True
    assert "FTS5" in result.detail


def test_the_ffplay_check_reports_a_missing_binary_truthfully(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never fatal: export still works, only playback does not."""
    from rytp.commands import HEALTH_CHECKS

    monkeypatch.setattr(export, "_ffplay_binary", lambda: None)
    result = HEALTH_CHECKS["ffplay"].run(db)
    # ok tells the truth about what was found; required=False on the
    # registration is what makes it non-fatal (contracts §5).
    assert result.ok is False
    assert HEALTH_CHECKS["ffplay"].required is False
    assert "export" in result.detail
    assert result.remedy and "ffmpeg" in result.remedy


def test_the_ffplay_check_passes_when_it_is_there(db: Database) -> None:
    from rytp.commands import HEALTH_CHECKS

    assert HEALTH_CHECKS["ffplay"].run(db).ok is True


def test_a_check_never_raises(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    """Contracts §5: a check never raises and never blocks."""
    from rytp.commands import HEALTH_CHECKS

    monkeypatch.setattr(export, "_ffplay_binary", lambda: None)
    for name in ("ffplay", "fts5"):
        result = HEALTH_CHECKS[name].run(db)
        assert isinstance(result.ok, bool)
        assert result.detail


# --- transcript.build ------------------------------------------------


def test_transcript_build_writes_the_file_and_names_it(db: Database) -> None:
    from rytp.config import paths

    video_id = corpus(db, "Добрый вечер")
    result = resolve("transcript.build").handler(db, video=str(video_id))
    path = paths().transcript(video_id)
    assert path.exists()
    assert str(path) in (result.message or "")


def test_transcript_build_tells_you_to_index_first(db: Database) -> None:
    video_id = make_video(db)
    add_words(db, video_id, "Добрый вечер")
    with pytest.raises(RytpError, match="index"):
        resolve("transcript.build").handler(db, video=str(video_id))
