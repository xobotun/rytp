"""Playback and audio export (design §7). No test spawns a player."""

from __future__ import annotations

import itertools
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.config import paths
from rytp.db import Database
from rytp.db.queries import insert_asset
from rytp.index import export
from rytp.index.export import (
    ClipError,
    MediaToolMissing,
    build_clip_command,
    build_play_command,
    clip_source,
    export_clip,
    padded,
    play_clip,
    render_transcript,
    timestamp,
    transcript_blocks,
    write_transcript,
)
from rytp.index.search import span_for_anchor
from rytp.index.utterances import index_video
from rytp.models import NotFoundError, RytpError
from tests.test_index_search import add_words, corpus, name_speaker
from tests.test_index_utterances import make_speaker, make_video


@pytest.fixture(autouse=True)
def never_spawn_a_player(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> list[list[str]]:
    """Replace the one function that runs a subprocess with a recorder.

    Autouse and opt-out rather than opt-in: a new test that forgets to
    patch the seam records a command instead of opening a window.
    """
    recorded: list[list[str]] = []
    if request.node.get_closest_marker("real_ffmpeg") is not None:
        return recorded

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        recorded.append(list(cmd))
        if cmd[-1].endswith(".wav"):
            # Stand in for the file ffmpeg would have written, so callers
            # that check for the output still see it.
            Path(cmd[-1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(export, "_run", fake_run)
    monkeypatch.setattr(export, "_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(export, "_ffplay_binary", lambda: "/usr/bin/ffplay")
    return recorded


def write_silence(path: Path, seconds: float = 1.0) -> Path:
    """A real 16 kHz mono PCM WAV, using nothing but the stdlib."""
    frames = int(C.AUDIO_SAMPLE_RATE_HZ * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(C.AUDIO_CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(C.AUDIO_SAMPLE_RATE_HZ)
        handle.writeframes(struct.pack("<h", 0) * frames)
    return path


def cached_wav(video_id: int) -> Path:
    path = paths().cache_wav(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_silence(path)


# --- command shape ---------------------------------------------------


def test_the_clip_command_seeks_on_the_input_side() -> None:
    cmd = build_clip_command("ffmpeg", Path("in.wav"), Path("out.wav"), 1000, 1500)
    assert cmd.index("-ss") < cmd.index("-i")


def test_the_clip_command_uses_a_duration_not_an_end_timestamp() -> None:
    """`-to` means an absolute output time, which is not what we want once
    `-ss` has moved the origin."""
    cmd = build_clip_command("ffmpeg", Path("in.wav"), Path("out.wav"), 1000, 1500)
    assert "-to" not in cmd
    assert cmd[cmd.index("-ss") + 1] == "1.000"
    assert cmd[cmd.index("-t") + 1] == "0.500"


def test_the_clip_command_writes_the_project_audio_format() -> None:
    cmd = build_clip_command("ffmpeg", Path("in.wav"), Path("out.wav"), 0, 1000)
    assert cmd[cmd.index("-ac") + 1] == str(C.AUDIO_CHANNELS)
    assert cmd[cmd.index("-ar") + 1] == str(C.AUDIO_SAMPLE_RATE_HZ)
    assert cmd[-1] == "out.wav"
    assert "-vn" in cmd


def test_the_play_command_is_headless_and_exits_on_its_own() -> None:
    cmd = build_play_command("ffplay", Path("in.wav"), 2000, 2400)
    assert "-nodisp" in cmd
    assert "-autoexit" in cmd
    assert cmd[cmd.index("-ss") + 1] == "2.000"
    assert cmd[cmd.index("-t") + 1] == "0.400"


def test_a_zero_length_span_still_asks_for_a_positive_duration() -> None:
    cmd = build_clip_command("ffmpeg", Path("in.wav"), Path("out.wav"), 500, 500)
    assert float(cmd[cmd.index("-t") + 1]) > 0


def test_padding_widens_the_span_but_never_seeks_before_zero() -> None:
    assert padded(1000, 2000, 150) == (850, 2150)
    assert padded(50, 200, 150) == (0, 350)
    assert padded(1000, 2000, 0) == (1000, 2000)


# --- where the audio comes from --------------------------------------


def test_clip_source_prefers_the_cached_wav(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    wav = cached_wav(video_id)
    insert_asset(db, video_id=video_id, role="audio", path=str(data_dir / "other.m4a"))
    assert clip_source(db, video_id) == wav


def test_clip_source_falls_back_to_the_audio_asset(db: Database, data_dir: Path) -> None:
    """Contracts §7 makes the WAV a prunable cache; a prune must not break
    playback."""
    video_id = make_video(db)
    audio = write_silence(data_dir / "audio.wav")
    insert_asset(db, video_id=video_id, role="audio", path=str(audio))
    assert clip_source(db, video_id) == audio


def test_clip_source_falls_back_to_a_container(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    container = write_silence(data_dir / "whole.wav")
    insert_asset(db, video_id=video_id, role="container", path=str(container))
    assert clip_source(db, video_id) == container


def test_clip_source_raises_when_nothing_is_on_disk(db: Database) -> None:
    video_id = make_video(db)
    with pytest.raises(ClipError, match="no audio"):
        clip_source(db, video_id)


# --- running it ------------------------------------------------------


def test_export_clip_runs_ffmpeg_and_returns_the_path(
    db: Database, data_dir: Path, never_spawn_a_player: list[list[str]]
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    out = data_dir / "clips" / "hit.wav"
    assert export_clip(db, video_id, 1000, 1500, out) == out
    assert len(never_spawn_a_player) == 1
    assert never_spawn_a_player[0][0] == "/usr/bin/ffmpeg"


def test_export_clip_creates_the_output_directory(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    out = data_dir / "deep" / "nested" / "hit.wav"
    export_clip(db, video_id, 0, 500, out)
    assert out.parent.is_dir()


def test_export_clip_applies_the_padding(
    db: Database, data_dir: Path, never_spawn_a_player: list[list[str]]
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    export_clip(db, video_id, 1000, 1500, data_dir / "hit.wav", pad_ms=200)
    cmd = never_spawn_a_player[0]
    assert cmd[cmd.index("-ss") + 1] == "0.800"
    assert cmd[cmd.index("-t") + 1] == "0.900"


def test_export_clip_reports_a_missing_binary_in_one_line(
    db: Database, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    monkeypatch.setattr(export, "_ffmpeg_binary", lambda: None)
    with pytest.raises(MediaToolMissing, match="ffmpeg"):
        export_clip(db, video_id, 0, 500, data_dir / "hit.wav")


def test_export_clip_reports_an_ffmpeg_failure(
    db: Database, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)

    def failing(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, "", "Invalid data found")

    monkeypatch.setattr(export, "_run", failing)
    with pytest.raises(ClipError, match="Invalid data found"):
        export_clip(db, video_id, 0, 500, data_dir / "hit.wav")


def test_export_clip_maps_a_vanished_binary_to_a_domain_error(
    db: Database, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """which() said yes, exec() said no. One line, not a traceback."""
    video_id = make_video(db)
    cached_wav(video_id)

    def gone(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(export, "_run", gone)
    with pytest.raises(MediaToolMissing):
        export_clip(db, video_id, 0, 500, data_dir / "hit.wav")


def test_play_clip_invokes_ffplay_with_the_span(
    db: Database, never_spawn_a_player: list[list[str]]
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    play_clip(db, video_id, 2000, 2400, pad_ms=0)
    assert len(never_spawn_a_player) == 1
    cmd = never_spawn_a_player[0]
    assert cmd[0] == "/usr/bin/ffplay"
    assert cmd[cmd.index("-t") + 1] == "0.400"


def test_play_clip_reports_a_missing_ffplay(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    monkeypatch.setattr(export, "_ffplay_binary", lambda: None)
    with pytest.raises(MediaToolMissing, match="ffplay"):
        play_clip(db, video_id, 0, 500)


@pytest.mark.real_ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_a_real_export_produces_a_clip_of_the_right_length(
    db: Database, data_dir: Path
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    out = data_dir / "real.wav"
    export_clip(db, video_id, 200, 400, out, pad_ms=0)
    with wave.open(str(out), "rb") as handle:
        seconds = handle.getnframes() / handle.getframerate()
    assert 0.15 < seconds < 0.25


# --- markdown transcripts --------------------------------------------


def test_timestamps_are_hours_minutes_seconds_milliseconds() -> None:
    assert timestamp(0) == "00:00:00.000"
    assert timestamp(62_345) == "00:01:02.345"
    assert timestamp(3_723_004) == "01:02:03.004"


def test_there_is_one_block_per_utterance(db: Database) -> None:
    video_id = make_video(db, "Evening")
    ordinal, clock = add_words(db, video_id, "Добрый вечер")
    add_words(
        db,
        video_id,
        "Сегодня поговорим",
        first_ord=ordinal,
        start_ms=clock + C.UTTERANCE_SILENCE_GAP_MS,
    )
    index_video(db, video_id)
    blocks = transcript_blocks(db, video_id)
    assert [block.text for block in blocks] == ["Добрый вечер", "Сегодня поговорим"]


def test_every_block_carries_an_anchor_that_resolves_back_to_words(
    db: Database,
) -> None:
    """The whole AI-access story: read a block, point at the database."""
    video_id = corpus(db, "Добрый вечер дорогие друзья")
    block = transcript_blocks(db, video_id)[0]
    assert block.anchor == f"v{video_id}:0-3"
    span = span_for_anchor(db, block.anchor)
    assert span.text == "Добрый вечер дорогие друзья"


def test_a_block_anchor_is_the_same_string_search_prints(db: Database) -> None:
    from rytp.index.search import search

    video_id = corpus(db, "Добрый вечер")
    assert transcript_blocks(db, video_id)[0].anchor == search(
        db, "добрый вечер"
    ).hits[0].anchor


def test_the_block_speaker_prefers_the_roster_label(db: Database) -> None:
    video_id = make_video(db)
    local = make_speaker(db, video_id, "SPEAKER_00")
    name_speaker(db, local, "Ведущий")
    add_words(db, video_id, "Добрый вечер", speaker_id=local)
    index_video(db, video_id)
    assert transcript_blocks(db, video_id)[0].speaker == "Ведущий"


def test_an_unmapped_label_shows_the_raw_diarizer_label(db: Database) -> None:
    video_id = make_video(db)
    local = make_speaker(db, video_id, "SPEAKER_00")
    add_words(db, video_id, "Добрый вечер", speaker_id=local)
    index_video(db, video_id)
    assert transcript_blocks(db, video_id)[0].speaker == "SPEAKER_00"


def test_an_undiarized_block_shows_the_null_cell(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    assert transcript_blocks(db, video_id)[0].speaker == C.NULL_CELL


# -- entry 37: splitting a block into shorter rows, at a word boundary ------


def test_max_chars_none_is_unsplit_and_unchanged(db: Database) -> None:
    """The default (`transcript.build`'s durable file) is exactly the old
    one-block-per-utterance shape — splitting is opt-in via `max_chars`."""
    words = ["слово"] * 20 + ["конец"]
    video_id = corpus(db, " ".join(words))
    blocks = transcript_blocks(db, video_id)
    assert len(blocks) == 1
    assert blocks[0].text == " ".join(words)
    assert blocks[0].anchor == f"v{video_id}:0-20"


def test_a_long_block_splits_into_several_rows_at_word_boundaries(db: Database) -> None:
    words = ["слово"] * 20 + ["конец"]
    video_id = corpus(db, " ".join(words))
    blocks = transcript_blocks(db, video_id, max_chars=20)
    assert len(blocks) > 1
    for block in blocks:
        assert len(block.text) <= 20
        # Never a word cut in half: every row is whole words, space-joined.
        assert all(word in words for word in block.text.split())
    # Reassembling every row's words, in order, reproduces the utterance.
    assert " ".join(block.text for block in blocks) == " ".join(words)


def test_a_word_longer_than_max_chars_is_its_own_row(db: Database) -> None:
    """Never split a word, even one over the limit on its own (entry 37)."""
    video_id = corpus(db, "тут оченьдлинноеслововотакоечтобывышлозалимит еще")
    blocks = transcript_blocks(db, video_id, max_chars=5)
    texts = [block.text for block in blocks]
    assert "оченьдлинноеслововотакоечтобывышлозалимит" in texts


def test_splitting_gives_every_row_its_own_anchor(db: Database) -> None:
    """Entry 37's second decision: sub-anchors, not one anchor repeated.
    `search play` resolves an anchor purely by word-ordinal range
    (`span_for_anchor`, against `words`, never `utterances`), so a split
    row's own range was always resolvable — this is what starts handing
    one out per row instead of the whole utterance's."""
    words = ["слово"] * 20 + ["конец"]
    video_id = corpus(db, " ".join(words))
    blocks = transcript_blocks(db, video_id, max_chars=20)
    anchors = [block.anchor for block in blocks]
    assert len(anchors) == len(set(anchors))
    for block in blocks:
        span = span_for_anchor(db, block.anchor)
        assert span.start_ms == block.start_ms
        assert span.end_ms == block.end_ms
        assert span.text == block.text


def test_a_caption_tier_split_rows_end_falls_back_to_the_next_words_start(
    db: Database,
) -> None:
    """Caption words carry no `end_ms` (contracts §3). A split row that is
    not the utterance's last row ends where the next row starts — the same
    fallback `implied_end_ms` uses for a whole utterance, scoped to one."""
    words = ["слово"] * 10 + ["конец"]
    video_id = corpus(db, " ".join(words), source="caption")
    whole = transcript_blocks(db, video_id)[0]
    split = transcript_blocks(db, video_id, max_chars=20)
    assert len(split) > 1
    for row, next_row in itertools.pairwise(split):
        assert row.end_ms == next_row.start_ms
    # The last row's end is the *utterance's* own end — already computed
    # with the next utterance (or the caption fallback cap) in mind, which
    # a purely local, in-utterance fallback cannot see.
    assert split[-1].end_ms == whole.end_ms


def test_the_rendered_document_has_a_title_a_header_and_the_blocks(
    db: Database,
) -> None:
    video_id = corpus(db, "Добрый вечер", title="Evening")
    text = render_transcript(db, video_id)
    assert text.startswith("# Evening\n")
    assert f"- video: {video_id}" in text
    assert "- source: aligned" in text
    assert f"## [v{video_id}:0-1]" in text
    assert "00:00:00.000" in text
    assert "Добрый вечер" in text


def test_the_header_names_the_tier_so_cuttability_is_visible(db: Database) -> None:
    """Contracts §3's three tiers, each named as itself."""
    captioned = corpus(db, "Добрый вечер", source="caption", title="Captioned")
    timed = corpus(db, "Добрый вечер", source="timed", title="Timed")
    aligned = corpus(db, "Добрый вечер", source="aligned", title="Aligned")
    assert "- source: caption" in render_transcript(db, captioned)
    assert "- source: timed" in render_transcript(db, timed)
    assert "- source: aligned" in render_transcript(db, aligned)


def test_a_mixed_tier_video_says_so(db: Database) -> None:
    video_id = make_video(db, "Mixed")
    ordinal, clock = add_words(db, video_id, "Добрый")
    add_words(db, video_id, "вечер", first_ord=ordinal, start_ms=clock, source="caption")
    index_video(db, video_id)
    assert "- source: mixed" in render_transcript(db, video_id)


def test_the_document_says_it_is_regenerable(db: Database) -> None:
    """Design §3: markdown transcripts are output, never input."""
    video_id = corpus(db, "Добрый вечер")
    assert "regenerable" in render_transcript(db, video_id).lower()


def test_write_transcript_lands_in_the_data_tree(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    path = write_transcript(db, video_id)
    assert path == paths().transcript(video_id)
    assert path.read_text(encoding="utf-8").startswith("# ")


def test_write_transcript_uses_unix_newlines_on_every_platform(db: Database) -> None:
    """The file is committed to nothing and read by agents; CRLF would be
    noise, and this project's other output is \\n."""
    video_id = corpus(db, "Добрый вечер")
    raw = write_transcript(db, video_id).read_bytes()
    assert b"\r\n" not in raw


def test_rebuilding_overwrites_rather_than_appends(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    first = write_transcript(db, video_id).read_text(encoding="utf-8")
    second = write_transcript(db, video_id).read_text(encoding="utf-8")
    assert first.count("## [") == second.count("## [") == 1


def test_deleting_the_file_is_safe(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    path = write_transcript(db, video_id)
    path.unlink()
    assert write_transcript(db, video_id).exists()


def test_a_video_with_no_utterances_tells_you_to_index_first(db: Database) -> None:
    video_id = make_video(db)
    add_words(db, video_id, "Добрый вечер")
    with pytest.raises(RytpError, match="index"):
        render_transcript(db, video_id)


def test_an_unknown_video_raises_not_found(db: Database) -> None:
    with pytest.raises(NotFoundError):
        render_transcript(db, 404)
