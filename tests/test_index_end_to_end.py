"""Words in, searchable index and a readable transcript out.

Caption tier on purpose: Part 3's json3 ingest needs no model and no
binary, so this runs anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rytp import constants as C
from rytp.cli import build_app
from rytp.config import paths
from rytp.db import Database
from rytp.index.search import MatchTier, search
from rytp.index.utterances import index_video
from rytp.models import normalize_text, stem_text
from rytp.transcribe.captions import ingest_captions
from tests.test_index_utterances import make_video

LINES = (
    (0, "Добрый вечер дорогие друзья"),
    (4_000, "Сегодня у нас странное ощущение"),
    (9_000, "Ну что же поговорим об этом"),
)


def write_json3(path: Path) -> Path:
    """A minimal json3 caption track: one event per line, one seg per word."""
    events = [
        {
            "tStartMs": base,
            "segs": [
                {"utf8": token, "tOffsetMs": index * 400}
                for index, token in enumerate(line.split())
            ],
        }
        for base, line in LINES
    ]
    path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return path


@pytest.fixture()
def captioned(db: Database, data_dir: Path) -> int:
    video_id = make_video(db, "Evening")
    ingest_captions(db, video_id, write_json3(data_dir / "captions.json3"))
    return video_id


def test_captions_become_a_searchable_index_and_a_transcript(
    captioned: int, db: Database
) -> None:
    runner = CliRunner()
    app = build_app()

    built = runner.invoke(app, ["index", "build", "--video", str(captioned)])
    assert built.exit_code == 0, built.output
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] >= 3

    found = runner.invoke(app, ["search", "words", "добрый вечер"])
    assert found.exit_code == 0, found.output
    assert f"v{captioned}:0-1" in found.output
    assert "exact match" in found.output

    inflected = runner.invoke(app, ["search", "words", "ощущения"])
    assert inflected.exit_code == 0, inflected.output
    assert "stem match" in inflected.output

    # Caption words are searchable and never cuttable (contracts §3).
    uncuttable = runner.invoke(app, ["search", "words", "добрый вечер", "--cuttable"])
    assert uncuttable.exit_code == 0, uncuttable.output
    assert "no hits" in uncuttable.output

    missing = runner.invoke(app, ["search", "words", "совершенно другое"])
    assert missing.exit_code == 0, missing.output
    assert "no hits" in missing.output

    written = runner.invoke(app, ["transcript", "build", str(captioned)])
    assert written.exit_code == 0, written.output
    text = paths().transcript(captioned).read_text(encoding="utf-8")
    assert "# Evening" in text
    assert f"## [v{captioned}:" in text
    assert "- source: caption" in text


def test_the_cli_reports_a_bad_anchor_as_one_line_not_a_traceback(
    captioned: int,
) -> None:
    result = CliRunner().invoke(build_app(), ["search", "play", "not-an-anchor"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output


def test_reindexing_after_the_words_change_keeps_search_honest(
    captioned: int, db: Database, data_dir: Path
) -> None:
    """Contracts §4: utterances die with the words; Part 4 re-derives them."""
    index_video(db, captioned)
    assert len(search(db, "добрый вечер").hits) == 1
    with db.transaction():
        db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (captioned,))
        db.conn.execute("DELETE FROM words WHERE video_id = ?", (captioned,))
    assert search(db, "добрый вечер").hits == ()
    ingest_captions(db, captioned, data_dir / "captions.json3")
    index_video(db, captioned)
    assert len(search(db, "добрый вечер").hits) == 1


def bulk_words(db: Database, video_id: int, lines: list[str]) -> None:
    """Insert one words row per token of every line, in one transaction."""
    rows = []
    ordinal = 0
    clock = 0
    for line in lines:
        for token in line.split():
            normalized = normalize_text(token)
            rows.append(
                (
                    video_id,
                    ordinal,
                    clock,
                    clock + 250,
                    token,
                    normalized,
                    stem_text(normalized),
                    "aligned",
                    "fake",
                )
            )
            ordinal += 1
            clock += 300
    with db.transaction():
        db.conn.executemany(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text,"
            " normalized_text, stem, source, engine)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def test_the_index_still_answers_at_a_realistic_size(db: Database) -> None:
    """Four rows prove nothing about an FTS index. Build a few thousand
    words across several videos and find one phrase in the middle."""
    needle = "совершенно неповторимая фраза"
    for video_index in range(5):
        video_id = make_video(db, f"Bulk{video_index}")
        lines = [
            needle if (video_index == 3 and chunk == 100) else "и что то ещё"
            for chunk in range(200)
        ]
        bulk_words(db, video_id, lines)
        index_video(db, video_id)

    result = search(db, needle)
    assert result.tier is MatchTier.EXACT
    assert len(result.hits) == 1
    assert result.hits[0].video_title == "Bulk3"
    # And the stem tier reaches it through an inflection.
    assert search(db, "совершенно неповторимые фразы").tier is MatchTier.STEM


def test_a_query_of_only_common_words_stays_bounded(db: Database) -> None:
    """The walk anchors on the rarest token and stops at the anchor limit,
    so a query of nothing but function words cannot walk the corpus."""
    for video_index in range(3):
        video_id = make_video(db, f"Common{video_index}")
        bulk_words(db, video_id, ["и " * 100] * 3)
        index_video(db, video_id)
    result = search(db, "и и", limit=5)
    assert len(result.hits) <= 5
    assert C.SEARCH_WALK_ANCHOR_LIMIT > 0
