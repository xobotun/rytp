"""Same input, same bytes — and a walk whose cost is the target, not the corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp.assemble import AssembleControls, assemble_target, dumps_cutlist, write_cutlist
from rytp.assemble.match import MatchFilters, occurrence_query
from rytp.db import Database
from tests.assembly_corpus import add_video, add_words

CREATED = "2026-09-21T09:00:00+00:00"
TARGET = "мы все понимаем что это неизбежно"


#: Caption videos per aligned video. Design §6 pulls captions for the whole
#: catalogue and aligns only what you mean to cut from, so a realistic
#: corpus is mostly uncuttable — and a fixture that is not cannot show
#: whether the hot lookup scales with the cuttable part.
CAPTION_RATIO = 9


def build(db: Database, aligned_videos: int) -> None:
    """A caption-heavy corpus whose aligned minority says the target.

    Every video says the same words. Only the aligned ones are eligible,
    so if the occurrence lookup ever scans by token before filtering by
    tier, its cost rises with CAPTION_RATIO while the answer does not.
    """
    for index in range(aligned_videos):
        first = add_video(db, external_id=f"VIDEO_A{index}", title=f"A{index}")
        add_words(db, first, "мы все понимаем что это")
        second = add_video(db, external_id=f"VIDEO_B{index}", title=f"B{index}")
        add_words(db, second, "все понимаем что это неизбежно")
        for caption in range(CAPTION_RATIO):
            noise = add_video(db, external_id=f"VIDEO_C{index}_{caption}")
            add_words(db, noise, TARGET, source="caption")
            muffled = add_video(db, external_id=f"VIDEO_D{index}_{caption}")
            add_words(db, muffled, TARGET, source="timed")


def test_the_fixture_is_mostly_uncuttable(db: Database) -> None:
    """Guards the guard: if this fixture goes uniform, the budget test below
    stops measuring what it claims to."""
    build(db, 2)
    counts = dict(
        db.conn.execute("SELECT source, COUNT(*) FROM words GROUP BY source").fetchall()
    )
    assert counts["caption"] > counts["aligned"] * 4
    assert counts["timed"] > counts["aligned"] * 4


def test_two_runs_write_the_same_bytes(db: Database, tmp_path: Path) -> None:
    build(db, 3)
    first = write_cutlist(
        assemble_target(db, TARGET, name="демо", created_at=CREATED), tmp_path / "one.toml"
    )
    second = write_cutlist(
        assemble_target(db, TARGET, name="демо", created_at=CREATED), tmp_path / "two.toml"
    )
    assert first.read_bytes() == second.read_bytes()


def test_a_seed_changes_the_bytes_and_then_keeps_them(db: Database) -> None:
    build(db, 3)

    def dump(seed: int) -> str:
        return dumps_cutlist(
            assemble_target(
                db, TARGET, name="демо", controls=AssembleControls(seed=seed), created_at=CREATED
            )
        )

    unseeded = dump(0)
    assert dump(0) == unseeded
    seeded = {dump(seed) for seed in range(1, 12)}
    assert len(seeded) > 1
    assert dump(7) == dump(7)


def test_the_knob_changes_the_bytes(db: Database) -> None:
    build(db, 3)

    def dump(consistency: float) -> str:
        return dumps_cutlist(
            assemble_target(
                db,
                TARGET,
                name="демо",
                controls=AssembleControls(consistency=consistency),
                created_at=CREATED,
            )
        )

    assert dump(0.0) != dump(1.0)


def count_statements(path: Path, videos: int) -> int:
    """How many SQL statements one assemble_target costs on this corpus."""
    database = Database(path)
    try:
        database.migrate()
        build(database, videos)
        statements = 0

        def trace(_sql: str) -> None:
            nonlocal statements
            statements += 1

        database.conn.set_trace_callback(trace)
        assemble_target(database, TARGET, name="демо", created_at=CREATED)
        database.conn.set_trace_callback(None)
        return statements
    finally:
        database.close()


def test_the_query_count_follows_the_target_and_not_the_corpus(tmp_path: Path) -> None:
    """Design §7: the walk is a pointer walk, so "no n-gram table is needed"."""
    small = count_statements(tmp_path / "small.db", 3)
    large = count_statements(tmp_path / "large.db", 30)
    assert small == large
    assert small < 40, f"{small} statements for a six-word target is too many"


def test_the_occurrence_lookup_uses_the_partial_index(db: Database) -> None:
    """contracts §3: words_alignable exists so this lookup reads only the
    cuttable rows. Counting statements cannot see a rewrite that keeps one
    query and scans the caption corpus inside it; the query plan can."""
    build(db, 2)
    db.conn.execute("ANALYZE")
    sql, params = occurrence_query(MatchFilters())
    plan = " ".join(
        str(row["detail"]) for row in db.conn.execute("EXPLAIN QUERY PLAN " + sql, params)
    )
    assert "words_alignable" in plan, plan


def test_the_caption_corpus_does_not_change_the_answer(db: Database) -> None:
    """Whatever the captions say, only the aligned minority is cut."""
    build(db, 1)
    cutlist = assemble_target(db, TARGET, name="демо", created_at=CREATED)
    sources = {slot.video_id for slot in cutlist.fragments}
    aligned = {
        int(row["video_id"])
        for row in db.conn.execute(
            "SELECT DISTINCT video_id FROM words WHERE source = 'aligned'"
        )
    }
    assert sources
    assert sources <= aligned


@pytest.mark.parametrize("tier", ["caption", "timed"])
def test_an_uncuttable_corpus_assembles_nothing_at_all(db: Database, tier: str) -> None:
    """The one rule that must never leak: contracts §3, only `aligned` is cut.

    `timed` is the dangerous half. A caption row has no end_ms and would
    fail loudly downstream; a timed row has plausible-looking timings that
    simply are not cuttable, so nothing but the tier filter stops it.
    """
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, TARGET, source=tier)
    cutlist = assemble_target(db, TARGET, name="демо", created_at=CREATED)
    assert cutlist.fragments == ()
    assert len(cutlist.gaps) == len(TARGET.split())
    assert all(slot.substitutions == () for slot in cutlist.gaps)
