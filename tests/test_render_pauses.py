"""Between-word pause statistics and the zero-inflation guard (design §9)."""

from __future__ import annotations

from rytp import constants as C
from rytp.db import Database
from rytp.render import pauses as P
from tests.fakes import make_video


def add_words(
    db: Database,
    video_id: int,
    spans: list[tuple[int, int]],
    *,
    source: str = "aligned",
    video_speaker_id: int | None = None,
    first_ord: int = 0,
) -> None:
    """Insert words at the given (start_ms, end_ms) spans, ords contiguous."""
    rows = [
        (
            video_id, first_ord + i, start, end if source == "aligned" else None,
            "х", "х", "х", source, "fake", video_speaker_id,
        )
        for i, (start, end) in enumerate(spans)
    ]
    db.conn.executemany(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, "
        "stem, source, engine, video_speaker_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    db.conn.commit()


def add_speaker(db: Database, video_id: int, *, label: str, local: str = "SPEAKER_00") -> int:
    """Map one diarized label of one video to a roster person, idempotently."""
    db.conn.execute(
        "INSERT OR IGNORE INTO speakers (label, created_at) "
        "VALUES (?, '2026-01-01T00:00:00+00:00')",
        (label,),
    )
    speaker_id = db.conn.execute(
        "SELECT id FROM speakers WHERE label = ?", (label,)
    ).fetchone()[0]
    cursor = db.conn.execute(
        # `engine` is NOT NULL (contracts §3): a label always knows which
        # diarizer produced it, fixtures included.
        "INSERT INTO video_speakers (video_id, local_label, speaker_id, engine) "
        "VALUES (?, ?, ?, 'fake')",
        (video_id, local, speaker_id),
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def evenly_spaced(n: int, *, word_ms: int = 300, gap_ms: int = 200) -> list[tuple[int, int]]:
    spans = []
    clock = 0
    for _ in range(n):
        spans.append((clock, clock + word_ms))
        clock += word_ms + gap_ms
    return spans


def zero_gapped(n: int, *, word_ms: int = 300) -> list[tuple[int, int]]:
    return [(i * word_ms, (i + 1) * word_ms) for i in range(n)]


def test_a_healthy_distribution_takes_the_median(db: Database) -> None:
    vid = make_video(db)
    add_words(db, vid, evenly_spaced(100, gap_ms=200))
    stats = P.measure_pause_stats(db, video_id=vid)
    assert stats.degenerate is False
    assert stats.median_ms == 200
    assert stats.scope == "video"
    assert stats.n_samples == 99


def test_a_whisper_shaped_distribution_is_refused(db: Database) -> None:
    """Design §6: 78.7% of gaps at exactly zero. The median must not be believed."""
    vid = make_video(db)
    tail_base = 80 * 300
    spans = zero_gapped(80) + [
        (tail_base + 250 * i, tail_base + 250 * i + 200) for i in range(20)
    ]
    add_words(db, vid, spans)
    stats = P.measure_pause_stats(db, video_id=vid)
    assert stats.degenerate is True
    assert stats.zero_fraction > C.PAUSE_DEGENERATE_ZERO_FRACTION
    assert "zero" in stats.reason
    assert P.applied_gap_ms(stats) == C.PAUSE_FALLBACK_GAP_MS


def test_too_few_samples_is_degenerate_and_says_so(db: Database) -> None:
    vid = make_video(db)
    add_words(db, vid, evenly_spaced(5))
    stats = P.measure_pause_stats(db, video_id=vid)
    assert stats.degenerate is True
    assert "samples" in stats.reason
    assert P.applied_gap_ms(stats) == C.PAUSE_FALLBACK_GAP_MS


def test_caption_words_are_never_measured(db: Database) -> None:
    vid = make_video(db)
    add_words(db, vid, evenly_spaced(100), source="caption")
    stats = P.measure_pause_stats(db, video_id=vid)
    assert stats.n_samples == 0
    assert stats.degenerate is True


def test_turn_length_silences_are_excluded(db: Database) -> None:
    vid = make_video(db)
    spans = evenly_spaced(60, gap_ms=150)
    tail_start = spans[-1][1] + 30_000
    spans.append((tail_start, tail_start + 300))
    add_words(db, vid, spans)
    stats = P.measure_pause_stats(db, video_id=vid)
    assert stats.n_samples == 59  # the 30 s silence is not a between-word pause
    assert stats.median_ms == 150


def test_a_gap_in_the_ordinals_is_not_a_pause(db: Database) -> None:
    vid = make_video(db)
    add_words(db, vid, evenly_spaced(60, gap_ms=150))
    add_words(db, vid, evenly_spaced(60, gap_ms=150), first_ord=500)
    stats = P.measure_pause_stats(db, video_id=vid)
    assert stats.n_samples == 118  # 59 within each run, none across the break


def test_a_named_speaker_pools_across_videos(db: Database) -> None:
    first = make_video(db)
    second = make_video(
        db, external_id="VIDEO_B", url="https://example.invalid/watch/VIDEO_B"
    )
    vs_one = add_speaker(db, first, label="host")
    vs_two = add_speaker(db, second, label="host", local="SPEAKER_01")
    add_words(db, first, evenly_spaced(40, gap_ms=120), video_speaker_id=vs_one)
    add_words(db, second, evenly_spaced(40, gap_ms=120), video_speaker_id=vs_two)
    stats = P.measure_pause_stats(db, video_id=first, speaker_label="host")
    assert stats.scope == "speaker"
    assert stats.key == "host"
    assert stats.n_samples == 78
    assert stats.degenerate is False
    assert stats.median_ms == 120


def test_a_diarized_label_with_no_roster_entry_scopes_to_that_label(db: Database) -> None:
    vid = make_video(db)
    cursor = db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine) "
        "VALUES (?, 'SPEAKER_00', 'fake')",
        (vid,),
    )
    db.conn.commit()
    label_id = int(cursor.lastrowid)
    add_words(db, vid, evenly_spaced(60, gap_ms=90), video_speaker_id=label_id)
    add_words(db, vid, evenly_spaced(60, gap_ms=400), first_ord=500)
    stats = P.measure_pause_stats(db, video_id=vid, video_speaker_id=label_id)
    assert stats.scope == "video_speaker"
    assert stats.median_ms == 90


def test_an_unknown_speaker_falls_back_to_the_video(db: Database) -> None:
    vid = make_video(db)
    add_words(db, vid, evenly_spaced(80, gap_ms=210))
    stats = P.measure_pause_stats(db, video_id=vid, speaker_label="nobody")
    assert stats.scope == "video"
    assert stats.median_ms == 210


def test_no_labels_means_no_gaps_not_every_gap(db: Database) -> None:
    """An unmapped speaker matches nothing; an `IN ()` must not match all."""
    vid = make_video(db)
    add_words(db, vid, evenly_spaced(100, gap_ms=200))
    assert P.gaps_for_speaker_ids(db, ()) == []


def test_the_applied_gap_is_clamped_both_ways() -> None:
    tiny = P.PauseStats(
        scope="video", key="1", n_samples=999, median_ms=3,
        zero_fraction=0.0, degenerate=False, reason="",
    )
    huge = P.PauseStats(
        scope="video", key="1", n_samples=999, median_ms=1_900,
        zero_fraction=0.0, degenerate=False, reason="",
    )
    assert P.applied_gap_ms(tiny) == C.PAUSE_MIN_APPLIED_GAP_MS
    assert P.applied_gap_ms(huge) == C.PAUSE_MAX_APPLIED_GAP_MS


def test_summarize_is_deterministic() -> None:
    samples = [100, 200, 300, 400]
    first = P.summarize_gaps(samples, scope="video", key="1")
    second = P.summarize_gaps(samples, scope="video", key="1")
    assert first == second
    assert first.median_ms == 250
