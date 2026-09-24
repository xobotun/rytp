"""The render report: structured first, markdown second (design §9)."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

from rytp import constants as C
from rytp.render import report as R

FRAGMENTS = (
    R.FragmentReport(
        ord=0, video_id=3, video_title="Разговор о выборах", video_url=None,
        speaker="host", source_start_ms=612_340, source_end_ms=613_100,
        output_start_ms=0, output_end_ms=760, gap_after_ms=180,
        gap_origin="measured", text="мы всё",
    ),
    R.FragmentReport(
        ord=1, video_id=9, video_title="Другой | разговор",
        video_url="https://example.invalid/watch/VIDEO_B", speaker=None,
        source_start_ms=220_100, source_end_ms=220_780,
        output_start_ms=940, output_end_ms=1_620, gap_after_ms=0,
        gap_origin="measured", text="исправит",
    ),
)

SOURCES = (
    R.SourceReport(
        video_id=3, title="Разговор о выборах", url=None, fragment_count=1,
        used_ms=760, measured_lufs=-23.4, gain_db=7.4, first_output_ms=0,
        geometry="1280x720 @ 25 fps",
    ),
    R.SourceReport(
        video_id=9, title="Другой | разговор",
        url="https://example.invalid/watch/VIDEO_B", fragment_count=1,
        used_ms=680, measured_lufs=None, gain_db=None, first_output_ms=940,
        geometry="640x480 @ 25 fps",
    ),
)

REPORT = R.RenderReport(
    render_id="demo-1a2b3c4d",
    cutlist="demo",
    created_at="2026-09-21T09:00:00+00:00",
    output_path="data/output/demo-1a2b3c4d/output.mp4",
    duration_ms=1_620,
    canvas="1280x720 @ 25 fps (16:9)",
    loudnorm=True,
    gap_policy="measured from aligned transcripts",
    target_text="мы всё исправим",
    assembled_text="мы всё исправит",
    fragments=FRAGMENTS,
    sources=SOURCES,
    missing=(
        R.MissingWord(
            text="исправим",
            position=2,
            substitutions=(
                R.SubstitutionRef(
                    text="исправит", reason="edit", video_id=9,
                    start_ms=220_100, end_ms=220_780, occurrences=3,
                ),
            ),
        ),
    ),
    notes=("speaker host: 81% of gaps are exactly zero; fell back to 180 ms",),
)


def test_timecodes_are_hours_minutes_seconds_and_milliseconds() -> None:
    assert R.format_timecode(0) == "0:00:00.000"
    assert R.format_timecode(612_340) == "0:10:12.340"
    assert R.format_timecode(3_661_001) == "1:01:01.001"


def test_the_clock_form_drops_the_hour_when_there_is_none() -> None:
    assert R.format_clock(0) == "0:00"
    assert R.format_clock(612_340) == "10:12"
    assert R.format_clock(3_661_001) == "1:01:01"


def test_the_markdown_opens_with_the_render_and_carries_the_text() -> None:
    text = R.render_markdown(REPORT)
    assert text.startswith("# Render demo-1a2b3c4d")
    assert "мы всё исправит" in text
    assert "1280x720 @ 25 fps (16:9)" in text


def test_every_fragment_appears_with_its_source_and_both_timelines() -> None:
    text = R.render_markdown(REPORT)
    assert "0:10:12.340" in text  # source in-point
    assert "0:00:00.940" in text  # output in-point of the second fragment
    assert "video 3" in text and "video 9" in text


def test_a_pipe_in_a_title_does_not_break_the_table() -> None:
    text = R.render_markdown(REPORT)
    row = next(line for line in text.splitlines() if "Другой" in line and "|" in line)
    assert "Другой \\| разговор" in row


def test_missing_words_are_listed_with_their_substitutions() -> None:
    text = R.render_markdown(REPORT)
    assert "## Words not found" in text
    assert "исправим" in text
    assert "исправит" in text
    assert "edit" in text


def test_a_render_with_nothing_missing_says_so() -> None:
    complete = replace(REPORT, missing=())
    assert "every word was found" in R.render_markdown(complete).lower()


def test_the_description_block_lists_each_source_once_in_appearance_order() -> None:
    block = R.description_block(REPORT)
    assert C.RENDER_DESCRIPTION_HEADER in block
    assert block.count("Разговор о выборах") == 1
    assert block.index("Разговор о выборах") < block.index("Другой | разговор")
    assert block.splitlines()[0] == "мы всё исправит"


def test_the_description_block_is_fenced_inside_the_markdown() -> None:
    text = R.render_markdown(REPORT)
    assert "## Description" in text
    assert "```" in text[text.index("## Description") :]


def test_notes_are_rendered_so_a_degenerate_fallback_is_visible() -> None:
    assert "fell back to 180 ms" in R.render_markdown(REPORT)


def test_the_whole_report_is_json_serialisable() -> None:
    """Design §9: JSON and subtitle output must be addable without rework."""
    payload = json.dumps(asdict(REPORT), ensure_ascii=False)
    round_tripped = json.loads(payload)
    assert round_tripped["fragments"][1]["output_start_ms"] == 940
    assert round_tripped["missing"][0]["substitutions"][0]["reason"] == "edit"
