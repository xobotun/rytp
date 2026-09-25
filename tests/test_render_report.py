"""The render report: structured first, markdown second (design §9)."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from rytp import constants as C
from rytp.render import report as R
from rytp.render import run as RUN
from rytp.render.canvas import Canvas, SourceGeometry

FRAGMENTS = (
    R.FragmentReport(
        ord=0, video_id=3, video_title="Разговор о выборах", video_url=None,
        speaker="host", source_start_ms=612_340, source_end_ms=613_100,
        output_start_ms=0, output_end_ms=760, gap_after_ms=180,
        gap_origin="measured", text="мы всё", tier="aligned",
    ),
    R.FragmentReport(
        ord=1, video_id=9, video_title="Другой | разговор",
        video_url="https://example.invalid/watch/VIDEO_B", speaker=None,
        source_start_ms=220_100, source_end_ms=220_780,
        output_start_ms=940, output_end_ms=1_620, gap_after_ms=0,
        gap_origin="measured", text="исправит", tier="timed",
    ),
)

SOURCES = (
    R.SourceReport(
        video_id=3, title="Разговор о выборах", url=None, fragment_count=1,
        used_ms=760, measured_lufs=-23.4, gain_db=7.4, first_output_ms=0,
        geometry="1280x720 @ 25 fps", timed_fragment_count=0,
    ),
    R.SourceReport(
        video_id=9, title="Другой | разговор",
        url="https://example.invalid/watch/VIDEO_B", fragment_count=1,
        used_ms=680, measured_lufs=None, gain_db=None, first_output_ms=940,
        geometry="640x480 @ 25 fps", timed_fragment_count=1,
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


def test_a_timed_tier_fragment_is_marked_in_the_source_list() -> None:
    """D1: `--allow-timed` has no quality check, so the recorded tier is the
    only signal a cut came from unaligned data — the render's source list is
    what the owner publishes, so it must say so (BUGS.md entry 31)."""
    text = R.render_markdown(REPORT)
    fragments_section = text[text.index("## Fragments") : text.index("## Words not found")]
    rows = [line for line in fragments_section.splitlines() if line.startswith("|")]
    aligned_row = next(r for r in rows if "мы всё" in r)
    timed_row = next(r for r in rows if "исправит" in r)
    assert "| aligned |" in aligned_row
    assert "| timed |" in timed_row

    sources_section = text[text.index("## Sources") : text.index("## Description")]
    source_rows = [line for line in sources_section.splitlines() if line.startswith("|")]
    row_3 = next(r for r in source_rows if r.startswith("| 3 "))
    row_9 = next(r for r in source_rows if r.startswith("| 9 "))
    assert "timed" not in row_3
    assert "1 (1 timed)" in row_9


def test_a_fragment_with_no_recorded_tier_shows_the_null_placeholder() -> None:
    """A fragment carrying no tier at all must not be misreported as
    ``aligned`` — that would hide exactly what entry 31 exists to show."""
    from dataclasses import replace

    untiered = replace(FRAGMENTS[0], tier=None)
    report = replace(REPORT, fragments=(untiered, FRAGMENTS[1]))
    text = R.render_markdown(report)
    fragments_section = text[text.index("## Fragments") : text.index("## Words not found")]
    row = next(
        line
        for line in fragments_section.splitlines()
        if line.startswith("|") and "мы всё" in line
    )
    assert "| aligned |" not in row
    assert "| — |" in row


def test_request_from_cutlist_carries_each_slots_tier() -> None:
    """The adapter half of the plumbing (BUGS.md entry 31 / D1): a real cut
    list slot's ``tier`` must reach :class:`~rytp.render.run.RenderFragment`
    unchanged, and a slot with no ``tier`` attribute at all — not something
    Task 6 produces, but the adapter must not invent one — must not be
    misreported as ``aligned``."""
    cutlist = SimpleNamespace(
        name="demo",
        target="мы всё",
        slots=(
            SimpleNamespace(
                kind="fragment", target_first=0, target_last=0, text="мы",
                video_id=3, first_word_ord=1, last_word_ord=1,
                start_ms=0, end_ms=200, tier="aligned",
            ),
            SimpleNamespace(
                kind="fragment", target_first=1, target_last=1, text="всё",
                video_id=3, first_word_ord=2, last_word_ord=2,
                start_ms=200, end_ms=400, tier="timed",
            ),
            SimpleNamespace(
                kind="fragment", target_first=2, target_last=2, text="?",
                video_id=3, first_word_ord=3, last_word_ord=3,
                start_ms=400, end_ms=600,
                # deliberately no `tier` attribute
            ),
        ),
    )
    request = RUN.request_from_cutlist(cutlist)
    assert [f.tier for f in request.fragments] == ["aligned", "timed", None]


def test_build_report_marks_a_timed_fragment_in_the_source_list() -> None:
    """The render side of D1: a fragment cut under `--allow-timed` must show
    up in the render's own report, not only at plan time, and a source made
    entirely of `aligned` fragments must report zero timed fragments."""
    geometry = SourceGeometry(width=1280, height=720, fps=25.0)
    source = RUN.SourceMedia(
        video_id=3, title="Разговор", url=None,
        video_path=Path("video.mp4"), audio_path=Path("audio.m4a"),
        geometry=geometry,
    )
    aligned_fragment = RUN.RenderFragment(
        video_id=3, first_word_ord=1, last_word_ord=1,
        start_ms=0, end_ms=200, text="мы", tier="aligned",
    )
    timed_fragment = RUN.RenderFragment(
        video_id=3, first_word_ord=2, last_word_ord=2,
        start_ms=200, end_ms=400, text="всё", tier="timed",
    )
    plan = RUN.RenderPlan(
        render_id="demo-1a2b3c4d",
        request=RUN.RenderRequest(name="demo", fragments=(aligned_fragment, timed_fragment)),
        options=RUN.RenderOptions(),
        canvas=Canvas(width=1280, height=720, fps=25, mode="letterbox"),
        fragments=(
            RUN.PlannedFragment(
                ord=0, fragment=aligned_fragment, source=source,
                gap_after_ms=0, gap_origin="measured",
                output_start_ms=0, output_end_ms=200,
                intermediate=Path("0.mp4"),
            ),
            RUN.PlannedFragment(
                ord=1, fragment=timed_fragment, source=source,
                gap_after_ms=0, gap_origin="measured",
                output_start_ms=200, output_end_ms=400,
                intermediate=Path("1.mp4"),
            ),
        ),
        sources=(source,),
        output_dir=Path("out"), output_path=Path("out/output.mp4"),
        report_path=Path("out/report.md"), list_file=Path("out/concat.txt"),
        duration_ms=400, notes=(),
    )
    report = RUN.build_report(plan, notes=(), created_at="2026-09-25T00:00:00+00:00")
    assert report.fragments[0].tier == "aligned"
    assert report.fragments[1].tier == "timed"
    assert report.sources[0].timed_fragment_count == 1

    text = R.render_markdown(report)
    sources_section = text[text.index("## Sources") : text.index("## Description")]
    row = next(line for line in sources_section.splitlines() if line.startswith("| 3 "))
    assert "1 timed" in row


def test_a_source_cut_entirely_from_aligned_fragments_reports_no_timed() -> None:
    text = R.render_markdown(REPORT)  # both REPORT sources: 1 aligned, 1 timed
    sources_section = text[text.index("## Sources") : text.index("## Description")]
    row_3 = next(line for line in sources_section.splitlines() if line.startswith("| 3 "))
    assert "timed" not in row_3


def test_the_whole_report_is_json_serialisable() -> None:
    """Design §9: JSON and subtitle output must be addable without rework."""
    payload = json.dumps(asdict(REPORT), ensure_ascii=False)
    round_tripped = json.loads(payload)
    assert round_tripped["fragments"][1]["output_start_ms"] == 940
    assert round_tripped["missing"][0]["substitutions"][0]["reason"] == "edit"
