"""The report that ships beside the output file. design §9.

The owner's actual deliverable is two files: something to upload and a
list of where every second of it came from. This module owns the second
one.

It is a dataclass tree with a markdown renderer bolted on, rather than
markdown assembled inline, because design §9 asks that JSON and subtitle
output be addable later without rework. Every field here is a scalar, a
string, or a tuple of those, so ``dataclasses.asdict`` is already
JSON-serialisable, and ``FragmentReport`` already carries output-timeline
timings — which is exactly what a subtitle writer needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from rytp import constants as C

_MS_PER_MINUTE = 60_000
_MS_PER_HOUR = 3_600_000


@dataclass(frozen=True)
class SubstitutionRef:
    """A ranked stand-in for a word the corpus does not contain."""

    text: str
    reason: str  # "stem" | "edit"
    video_id: int
    start_ms: int
    end_ms: int
    occurrences: int = 0


@dataclass(frozen=True)
class MissingWord:
    """A word of the target that no cuttable fragment covers."""

    text: str
    position: int  # index into the target's token sequence
    substitutions: tuple[SubstitutionRef, ...] = ()


@dataclass(frozen=True)
class FragmentReport:
    """One cut, with both timelines: where it came from and where it landed."""

    ord: int
    video_id: int
    video_title: str
    video_url: str | None
    speaker: str | None
    source_start_ms: int
    source_end_ms: int
    output_start_ms: int
    output_end_ms: int
    gap_after_ms: int
    gap_origin: str
    text: str


@dataclass(frozen=True)
class SourceReport:
    """One source video's whole contribution."""

    video_id: int
    title: str
    url: str | None
    fragment_count: int
    used_ms: int
    measured_lufs: float | None
    gain_db: float | None
    first_output_ms: int
    geometry: str


@dataclass(frozen=True)
class RenderReport:
    """Everything the render did, structured."""

    render_id: str
    cutlist: str
    created_at: str
    output_path: str
    duration_ms: int
    canvas: str
    loudnorm: bool
    gap_policy: str
    target_text: str
    assembled_text: str
    fragments: tuple[FragmentReport, ...]
    sources: tuple[SourceReport, ...]
    missing: tuple[MissingWord, ...]
    notes: tuple[str, ...]


def format_timecode(ms: int) -> str:
    """``H:MM:SS.mmm`` — precise enough to seek to in a player."""
    ms = max(0, int(ms))
    hours, rest = divmod(ms, _MS_PER_HOUR)
    minutes, rest = divmod(rest, _MS_PER_MINUTE)
    secs, millis = divmod(rest, C.MS_PER_SECOND)
    return f"{hours}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_clock(ms: int) -> str:
    """``M:SS``, or ``H:MM:SS`` past an hour — the form a description wants."""
    ms = max(0, int(ms))
    hours, rest = divmod(ms, _MS_PER_HOUR)
    minutes, rest = divmod(rest, _MS_PER_MINUTE)
    secs = rest // C.MS_PER_SECOND
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _cell(text: str | None) -> str:
    """Make a value safe inside a markdown table cell."""
    if text is None:
        return "—"
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _number(value: float | None, *, suffix: str = "") -> str:
    return "—" if value is None else f"{value:+.1f}{suffix}"


def description_block(report: RenderReport) -> str:
    """The paste-ready block for a video description.

    The assembled text, then each source once, in the order it first
    appears, with the output timestamp where it does. One line per
    source rather than per fragment: a description listing forty
    entries is not a description.
    """
    lines = [report.assembled_text.strip(), "", C.RENDER_DESCRIPTION_HEADER]
    for source in sorted(report.sources, key=lambda s: s.first_output_ms):
        tail = f" — {source.url}" if source.url else ""
        lines.append(f"{format_clock(source.first_output_ms)} — {source.title}{tail}")
    return "\n".join(lines).strip() + "\n"


def render_markdown(report: RenderReport) -> str:
    """The whole report as one markdown document."""
    out: list[str] = [
        f"# Render {report.render_id}",
        "",
        f"Cut list `{report.cutlist}` · rendered {report.created_at} · "
        f"{format_timecode(report.duration_ms)} long",
        "",
        f"- **Output** `{report.output_path}`",
        f"- **Canvas** {report.canvas}",
        f"- **Loudness** {'normalized' if report.loudnorm else 'left alone'}",
        f"- **Gaps** {report.gap_policy}",
        "",
        "## Text",
        "",
        f"> {report.assembled_text.strip() or '—'}",
        "",
    ]
    if report.target_text.strip() != report.assembled_text.strip():
        out += ["Asked for:", "", f"> {report.target_text.strip()}", ""]

    out += [
        "## Fragments",
        "",
        "| # | output | source | speaker | source in → out | gap after | text |",
        "|--:|---|---|---|---|--:|---|",
    ]
    for fragment in report.fragments:
        out.append(
            f"| {fragment.ord + 1} "
            f"| {format_timecode(fragment.output_start_ms)} → "
            f"{format_timecode(fragment.output_end_ms)} "
            f"| video {fragment.video_id} — {_cell(fragment.video_title)} "
            f"| {_cell(fragment.speaker)} "
            f"| {format_timecode(fragment.source_start_ms)} → "
            f"{format_timecode(fragment.source_end_ms)} "
            f"| {fragment.gap_after_ms} ms ({fragment.gap_origin}) "
            f"| {_cell(fragment.text)} |"
        )

    out += ["", "## Words not found", ""]
    if not report.missing:
        out.append("None — every word was found in the corpus.")
    else:
        out += ["| word | position | suggested instead |", "|---|--:|---|"]
        for missing in report.missing:
            suggestions = (
                "; ".join(
                    f"{_cell(s.text)} ({s.reason}, {s.occurrences}×, "
                    f"video {s.video_id} @ {format_timecode(s.start_ms)})"
                    for s in missing.substitutions
                )
                or "—"
            )
            out.append(f"| {_cell(missing.text)} | {missing.position} | {suggestions} |")

    out += [
        "",
        "## Sources",
        "",
        "| video | title | fragments | used | geometry | measured | gain |",
        "|--:|---|--:|--:|---|---|---|",
    ]
    for source in report.sources:
        out.append(
            f"| {source.video_id} "
            f"| {_cell(source.title)} "
            f"| {source.fragment_count} "
            f"| {format_timecode(source.used_ms)} "
            f"| {_cell(source.geometry)} "
            f"| {_number(source.measured_lufs, suffix=' LUFS')} "
            f"| {_number(source.gain_db, suffix=' dB')} |"
        )

    if report.notes:
        out += ["", "## Notes", ""]
        out += [f"- {note}" for note in report.notes]

    out += [
        "",
        "## Description",
        "",
        "```",
        description_block(report).rstrip("\n"),
        "```",
        "",
    ]
    return "\n".join(out)
