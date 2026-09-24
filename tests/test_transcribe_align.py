"""Aligner adapters, and the pure parsing they are built on."""
from __future__ import annotations

import os

from rytp import constants as C
from rytp.transcribe import registry
from rytp.transcribe.align.mfa import (
    MfaAligner,
    mfa_binary,
    parse_textgrid,
    spans_from_intervals,
)
from rytp.transcribe.align.wav2vec2 import Wav2Vec2Aligner, spans_from_frames

TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.5
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.5
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.25
            text = ""
        intervals [2]:
            xmin = 0.25
            xmax = 0.80
            text = "привет"
        intervals [3]:
            xmin = 0.90
            xmax = 1.50
            text = "мир"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.5
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1.5
            text = "p"
'''


def test_parse_textgrid_returns_the_words_tier_without_the_silences() -> None:
    assert parse_textgrid(TEXTGRID) == [(0.25, 0.80, "привет"), (0.90, 1.50, "мир")]


def test_parse_textgrid_ignores_other_tiers() -> None:
    assert all(label in {"привет", "мир"} for _, _, label in parse_textgrid(TEXTGRID))


def test_parse_textgrid_of_rubbish_is_empty() -> None:
    assert parse_textgrid("not a textgrid") == []


def test_spans_from_intervals_offsets_onto_the_whole_file_timeline() -> None:
    spans = spans_from_intervals(parse_textgrid(TEXTGRID), offset_ms=10_000)
    assert [(s.start_ms, s.end_ms) for s in spans] == [(10_250, 10_800), (10_900, 11_500)]
    assert spans[0].score is None


def test_mfa_binary_sits_beside_the_configured_interpreter() -> None:
    derived = mfa_binary(os.path.join("opt", "mfa-env", "bin", "python"))
    assert derived.endswith("mfa.exe" if os.name == "nt" else "mfa")
    assert "mfa-env" in derived


def test_spans_from_frames_uses_the_ctc_stride() -> None:
    spans = spans_from_frames(
        [(0, 4, 0.9), (10, 14, None)], stride_ms=C.WAV2VEC2_STRIDE_MS, offset_ms=1_000
    )
    assert [(s.start_ms, s.end_ms) for s in spans] == [(1_000, 1_100), (1_200, 1_300)]
    assert spans[0].score == 0.9


def test_spans_from_frames_never_produces_a_zero_length_span() -> None:
    spans = spans_from_frames([(5, 5, None)], stride_ms=20.0, offset_ms=0)
    assert spans[0].end_ms > spans[0].start_ms


def test_importing_the_align_package_registers_both_aligners() -> None:
    import rytp.transcribe.align  # noqa: F401

    assert "mfa" in registry.ALIGNERS
    assert "wav2vec2" in registry.ALIGNERS


def test_both_aligners_run_out_of_process_and_need_no_token() -> None:
    for cls in (MfaAligner, Wav2Vec2Aligner):
        assert cls.out_of_process is True
        assert cls.requires_hf_token is False
