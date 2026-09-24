"""Aligner adapters, and the pure parsing they are built on."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.models import RytpError
from rytp.transcribe import registry
from rytp.transcribe.align import wav2vec2 as w2v
from rytp.transcribe.align.mfa import (
    MfaAligner,
    mfa_binary,
    parse_textgrid,
    spans_from_intervals,
)
from rytp.transcribe.align.wav2vec2 import (
    Wav2Vec2Aligner,
    build_target_ids,
    spans_from_frames,
    word_frames,
)

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


# --- Entry 36: word_frames actually groups the CTC path (2026-09-25 batch) ---

# A path built by hand to realise BUGS.md entry 36's own demonstration: three
# words genuinely occupying 2, 2 and 10 frames, each a distinct target token
# so no doubled-letter merging is needed inside a word — that is covered by
# its own test below. targets: word "а" (token 1), delimiter (99), word "бб"
# as a single token (2) for width, delimiter (99), word "ввввв" as five
# distinct characters (3, 4, 5, 6, 7), each held for two frames.
_DELIM = 99
_BUGS_TARGETS = [1, _DELIM, 2, _DELIM, 3, 4, 5, 6, 7]
_BUGS_PATH = [
    1, 1, 0, _DELIM, 0,          # word "а": frames 0-1
    2, 2, 0, _DELIM, 0,          # word "бб": frames 5-6
    3, 3, 4, 4, 5, 5, 6, 6, 7, 7,  # word "ввввв": frames 10-19
]
_BUGS_SCORES = [
    -1.0, -2.0, 0.0, -9.0, 0.0,
    -3.0, -4.0, 0.0, -9.0, 0.0,
    -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
]

_AUDIO = Path("unused.wav")


def test_word_frames_reproduces_the_bugs_md_2_2_10_demonstration() -> None:
    """The exact regression BUGS.md entry 36 was written from.

    The old code returned three 6-frame spans with an identical -0.1 score
    for this input — equal widths, uniform score, no relationship to the
    path. This asserts the real shape: 2, 2 and 10 frames, three distinct
    scores.
    """
    frames, notes = word_frames(
        _BUGS_PATH, _BUGS_SCORES, targets=_BUGS_TARGETS, delimiter=_DELIM
    )
    assert notes == []
    widths = [entry["last_frame"] - entry["first_frame"] + 1 for entry in frames]
    assert widths == [2, 2, 10]
    assert len(widths) > 1 and len(set(widths)) > 1  # not equal division
    scores = [entry["score"] for entry in frames]
    assert scores == [-1.5, -3.5, -5.0]
    assert len(set(scores)) == len(scores)  # not the same mean-of-everything


def test_word_frames_via_spans_from_frames_matches_the_stride() -> None:
    frames, _notes = word_frames(
        _BUGS_PATH, _BUGS_SCORES, targets=_BUGS_TARGETS, delimiter=_DELIM
    )
    spans = spans_from_frames(
        [(f["first_frame"], f["last_frame"], f["score"]) for f in frames],
        stride_ms=C.WAV2VEC2_STRIDE_MS,
        offset_ms=0,
    )
    assert [(s.start_ms, s.end_ms) for s in spans] == [(0, 40), (100, 140), (200, 400)]


def test_word_frames_does_not_double_advance_on_a_doubled_letter() -> None:
    """"аа" is two adjacent, identical targets, separated by a forced-align
    blank. The cursor must advance exactly once per occurrence — landing both
    frames of the first occurrence and both of the second in the same word's
    bucket, not splitting them into two words and not collapsing them into
    one occurrence.
    """
    # word "аа" -> tokens [7, 7], single word, no delimiter needed.
    targets = [7, 7]
    path = [7, 7, 0, 7, 7]  # first occurrence x2, blank, second occurrence x2
    scores = [0.1, 0.2, 0.0, 0.3, 0.4]
    frames, notes = word_frames(path, scores, targets=targets, delimiter=_DELIM)
    assert notes == []
    assert len(frames) == 1  # one word
    assert frames[0]["first_frame"] == 0
    assert frames[0]["last_frame"] == 4
    assert frames[0]["score"] == pytest.approx((0.1 + 0.2 + 0.3 + 0.4) / 4)


def test_word_frames_tolerates_several_consecutive_delimiter_frames() -> None:
    """A word delimiter token held over several frames is still one boundary."""
    targets = [1, _DELIM, 2]
    path = [1, 1, _DELIM, _DELIM, _DELIM, 2, 2]
    scores = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    frames, notes = word_frames(path, scores, targets=targets, delimiter=_DELIM)
    assert notes == []
    assert len(frames) == 2
    assert frames[0]["last_frame"] == 1
    assert frames[1]["first_frame"] == 5
    assert frames[1]["last_frame"] == 6


def test_word_frames_interpolates_a_word_that_received_no_frames() -> None:
    """The path ends before ever reaching the last word's own target.

    This is the narrow, observable fallback: the missing word's boundary is
    interpolated from its neighbour, and a note names which word it was —
    never silent, unlike the equal-division bug this replaces.
    """
    targets = [1, _DELIM, 2, _DELIM, 3]
    path = [1, 1, 0, _DELIM, 0, 2, 2]  # never reaches token 3
    scores = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    frames, notes = word_frames(path, scores, targets=targets, delimiter=_DELIM)
    assert len(frames) == 3
    assert frames[0]["first_frame"] == 0 and frames[0]["last_frame"] == 1
    assert frames[1]["first_frame"] == 5 and frames[1]["last_frame"] == 6
    # interpolated: right after word 1's last frame, single frame wide.
    assert frames[2] == {"first_frame": 7, "last_frame": 7, "score": None}
    assert len(notes) == 1
    assert "word 2" in notes[0]


def test_word_frames_with_no_frames_at_all_falls_back_to_frame_zero() -> None:
    frames, notes = word_frames([0, 0, 0], [0.0, 0.0, 0.0], targets=[1], delimiter=_DELIM)
    assert frames == [{"first_frame": 0, "last_frame": 0, "score": None}]
    assert len(notes) == 1


# --- The upstream defect: the delimiter and the vocabulary case ---


def test_resolve_delimiter_prefers_the_tokenizer_attribute() -> None:
    class _Tokenizer:
        word_delimiter_token_id = 42

    assert w2v._resolve_delimiter(_Tokenizer(), {}, "some-model") == 42


def test_resolve_delimiter_falls_back_to_the_pipe_token() -> None:
    class _Tokenizer:
        pass

    assert w2v._resolve_delimiter(_Tokenizer(), {"|": 5}, "some-model") == 5


def test_resolve_delimiter_raises_naming_the_model_when_neither_resolves() -> None:
    class _Tokenizer:
        pass

    with pytest.raises(RytpError, match="some-model"):
        w2v._resolve_delimiter(_Tokenizer(), {}, "some-model")


def test_build_target_ids_uses_the_delimiter_only_between_words() -> None:
    vocabulary = {"а": 1, "б": 2, "|": 9}
    assert build_target_ids(["а", "б"], vocabulary, delimiter=9) == [1, 9, 2]
    assert build_target_ids(["а"], vocabulary, delimiter=9) == [1]


def test_build_target_ids_case_folds_to_the_vocabulary() -> None:
    # Vocabulary is lower-case only; an upper-case word must still encode.
    # "а" is the letter _vocabulary_case probes on to decide the model's case.
    vocabulary = {"а": 0, "п": 1, "р": 2, "и": 3, "в": 4, "е": 5, "т": 6, "|": 9}
    assert build_target_ids(["ПРИВЕТ"], vocabulary, delimiter=9) == [1, 2, 3, 4, 5, 6]


def test_build_target_ids_raises_naming_the_word_with_no_vocabulary_tokens() -> None:
    vocabulary = {"а": 1, "|": 9}
    with pytest.raises(RytpError, match="бб"):
        build_target_ids(["а", "бб"], vocabulary, delimiter=9)


def test_vocabulary_case_is_probed_not_assumed() -> None:
    assert w2v._vocabulary_case({"а": 1}) == "lower"
    assert w2v._vocabulary_case({"А": 1}) == "upper"
    assert w2v._vocabulary_case({}) == "asis"


# --- The strengthened guard: span count was always right, shape never was ---


def test_wav2vec2_aligner_rejects_equal_width_spans_for_differently_sized_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A regression to equal division must be caught even if it slips past
    the child, because the count-only guard (``AlignmentMismatchError``)
    never would have.
    """

    def _fake_run_child(**_kwargs: object) -> dict[str, object]:
        # Three same-width frames for three words of very different length —
        # exactly what the old, broken code produced.
        return {
            "frames": [
                {"first_frame": 0, "last_frame": 5, "score": -0.1},
                {"first_frame": 6, "last_frame": 11, "score": -0.1},
                {"first_frame": 12, "last_frame": 17, "score": -0.1},
            ],
            "notes": [],
        }

    monkeypatch.setattr(w2v, "run_child", _fake_run_child)
    aligner = Wav2Vec2Aligner(interpreter="fake-interpreter")
    with pytest.raises(RytpError, match="equal-width"):
        aligner.align(
            _AUDIO, ["а", "бб", "ввввв"], start_ms=0, end_ms=1000
        )


def test_wav2vec2_aligner_accepts_a_genuinely_unequal_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run_child(**_kwargs: object) -> dict[str, object]:
        return {
            "frames": [
                {"first_frame": 0, "last_frame": 1, "score": -1.5},
                {"first_frame": 5, "last_frame": 6, "score": -3.5},
                {"first_frame": 10, "last_frame": 19, "score": -5.0},
            ],
            "notes": [],
        }

    monkeypatch.setattr(w2v, "run_child", _fake_run_child)
    aligner = Wav2Vec2Aligner(interpreter="fake-interpreter")
    spans = aligner.align(
        _AUDIO, ["а", "бб", "ввввв"], start_ms=0, end_ms=1000
    )
    assert len({s.end_ms - s.start_ms for s in spans}) > 1


def test_wav2vec2_aligner_drains_child_notes_onto_the_notes_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run_child(**_kwargs: object) -> dict[str, object]:
        return {
            "frames": [
                {"first_frame": 0, "last_frame": 1, "score": -1.0},
                {"first_frame": 2, "last_frame": 3, "score": -2.0},
            ],
            "notes": ["wav2vec2: word 1 received no CTC frames; interpolated"],
        }

    monkeypatch.setattr(w2v, "run_child", _fake_run_child)
    aligner = Wav2Vec2Aligner(interpreter="fake-interpreter")
    aligner.align(_AUDIO, ["а", "б"], start_ms=0, end_ms=1000)
    assert aligner.notes == ["wav2vec2: word 1 received no CTC frames; interpolated"]
    # A second, clean call clears the previous call's notes.
    monkeypatch.setattr(
        w2v,
        "run_child",
        lambda **_kwargs: {
            "frames": [
                {"first_frame": 0, "last_frame": 1, "score": -1.0},
                {"first_frame": 2, "last_frame": 3, "score": -2.0},
            ],
            "notes": [],
        },
    )
    aligner.align(_AUDIO, ["а", "б"], start_ms=0, end_ms=1000)
    assert aligner.notes == []


def test_wav2vec2_aligner_still_rejects_a_word_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run_child(**_kwargs: object) -> dict[str, object]:
        return {"frames": [{"first_frame": 0, "last_frame": 1, "score": -1.0}], "notes": []}

    monkeypatch.setattr(w2v, "run_child", _fake_run_child)
    aligner = Wav2Vec2Aligner(interpreter="fake-interpreter")
    with pytest.raises(RytpError, match="aligned 1 of 2 words"):
        aligner.align(_AUDIO, ["а", "б"], start_ms=0, end_ms=1000)


# --- MFA audit (entry 36): the analogous risks in a separate adapter ---


_TEXTGRID_WITH_SILENCE_MARKERS = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2.0
        intervals: size = 5
        intervals [1]:
            xmin = 0
            xmax = 0.10
            text = "sil"
        intervals [2]:
            xmin = 0.10
            xmax = 0.60
            text = "привет"
        intervals [3]:
            xmin = 0.60
            xmax = 0.65
            text = "sp"
        intervals [4]:
            xmin = 0.65
            xmax = 1.20
            text = "мир"
        intervals [5]:
            xmin = 1.20
            xmax = 2.0
            text = "sil"
'''


def test_parse_textgrid_drops_silence_and_short_pause_markers() -> None:
    """Audit finding: MFA can write ``sil``/``sp``/``spn`` as labelled
    (non-empty) intervals directly in the words tier, not only as empty
    ones. Left in, they shift the interval count against the word count and
    either raise a spurious mismatch or misalign every following word.
    """
    intervals = parse_textgrid(_TEXTGRID_WITH_SILENCE_MARKERS)
    assert [label for _s, _e, label in intervals] == ["привет", "мир"]


def test_mfa_aligner_still_raises_on_a_genuine_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AlignmentMismatchError`` (entry 36's other adapter) already checks
    the count correctly here — MFA has no equal-division defect because it
    takes one TextGrid interval per word. Confirmed directly rather than
    assumed.
    """
    import rytp.transcribe.align.mfa as mfa_module

    def _fake_run_child(**_kwargs: object) -> dict[str, object]:
        return {
            "spans": [{"start_ms": 0, "end_ms": 100, "score": None}],
        }

    monkeypatch.setattr(mfa_module, "run_child", _fake_run_child)
    aligner = MfaAligner(interpreter="fake-interpreter")
    with pytest.raises(RytpError, match="aligned 1 of 2 words"):
        aligner.align(_AUDIO, ["привет", "мир"], start_ms=0, end_ms=1000)


# --- Score scale and device (plan §1a/§1b, BUGS.md entries 13, 34) ---


def test_wav2vec2_reports_the_logprob_scale() -> None:
    assert Wav2Vec2Aligner.score_scale == C.ALIGN_SCALE_LOGPROB


def test_mfa_reports_the_none_scale_and_no_gpu_path() -> None:
    assert MfaAligner.score_scale == C.ALIGN_SCALE_NONE
    assert MfaAligner.device == "n/a"


def test_mfa_accepts_and_ignores_a_device_kwarg() -> None:
    # Plan §1b: every adapter takes the parameter uniformly, even one with
    # no GPU path at all.
    aligner = MfaAligner(interpreter="fake-interpreter", device="cuda")
    assert aligner.device == "n/a"


def test_wav2vec2_threads_the_requested_device_and_reports_the_chosen_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_child(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs["request"])  # type: ignore[arg-type]
        return {
            "frames": [{"first_frame": 0, "last_frame": 1, "score": -1.0}],
            "notes": [],
            "device": "cpu",
        }

    monkeypatch.setattr(w2v, "run_child", _fake_run_child)
    aligner = Wav2Vec2Aligner(interpreter="fake-interpreter", device="cpu")
    assert aligner.device == "cpu"
    aligner.align(_AUDIO, ["а"], start_ms=0, end_ms=1000)
    assert captured["device"] == "cpu"
    assert aligner.device == "cpu"


def test_wav2vec2_device_defaults_to_auto_until_a_call_resolves_it() -> None:
    aligner = Wav2Vec2Aligner(interpreter="fake-interpreter")
    assert aligner.device == C.ENGINE_DEFAULT_DEVICE


# --- base.py stays import-light (contracts §6, module docstring) ---


def test_base_module_imports_nothing_outside_the_standard_library_and_models() -> None:
    """The out-of-process seam depends on this: `base.py` is imported again
    inside a foreign interpreter where only the standard library and
    :mod:`rytp.models` are guaranteed to load.
    """
    import ast
    import inspect
    import sys

    import rytp.transcribe.base as base_module

    source = inspect.getsource(base_module)
    tree = ast.parse(source)
    allowed_top_level = set(sys.stdlib_module_names) | {"rytp"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top in allowed_top_level, f"base.py imports {alias.name!r}"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            if node.module.startswith("rytp"):
                assert node.module == "rytp.models", f"base.py imports {node.module!r}"
            else:
                top = node.module.split(".")[0]
                assert top in allowed_top_level, f"base.py imports {node.module!r}"


# --- MFA's binary check tells the truth (BUGS.md entry 7) ---


def test_mfa_declares_the_binary_it_needs_not_a_module() -> None:
    # MFA is a conda binary, never a `pip`-importable module (BUGS.md entry
    # 7's exception): `required_module` stays `None` and the sibling
    # attribute names the binary instead.
    assert MfaAligner.required_module is None
    assert MfaAligner.required_binary == "mfa"


def test_availability_reports_ready_when_the_mfa_binary_is_found(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = tmp_path / "python"
    interpreter.write_text("", encoding="utf-8")
    (tmp_path / "mfa").write_text("", encoding="utf-8")
    monkeypatch.setattr(registry, "interpreter_for", lambda db, name: str(interpreter))
    state = registry.availability(db, MfaAligner)
    assert state == "ready"


def test_availability_names_the_conda_remedy_when_the_mfa_binary_is_absent(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = tmp_path / "python"
    interpreter.write_text("", encoding="utf-8")
    monkeypatch.setattr(registry, "interpreter_for", lambda db, name: str(interpreter))
    monkeypatch.setattr("shutil.which", lambda name: None)
    state = registry.availability(db, MfaAligner)
    assert "mfa" in state
    assert "conda" in state
    assert state != "interpreter ok"


def test_check_available_raises_the_conda_remedy_when_the_mfa_binary_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.transcribe.base import EngineUnavailable

    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(EngineUnavailable) as excinfo:
        registry.check_available(MfaAligner, interpreter=str(tmp_path / "python"))
    assert "conda" in str(excinfo.value)
