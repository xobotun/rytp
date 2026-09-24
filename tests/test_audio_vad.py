"""Energy-based voice activity detection."""
from __future__ import annotations

from rytp import constants as C
from rytp.audio.vad import SpeechSegment, detect_speech
from tests.synth_audio import SR, concat, silence, tone


def test_two_utterances_separated_by_silence_are_two_segments() -> None:
    samples = concat(tone(400), silence(400, amp=0.001, seed=2), tone(400))
    segments = detect_speech(samples, SR, pad_ms=0)
    assert len(segments) == 2
    assert abs(segments[0].start_ms - 0) <= 30
    assert abs(segments[0].end_ms - 400) <= 30
    assert abs(segments[1].start_ms - 800) <= 30
    assert abs(segments[1].end_ms - 1200) <= 30


def test_a_gap_shorter_than_the_minimum_silence_is_swallowed() -> None:
    gap = C.VAD_MIN_SILENCE_MS - 60
    samples = concat(tone(400), silence(gap, amp=0.001, seed=3), tone(400))
    segments = detect_speech(samples, SR, pad_ms=0)
    assert len(segments) == 1


def test_a_burst_shorter_than_the_minimum_speech_is_discarded() -> None:
    burst = C.VAD_MIN_SPEECH_MS - 60
    samples = concat(
        silence(400, amp=0.001, seed=4), tone(burst), silence(400, amp=0.001, seed=5)
    )
    assert detect_speech(samples, SR, pad_ms=0) == []


def test_digital_silence_has_no_speech() -> None:
    assert detect_speech(silence(1000), SR) == []


def test_continuous_speech_with_no_silence_is_one_segment() -> None:
    # The percentile noise floor sits inside speech here, so the hysteresis
    # band finds nothing. Treating the whole file as speech is the safe answer:
    # chunk planning still works and the transcriber sees everything.
    samples = tone(2000)
    segments = detect_speech(samples, SR)
    assert segments == [SpeechSegment(start_ms=0, end_ms=2000)]


def test_padding_widens_a_segment_and_is_clipped_to_the_audio() -> None:
    samples = concat(tone(400), silence(400, amp=0.001, seed=6), tone(400))
    tight = detect_speech(samples, SR, pad_ms=0)
    padded = detect_speech(samples, SR, pad_ms=50)
    assert padded[0].start_ms == 0
    assert padded[0].end_ms >= tight[0].end_ms
    assert padded[-1].end_ms <= 1200


def test_detection_is_deterministic() -> None:
    samples = concat(tone(400), silence(400, amp=0.001, seed=7), tone(400))
    assert detect_speech(samples, SR) == detect_speech(samples, SR)
