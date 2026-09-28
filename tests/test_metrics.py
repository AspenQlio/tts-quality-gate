from __future__ import annotations

from tts_quality_gate.metrics import calculate_wer


def test_wer_is_zero_for_perfect_match():
    assert calculate_wer("hello world", "hello world") == 0.0


def test_wer_ignores_case_and_punctuation():
    assert calculate_wer("Hello, World!", "hello world") == 0.0


def test_wer_calculates_substitutions():
    # 1 substitution in 2 words = 0.5
    assert calculate_wer("hello world", "hello there") == 0.5


def test_wer_handles_empty_reference():
    assert calculate_wer("", "noise") == 1.0
    assert calculate_wer("", "") == 0.0
