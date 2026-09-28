from __future__ import annotations

from tts_quality_gate.domain import EvaluationConfig, EvalCase
from tts_quality_gate.runner import run_evaluation
from tts_quality_gate.transcriber import MockTranscriber


def test_run_evaluation_passes_when_wer_is_below_threshold():
    cases = [
        EvalCase(id="c1", reference_text="hello world", audio_path="c1.wav"),
        EvalCase(id="c2", reference_text="goodbye", audio_path="c2.wav"),
    ]
    transcriber = MockTranscriber(
        forced_responses={
            "c1.wav": "hello world",
            "c2.wav": "goodbye",
        }
    )
    config = EvaluationConfig(wer_threshold=0.0)

    report = run_evaluation(cases, transcriber, config)

    assert report.success is True
    assert report.aggregate_wer == 0.0
    assert report.passed_cases == 2


def test_run_evaluation_fails_when_wer_exceeds_threshold():
    cases = [
        EvalCase(id="c1", reference_text="hello world", audio_path="c1.wav"),
    ]
    transcriber = MockTranscriber(
        forced_responses={
            "c1.wav": "hello there",  # 0.5 WER
        }
    )
    config = EvaluationConfig(wer_threshold=0.1)

    report = run_evaluation(cases, transcriber, config)

    assert report.success is False
    assert report.aggregate_wer == 0.5
    assert report.passed_cases == 0
