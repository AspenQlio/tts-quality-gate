from __future__ import annotations

from collections.abc import Sequence

from tts_quality_gate.domain import CaseResult, EvaluationConfig, EvaluationReport, EvalCase
from tts_quality_gate.metrics import calculate_wer
from tts_quality_gate.transcriber import Transcriber


def run_evaluation(
    cases: Sequence[EvalCase],
    transcriber: Transcriber,
    config: EvaluationConfig,
) -> EvaluationReport:
    """Runs the full evaluation pipeline over a dataset."""
    results: list[CaseResult] = []
    total_wer = 0.0

    for case in cases:
        if not case.audio_path:
            raise ValueError(f"Case {case.id} is missing an audio path to evaluate.")

        hypothesis = transcriber.transcribe(case.audio_path)
        wer = calculate_wer(case.reference_text, hypothesis)

        passed = wer <= config.wer_threshold

        results.append(
            CaseResult(
                case_id=case.id,
                reference=case.reference_text,
                hypothesis=hypothesis,
                wer=wer,
                passed=passed,
            )
        )
        total_wer += wer

    aggregate_wer = total_wer / len(cases) if cases else 0.0
    passed_cases = sum(1 for r in results if r.passed)

    return EvaluationReport(
        total_cases=len(cases),
        passed_cases=passed_cases,
        aggregate_wer=aggregate_wer,
        results=tuple(results),
    )
