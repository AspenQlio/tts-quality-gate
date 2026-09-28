from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tts_quality_gate.domain import EvaluationConfig, EvalCase
from tts_quality_gate.runner import run_evaluation
from tts_quality_gate.transcriber import build_transcriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts-quality-gate",
        description="Automated quality gate for TTS models.",
    )
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset.")
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="Maximum allowed WER (0.0 to 1.0)."
    )
    parser.add_argument("--backend", default="mock", help="Transcription backend to use.")
    parser.add_argument("--json", dest="json_path", help="Path to write the JSON report.")
    return parser


def load_dataset(path: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            cases.append(EvalCase.model_validate(data))
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cases = load_dataset(args.dataset)
        config = EvaluationConfig(
            wer_threshold=args.threshold,
            transcription_backend=args.backend,
        )
        transcriber = build_transcriber(config.transcription_backend)

        report = run_evaluation(cases, transcriber, config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json_path:
        target = Path(args.json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")

    print(f"Total cases: {report.total_cases}")
    print(f"Passed: {report.passed_cases}/{report.total_cases}")
    print(f"Aggregate WER: {report.aggregate_wer:.4f}")

    return 0 if report.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
