from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    reference_text: str
    audio_path: str | None = None
    group: str = "default"


class EvaluationConfig(BaseModel):
    wer_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    transcription_backend: str = "mock"


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    reference: str
    hypothesis: str
    wer: float
    passed: bool


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    total_cases: int
    passed_cases: int
    aggregate_wer: float
    results: tuple[CaseResult, ...]

    @property
    def success(self) -> bool:
        return self.total_cases > 0 and self.passed_cases == self.total_cases

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "aggregate_wer": round(self.aggregate_wer, 4),
            "success": self.success,
            "results": [
                {
                    "case_id": r.case_id,
                    "reference": r.reference,
                    "hypothesis": r.hypothesis,
                    "wer": round(r.wer, 4),
                    "passed": r.passed,
                }
                for r in self.results
            ],
        }
