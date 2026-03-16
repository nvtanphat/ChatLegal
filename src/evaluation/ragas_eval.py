from __future__ import annotations

from dataclasses import dataclass

from src.inference.inference_engine import InferenceEngine

FAILURE_MARKERS = (
    "tam thoi gap loi",
    "dang gap loi",
    "khong phan hoi",
    "kiem tra ollama",
    "khong the",
    "toi chua tim thay can cu",
)


@dataclass(slots=True)
class EvalResult:
    total: int
    answered: int
    answer_rate: float


def _is_answered(answer: str) -> bool:
    normalized = (answer or "").strip().lower()
    if not normalized:
        return False
    return not any(marker in normalized for marker in FAILURE_MARKERS)


def run_basic_eval(engine: InferenceEngine, golden_samples: list[dict]) -> EvalResult:
    if not golden_samples:
        return EvalResult(total=0, answered=0, answer_rate=0.0)

    answered = 0
    for sample in golden_samples:
        query = sample.get("question", "")
        output = engine.ask(query, session_id="eval")
        if _is_answered(output.answer):
            answered += 1
    total = len(golden_samples)
    return EvalResult(total=total, answered=answered, answer_rate=answered / total)
