from typing import List
from .base import Judge
from .factory import JudgeFactory
from .schemas import JudgeResult


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


@JudgeFactory.register("simple")
class SimpleJudge(Judge):
    """Heuristic judge: token-normalized equality and boolean equivalence.

    Rules:
    - exact normalized match => equivalent=True
    - else use a simple overlap heuristic; equivalent=True if overlap ratio >= 0.8
    - score is always boolean and mirrors `equivalent`
    """

    def compare(self, answer_a: str, answer_b: str, prompt: str) -> bool:
        na, nb = _normalize(answer_a), _normalize(answer_b)
        if na == nb:
            return True
        else:
            # very simple similarity: overlap of characters / max length
            overlap = len(set(na) & set(nb))
            denom = max(len(na), len(nb)) or 1
            ratio = overlap / denom
            return ratio >= 0.8

    def compare_batch(self, answers_a: List[str], answers_b: List[str], prompts: List[str]) -> JudgeResult:
        return super().compare_batch(answers_a, answers_b, prompts)
