from abc import ABC, abstractmethod
from typing import List
from .schemas import JudgeResult
import concurrent.futures
from tqdm import tqdm


class Judge(ABC):
    """Abstract judge interface.

    Contract:
    - compare(answer_a, answer_b, prompt): returns a boolean indicating equivalence.
    - compare_batch(answers_a, answers_b, prompts): returns a `JudgeResult` which is a list of
      booleans (one per pair) indicating equivalence, processed in parallel.
    """

    @abstractmethod
    def compare(self, answer_a: str, answer_b: str, prompt: str) -> bool:
        pass

    def compare_batch(self, answers_a: List[str], answers_b: List[str], prompts: List[str]) -> JudgeResult:
        if not (len(answers_a) == len(answers_b) == len(prompts)):
            raise ValueError("Answer and prompt lists must be the same length")

        if not prompts:
            return JudgeResult(scores=[])

        max_workers = min(32, len(prompts))
        scores: List[bool] = [False] * len(prompts)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.compare, a, b, p): i
                for i, (a, b, p) in enumerate(zip(answers_a, answers_b, prompts))
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(prompts), desc="Judging"):
                index = future_to_index[future]
                try:
                    scores[index] = future.result()
                except Exception:
                    scores[index] = False  # Default to False on error

        return JudgeResult(scores=scores)
