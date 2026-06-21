"""
evaluator/pipeline.py — Runs all 4 metrics on a RAG dataset.

Input: list of { question, context, answer, ground_truth } dicts
Output: EvaluationResult with per-sample and aggregate scores
"""

from dataclasses import dataclass, field
from evaluator.metrics.faithfulness import evaluate_faithfulness, FaithfulnessResult
from evaluator.metrics.relevance import evaluate_relevance, RelevanceResult
from evaluator.metrics.completeness import evaluate_completeness, CompletenessResult
from evaluator.metrics.precision import evaluate_precision, PrecisionResult


@dataclass
class SampleResult:
    id: str
    question: str
    answer: str
    faithfulness: FaithfulnessResult
    relevance: RelevanceResult
    completeness: CompletenessResult
    precision: PrecisionResult

    @property
    def overall(self) -> float:
        return round(
            (self.faithfulness.score + self.relevance.score +
             self.completeness.score + self.precision.score) / 4,
            4
        )


@dataclass
class AggregateScores:
    faithfulness: float
    relevance: float
    completeness: float
    precision: float
    overall: float
    n_samples: int


@dataclass
class EvaluationResult:
    samples: list[SampleResult] = field(default_factory=list)
    aggregate: AggregateScores = None

    def compute_aggregate(self):
        if not self.samples:
            return

        n = len(self.samples)
        self.aggregate = AggregateScores(
            faithfulness=round(sum(s.faithfulness.score for s in self.samples) / n, 4),
            relevance=round(sum(s.relevance.score for s in self.samples) / n, 4),
            completeness=round(sum(s.completeness.score for s in self.samples) / n, 4),
            precision=round(sum(s.precision.score for s in self.samples) / n, 4),
            overall=round(sum(s.overall for s in self.samples) / n, 4),
            n_samples=n,
        )


def evaluate_dataset(dataset: list[dict]) -> EvaluationResult:
    """
    Run all 4 metrics on every sample in the dataset.

    Each sample must have:
        - id: str
        - question: str
        - context: list[str]
        - answer: str
        - ground_truth: str

    Returns:
        EvaluationResult with per-sample results and aggregate scores
    """
    result = EvaluationResult()

    for i, sample in enumerate(dataset):
        sample_id = sample.get("id", str(i))
        question = sample.get("question", "")
        context = sample.get("context", [])
        answer = sample.get("answer", "")
        ground_truth = sample.get("ground_truth", "")

        faith = evaluate_faithfulness(answer, context)
        rel = evaluate_relevance(question, context)
        comp = evaluate_completeness(answer, ground_truth)
        prec = evaluate_precision(answer, context)

        result.samples.append(SampleResult(
            id=sample_id,
            question=question,
            answer=answer,
            faithfulness=faith,
            relevance=rel,
            completeness=comp,
            precision=prec,
        ))

    result.compute_aggregate()
    return result