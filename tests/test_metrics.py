"""
tests/test_metrics.py — Unit tests for all 4 evaluation metrics.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluator.metrics.faithfulness import evaluate_faithfulness
from evaluator.metrics.relevance import evaluate_relevance
from evaluator.metrics.completeness import evaluate_completeness
from evaluator.metrics.precision import evaluate_precision
from evaluator.pipeline import evaluate_dataset


# ─────────────────────────────────────────────
# Faithfulness
# ─────────────────────────────────────────────

class TestFaithfulness:
    def test_fully_supported_answer(self):
        context = ["Paris is the capital of France."]
        answer = "The capital of France is Paris."
        result = evaluate_faithfulness(answer, context)
        assert result.score > 0.5

    def test_fully_hallucinated_answer(self):
        context = ["Paris is the capital of France."]
        answer = "Mount Everest is the tallest mountain in the world."
        result = evaluate_faithfulness(answer, context)
        assert result.score < 0.5
        assert len(result.unsupported_sentences) > 0

    def test_empty_answer(self):
        result = evaluate_faithfulness("", ["some context"])
        assert result.score == 1.0
        assert result.total == 0

    def test_empty_context(self):
        result = evaluate_faithfulness("An answer.", [])
        assert result.score < 1.0

    def test_score_range(self):
        result = evaluate_faithfulness("test answer", ["test context"])
        assert 0.0 <= result.score <= 1.0


# ─────────────────────────────────────────────
# Relevance
# ─────────────────────────────────────────────

class TestRelevance:
    def test_relevant_context(self):
        question = "What is the capital of France?"
        context = ["Paris is the capital of France."]
        result = evaluate_relevance(question, context)
        assert result.score > 0.3

    def test_irrelevant_context(self):
        question = "What is the capital of France?"
        context = ["The stock market rose yesterday by two points."]
        result = evaluate_relevance(question, context)
        assert result.score < 0.3
        assert 0 in result.irrelevant_chunks

    def test_empty_context(self):
        result = evaluate_relevance("a question", [])
        assert result.score == 0.0

    def test_chunk_scores_length(self):
        context = ["chunk one", "chunk two", "chunk three"]
        result = evaluate_relevance("question", context)
        assert len(result.chunk_scores) == 3

    def test_score_range(self):
        result = evaluate_relevance("question", ["context"])
        assert 0.0 <= result.score <= 1.0


# ─────────────────────────────────────────────
# Completeness
# ─────────────────────────────────────────────

class TestCompleteness:
    def test_complete_answer(self):
        answer = "Alexander Graham Bell invented the telephone in 1876."
        ground_truth = "Alexander Graham Bell invented the telephone in 1876."
        result = evaluate_completeness(answer, ground_truth)
        assert result.score == 1.0

    def test_partial_answer(self):
        answer = "Bell invented the telephone."
        ground_truth = "Alexander Graham Bell invented the telephone in 1876."
        result = evaluate_completeness(answer, ground_truth)
        assert 0.0 < result.score < 1.0

    def test_empty_ground_truth(self):
        result = evaluate_completeness("some answer", "")
        assert result.score == 1.0

    def test_missing_terms_identified(self):
        answer = "Bell invented the telephone."
        ground_truth = "Alexander Graham Bell invented the telephone in 1876."
        result = evaluate_completeness(answer, ground_truth)
        assert len(result.missing_terms) > 0

    def test_score_range(self):
        result = evaluate_completeness("answer", "ground truth answer")
        assert 0.0 <= result.score <= 1.0


# ─────────────────────────────────────────────
# Precision
# ─────────────────────────────────────────────

class TestPrecision:
    def test_high_precision_context(self):
        answer = "Paris is the capital of France."
        context = ["Paris is the capital of France."]
        result = evaluate_precision(answer, context)
        assert result.score > 0.5

    def test_noisy_context(self):
        answer = "Paris is the capital."
        context = ["Completely unrelated text about something else entirely different."]
        result = evaluate_precision(answer, context)
        assert result.score < 0.3
        assert 0 in result.noisy_chunks

    def test_empty_context(self):
        result = evaluate_precision("answer", [])
        assert result.score == 0.0

    def test_chunk_scores_length(self):
        context = ["chunk a", "chunk b"]
        result = evaluate_precision("answer text", context)
        assert len(result.chunk_scores) == 2

    def test_score_range(self):
        result = evaluate_precision("answer", ["context"])
        assert 0.0 <= result.score <= 1.0


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

class TestPipeline:
    def sample_dataset(self):
        return [
            {
                "id": "q1",
                "question": "What is the capital of France?",
                "context": ["Paris is the capital of France."],
                "answer": "The capital of France is Paris.",
                "ground_truth": "Paris is the capital of France."
            },
            {
                "id": "q2",
                "question": "Who invented the telephone?",
                "context": ["Alexander Graham Bell invented the telephone in 1876."],
                "answer": "Bell invented the telephone.",
                "ground_truth": "Alexander Graham Bell invented the telephone."
            }
        ]

    def test_correct_number_of_samples(self):
        result = evaluate_dataset(self.sample_dataset())
        assert len(result.samples) == 2

    def test_aggregate_computed(self):
        result = evaluate_dataset(self.sample_dataset())
        assert result.aggregate is not None
        assert result.aggregate.n_samples == 2

    def test_aggregate_scores_in_range(self):
        result = evaluate_dataset(self.sample_dataset())
        agg = result.aggregate
        for score in [agg.faithfulness, agg.relevance, agg.completeness, agg.precision, agg.overall]:
            assert 0.0 <= score <= 1.0

    def test_sample_ids_preserved(self):
        result = evaluate_dataset(self.sample_dataset())
        ids = [s.id for s in result.samples]
        assert "q1" in ids and "q2" in ids

    def test_empty_dataset(self):
        result = evaluate_dataset([])
        assert len(result.samples) == 0
        assert result.aggregate is None