"""
metrics/completeness.py — Measures answer completeness vs ground truth.

Question: Does the answer cover everything it should?

Method:
  - Compare answer tokens against ground truth tokens
  - Score = answer_tokens_in_ground_truth / ground_truth_tokens
  - Missing key terms are flagged

Score range: 0.0 (answer missed everything) → 1.0 (answer covered everything)
"""

import re
from dataclasses import dataclass


@dataclass
class CompletenessResult:
    score: float            # 0.0 - 1.0
    covered: int            # ground truth tokens covered by answer
    total: int              # total ground truth tokens
    missing_terms: list[str]  # key terms in ground truth not in answer


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))


def evaluate_completeness(
    answer: str,
    ground_truth: str,
) -> CompletenessResult:
    """
    Evaluate how completely the answer covers the ground truth.

    Args:
        answer: the RAG system's generated answer
        ground_truth: the expected / reference answer

    Returns:
        CompletenessResult with score and missing terms
    """
    answer_tokens = _tokenize(answer)
    truth_tokens = _tokenize(ground_truth)

    if not truth_tokens:
        return CompletenessResult(score=1.0, covered=0, total=0, missing_terms=[])

    covered = answer_tokens & truth_tokens
    missing = truth_tokens - answer_tokens

    score = len(covered) / len(truth_tokens)

    return CompletenessResult(
        score=round(score, 4),
        covered=len(covered),
        total=len(truth_tokens),
        missing_terms=sorted(missing),
    )