"""
metrics/relevance.py — Measures context relevance to the question.

Question: Is the retrieved context actually related to the question?

Method:
  - Compute token overlap between question and each context chunk
  - Score = average overlap across all chunks
  - Chunks with zero overlap are flagged as irrelevant

Score range: 0.0 (context totally unrelated) → 1.0 (context perfectly aligned)
"""

import re
from dataclasses import dataclass, field


@dataclass
class RelevanceResult:
    score: float                    # 0.0 - 1.0 average across chunks
    chunk_scores: list[float]       # per-chunk relevance scores
    irrelevant_chunks: list[int]    # indices of chunks with score < threshold


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))


def _chunk_relevance(question: str, chunk: str) -> float:
    """
    Compute relevance of one chunk to the question.

    Uses token overlap: how many question keywords appear in the chunk?
    Score = matching_tokens / question_tokens
    """
    q_tokens = _tokenize(question)
    c_tokens = _tokenize(chunk)

    if not q_tokens:
        return 0.0

    overlap = len(q_tokens & c_tokens)
    return round(overlap / len(q_tokens), 4)


def evaluate_relevance(
    question: str,
    context: list[str],
    irrelevance_threshold: float = 0.1,
) -> RelevanceResult:
    """
    Evaluate how relevant the retrieved context is to the question.

    Args:
        question: the user's question
        context: list of retrieved context chunks
        irrelevance_threshold: chunks below this score are flagged irrelevant

    Returns:
        RelevanceResult with overall score and per-chunk breakdown
    """
    if not context:
        return RelevanceResult(score=0.0, chunk_scores=[], irrelevant_chunks=[])

    chunk_scores = [_chunk_relevance(question, chunk) for chunk in context]
    avg_score = sum(chunk_scores) / len(chunk_scores)
    irrelevant = [i for i, s in enumerate(chunk_scores) if s < irrelevance_threshold]

    return RelevanceResult(
        score=round(avg_score, 4),
        chunk_scores=chunk_scores,
        irrelevant_chunks=irrelevant,
    )