"""
metrics/precision.py — Measures context precision (signal vs noise).

Question: Is the retrieved context clean, or is it full of irrelevant chunks?

Method:
  - For each context chunk, measure how much of it is used in the answer
  - Score = answer_tokens_in_chunk / chunk_tokens (averaged across chunks)
  - Low precision = lots of context that didn't contribute to the answer

Score range: 0.0 (context all noise) → 1.0 (every context word used in answer)
"""

import re
from dataclasses import dataclass


@dataclass
class PrecisionResult:
    score: float              # 0.0 - 1.0
    chunk_scores: list[float] # per-chunk precision
    noisy_chunks: list[int]   # indices of low-precision chunks


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))


def _chunk_precision(answer: str, chunk: str) -> float:
    """
    What fraction of the chunk's content actually appears in the answer?
    """
    chunk_tokens = _tokenize(chunk)
    answer_tokens = _tokenize(answer)

    if not chunk_tokens:
        return 0.0

    used = chunk_tokens & answer_tokens
    return round(len(used) / len(chunk_tokens), 4)


def evaluate_precision(
    answer: str,
    context: list[str],
    noise_threshold: float = 0.1,
) -> PrecisionResult:
    """
    Evaluate how much of the retrieved context was actually used.

    Args:
        answer: the RAG system's generated answer
        context: list of retrieved context chunks
        noise_threshold: chunks below this score are flagged as noisy

    Returns:
        PrecisionResult with overall score and per-chunk breakdown
    """
    if not context:
        return PrecisionResult(score=0.0, chunk_scores=[], noisy_chunks=[])

    chunk_scores = [_chunk_precision(answer, chunk) for chunk in context]
    avg_score = sum(chunk_scores) / len(chunk_scores)
    noisy = [i for i, s in enumerate(chunk_scores) if s < noise_threshold]

    return PrecisionResult(
        score=round(avg_score, 4),
        chunk_scores=chunk_scores,
        noisy_chunks=noisy,
    )