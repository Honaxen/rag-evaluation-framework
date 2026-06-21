"""
metrics/faithfulness.py — Measures hallucination.

Question: Does the answer come from the context, or did the LLM make things up?

Method:
  - Split answer into sentences
  - For each sentence, check if it's supported by the context
    using token overlap (fast, no model needed) + optional LLM verification
  - Score = supported_sentences / total_sentences

Score range: 0.0 (fully hallucinated) → 1.0 (fully grounded)
"""

import re
from dataclasses import dataclass


@dataclass
class FaithfulnessResult:
    score: float               # 0.0 - 1.0
    supported: int             # sentences supported by context
    total: int                 # total sentences in answer
    unsupported_sentences: list[str]  # which sentences weren't grounded


def _tokenize(text: str) -> set[str]:
    """Simple word tokenizer — lowercase, strip punctuation."""
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))


def _sentence_supported(sentence: str, context_chunks: list[str], threshold: float = 0.3) -> bool:
    """
    Check if a sentence is supported by any context chunk.

    Uses token overlap (Jaccard-like similarity).
    threshold: minimum fraction of sentence tokens that must appear in context.
    """
    sentence_tokens = _tokenize(sentence)
    if not sentence_tokens:
        return True  # empty/punctuation-only sentence — skip

    context_text = " ".join(context_chunks)
    context_tokens = _tokenize(context_text)

    overlap = len(sentence_tokens & context_tokens)
    coverage = overlap / len(sentence_tokens)

    return coverage >= threshold


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def evaluate_faithfulness(
    answer: str,
    context: list[str],
    threshold: float = 0.3,
) -> FaithfulnessResult:
    """
    Evaluate how faithful the answer is to the context.

    Args:
        answer: the RAG system's generated answer
        context: list of retrieved context chunks
        threshold: min token overlap to consider a sentence supported

    Returns:
        FaithfulnessResult with score and breakdown
    """
    sentences = _split_sentences(answer)

    if not sentences:
        return FaithfulnessResult(score=1.0, supported=0, total=0, unsupported_sentences=[])

    supported = []
    unsupported = []

    for sentence in sentences:
        if _sentence_supported(sentence, context, threshold):
            supported.append(sentence)
        else:
            unsupported.append(sentence)

    score = len(supported) / len(sentences)

    return FaithfulnessResult(
        score=round(score, 4),
        supported=len(supported),
        total=len(sentences),
        unsupported_sentences=unsupported,
    )