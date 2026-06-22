# RAG Evaluation Framework

A standalone evaluation tool for any RAG pipeline — measures 4 quality metrics and generates JSON + HTML reports. No external API, no paid service.

---

## Why This Project

`rag-system-from-scratch` built the pipeline. This project answers the question nobody asks enough:

**How do you know if your RAG is actually good?**

Most RAG projects measure nothing. This framework gives you numbers.

---

## Metrics

| Metric | Question it answers | Range |
|---|---|---|
| **Faithfulness** | Does the answer come from the context, or did the LLM hallucinate? | 0–1 |
| **Relevance** | Is the retrieved context actually related to the question? | 0–1 |
| **Completeness** | Does the answer cover everything the ground truth contains? | 0–1 |
| **Precision** | Is the context clean, or full of irrelevant chunks? | 0–1 |

---

## How It Works

```
Your RAG pipeline output (JSON)
      ↓
{ question, context, answer, ground_truth }
      ↓
[ Evaluation Framework ]
      ↓
  faithfulness:  0.80
  relevance:     0.52
  completeness:  0.75
  precision:     0.29
  overall:       0.59
      ↓
reports/eval_<timestamp>.json
reports/eval_<timestamp>.html
```

---

## Project Structure

```
rag-evaluation-framework/
├── evaluator/
│   ├── __init__.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── faithfulness.py  — hallucination detection via token overlap
│   │   ├── relevance.py     — context-question alignment
│   │   ├── completeness.py  — answer coverage vs ground truth
│   │   └── precision.py     — context noise detection
│   ├── pipeline.py          — runs all 4 metrics on a dataset
│   └── report.py            — JSON + HTML output generator
├── examples/
│   └── sample_rag_output.json
├── tests/
│   └── test_metrics.py      — 20 unit tests
├── main.py                  — CLI
├── requirements.txt
└── .gitignore
```

---

## Getting Started

```bash
pip install -r requirements.txt
```

### Run on sample data

```bash
python3 main.py examples/sample_rag_output.json
```

Output:

```
RAG Evaluation Framework
Evaluating 5 samples...

────────────────────────────────────────────
  Metric                Score  Bar
────────────────────────────────────────────
  Faithfulness           0.80  ████████████████░░░░
  Relevance              0.52  ██████████░░░░░░░░░░
  Completeness           0.75  ███████████████░░░░░
  Precision              0.29  █████░░░░░░░░░░░░░░░
────────────────────────────────────────────
  Overall                0.59

Per-sample results:
  [q1] overall=0.75  faith=1.00  rel=0.75  comp=1.00  prec=0.24
  [q5] overall=0.09  faith=0.00  rel=0.33  comp=0.00  prec=0.04 ⚠
```

`q5` scored 0.09 because the answer was completely off-topic — the framework caught the hallucination automatically.

### Run on your own RAG output

Your JSON file must follow this format:

```json
[
  {
    "id": "q1",
    "question": "What is the capital of France?",
    "context": ["Paris is the capital and most populous city of France."],
    "answer": "The capital of France is Paris.",
    "ground_truth": "Paris"
  }
]
```

```bash
python3 main.py my_rag_output.json --output-dir results/
```

### Run tests

```bash
pytest tests/ -v
```

---

## Stack

Python · pytest (no external dependencies for evaluation)

---

## What I Learned

Evaluation is harder than building.
Anyone can build a RAG pipeline. Knowing if it's good requires defining "good" precisely enough to measure it.

Token overlap is a surprisingly strong baseline.
No embeddings, no LLM calls — just word overlap correctly identified a fully hallucinated answer (`q5`) with a score of 0.09.

Each metric reveals a different failure mode.
Low faithfulness = hallucination. Low relevance = bad retrieval. Low completeness = incomplete answers. Low precision = too much noise in context. You need all four to understand where your pipeline is breaking.

---

## Related Projects

- [rag-system-from-scratch](https://github.com/Honaxen/rag-system-from-scratch) — the RAG pipeline this evaluates

---

## Author

[Honaxen](https://github.com/Honaxen)