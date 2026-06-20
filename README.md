# RAG Evaluation Framework

A standalone evaluation tool for any RAG pipeline — measures 4 quality metrics and generates JSON + HTML reports.

🚧 **Work in progress.** Full implementation coming soon.

---

## Why This Project

`rag-system-from-scratch` built the pipeline. This project answers the question nobody asks enough:

**How do you know if your RAG is actually good?**

Most RAG projects measure nothing. This framework gives you numbers.

---

## Planned Metrics

| Metric | Question it answers |
|---|---|
| **Faithfulness** | Does the answer come from the context, or did the LLM hallucinate? |
| **Relevance** | Is the retrieved context actually related to the question? |
| **Completeness** | Does the answer cover everything the context contains? |
| **Precision** | Is the context clean, or full of irrelevant chunks? |

---

## Planned Flow

```
Your RAG pipeline
      ↓
  { question, context, answer }
      ↓
[ Evaluation Framework ]
      ↓
  faithfulness_score: 0.91
  relevance_score:    0.87
  completeness_score: 0.73
  precision_score:    0.84
      ↓
  reports/eval_<timestamp>.json
  reports/eval_<timestamp>.html
```

---

## Build Checklist

- [ ] `evaluator/metrics/faithfulness.py` — hallucination detection
- [ ] `evaluator/metrics/relevance.py` — context-question alignment
- [ ] `evaluator/metrics/completeness.py` — answer coverage
- [ ] `evaluator/metrics/precision.py` — context noise detection
- [ ] `evaluator/pipeline.py` — runs all 4 metrics on a dataset
- [ ] `evaluator/report.py` — JSON + HTML output
- [ ] `examples/` — sample RAG output to test against
- [ ] `tests/` — unit tests for each metric
- [ ] `main.py` — CLI
- [ ] `requirements.txt` + `.gitignore`
- [ ] Replace this README with full documentation

---

## Planned Structure

```
rag-evaluation-framework/
├── evaluator/
│   ├── __init__.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── faithfulness.py
│   │   ├── relevance.py
│   │   ├── completeness.py
│   │   └── precision.py
│   ├── pipeline.py
│   └── report.py
├── examples/
│   └── sample_rag_output.json
├── tests/
│   └── test_metrics.py
├── main.py
├── requirements.txt
└── .gitignore
```

---

## Stack

Python · Ollama · sentence-transformers · numpy · pytest

---

## Related Projects

- [rag-system-from-scratch](https://github.com/Honaxen/rag-system-from-scratch) — the RAG pipeline this evaluates

---

## Author

[Honaxen](https://github.com/Honaxen)