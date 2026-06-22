"""
main.py — CLI for the RAG Evaluation Framework.

Usage:
    python3 main.py examples/sample_rag_output.json
    python3 main.py my_rag_output.json --output-dir results/
"""

import argparse
import json
import sys

from evaluator.pipeline import evaluate_dataset
from evaluator.report import generate_reports

RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED   = "\033[31m"
CYAN  = "\033[36m"
DIM   = "\033[2m"


def _color(score: float) -> str:
    if score >= 0.8:
        return GREEN
    elif score >= 0.5:
        return YELLOW
    return RED


def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG pipeline output")
    parser.add_argument("input", help="Path to RAG output JSON file")
    parser.add_argument("--output-dir", default="reports", help="Directory for reports")
    args = parser.parse_args()

    # Load dataset
    try:
        with open(args.input) as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{BOLD}RAG Evaluation Framework{RESET}")
    print(f"{DIM}Evaluating {len(dataset)} samples...{RESET}\n")

    # Run evaluation
    result = evaluate_dataset(dataset)
    agg = result.aggregate

    # Print results
    print(f"{CYAN}{BOLD}{'─'*44}{RESET}")
    print(f"  {'Metric':<18} {'Score':>8}  {'Bar'}")
    print(f"{CYAN}{'─'*44}{RESET}")

    metrics = [
        ("Faithfulness", agg.faithfulness),
        ("Relevance", agg.relevance),
        ("Completeness", agg.completeness),
        ("Precision", agg.precision),
    ]

    for name, score in metrics:
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        c = _color(score)
        print(f"  {name:<18} {c}{score:>6.2f}{RESET}  {c}{bar}{RESET}")

    print(f"{CYAN}{'─'*44}{RESET}")
    c = _color(agg.overall)
    print(f"  {'Overall':<18} {c}{BOLD}{agg.overall:>6.2f}{RESET}")
    print(f"{CYAN}{BOLD}{'─'*44}{RESET}\n")

    # Per-sample breakdown
    print(f"{BOLD}Per-sample results:{RESET}")
    for s in result.samples:
        c = _color(s.overall)
        flag = " ⚠" if s.faithfulness.unsupported_sentences else ""
        print(f"  [{s.id}] overall={c}{s.overall:.2f}{RESET}  "
              f"faith={s.faithfulness.score:.2f}  "
              f"rel={s.relevance.score:.2f}  "
              f"comp={s.completeness.score:.2f}  "
              f"prec={s.precision.score:.2f}{flag}")

    print()

    # Save reports
    json_path, html_path = generate_reports(result, args.output_dir)
    print(f"{GREEN}Reports saved:{RESET}")
    print(f"  JSON  → {json_path}")
    print(f"  HTML  → {html_path}\n")


if __name__ == "__main__":
    main()