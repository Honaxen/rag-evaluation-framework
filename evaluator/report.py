"""
evaluator/report.py — Generates JSON and HTML reports from evaluation results.
"""

import json
import os
from datetime import datetime
from evaluator.pipeline import EvaluationResult


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "#3fb950"   # green
    elif score >= 0.5:
        return "#f0883e"   # orange
    else:
        return "#f85149"   # red


def _score_bar(score: float, width: int = 200) -> str:
    filled = int(score * width)
    color = _score_color(score)
    return (
        f'<div style="background:#21262d;border-radius:4px;width:{width}px;height:12px;display:inline-block;">'
        f'<div style="background:{color};width:{filled}px;height:12px;border-radius:4px;"></div>'
        f'</div>'
    )


def save_json(result: EvaluationResult, path: str) -> str:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "aggregate": {
            "faithfulness": result.aggregate.faithfulness,
            "relevance": result.aggregate.relevance,
            "completeness": result.aggregate.completeness,
            "precision": result.aggregate.precision,
            "overall": result.aggregate.overall,
            "n_samples": result.aggregate.n_samples,
        },
        "samples": [
            {
                "id": s.id,
                "question": s.question,
                "answer": s.answer,
                "overall": s.overall,
                "scores": {
                    "faithfulness": s.faithfulness.score,
                    "relevance": s.relevance.score,
                    "completeness": s.completeness.score,
                    "precision": s.precision.score,
                },
                "details": {
                    "faithfulness": {
                        "supported": s.faithfulness.supported,
                        "total": s.faithfulness.total,
                        "unsupported": s.faithfulness.unsupported_sentences,
                    },
                    "relevance": {
                        "chunk_scores": s.relevance.chunk_scores,
                        "irrelevant_chunks": s.relevance.irrelevant_chunks,
                    },
                    "completeness": {
                        "covered": s.completeness.covered,
                        "total": s.completeness.total,
                        "missing_terms": s.completeness.missing_terms,
                    },
                    "precision": {
                        "chunk_scores": s.precision.chunk_scores,
                        "noisy_chunks": s.precision.noisy_chunks,
                    },
                },
            }
            for s in result.samples
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def save_html(result: EvaluationResult, path: str) -> str:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    agg = result.aggregate
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    rows = ""
    for s in result.samples:
        rows += f"""
        <tr>
          <td style="color:#7d8590;font-size:12px;">{s.id}</td>
          <td style="color:#cdd9e5;font-size:12px;">{s.question[:60]}{'...' if len(s.question)>60 else ''}</td>
          <td style="text-align:center;">{_score_bar(s.faithfulness.score, 80)} <span style="color:{_score_color(s.faithfulness.score)};font-size:11px;margin-left:4px;">{s.faithfulness.score:.2f}</span></td>
          <td style="text-align:center;">{_score_bar(s.relevance.score, 80)} <span style="color:{_score_color(s.relevance.score)};font-size:11px;margin-left:4px;">{s.relevance.score:.2f}</span></td>
          <td style="text-align:center;">{_score_bar(s.completeness.score, 80)} <span style="color:{_score_color(s.completeness.score)};font-size:11px;margin-left:4px;">{s.completeness.score:.2f}</span></td>
          <td style="text-align:center;">{_score_bar(s.precision.score, 80)} <span style="color:{_score_color(s.precision.score)};font-size:11px;margin-left:4px;">{s.precision.score:.2f}</span></td>
          <td style="text-align:center;font-weight:bold;color:{_score_color(s.overall)};">{s.overall:.2f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>RAG Evaluation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, sans-serif; padding: 2rem; }}
  h1 {{ font-size: 20px; font-weight: 700; margin-bottom: 4px; }}
  .meta {{ font-size: 12px; color: #7d8590; margin-bottom: 2rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 2rem; }}
  .card {{ background: #161b22; border: 0.5px solid #30363d; border-radius: 8px; padding: 14px; text-align: center; }}
  .card-label {{ font-size: 11px; color: #7d8590; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }}
  .card-score {{ font-size: 28px; font-weight: 700; }}
  table {{ width: 100%; border-collapse: collapse; background: #161b22; border: 0.5px solid #30363d; border-radius: 8px; overflow: hidden; font-family: monospace; }}
  th {{ background: #21262d; padding: 10px 12px; font-size: 11px; color: #7d8590; text-align: left; letter-spacing: 0.5px; }}
  td {{ padding: 10px 12px; border-top: 0.5px solid #21262d; vertical-align: middle; }}
  tr:hover td {{ background: #1c2128; }}
</style>
</head>
<body>
<h1>RAG Evaluation Report</h1>
<p class="meta">Generated: {ts} &nbsp;·&nbsp; {agg.n_samples} samples evaluated</p>

<div class="cards">
  <div class="card">
    <div class="card-label">Faithfulness</div>
    <div class="card-score" style="color:{_score_color(agg.faithfulness)};">{agg.faithfulness:.2f}</div>
  </div>
  <div class="card">
    <div class="card-label">Relevance</div>
    <div class="card-score" style="color:{_score_color(agg.relevance)};">{agg.relevance:.2f}</div>
  </div>
  <div class="card">
    <div class="card-label">Completeness</div>
    <div class="card-score" style="color:{_score_color(agg.completeness)};">{agg.completeness:.2f}</div>
  </div>
  <div class="card">
    <div class="card-label">Precision</div>
    <div class="card-score" style="color:{_score_color(agg.precision)};">{agg.precision:.2f}</div>
  </div>
  <div class="card">
    <div class="card-label">Overall</div>
    <div class="card-score" style="color:{_score_color(agg.overall)};">{agg.overall:.2f}</div>
  </div>
</div>

<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Question</th>
      <th>Faithfulness</th>
      <th>Relevance</th>
      <th>Completeness</th>
      <th>Precision</th>
      <th>Overall</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)

    return path


def generate_reports(result: EvaluationResult, output_dir: str = "reports") -> tuple[str, str]:
    """Generate both JSON and HTML reports. Returns (json_path, html_path)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = save_json(result, f"{output_dir}/eval_{ts}.json")
    html_path = save_html(result, f"{output_dir}/eval_{ts}.html")
    return json_path, html_path