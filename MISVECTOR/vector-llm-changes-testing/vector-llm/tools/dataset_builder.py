"""
tools/dataset_builder.py
========================
Generates synthetic and domain-specific dataset pairs (Alpaca/ShareGPT format)
to fine-tune a local LLM (Qwen 2.5 7B, Llama 3.1 8B, or Mistral 7B) for Vector.

OUTPUT FORMAT (Alpaca JSONL):
[
  {
    "instruction": "System prompt / Task definition",
    "input": "User query / context data",
    "output": "Exact expected JSON or completion"
  }
]

USAGE:
  python tools/dataset_builder.py --output_dir ./tools/data --num_samples 200
"""

import os
import json
import argparse
import random


INTENT_PATTERNS = [
    ("show me monthly revenue as a bar chart", {"intent": "dashboard", "confidence": 0.95, "refined_query": "Show monthly revenue as a bar chart"}),
    ("should we expand sales team in Q4?", {"intent": "decision", "confidence": 0.90, "refined_query": "Should we expand sales team in Q4?"}),
    ("what was total profit in March?", {"intent": "question", "confidence": 0.95, "refined_query": "What was total profit in March?"}),
    ("forecast customer churn for next month", {"intent": "forecast", "confidence": 0.90, "refined_query": "Forecast customer churn for next month"}),
    ("give me an overview of Q2 sales performance", {"intent": "summary", "confidence": 0.92, "refined_query": "Give an overview of Q2 sales performance"}),
]

PLOTLY_TEMPLATES = [
    {
        "query": "Display monthly sales trend",
        "context": "Month,Sales\nJan,45000\nFeb,52000\nMar,61000\nApr,58000",
        "output": {
            "chart_type": "line",
            "title": "Monthly Sales Trend",
            "x_label": "Month",
            "y_label": "Sales ($)",
            "data": [{"x": ["Jan", "Feb", "Mar", "Apr"], "y": [45000, 52000, 61000, 58000], "type": "line", "name": "Sales"}],
            "layout": {"showlegend": True, "colorway": ["#6c63ff"]},
            "insight": "Sales grew consistently from Jan to Mar, peaking at $61,000 before a minor dip in Apr."
        }
    },
    {
        "query": "Bar chart comparing regional revenue",
        "context": "Region,Revenue\nNorth,120000\nSouth,85000\nEast,95000\nWest,110000",
        "output": {
            "chart_type": "bar",
            "title": "Regional Revenue Comparison",
            "x_label": "Region",
            "y_label": "Revenue ($)",
            "data": [{"x": ["North", "South", "East", "West"], "y": [120000, 85000, 95000, 110000], "type": "bar", "name": "Revenue"}],
            "layout": {"showlegend": False, "colorway": ["#22d3a5"]},
            "insight": "North region generated the highest revenue at $120,000, followed closely by West."
        }
    }
]

DECISION_TEMPLATES = [
    {
        "query": "Should we focus marketing budget on Product A or Product B?",
        "stats": {
            "row_count": 100,
            "columns": ["product", "revenue", "roi"],
            "group_summary": {"revenue": {"Product A": 150000, "Product B": 80000}, "roi": {"Product A": 3.2, "Product B": 1.5}}
        },
        "output": {
            "recommendation": "Allocate marketing budget primarily to Product A.",
            "reasoning": "Product A generated $150,000 in revenue with an ROI of 3.2x compared to Product B's $80,000 and 1.5x ROI.",
            "confidence": 0.94,
            "risk": "low",
            "alternatives": ["Split budget 70% Product A / 30% Product B", "Perform A/B test on Product B messaging"],
            "key_metrics": {"product_a_revenue": 150000, "product_a_roi": 3.2, "product_b_roi": 1.5},
            "needs_more_data": False
        }
    }
]


def generate_dataset(output_dir: str, num_samples: int):
    os.makedirs(output_dir, exist_ok=True)
    dataset = []

    # 1. Router intent samples
    for query, target in INTENT_PATTERNS:
        dataset.append({
            "instruction": "You are a query classifier for Vector. Classify query into intent JSON.",
            "input": f"User query: \"{query}\"",
            "output": json.dumps(target)
        })

    # 2. Dashboard Plotly samples
    for t in PLOTLY_TEMPLATES:
        dataset.append({
            "instruction": "You are a data visualization expert for Vector. Return a Plotly chart spec JSON.",
            "input": f"User request: \"{t['query']}\"\nData context:\n{t['context']}",
            "output": json.dumps(t['output'])
        })

    # 3. Decision recommendation samples
    for d in DECISION_TEMPLATES:
        dataset.append({
            "instruction": "You are a senior business analyst for Vector. Return a structured decision JSON.",
            "input": f"User question: \"{d['query']}\"\nPre-computed statistics:\n{json.dumps(d['stats'])}",
            "output": json.dumps(d['output'])
        })

    # Duplicate / augment to reach requested sample count if needed
    full_dataset = (dataset * (num_samples // len(dataset) + 1))[:num_samples]

    out_file = os.path.join(output_dir, "vector_llm_train.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in full_dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"[dataset_builder] Created {len(full_dataset)} training samples at {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build fine-tuning dataset for Vector LLM")
    parser.add_argument("--output_dir", default="./tools/data", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    args = parser.parse_args()
    generate_dataset(args.output_dir, args.num_samples)
