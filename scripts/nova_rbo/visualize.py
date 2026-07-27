#!/usr/bin/env python3
"""Create an interactive HTML report from summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.result_dir / "summary.csv")
    data["scenario"] = data["filter_type"] + " / " + (data["selectivity"] * 100).map(lambda value: f"{value:g}%")

    post = data[data["plan"] == "post_filter"].copy()
    correctness = px.line(
        post,
        x="amp_mul",
        y="min_result_count",
        color="scenario",
        markers=True,
        title="Post-Filter result completeness",
        labels={"amp_mul": "nova_topk_amp_mul", "min_result_count": "Minimum returned rows"},
    )
    correctness.add_hline(y=100, line_dash="dash", annotation_text="TopK=100")

    recall = px.line(
        post,
        x="amp_mul",
        y="mean_recall_at_k",
        color="scenario",
        markers=True,
        title="Post-Filter Recall@K",
        labels={"amp_mul": "nova_topk_amp_mul", "mean_recall_at_k": "Mean Recall@K"},
    )

    comparison = px.bar(
        data,
        x="scenario",
        y="latency_p95_ms",
        color="plan",
        barmode="group",
        facet_row="amp_mul",
        title="P95 latency by plan and scenario",
        labels={"latency_p95_ms": "P95 latency (ms)", "scenario": ""},
    )

    parts = [
        correctness.to_html(full_html=False, include_plotlyjs="cdn"),
        recall.to_html(full_html=False, include_plotlyjs=False),
        comparison.to_html(full_html=False, include_plotlyjs=False),
        data.to_html(index=False, classes="summary", float_format=lambda value: f"{value:.4f}"),
    ]
    output = args.result_dir / "report.html"
    output.write_text(
        "<!doctype html><meta charset='utf-8'><title>Nova RBO experiment</title>"
        "<style>body{font-family:system-ui;margin:24px}.summary{border-collapse:collapse;font-size:12px}"
        ".summary td,.summary th{border:1px solid #ccc;padding:4px 6px}</style>"
        + "\n".join(parts),
    )
    print(output)


if __name__ == "__main__":
    main()
