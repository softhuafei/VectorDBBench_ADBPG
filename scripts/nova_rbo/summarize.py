#!/usr/bin/env python3
"""Aggregate Nova RBO raw query results into summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    raw = pd.read_csv(args.result_dir / "raw_queries.csv")
    keys = ["filter_type", "selectivity", "plan", "amp_mul"]
    summary = (
        raw.groupby(keys, as_index=False)
        .agg(
            query_runs=("query_index", "size"),
            distinct_queries=("query_index", "nunique"),
            estimated_selectivity=("estimated_selectivity", "mean"),
            min_result_count=("returned_count", "min"),
            insufficient_query_rate=("is_insufficient", "mean"),
            mean_recall_at_k=("recall_at_k", "mean"),
            min_recall_at_k=("recall_at_k", "min"),
            latency_mean_ms=("latency_ms", "mean"),
            latency_p50_ms=("latency_ms", "median"),
            latency_p95_ms=("latency_ms", lambda values: values.quantile(0.95)),
        )
        .sort_values(keys)
    )
    summary.to_csv(args.result_dir / "summary.csv", index=False)
    print(args.result_dir / "summary.csv")


if __name__ == "__main__":
    main()
