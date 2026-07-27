#!/usr/bin/env python3
"""Compute the ideal independent-filter MPP shortage probability."""

from __future__ import annotations

import argparse
import math

import numpy as np


def binomial_pmf(n: int, probability: float) -> np.ndarray:
    values = np.arange(n + 1)
    logs = np.array(
        [
            math.lgamma(n + 1)
            - math.lgamma(int(value) + 1)
            - math.lgamma(n - int(value) + 1)
            + value * math.log(probability)
            + (n - value) * math.log1p(-probability)
            for value in values
        ],
    )
    result = np.exp(logs)
    return result / result.sum()


def shortage_probability(topk: int, selectivity: float, nodes: int, multiplier: float) -> float:
    candidates = max(1, math.floor(topk / selectivity * multiplier))
    local = binomial_pmf(candidates, selectivity)
    # A segment sends at most topk rows to the coordinator.
    capped = np.zeros(topk + 1)
    uncapped_count = min(topk, len(local))
    capped[:uncapped_count] = local[:uncapped_count]
    if len(local) > topk:
        capped[topk] = local[topk:].sum()
    total = capped
    for _ in range(nodes - 1):
        total = np.convolve(total, capped)
    return float(total[:topk].sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.2, 0.3, 0.55, 0.9])
    parser.add_argument("--multipliers", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.75, 1.0])
    args = parser.parse_args()
    print("selectivity,multiplier,shortage_probability")
    for rate in args.rates:
        for multiplier in args.multipliers:
            probability = shortage_probability(args.topk, rate, args.nodes, multiplier)
            print(f"{rate},{multiplier},{probability:.12g}")


if __name__ == "__main__":
    main()
