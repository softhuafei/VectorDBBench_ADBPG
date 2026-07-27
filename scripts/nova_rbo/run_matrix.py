#!/usr/bin/env python3
"""Run a short NovaMR post-filter/bitmap matrix and emit tidy CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from vectordb_bench.backend.clients.adbpg import hybrid_synth
from vectordb_bench.backend.filter import hybrid_rate_label
from vectordb_bench.metric import calc_recall


RAW_FIELDS = [
    "filter_type",
    "selectivity",
    "estimated_selectivity",
    "plan",
    "amp_mul",
    "query_index",
    "repeat",
    "topk",
    "returned_count",
    "is_insufficient",
    "recall_at_k",
    "latency_ms",
    "plan_text",
]


def set_gucs(
    conn: psycopg.Connection[Any],
    plan: str,
    amp_mul: float,
    ef_search: int,
    max_scan_points: int,
    quantize_rescore_amp: float,
    optimizer: str,
) -> None:
    common: dict[str, Any] = {
        "fastann.nova_topk_amp_mul": amp_mul,
        "fastann.hnsw_ef_search": ef_search,
        "fastann.hnsw_max_scan_points": max_scan_points,
        "fastann.quantize_rescore_amp": quantize_rescore_amp,
        "fastann.index_scan_mode": "snapshot",
        "optimizer": optimizer,
        "elog_process_parameters": "off",
        "plan_cache_mode": "force_custom_plan",
        "adbpg_fastann_hybrid_brute_force_with_row_threshold": 0,
        "adbpg_fastann_hybrid_brute_force_with_table_threshold": 0,
        "adbpg_fastann_hybrid_brute_force_selectivity_threshold": 0,
    }
    if plan == "post_filter":
        common["adbpg_fastann_hybrid_expression_pushdown_selectivity_threshold"] = 0
    elif plan == "bitmap_pushdown":
        common.update(
            {
                "adbpg_fastann_hybrid_expression_pushdown_selectivity_threshold": 1.0,
                "adbpg_enable_fastann_hybrid_bitmap_pushdown": "on",
                "adbpg_fastann_bitmap_pushdown_selectivity_threshold": 1.0,
                "adbpg_fastann_bitmap_pushdown_rows_threshold": -1,
                "adbpg_fastann_bitmap_pushdown_bitmapcost_rate_threshold": 1.0,
            },
        )
    elif plan == "auto_20":
        common["adbpg_fastann_hybrid_expression_pushdown_selectivity_threshold"] = 0.2
    else:
        raise ValueError(f"unsupported plan {plan}")
    with conn.cursor() as cur:
        # Prevent plan-forcing GUCs from a previous matrix cell leaking into
        # Auto or a different forced plan.
        cur.execute("RESET ALL")
        for name, value in common.items():
            cur.execute(
                sql.SQL("SET {name} = {value}").format(
                    name=sql.Identifier(name),
                    value=sql.Literal(str(value)),
                ),
            )


def predicate(filter_type: str, rate: float) -> sql.Composed:
    threshold = hybrid_synth.unified_threshold_for_rate(rate)
    marker = hybrid_synth.unified_rate_marker_for_rate(rate)
    if filter_type == "scalar":
        return sql.SQL("percentile < {}").format(sql.Literal(threshold))
    if filter_type == "array":
        return sql.SQL("user_array @> ARRAY[{}]::text[]").format(sql.Literal(marker))
    payload = json.dumps({"rates": [marker]}, separators=(",", ":"))
    return sql.SQL("payload @> {}::jsonb").format(sql.Literal(payload))


def find_ann_node(plan: dict[str, Any]) -> dict[str, Any]:
    if str(plan.get("Node Type", "")).startswith("Ann Index Scan"):
        return plan
    for child in plan.get("Plans", []):
        try:
            return find_ann_node(child)
        except LookupError:
            pass
    raise LookupError("EXPLAIN JSON did not contain an Ann Index Scan")


def load_queries(dataset_dir: Path, rates: list[float], query_count: int) -> tuple[list[np.ndarray], dict[float, list[list[int]]]]:
    test = pq.read_table(dataset_dir / "test.parquet")
    count = min(query_count, test.num_rows)
    queries = [
        np.asarray(value, dtype=np.float32)
        for value in test.slice(0, count).column("emb").to_pylist()
    ]
    ground_truth: dict[float, list[list[int]]] = {}
    for rate in rates:
        path = dataset_dir / f"neighbors_hybrid_{hybrid_rate_label(rate)}.parquet"
        rows = pq.read_table(path).slice(0, count).column("neighbors_id").to_pylist()
        ground_truth[rate] = [[int(item) for item in row] for row in rows]
    return queries, ground_truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=56606)
    parser.add_argument("--db-name", default="postgres")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password-env", default="PGPASSWORD")
    parser.add_argument("--table", required=True)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.2, 0.3, 0.55, 0.9])
    parser.add_argument("--filter-types", nargs="+", choices=["scalar", "array", "json"], default=["scalar", "array", "json"])
    parser.add_argument("--post-filter-muls", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--bitmap-muls", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--plans",
        nargs="+",
        choices=["post_filter", "bitmap_pushdown", "auto_20"],
        default=["post_filter", "bitmap_pushdown"],
    )
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=150)
    parser.add_argument("--max-scan-points", type=int, default=20_000)
    parser.add_argument("--quantize-rescore-amp", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["on", "off"], default="off")
    parser.add_argument(
        "--rotate-plan-order",
        action="store_true",
        help="Rotate the plan execution order between adjacent filter/rate cells.",
    )
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    password = os.environ.get(args.password_env, "")
    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.db_name,
        user=args.user,
        password=password,
        autocommit=True,
    )
    register_vector(conn)
    queries, ground_truth = load_queries(args.dataset_dir, args.rates, args.queries)
    table = sql.Identifier(args.table)

    # Fail before the matrix if fastann wasn't initialized via shared preload.
    with conn.cursor() as cur:
        cur.execute("SHOW fastann.nova_topk_amp_mul")
        cur.execute(
            "SELECT count(*) FROM gp_segment_configuration WHERE role='p' AND content >= 0 AND status='u'",
        )
        node_count = int(cur.fetchone()[0])
        cur.execute(sql.SQL("SELECT count(*) FROM {}").format(table))
        table_size = int(cur.fetchone()[0])

    metadata = {
        "table": args.table,
        "table_size": table_size,
        "node_count": node_count,
        "topk": args.topk,
        "ef_search": args.ef_search,
        "max_scan_points": args.max_scan_points,
        "quantize_rescore_amp": args.quantize_rescore_amp,
        "optimizer": args.optimizer,
        "rotate_plan_order": args.rotate_plan_order,
        "queries": len(queries),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "rates": args.rates,
        "filter_types": args.filter_types,
        "post_filter_muls": args.post_filter_muls,
        "bitmap_muls": args.bitmap_muls,
        "plans": args.plans,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    with (args.output_dir / "raw_segments.csv").open("w", newline="") as segment_file:
        writer = csv.DictWriter(segment_file, fieldnames=["filter_type", "selectivity", "segment_id", "matched_rows"])
        writer.writeheader()
        with conn.cursor() as cur:
            for filter_type in args.filter_types:
                for rate in args.rates:
                    cur.execute(
                        sql.SQL("SELECT gp_segment_id, count(*) FROM {} WHERE {} GROUP BY gp_segment_id ORDER BY 1").format(
                            table,
                            predicate(filter_type, rate),
                        ),
                    )
                    for segment_id, matched_rows in cur.fetchall():
                        writer.writerow(
                            {
                                "filter_type": filter_type,
                                "selectivity": rate,
                                "segment_id": segment_id,
                                "matched_rows": matched_rows,
                            },
                        )

    raw_path = args.output_dir / "raw_queries.csv"
    with raw_path.open("w", newline="") as raw_file:
        writer = csv.DictWriter(raw_file, fieldnames=RAW_FIELDS)
        writer.writeheader()
        scenario_index = 0
        for filter_type in args.filter_types:
            for rate in args.rates:
                where = predicate(filter_type, rate)
                query = sql.SQL(
                    "SELECT id FROM {} WHERE {} ORDER BY embedding <=> %s::vector({}) LIMIT %s::int",
                ).format(table, where, sql.Literal(args.dim))
                plan_multipliers = {
                    "post_filter": args.post_filter_muls,
                    "bitmap_pushdown": args.bitmap_muls,
                    "auto_20": args.post_filter_muls,
                }
                plan_order = list(args.plans)
                if args.rotate_plan_order and plan_order:
                    shift = scenario_index % len(plan_order)
                    plan_order = plan_order[shift:] + plan_order[:shift]
                scenario_index += 1
                for plan in plan_order:
                    multipliers = plan_multipliers[plan]
                    for amp_mul in multipliers:
                        set_gucs(
                            conn,
                            plan,
                            amp_mul,
                            args.ef_search,
                            args.max_scan_points,
                            args.quantize_rescore_amp,
                            args.optimizer,
                        )
                        with conn.cursor() as cur:
                            cur.execute(sql.SQL("EXPLAIN (FORMAT JSON) ") + query, (queries[0], args.topk))
                            explain = cur.fetchone()[0]
                            if isinstance(explain, str):
                                explain = json.loads(explain)
                            ann_node = find_ann_node(explain[0]["Plan"])
                            plan_text = str(ann_node["Node Type"])
                            estimated_selectivity = min(
                                1.0,
                                float(ann_node["Plan Rows"]) / (table_size / node_count),
                            )
                            for _ in range(args.warmups):
                                cur.execute(query, (queries[0], args.topk))
                                cur.fetchall()
                            for query_index, vector in enumerate(queries):
                                for repeat in range(args.repeats):
                                    started = time.perf_counter()
                                    cur.execute(query, (vector, args.topk))
                                    result = [int(row[0]) for row in cur.fetchall()]
                                    latency_ms = (time.perf_counter() - started) * 1_000
                                    writer.writerow(
                                        {
                                            "filter_type": filter_type,
                                            "selectivity": rate,
                                            "estimated_selectivity": estimated_selectivity,
                                            "plan": plan,
                                            "amp_mul": amp_mul,
                                            "query_index": query_index,
                                            "repeat": repeat,
                                            "topk": args.topk,
                                            "returned_count": len(result),
                                            "is_insufficient": int(len(result) < args.topk),
                                            "recall_at_k": calc_recall(
                                                args.topk,
                                                ground_truth[rate][query_index][: args.topk],
                                                result,
                                            ),
                                            "latency_ms": round(latency_ms, 3),
                                            "plan_text": plan_text,
                                        },
                                    )
                                    raw_file.flush()
    conn.close()
    print(raw_path)


if __name__ == "__main__":
    main()
