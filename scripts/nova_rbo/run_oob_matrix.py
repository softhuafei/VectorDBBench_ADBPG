#!/usr/bin/env python3
"""Run the old/fast/new Nova out-of-box hybrid-search matrix.

Unlike run_matrix.py, this runner never forces a physical plan. It changes
only the intended profile knobs and records the plan chosen by the optimizer,
including non-ANN plans at very low selectivities.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from vectordb_bench.backend.clients.adbpg import hybrid_synth
from vectordb_bench.backend.filter import hybrid_rate_label
from vectordb_bench.metric import calc_recall


PROFILES = {
    "old_off_55": {"fast_tidbitmap": "off", "threshold": 0.55},
    "fast_on_55": {"fast_tidbitmap": "on", "threshold": 0.55},
    "new_on_20": {"fast_tidbitmap": "on", "threshold": 0.20},
}

RAW_FIELDS = [
    "profile",
    "filter_type",
    "selectivity",
    "actual_selectivity",
    "plan_class",
    "plan_signature",
    "amp_mul",
    "query_index",
    "repeat",
    "topk",
    "returned_count",
    "is_insufficient",
    "recall_at_k",
    "latency_ms",
]


def set_profile(
    conn: psycopg.Connection[Any],
    profile: str,
    amp_mul: float,
    ef_search: int,
    max_scan_points: int,
    quantize_rescore_amp: float,
) -> None:
    cfg = PROFILES[profile]
    settings: dict[str, Any] = {
        "enable_fast_tidbitmap": cfg["fast_tidbitmap"],
        "adbpg_fastann_hybrid_expression_pushdown_selectivity_threshold": cfg["threshold"],
        "fastann.nova_topk_amp_mul": amp_mul,
        "fastann.hnsw_ef_search": ef_search,
        "fastann.hnsw_max_scan_points": max_scan_points,
        "fastann.quantize_rescore_amp": quantize_rescore_amp,
        "fastann.index_scan_mode": "snapshot",
        "plan_cache_mode": "force_custom_plan",
        "optimizer": "on",
    }
    with conn.cursor() as cur:
        cur.execute("RESET ALL")
        for name, value in settings.items():
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


def walk_plan(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield node
    for child in node.get("Plans", []):
        yield from walk_plan(child)


def classify_plan(root: dict[str, Any]) -> tuple[str, str]:
    nodes = list(walk_plan(root))
    names = [str(node.get("Node Type", "")) for node in nodes]
    ann_names = [name for name in names if name.startswith("Ann Index Scan")]
    if ann_names:
        ann = ann_names[0].lower()
        if "bitmap push-down" in ann:
            plan_class = "ann_bitmap_pushdown"
        elif "post-filter" in ann:
            plan_class = "ann_post_filter"
        else:
            plan_class = "ann_other"
    elif any("Bitmap" in name for name in names):
        plan_class = "filter_index_sort"
    elif any(name in {"Index Scan", "Index Only Scan"} for name in names):
        plan_class = "filter_index_sort"
    elif any(name == "Seq Scan" for name in names):
        plan_class = "seq_scan_sort"
    else:
        plan_class = "other"
    signature = " > ".join(dict.fromkeys(names))
    return plan_class, signature


def load_queries(
    dataset_dir: Path,
    rates: list[float],
    query_count: int,
) -> tuple[list[np.ndarray], dict[float, list[list[int]]]]:
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
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--db-name", default="postgres")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password-env", default="PGPASSWORD")
    parser.add_argument("--table", required=True)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.55, 0.90],
    )
    parser.add_argument(
        "--filter-types",
        nargs="+",
        choices=["scalar", "array", "json"],
        default=["scalar", "array", "json"],
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=list(PROFILES),
        default=list(PROFILES),
    )
    parser.add_argument("--amp-mul", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=150)
    parser.add_argument("--max-scan-points", type=int, default=20_000)
    parser.add_argument("--quantize-rescore-amp", type=float, default=0.0)
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.db_name,
        user=args.user,
        password=os.environ.get(args.password_env, ""),
        autocommit=True,
    )
    register_vector(conn)
    queries, ground_truth = load_queries(args.dataset_dir, args.rates, args.queries)
    table = sql.Identifier(args.table)

    with conn.cursor() as cur:
        cur.execute("SHOW fastann.nova_topk_amp_mul")
        cur.execute(
            "SELECT count(*) FROM gp_segment_configuration "
            "WHERE role='p' AND content >= 0 AND status='u'",
        )
        node_count = int(cur.fetchone()[0])
        cur.execute(sql.SQL("SELECT count(*) FROM {}").format(table))
        table_size = int(cur.fetchone()[0])
        cur.execute("SHOW shared_buffers")
        shared_buffers = cur.fetchone()[0]

    metadata = {
        "table": args.table,
        "table_size": table_size,
        "node_count": node_count,
        "shared_buffers": shared_buffers,
        "topk": args.topk,
        "ef_search": args.ef_search,
        "max_scan_points": args.max_scan_points,
        "quantize_rescore_amp": args.quantize_rescore_amp,
        "amp_mul": args.amp_mul,
        "queries": len(queries),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "rates": args.rates,
        "filter_types": args.filter_types,
        "profiles": {name: PROFILES[name] for name in args.profiles},
        "profile_execution_order": "rotated by filter/selectivity scenario",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
    )

    actual_rates: dict[tuple[str, float], float] = {}
    with (args.output_dir / "raw_segments.csv").open("w", newline="") as segment_file:
        segment_writer = csv.DictWriter(
            segment_file,
            fieldnames=["filter_type", "selectivity", "segment_id", "matched_rows"],
        )
        segment_writer.writeheader()
        with conn.cursor() as cur:
            for filter_type in args.filter_types:
                for rate in args.rates:
                    cur.execute(
                        sql.SQL(
                            "SELECT gp_segment_id, count(*) FROM {} WHERE {} "
                            "GROUP BY gp_segment_id ORDER BY 1",
                        ).format(table, predicate(filter_type, rate)),
                    )
                    total = 0
                    for segment_id, matched_rows in cur.fetchall():
                        total += int(matched_rows)
                        segment_writer.writerow(
                            {
                                "filter_type": filter_type,
                                "selectivity": rate,
                                "segment_id": segment_id,
                                "matched_rows": matched_rows,
                            },
                        )
                    actual_rates[(filter_type, rate)] = total / table_size

    raw_path = args.output_dir / "raw_queries.csv"
    plan_path = args.output_dir / "plans.jsonl"
    with raw_path.open("w", newline="") as raw_file, plan_path.open("w") as plan_file:
        writer = csv.DictWriter(raw_file, fieldnames=RAW_FIELDS)
        writer.writeheader()
        scenario_index = 0
        for filter_type in args.filter_types:
            for rate in args.rates:
                where = predicate(filter_type, rate)
                query = sql.SQL(
                    "SELECT id FROM {} WHERE {} "
                    "ORDER BY embedding <=> %s::vector({}) LIMIT %s::int",
                ).format(table, where, sql.Literal(args.dim))
                shift = scenario_index % len(args.profiles)
                profile_order = args.profiles[shift:] + args.profiles[:shift]
                scenario_index += 1
                for profile in profile_order:
                    set_profile(
                        conn,
                        profile,
                        args.amp_mul,
                        args.ef_search,
                        args.max_scan_points,
                        args.quantize_rescore_amp,
                    )
                    with conn.cursor() as cur:
                        cur.execute(
                            sql.SQL("EXPLAIN (FORMAT JSON) ") + query,
                            (queries[0], args.topk),
                        )
                        explain = cur.fetchone()[0]
                        if isinstance(explain, str):
                            explain = json.loads(explain)
                        root = explain[0]["Plan"]
                        plan_class, plan_signature = classify_plan(root)
                        plan_file.write(
                            json.dumps(
                                {
                                    "profile": profile,
                                    "filter_type": filter_type,
                                    "selectivity": rate,
                                    "actual_selectivity": actual_rates[(filter_type, rate)],
                                    "plan_class": plan_class,
                                    "plan_signature": plan_signature,
                                    "plan": root,
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n",
                        )
                        plan_file.flush()
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
                                        "profile": profile,
                                        "filter_type": filter_type,
                                        "selectivity": rate,
                                        "actual_selectivity": actual_rates[(filter_type, rate)],
                                        "plan_class": plan_class,
                                        "plan_signature": plan_signature,
                                        "amp_mul": args.amp_mul,
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
                                    },
                                )
                                raw_file.flush()
    conn.close()
    print(raw_path)


if __name__ == "__main__":
    main()
