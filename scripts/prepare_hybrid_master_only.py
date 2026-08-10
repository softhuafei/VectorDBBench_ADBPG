#!/usr/bin/env python3
"""Prepare a coordinator-local unified hybrid table for Nova benchmarks.

The benchmark client intentionally remains query-focused. This script creates
a real ADBPG entry/master-only table, loads Parquet shards with independent
COPY connections, builds scalar/GIN indexes after loading, and finally builds
the ANN index with Nova's internal parallelism.
"""

# ruff: noqa: E402, EM102, SIM117, T201

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg
import pyarrow.parquet as pq
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb

from vectordb_bench.backend.clients.adbpg import hybrid_synth


def connection_kwargs(args: argparse.Namespace, application_name: str) -> dict[str, Any]:
    return {
        "host": args.host,
        "port": args.port,
        "dbname": args.db_name,
        "user": args.user,
        "password": os.environ.get(args.password_env, ""),
        "application_name": application_name,
    }


def connect(kwargs: dict[str, Any], *, autocommit: bool = False) -> psycopg.Connection[Any]:
    conn = psycopg.connect(**kwargs, autocommit=autocommit)
    register_vector(conn)
    return conn


def discover_shards(dataset_dir: Path, pattern: str, max_shards: int | None) -> list[Path]:
    shards = sorted(path for path in dataset_dir.glob(pattern) if path.is_file())
    if max_shards is not None:
        shards = shards[:max_shards]
    if not shards:
        raise FileNotFoundError(f"no Parquet shards match {dataset_dir / pattern}")
    return shards


def create_master_only_table(args: argparse.Namespace) -> None:
    table = sql.Identifier(args.table)
    qualified = f"public.{args.table}"
    with connect(connection_kwargs(args, "nova-master-only-ddl")) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (qualified,))
            exists = cur.fetchone()[0] is not None
            if exists and not args.drop_existing:
                raise RuntimeError(
                    f"{qualified} already exists; pass --drop-existing to replace it",
                )
            if exists:
                cur.execute(sql.SQL("DROP TABLE public.{} CASCADE").format(table))
            cur.execute(
                sql.SQL(
                    "CREATE TABLE public.{} ("
                    "id BIGINT NOT NULL, embedding vector({}), percentile INTEGER NOT NULL, "
                    "user_array TEXT[] NOT NULL, payload JSONB NOT NULL) DISTRIBUTED REPLICATED",
                ).format(table, sql.Literal(args.dim)),
            )
            cur.execute("SET LOCAL allow_system_table_mods = true")
            cur.execute(
                "DELETE FROM gp_distribution_policy WHERE localoid = %s::regclass",
                (qualified,),
            )
            cur.execute(
                sql.SQL("ALTER TABLE public.{} ALTER COLUMN embedding SET STORAGE PLAIN").format(table),
            )
        conn.commit()
    print(f"created master-only table {qualified}", flush=True)


def load_shard(
    shard: str,
    table_name: str,
    batch_size: int,
    rows_per_shard: int | None,
    conn_kwargs: dict[str, Any],
) -> tuple[str, int, float]:
    started = time.perf_counter()
    path = Path(shard)
    loaded = 0
    conn_kwargs = dict(conn_kwargs)
    conn_kwargs["application_name"] = f"nova-master-only-copy:{path.name}"
    with connect(conn_kwargs) as conn:
        with conn.cursor() as cur:
            with cur.copy(
                sql.SQL("COPY public.{} FROM STDIN (FORMAT BINARY)").format(sql.Identifier(table_name)),
            ) as copy:
                copy.set_types(["bigint", "vector", "integer", 1009, "jsonb"])
                parquet = pq.ParquetFile(path)
                for batch in parquet.iter_batches(batch_size=batch_size, columns=["id", "emb"]):
                    ids = batch.column("id").to_pylist()
                    embeddings = batch.column("emb").to_pylist()
                    for rid, embedding in zip(ids, embeddings, strict=True):
                        if rows_per_shard is not None and loaded >= rows_per_shard:
                            break
                        row_id = int(rid)
                        copy.write_row(
                            (
                                row_id,
                                embedding,
                                hybrid_synth.filter_percentile_for(row_id),
                                hybrid_synth.unified_user_array_for(row_id),
                                Jsonb(hybrid_synth.unified_payload_for(row_id)),
                            ),
                        )
                        loaded += 1
                    if rows_per_shard is not None and loaded >= rows_per_shard:
                        break
        conn.commit()
    return path.name, loaded, time.perf_counter() - started


def parallel_load(args: argparse.Namespace, shards: list[Path]) -> int:
    started = time.perf_counter()
    workers = min(args.workers, len(shards))
    base_kwargs = connection_kwargs(args, "nova-master-only-copy")
    loaded = 0
    print(f"loading {len(shards)} shards with {workers} COPY workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                load_shard,
                str(shard),
                args.table,
                args.batch_size,
                args.rows_per_shard,
                base_kwargs,
            )
            for shard in shards
        ]
        for future in as_completed(futures):
            name, count, duration = future.result()
            loaded += count
            print(
                f"loaded shard={name} rows={count} shard_s={duration:.1f} total_rows={loaded}",
                flush=True,
            )
    if args.expected_rows is not None and loaded != args.expected_rows:
        raise RuntimeError(f"loaded {loaded} rows, expected {args.expected_rows}")
    print(f"parallel load complete rows={loaded} elapsed_s={time.perf_counter() - started:.1f}", flush=True)
    return loaded


def build_aux_indexes(args: argparse.Namespace) -> None:
    table = sql.Identifier(args.table)
    index_specs = [
        (
            "primary key",
            sql.SQL("ALTER TABLE public.{} ADD CONSTRAINT {} PRIMARY KEY (id)").format(
                table,
                sql.Identifier(f"{args.table}_pkey"),
            ),
        ),
        (
            "percentile btree",
            sql.SQL("CREATE INDEX {} ON public.{} (percentile)").format(
                sql.Identifier(f"{args.table}_percentile_btree"),
                table,
            ),
        ),
        (
            "array gin",
            sql.SQL("CREATE INDEX {} ON public.{} USING gin (user_array)").format(
                sql.Identifier(f"{args.table}_user_array_gin"),
                table,
            ),
        ),
        (
            "json gin",
            sql.SQL("CREATE INDEX {} ON public.{} USING gin (payload jsonb_path_ops)").format(
                sql.Identifier(f"{args.table}_payload_gin"),
                table,
            ),
        ),
    ]
    with connect(connection_kwargs(args, "nova-master-only-aux-index")) as conn:
        for label, statement in index_specs:
            started = time.perf_counter()
            print(f"building {label}", flush=True)
            with conn.cursor() as cur:
                cur.execute(statement)
            conn.commit()
            print(f"built {label} elapsed_s={time.perf_counter() - started:.1f}", flush=True)


def build_ann_index(args: argparse.Namespace) -> None:
    table = sql.Identifier(args.table)
    index = sql.Identifier(f"{args.table}_{args.algorithm}_index")
    started = time.perf_counter()
    with connect(connection_kwargs(args, "nova-master-only-ann-index")) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SET fastann.build_parallel_processes = {}").format(
                    sql.Literal(args.build_parallel_processes),
                ),
            )
            cur.execute(
                sql.SQL("SET fastann.nova_build_optimize_level = {}").format(
                    sql.Literal(args.nova_build_optimize_level),
                ),
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX {} ON public.{} USING ann (embedding) INCLUDE (id) "
                    "WITH (algorithm={}, dim={}, distancemeasure=cosine, hnsw_m={}, "
                    "hnsw_ef_construction={}, rabitq_bits={}, max_key_len=1)",
                ).format(
                    index,
                    table,
                    sql.Literal(args.algorithm),
                    sql.Literal(args.dim),
                    sql.Literal(args.hnsw_m),
                    sql.Literal(args.ef_construction),
                    sql.Literal(args.rabitq_bits),
                ),
            )
        conn.commit()
    print(f"built ANN index elapsed_s={time.perf_counter() - started:.1f}", flush=True)


def validate_master_only(args: argparse.Namespace) -> None:
    qualified = f"public.{args.table}"
    table = sql.Identifier(args.table)
    with connect(connection_kwargs(args, "nova-master-only-validate"), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM gp_distribution_policy WHERE localoid = %s::regclass",
                (qualified,),
            )
            if cur.fetchone()[0] != 0:
                raise RuntimeError(f"{qualified} is not master-only")
            cur.execute(sql.SQL("SELECT gp_segment_id FROM public.{} LIMIT 1").format(table))
            row = cur.fetchone()
            if row is not None and row[0] != -1:
                raise RuntimeError(f"{qualified} contains non-coordinator rows: gp_segment_id={row[0]}")
            cur.execute(sql.SQL("ANALYZE public.{}").format(table))
    print(f"validated {qualified}: policy=entry, gp_segment_id=-1", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--train-glob", default="shuffle_train-*-of-*.parquet")
    parser.add_argument("--table", default="vector")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--db-name", default="postgres")
    parser.add_argument("--user", default=os.environ.get("PGUSER", os.environ.get("USER", "postgres")))
    parser.add_argument("--password-env", default="PGPASSWORD")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--rows-per-shard", type=int)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--skip-aux-indexes", action="store_true")
    parser.add_argument("--skip-ann-index", action="store_true")
    parser.add_argument("--algorithm", default="novamr")
    parser.add_argument("--rabitq-bits", type=int, default=7)
    parser.add_argument("--hnsw-m", type=int, default=48)
    parser.add_argument("--ef-construction", type=int, default=600)
    parser.add_argument("--build-parallel-processes", type=int, default=32)
    parser.add_argument("--nova-build-optimize-level", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("--workers and --batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    shards = discover_shards(args.dataset_dir, args.train_glob, args.max_shards)
    create_master_only_table(args)
    parallel_load(args, shards)
    if not args.skip_aux_indexes:
        build_aux_indexes(args)
    if not args.skip_ann_index:
        build_ann_index(args)
    validate_master_only(args)


if __name__ == "__main__":
    main()
