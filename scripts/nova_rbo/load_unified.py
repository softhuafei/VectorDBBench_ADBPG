#!/usr/bin/env python3
"""Load the unified 1M table and build its NovaMR covering index."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb

from vectordb_bench.backend.clients.adbpg import hybrid_synth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--table", default="nova_unified_1m")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=56606)
    parser.add_argument("--db-name", default="postgres")
    parser.add_argument("--user", default="linzhi.wzw")
    parser.add_argument("--password-env", default="PGPASSWORD")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.db_name,
        user=args.user,
        password=os.environ.get(args.password_env, ""),
        autocommit=False,
    )
    register_vector(conn)
    table = sql.Identifier(args.table)
    index = sql.Identifier(f"{args.table}_novamr_idx")
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS public.{} CASCADE").format(table))
        cur.execute(
            sql.SQL(
                "CREATE TABLE public.{} (id BIGINT PRIMARY KEY, embedding vector({}), "
                "percentile INTEGER NOT NULL, user_array TEXT[] NOT NULL, payload JSONB NOT NULL) DISTRIBUTED BY (id)"
            ).format(table, sql.Literal(args.dim)),
        )
        cur.execute(
            sql.SQL("CREATE INDEX {} ON public.{} (percentile)").format(
                sql.Identifier(f"{args.table}_percentile_idx"), table,
            ),
        )
        cur.execute(
            sql.SQL("CREATE INDEX {} ON public.{} USING gin (user_array)").format(
                sql.Identifier(f"{args.table}_array_idx"), table,
            ),
        )
        cur.execute(
            sql.SQL("CREATE INDEX {} ON public.{} USING gin (payload jsonb_path_ops)").format(
                sql.Identifier(f"{args.table}_json_idx"), table,
            ),
        )
    conn.commit()

    parquet = pq.ParquetFile(args.src)
    loaded = 0
    started = time.perf_counter()
    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=["id", "emb"]):
        ids = batch.column("id").to_pylist()
        embeddings = batch.column("emb").to_pylist()
        with conn.cursor() as cur:
            with cur.copy(sql.SQL("COPY public.{} FROM STDIN (FORMAT BINARY)").format(table)) as copy:
                copy.set_types(["bigint", "vector", "integer", 1009, "jsonb"])
                for rid, embedding in zip(ids, embeddings):
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
        conn.commit()
        loaded += len(ids)
        if loaded % 100_000 < len(ids):
            print(f"loaded={loaded} elapsed_s={time.perf_counter() - started:.1f}", flush=True)

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "CREATE INDEX {} ON public.{} USING ann (embedding) INCLUDE (id) "
                "WITH (algorithm=novamr, dim={}, distancemeasure=cosine, "
                "hnsw_m=48, hnsw_ef_construction=600, rabitq_bits=7, max_key_len=1)"
            ).format(index, table, sql.Literal(args.dim)),
        )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(sql.SQL("ANALYZE public.{}").format(table))
    conn.commit()
    print(f"loaded={loaded} total_s={time.perf_counter() - started:.1f}")
    conn.close()


if __name__ == "__main__":
    main()
