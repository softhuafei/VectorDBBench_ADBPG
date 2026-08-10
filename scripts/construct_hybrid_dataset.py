"""Offline ground-truth construction for hybrid-search benchmarks.

Reads a Bioasq-like dataset (train.parquet with id+emb, test.parquet with
id+emb) and writes:
  - shuffle_train.parquet  (optional streamed copy / subset of input)
  - test.parquet           (copy / subset of input)
  - neighbors.parquet      (full GT)
  - neighbors_hybrid_<rate>.parquet shared by scalar/array/json filters
  - neighbors_join_<rate>.parquet   for each rate in RATE_DENOMS

Supports either a single `--train-file` or a glob/list `--train-glob` for
sharded sources (e.g. shuffle_train-NN-of-10.parquet from the 10M source).
GT is computed via a streaming chunked top-k so that 10M-scale data does
not need to fit in RAM at once.

The hybrid columns themselves are synthesized at load-time by the adbpg
client using the same rate markers (see vectordb_bench/backend/clients/
adbpg/hybrid_synth.py). Therefore this script does not write user_array
or tags columns -- it only needs to produce filtered ground-truth.

Usage
=====
    # 1M from a single shuffle file
    python scripts/construct_hybrid_dataset.py \\
        --src /data/bioasq/bioasq_medium_1m \\
        --dst /tmp/vectordb_bench/dataset/bioasq_hybrid_medium_1m \\
        --train-file shuffle_train.parquet

    # 10M from sharded shuffle files
    python scripts/construct_hybrid_dataset.py \\
        --src /data/bioasq/bioasq_large_10m \\
        --dst /tmp/vectordb_bench/dataset/bioasq_hybrid_large_10m \\
        --train-glob 'shuffle_train-*-of-10.parquet'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Prefer this checkout over an older globally installed vectordb_bench.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from vectordb_bench.backend.clients.adbpg import hybrid_synth  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

log = logging.getLogger("construct_hybrid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_join_mask(row_ids: np.ndarray, denom: int, chunks_per_doc: int) -> np.ndarray:
    doc_ids = row_ids // chunks_per_doc
    return (doc_ids % denom) == 0


def build_unified_percentiles(row_ids: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (hybrid_synth.filter_percentile_for(int(row_id)) for row_id in row_ids),
        dtype=np.int32,
        count=len(row_ids),
    )


def build_unified_mask(row_ids: np.ndarray, rate: float, percentiles: np.ndarray | None = None) -> np.ndarray:
    threshold = hybrid_synth.unified_threshold_for_rate(rate)
    if percentiles is None:
        percentiles = build_unified_percentiles(row_ids)
    return percentiles < threshold


def write_neighbors(path: Path, qids: np.ndarray, all_neighbors: list[np.ndarray]) -> None:
    table = pa.table({
        "id": pa.array(qids, type=pa.int64()),
        "neighbors_id": pa.array(all_neighbors, type=pa.list_(pa.int64())),
    })
    pq.write_table(table, str(path), compression="zstd")
    log.info("wrote %s (%d rows)", path, len(qids))


def select_hybrid_queries(test_tbl: pa.Table, test_size: int) -> pa.Table:
    """Select the stable 500-query hybrid set, then take a quick-test prefix."""
    query_ids = np.asarray(test_tbl.column("id").to_numpy()).astype(np.int64)
    if len(np.unique(query_ids)) != len(query_ids):
        raise ValueError("test query ids must be unique")
    order = sorted(
        range(len(query_ids)),
        key=lambda index: (
            hybrid_synth.hybrid_query_score(int(query_ids[index])),
            int(query_ids[index]),
        ),
    )
    fixed_count = min(hybrid_synth.HYBRID_QUERY_COUNT, len(order))
    selected = test_tbl.take(pa.array(order[:fixed_count], type=pa.int64()))
    if test_size > 0:
        selected = selected.slice(0, min(test_size, fixed_count))
    return selected


def _resolve_train_files(src: Path, train_file: str | None, train_glob: str | None) -> list[Path]:
    if train_glob:
        files = sorted(path for path in src.glob(train_glob) if path.is_file())
        if not files:
            msg = f"--train-glob '{train_glob}' matched 0 files under {src}"
            raise FileNotFoundError(msg)
        return files
    return [src / (train_file or "shuffle_train.parquet")]


def _iter_train_batches(
    files: list[Path],
    target_rows: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield id and embedding arrays without retaining earlier Parquet batches."""
    total = 0
    for path in files:
        log.info("streaming %s", path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["id", "emb"]):
            table = pa.Table.from_batches([batch])
            take = table.num_rows if target_rows <= 0 else min(table.num_rows, target_rows - total)
            if take <= 0:
                return
            table = table.slice(0, take)
            row_ids = np.asarray(table.column("id").to_numpy()).astype(np.int64)
            embeddings = np.stack(
                table.column("emb").to_numpy(zero_copy_only=False),
            ).astype(np.float32, copy=False)
            yield embeddings, row_ids
            total += take
            if target_rows > 0 and total >= target_rows:
                return


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32, copy=False)


def _merge_topk_batch(
    state: dict[str, tuple[np.ndarray, np.ndarray]],
    test_norm: np.ndarray,
    train_norm: np.ndarray,
    candidate_ids: np.ndarray,
    masks: dict[str, np.ndarray],
    k: int,
) -> None:
    """Merge one train batch into every filtered running top-k."""
    sims = test_norm @ train_norm.T
    broadcast_ids = np.broadcast_to(candidate_ids, (test_norm.shape[0], candidate_ids.size))
    rows_idx = np.arange(test_norm.shape[0])[:, None]

    for name, mask in masks.items():
        best_sims, best_ids = state[name]
        filtered_sims = sims if mask.all() else np.where(mask[None, :], sims, -np.inf)
        merged_sims = np.concatenate([best_sims, filtered_sims], axis=1)
        merged_ids = np.concatenate([best_ids, broadcast_ids], axis=1)
        part = np.argpartition(-merged_sims, k - 1, axis=1)[:, :k]
        state[name] = (merged_sims[rows_idx, part], merged_ids[rows_idx, part])


def _finish_topk_state(
    state: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = {}
    for name, (best_sims, best_ids) in state.items():
        rows: list[np.ndarray] = []
        for query_index in range(best_sims.shape[0]):
            order = np.lexsort((best_ids[query_index], -best_sims[query_index]))
            sorted_sims = best_sims[query_index][order]
            sorted_ids = best_ids[query_index][order]
            rows.append(sorted_ids[sorted_sims > -np.inf])
        result[name] = rows
    return result


def _stream_topk_filtered_many(
    train_files: list[Path],
    test_norm: np.ndarray,
    mask_builders: dict[str, Callable[[np.ndarray], np.ndarray]],
    k: int,
    target_rows: int,
    batch_size: int,
    train_output_path: Path | None = None,
) -> tuple[dict[str, list[np.ndarray]], dict[str, int], int]:
    """Compute all filtered GT sets in one true streaming Parquet scan."""
    state = {
        name: (
            np.full((test_norm.shape[0], k), -np.inf, dtype=np.float32),
            np.full((test_norm.shape[0], k), -1, dtype=np.int64),
        )
        for name in mask_builders
    }
    hit_counts = dict.fromkeys(mask_builders, 0)
    total = 0
    writer: pq.ParquetWriter | None = None
    try:
        for embeddings, row_ids in _iter_train_batches(train_files, target_rows, batch_size):
            if train_output_path is not None:
                output_table = pa.table({
                    "id": pa.array(row_ids, type=pa.int64()),
                    "emb": pa.array(
                        list(embeddings),
                        type=pa.list_(pa.float32(), embeddings.shape[1]),
                    ),
                })
                if writer is None:
                    writer = pq.ParquetWriter(train_output_path, output_table.schema, compression="zstd")
                writer.write_table(output_table)

            masks = {name: builder(row_ids) for name, builder in mask_builders.items()}
            for name, mask in masks.items():
                hit_counts[name] += int(mask.sum())
            _merge_topk_batch(
                state,
                test_norm,
                _normalize_rows(embeddings),
                row_ids,
                masks,
                k,
            )
            total += len(row_ids)
            log.info("  scanned %d rows for %d GT sets", total, len(mask_builders))
    finally:
        if writer is not None:
            writer.close()

    if total == 0:
        raise ValueError("no train rows were read")
    return _finish_topk_state(state), hit_counts, total


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source bioasq directory")
    ap.add_argument("--dst", required=True, help="destination directory")
    train_input = ap.add_mutually_exclusive_group()
    train_input.add_argument("--train-file", default=None)
    train_input.add_argument(
        "--train-glob",
        default=None,
        help="glob (relative to --src) matching multiple train shards",
    )
    ap.add_argument("--test-file", default="test.parquet")
    ap.add_argument("--train-size", type=int, default=0, help="0 = full")
    ap.add_argument(
        "--test-size",
        type=int,
        default=0,
        help="0 = fixed 500-query set; N > 0 = first N queries from that fixed set",
    )
    ap.add_argument("--topk", type=int, default=1_000)
    ap.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=None,
        help="rate values in (0,1]; defaults to each mode's configured grid",
    )
    ap.add_argument(
        "--modes", nargs="+", choices=["join", "unified"], default=["unified"],
    )
    ap.add_argument("--chunks-per-doc", type=int, default=hybrid_synth.CHUNKS_PER_DOC)
    ap.add_argument("--gt-chunk", type=int, default=200_000,
                    help="train rows per chunk during GT computation")
    ap.add_argument("--skip-train-write", action="store_true",
                    help="skip writing the (large) shuffle_train.parquet output -- only emit GT files")
    ap.add_argument("--skip-full-gt", action="store_true",
                    help="skip writing unfiltered neighbors.parquet")
    args = ap.parse_args()
    if args.train_size < 0 or args.test_size < 0:
        ap.error("--train-size and --test-size must be non-negative")
    if args.topk < 1 or args.gt_chunk < 1:
        ap.error("--topk and --gt-chunk must be positive")
    if args.rates is not None and any(not 0 < rate <= 1 for rate in args.rates):
        ap.error("every --rates value must be in (0, 1]")

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    train_files = _resolve_train_files(src, args.train_file, args.train_glob)
    test_path = src / args.test_file

    log.info("loading test %s", test_path)
    test_tbl = pq.read_table(test_path)
    test_tbl = select_hybrid_queries(test_tbl, args.test_size)
    test_ids = np.asarray(test_tbl.column("id").to_numpy()).astype(np.int64)
    test_emb = np.stack(test_tbl.column("emb").to_numpy(zero_copy_only=False)).astype(np.float32)
    test_norm = _normalize_rows(test_emb)

    out_test = pa.table({
        "id": pa.array(test_ids, type=pa.int64()),
        "emb": pa.array(list(test_emb), type=pa.list_(pa.float32(), test_emb.shape[1])),
    })
    pq.write_table(out_test, str(dst / "test.parquet"), compression="zstd")
    (dst / "query_ids.json").write_text(
        json.dumps(test_ids.tolist(), indent=2) + "\n",
        encoding="utf-8",
    )
    log.info("wrote %s (%d rows)", dst / "test.parquet", len(test_ids))

    mask_builders: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    output_paths: dict[str, Path] = {}
    if not args.skip_full_gt:
        mask_builders["full"] = lambda row_ids: np.ones(len(row_ids), dtype=bool)
        output_paths["full"] = dst / "neighbors.parquet"

    selected_rates: dict[str, list[float]] = {}
    for mode in args.modes:
        if args.rates is not None:
            mode_rates = args.rates
        elif mode == "unified":
            mode_rates = list(hybrid_synth.UNIFIED_RATES)
        else:
            mode_rates = [1.0 / d for d in hybrid_synth.RATE_DENOMS]
        selected_rates[mode] = list(mode_rates)
        for rate in mode_rates:
            label = _rate_label(rate)
            key = f"{mode}:{label}"
            if mode == "join":
                d = round(1.0 / rate)
                if abs(rate - 1.0 / d) > 1e-9:
                    msg = f"legacy join mode requires rate=1/denom, got {rate}"
                    raise ValueError(msg)
                mask_builders[key] = (
                    lambda row_ids, denom=d: build_join_mask(row_ids, denom, args.chunks_per_doc)
                )
                output_name = f"neighbors_join_{label}.parquet"
            else:
                # Scalar/Array/JSON predicates all use this exact mask and GT.
                hybrid_synth.unified_rate_marker_for_rate(rate)
                mask_builders[key] = (
                    lambda row_ids, filter_rate=rate: build_unified_mask(row_ids, filter_rate)
                )
                output_name = f"neighbors_hybrid_{label}.parquet"
            output_paths[key] = dst / output_name
            log.info("prepared GT mode=%s rate=%s", mode, label)

    log.info("computing %d GT sets in one streaming similarity scan", len(mask_builders))
    all_neighbors, hit_counts, n_train = _stream_topk_filtered_many(
        train_files,
        test_norm,
        mask_builders,
        args.topk,
        args.train_size,
        args.gt_chunk,
        None if args.skip_train_write else dst / "shuffle_train.parquet",
    )
    for key, neighbors in all_neighbors.items():
        write_neighbors(output_paths[key], test_ids, neighbors)
        log.info("GT %s hits=%d", key, hit_counts[key])

    metadata = {
        "train_rows": n_train,
        "query_rows": len(test_ids),
        "query_selection": "hybrid-query-v1",
        "filter_domain_size": hybrid_synth.FILTER_DOMAIN_SIZE,
        "modes": list(args.modes),
        "rates": selected_rates,
        "topk": args.topk,
    }
    (dst / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rate_label(rate: float) -> str:
    percent = rate * 100
    if abs(percent - round(percent)) < 1e-9:
        return f"{round(percent)}p"
    return f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "_") + "p"


if __name__ == "__main__":
    main()
