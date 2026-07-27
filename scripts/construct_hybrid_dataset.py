"""Offline ground-truth construction for hybrid-search benchmarks.

Reads a Bioasq-like dataset (train.parquet with id+emb, test.parquet with
id+emb) and writes:
  - shuffle_train.parquet  (re-keyed copy / subset of input)
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
import glob as _glob
import logging
import sys
from pathlib import Path

# Prefer this checkout over an older globally installed vectordb_bench.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vectordb_bench.backend.clients.adbpg import hybrid_synth

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


def _resolve_train_files(src: Path, train_file: str | None, train_glob: str | None) -> list[Path]:
    if train_glob:
        files = sorted(Path(p) for p in _glob.glob(str(src / train_glob)))
        if not files:
            raise FileNotFoundError(f"--train-glob '{train_glob}' matched 0 files under {src}")
        return files
    return [src / (train_file or "shuffle_train.parquet")]


def _stream_train(files: list[Path], target_rows: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Read up to target_rows of id+emb. target_rows<=0 means read all."""
    embedding_chunks: list[np.ndarray] = []
    id_chunks: list[np.ndarray] = []
    total = 0
    for f in files:
        log.info("loading %s", f)
        tbl = pq.read_table(f, columns=["id", "emb"])
        rows = tbl.num_rows
        take = rows if target_rows <= 0 else min(rows, target_rows - total)
        if take <= 0:
            break
        emb = np.stack(tbl.slice(0, take).column("emb").to_numpy(zero_copy_only=False)).astype(np.float32, copy=False)
        embedding_chunks.append(emb)
        id_chunks.append(np.asarray(tbl.slice(0, take).column("id").to_numpy()).astype(np.int64))
        total += take
        log.info("  +%d rows (running total=%d)", take, total)
        if target_rows > 0 and total >= target_rows:
            break
    embeddings = (
        np.concatenate(embedding_chunks, axis=0)
        if len(embedding_chunks) > 1
        else embedding_chunks[0]
    )
    row_ids = np.concatenate(id_chunks, axis=0) if len(id_chunks) > 1 else id_chunks[0]
    return embeddings, row_ids, total


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32, copy=False)


def _topk_filtered(
    train_norm: np.ndarray,
    test_norm: np.ndarray,
    mask: np.ndarray,
    k: int,
    chunk: int = 200_000,
    candidate_ids: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Compute filtered top-k cosine for every test row.

    Streams chunks of train rows; per chunk, masks out non-matching rows
    by setting their similarity to -inf, then merges into a running
    per-test top-k via partial sort. Memory: O(chunk*test_rows + n_test*k).
    """
    n_train = train_norm.shape[0]
    n_test = test_norm.shape[0]

    # Per-test running top-k buffers (sims, ids).
    best_sims = np.full((n_test, k), -np.inf, dtype=np.float32)
    best_ids = np.full((n_test, k), -1, dtype=np.int64)

    for start in range(0, n_train, chunk):
        end = min(start + chunk, n_train)
        sub = train_norm[start:end]
        sub_mask = mask[start:end]
        # (n_test, sub) similarity
        sims = test_norm @ sub.T  # already normalized
        # Mask out non-matching rows so they cannot make the top-k.
        if not sub_mask.all():
            sims[:, ~sub_mask] = -np.inf

        # Merge with running best_sims/best_ids.
        merged_sims = np.concatenate([best_sims, sims], axis=1)
        sub_ids = (
            np.arange(start, end, dtype=np.int64)
            if candidate_ids is None
            else candidate_ids[start:end]
        )
        merged_ids = np.concatenate(
            [best_ids, np.broadcast_to(sub_ids, (n_test, sub_ids.size))],
            axis=1,
        )

        # argpartition to keep top-k.
        if merged_sims.shape[1] > k:
            part = np.argpartition(-merged_sims, k - 1, axis=1)[:, :k]
            rows_idx = np.arange(n_test)[:, None]
            best_sims = merged_sims[rows_idx, part]
            best_ids = merged_ids[rows_idx, part]
        else:
            best_sims = merged_sims
            best_ids = merged_ids

        if start % (chunk * 5) == 0 or end == n_train:
            log.info("  scanned %d / %d rows", end, n_train)

    # Final sort by descending similarity per test row. Drop -inf entries
    # (which correspond to test rows whose mask had < k hits).
    out: list[np.ndarray] = []
    for i in range(n_test):
        sims = best_sims[i]
        ids = best_ids[i]
        order = np.argsort(-sims)
        sims = sims[order]
        ids = ids[order]
        valid = sims > -np.inf
        out.append(ids[valid])
    return out


def _topk_filtered_many(
    train_norm: np.ndarray,
    test_norm: np.ndarray,
    masks: dict[str, np.ndarray],
    k: int,
    chunk: int = 200_000,
    candidate_ids: np.ndarray | None = None,
) -> dict[str, list[np.ndarray]]:
    """Compute many filtered GT sets in one similarity scan.

    Matrix multiplication dominates GT construction. Reusing each chunk's
    similarity matrix across all selectivities/modes makes a grid run much
    closer to the cost of one GT than N independent runs.
    """
    n_train = train_norm.shape[0]
    n_test = test_norm.shape[0]
    state = {
        name: (
            np.full((n_test, k), -np.inf, dtype=np.float32),
            np.full((n_test, k), -1, dtype=np.int64),
        )
        for name in masks
    }

    for start in range(0, n_train, chunk):
        end = min(start + chunk, n_train)
        sims = test_norm @ train_norm[start:end].T
        sub_ids = (
            np.arange(start, end, dtype=np.int64)
            if candidate_ids is None
            else candidate_ids[start:end]
        )
        broadcast_ids = np.broadcast_to(sub_ids, (n_test, sub_ids.size))
        rows_idx = np.arange(n_test)[:, None]

        for name, mask in masks.items():
            best_sims, best_ids = state[name]
            sub_mask = mask[start:end]
            filtered_sims = sims if sub_mask.all() else np.where(sub_mask[None, :], sims, -np.inf)
            merged_sims = np.concatenate([best_sims, filtered_sims], axis=1)
            merged_ids = np.concatenate([best_ids, broadcast_ids], axis=1)
            part = np.argpartition(-merged_sims, k - 1, axis=1)[:, :k]
            state[name] = (merged_sims[rows_idx, part], merged_ids[rows_idx, part])

        if start % (chunk * 5) == 0 or end == n_train:
            log.info("  scanned %d / %d rows for %d GT sets", end, n_train, len(masks))

    result: dict[str, list[np.ndarray]] = {}
    for name, (best_sims, best_ids) in state.items():
        rows: list[np.ndarray] = []
        for i in range(n_test):
            order = np.argsort(-best_sims[i])
            sorted_sims = best_sims[i][order]
            sorted_ids = best_ids[i][order]
            rows.append(sorted_ids[sorted_sims > -np.inf])
        result[name] = rows
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source bioasq directory")
    ap.add_argument("--dst", required=True, help="destination directory")
    ap.add_argument("--train-file", default=None)
    ap.add_argument("--train-glob", default=None,
                    help="glob (relative to --src) matching multiple train shards")
    ap.add_argument("--test-file", default="test.parquet")
    ap.add_argument("--gt-file", default="neighbors.parquet")
    ap.add_argument("--train-size", type=int, default=0, help="0 = full")
    ap.add_argument("--test-size", type=int, default=0, help="0 = full")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument(
        "--rates",
        nargs="*",
        type=float,
        default=None,
        help="rate values in (0,1]; defaults to each mode's configured grid",
    )
    ap.add_argument(
        "--modes", nargs="*", choices=["join", "unified"], default=["unified"],
    )
    ap.add_argument("--chunks-per-doc", type=int, default=hybrid_synth.CHUNKS_PER_DOC)
    ap.add_argument("--gt-chunk", type=int, default=200_000,
                    help="train rows per chunk during GT computation")
    ap.add_argument("--skip-train-write", action="store_true",
                    help="skip writing the (large) shuffle_train.parquet output -- only emit GT files")
    ap.add_argument("--skip-full-gt", action="store_true",
                    help="skip writing unfiltered neighbors.parquet")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    train_files = _resolve_train_files(src, args.train_file, args.train_glob)
    test_path = src / args.test_file

    log.info("loading test %s", test_path)
    test_tbl = pq.read_table(test_path)
    n_test = test_tbl.num_rows if args.test_size <= 0 else min(args.test_size, test_tbl.num_rows)
    test_tbl = test_tbl.slice(0, n_test)
    test_ids = np.asarray(test_tbl.column("id").to_numpy()).astype(np.int64)[:n_test]
    test_emb = np.stack(test_tbl.column("emb").to_numpy(zero_copy_only=False)).astype(np.float32)
    test_norm = _normalize_rows(test_emb)

    train_emb, train_ids, n_train = _stream_train(train_files, args.train_size)
    log.info("train loaded: %d rows, dim=%d, dtype=%s", n_train, train_emb.shape[1], train_emb.dtype)

    if not args.skip_train_write:
        out_train = pa.table({
            "id": pa.array(train_ids, type=pa.int64()),
            "emb": pa.array(list(train_emb), type=pa.list_(pa.float32(), train_emb.shape[1])),
        })
        out_path = dst / "shuffle_train.parquet"
        pq.write_table(out_train, str(out_path), compression="zstd")
        log.info("wrote %s (%d rows)", out_path, n_train)

    out_test = pa.table({
        "id": pa.array(test_ids, type=pa.int64()),
        "emb": pa.array(list(test_emb), type=pa.list_(pa.float32(), test_emb.shape[1])),
    })
    pq.write_table(out_test, str(dst / "test.parquet"), compression="zstd")
    log.info("wrote %s (%d rows)", dst / "test.parquet", n_test)

    # Pre-normalize train rows so chunked dot products yield cosine sim.
    log.info("normalizing %d train rows", n_train)
    train_norm = _normalize_rows(train_emb)
    del train_emb

    masks: dict[str, np.ndarray] = {}
    output_paths: dict[str, Path] = {}
    unified_percentiles = build_unified_percentiles(train_ids) if "unified" in args.modes else None
    if not args.skip_full_gt:
        masks["full"] = np.ones(n_train, dtype=bool)
        output_paths["full"] = dst / "neighbors.parquet"

    for mode in args.modes:
        if args.rates is not None:
            mode_rates = args.rates
        elif mode == "unified":
            mode_rates = [threshold / 10_000 for threshold in hybrid_synth.UNIFIED_RATE_THRESHOLDS]
        else:
            mode_rates = [1.0 / d for d in hybrid_synth.RATE_DENOMS]
        for rate in mode_rates:
            label = _rate_label(rate)
            if mode == "join":
                d = int(round(1.0 / rate))
                if abs(rate - 1.0 / d) > 1e-9:
                    raise ValueError(f"legacy join mode requires rate=1/denom, got {rate}")
                mask = build_join_mask(train_ids, d, args.chunks_per_doc)
                output_name = f"neighbors_join_{label}.parquet"
            else:
                # Scalar/Array/JSON predicates all use this exact mask and GT.
                hybrid_synth.unified_rate_marker_for_rate(rate)
                mask = build_unified_mask(train_ids, rate, unified_percentiles)
                output_name = f"neighbors_hybrid_{label}.parquet"
            key = f"{mode}:{label}"
            masks[key] = mask
            output_paths[key] = dst / output_name
            log.info("prepared GT mode=%s rate=%s hits=%d", mode, label, int(mask.sum()))

    log.info("computing %d GT sets in one similarity scan", len(masks))
    all_neighbors = _topk_filtered_many(
        train_norm,
        test_norm,
        masks,
        args.topk,
        chunk=args.gt_chunk,
        candidate_ids=train_ids,
    )
    for key, neighbors in all_neighbors.items():
        write_neighbors(output_paths[key], test_ids, neighbors)


def _rate_label(rate: float) -> str:
    percent = rate * 100
    if abs(percent - round(percent)) < 1e-9:
        return f"{int(round(percent))}p"
    return f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "_") + "p"


if __name__ == "__main__":
    main()
