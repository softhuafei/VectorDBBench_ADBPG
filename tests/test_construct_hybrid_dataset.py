from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import construct_hybrid_dataset as construct


def exact_topk(
    train: np.ndarray,
    queries: np.ndarray,
    row_ids: np.ndarray,
    masks: dict[str, np.ndarray],
    k: int,
) -> dict[str, list[np.ndarray]]:
    similarities = construct._normalize_rows(queries) @ construct._normalize_rows(train).T
    result: dict[str, list[np.ndarray]] = {}
    for name, mask in masks.items():
        rows: list[np.ndarray] = []
        for query_similarities in similarities:
            filtered_ids = row_ids[mask]
            filtered_similarities = query_similarities[mask]
            order = np.lexsort((filtered_ids, -filtered_similarities))
            rows.append(filtered_ids[order[:k]])
        result[name] = rows
    return result


def test_fixed_query_selection_and_quick_prefix():
    query_ids = np.arange(3_106, dtype=np.int64)
    embeddings = np.column_stack((query_ids, query_ids + 1)).astype(np.float32)
    table = pa.table({
        "id": pa.array(query_ids),
        "emb": pa.array(list(embeddings), type=pa.list_(pa.float32(), 2)),
    })

    full = construct.select_hybrid_queries(table, test_size=0)
    repeated = construct.select_hybrid_queries(table, test_size=0)
    quick = construct.select_hybrid_queries(table, test_size=5)

    full_ids = full.column("id").to_pylist()
    assert len(full_ids) == 500
    assert full_ids == repeated.column("id").to_pylist()
    assert quick.column("id").to_pylist() == full_ids[:5]


def test_streaming_topk_matches_in_memory_exact_result(tmp_path: Path):
    rng = np.random.default_rng(7)
    row_ids = np.arange(10_000, dtype=np.int64)
    train = rng.normal(size=(len(row_ids), 16)).astype(np.float32)
    queries = rng.normal(size=(5, 16)).astype(np.float32)
    train_path = tmp_path / "train.parquet"
    pq.write_table(
        pa.table({
            "id": pa.array(row_ids),
            "emb": pa.array(list(train), type=pa.list_(pa.float32(), 16)),
        }),
        train_path,
    )

    rates = (0.001, 0.01)
    builders = {
        str(rate): (
            lambda ids, filter_rate=rate: construct.build_unified_mask(ids, filter_rate)
        )
        for rate in rates
    }
    query_norm = construct._normalize_rows(queries)
    streamed, hit_counts, total = construct._stream_topk_filtered_many(
        [train_path],
        query_norm,
        builders,
        k=3,
        target_rows=0,
        batch_size=777,
    )

    masks = {name: builder(row_ids) for name, builder in builders.items()}
    exact = exact_topk(
        train,
        queries,
        row_ids,
        masks,
        k=3,
    )

    assert total == len(row_ids)
    assert hit_counts == {name: int(mask.sum()) for name, mask in masks.items()}
    for name in builders:
        for streamed_ids, exact_ids in zip(streamed[name], exact[name], strict=True):
            np.testing.assert_array_equal(streamed_ids, exact_ids)
