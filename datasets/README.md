# Hybrid Search Ground-Truth Datasets

Pre-computed hybrid GT for BioASQ 1024-d cosine, two sizes × array/join filter mode × five selectivity points.

## Layout

```
datasets/
├── bioasq_hybrid_medium_1m/   # base = 1,000,000 vectors
│   ├── test.parquet                          # 1000 query vectors
│   ├── neighbors.parquet                     # pure ANN top-100 GT
│   ├── neighbors_array_{50p,10p,1p,0.10p,0.01p}.parquet
│   └── neighbors_join_{50p,10p,1p,0.10p,0.01p}.parquet
└── bioasq_hybrid_large_10m/   # base = 10,000,000 vectors
    ├── test.parquet
    ├── neighbors_array_{50p,10p,1p,0.10p,0.01p}.parquet
    └── neighbors_join_{50p,10p,1p,0.10p,0.01p}.parquet
```

`*p` in the filename is the post-filter selectivity (the fraction of rows that pass the scalar predicate before the top-k cut):

| Suffix | Selectivity | `--hybrid-rate` |
|---|---|---|
| `50p`   | 50%   | 0.5    |
| `10p`   | 10%   | 0.1    |
| `1p`    | 1%    | 0.01   |
| `0.10p` | 0.1%  | 0.001  |
| `0.01p` | 0.01% | 0.0001 |

The `array` files target `HybridArrayPerformanceCase` (single-table `user_array @> '{tag}'::text[]`); the `join` files target `HybridJoinPerformanceCase` (vector ⋈ side-table `array_overlap`).

## Train data is NOT included

`shuffle_train*.parquet` (the actual 1M / 10M base vectors) is excluded — those total ~40 GB and are not tracked in git. Generate them via `scripts/construct_hybrid_dataset.py` or place pre-built train shards next to these GT files when running bench.

## Usage with VectorDBBench

```bash
export DATASET_LOCAL_DIR=$(pwd)/datasets   # parent dir of the two case dirs

vectordbbench adbpgnova ... \
  --custom-dataset-name bioasq_hybrid_medium_1m \
  --custom-dataset-dir . \
  --custom-dataset-with-gt \
  --case-type HybridArrayPerformanceCase \
  --hybrid-rate 0.1
```

The runner picks the right `neighbors_<mode>_<rate>p.parquet` based on `--hybrid-mode {array,join}` and `--hybrid-rate`.
