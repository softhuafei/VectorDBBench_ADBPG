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

## Train shards: download from VectorDBBench's Aliyun OSS

The base vectors (`shuffle_train*.parquet`) are NOT tracked in git — together they are ~40 GB.
They come from the same BioASQ corpus that upstream VectorDBBench publishes on the public
Aliyun OSS bucket (anonymous read, no key required):

```
oss://assets.zilliz.com.cn/benchmark/bioasq_medium_1m/shuffle_train.parquet
oss://assets.zilliz.com.cn/benchmark/bioasq_large_10m/shuffle_train-{00..09}-of-10.parquet
```

This is the same endpoint `AliyunOSSReader` already uses (see
`vectordb_bench/__init__.py` → `ALIYUN_OSS_URL = "assets.zilliz.com.cn/benchmark/"`
and `vectordb_bench/backend/data_source.py::AliyunOSSReader`).

### Option A — let VectorDBBench fetch on first run (recommended)

Switch the dataset source to AliyunOSS and point `DATASET_LOCAL_DIR` at this folder; the CLI
(or the web UI's "Dataset from Aliyun (Shanghai)" checkbox) will download missing train
shards into `bioasq_hybrid_medium_1m/` (or `bioasq_hybrid_large_10m/`) on first run.

```bash
export VECTORDB_BENCH_DATASET_SOURCE=AliyunOSS
export DATASET_LOCAL_DIR=<repo>/datasets
```

### Option B — pre-download with ossutil

```bash
# 1M
mkdir -p <repo>/datasets/bioasq_hybrid_medium_1m
ossutil cp -r oss://assets.zilliz.com.cn/benchmark/bioasq_medium_1m/ \
  <repo>/datasets/bioasq_hybrid_medium_1m/ --include "shuffle_train*.parquet"

# 10M
mkdir -p <repo>/datasets/bioasq_hybrid_large_10m
ossutil cp -r oss://assets.zilliz.com.cn/benchmark/bioasq_large_10m/ \
  <repo>/datasets/bioasq_hybrid_large_10m/ --include "shuffle_train*.parquet"
```

> Note: upstream stores the train shards under `bioasq_medium_1m/` / `bioasq_large_10m/`
> (no `_hybrid_` suffix). The `_hybrid_` directories in this repo are the bench's custom
> case dirs — download the shards into them, or symlink an existing upstream copy in. The
> `neighbors_*` / `test.parquet` files that ship with this repo do NOT need to be
> re-downloaded.

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
