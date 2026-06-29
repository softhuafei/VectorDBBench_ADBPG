# ADBPG7 novad In-Filter 混合检索性能测试报告

## 测试环境

- **数据集：** BioASQ 1024维 100万行，COSINE 距离
- **索引：** novad（nlist=1024, rabitq_bits=3）
- **查询参数：** k=100，500 条测试查询（从 3106 条截断）
- **实例：** gp-2zes1gv12rb579521-master.gpdb.rds.aliyuncs.com:5432
- **数据库：** tester_novad（utility mode）
- **表结构：** `vector(id bigint, embedding vector(1024), user_array text[])`
- **索引：** novad（embedding, INCLUDE id）、GIN（user_array）
- **过滤标记：** r_2（50%）、r_10（10%）、r_100（1%）、r_1000（0.1%）、r_10000（0.01%）
- **Load 耗时：** insert=277s, index=20s
- **测试日期：** 2026-06-25

## 执行计划

novad in-filter 对应 `Ann Index Scan with expression`，通过以下 GUC 触发：
- `adbpg_enable_fastann_hybrid_bitmap_pushdown = off`
- `adbpg_fastann_hybrid_brute_force_selectivity_threshold = 0`
- `adbpg_enable_fastann_hybrid_expression_pushdown = on`（保持开启）

## 关键参数说明

| 参数 | 含义 | 调参方向 |
|------|------|---------|
| `fastann.nova_nprobe` | IVF 探查的 cluster 数量 | 增大 → 候选覆盖率↑, RT↑ |
| `fastann.quantize_rescore_amp` | 量化粗排后全精度 rescore 的放大系数 | 增大 → 排序精度↑, RT↑ |

**调参规律：**
- nprobe 决定候选覆盖率上限（能触达多少 cluster）
- rescore_amp 决定已覆盖候选中的排序精度
- nprobe 不够时，加大 rescore_amp 无效（候选池中根本没有目标行）
- rescore_amp 从 0→1.6 有显著提升（+4%），1.6 之后在当前 nprobe 下饱和

---

## 1. rescore_amp 效果验证（固定 nprobe=50, r=0.5）

| rescore_amp | Recall | Avg RT | P95 |
|-------------|--------|--------|-----|
| 0 | 82.5% | 4.6ms | 11.4ms |
| 1.6 | **86.8%** | 5.8ms | 14.5ms |
| 3.2 | 86.8% | 7.9ms | 20.0ms |
| 5.0 | 86.8% | 9.2ms | 20.5ms |

rescore_amp > 1.6 后 recall 不再提升，仅 RT 增加。瓶颈在 nprobe 覆盖率。

---

## 2. rescore_amp 在低 nprobe(=32) 下的效果

| 选择率 | rescore_amp | Recall | Avg RT |
|--------|-------------|--------|--------|
| r=0.5 | 3.2 | 81.4% | 11.0ms |
| r=0.5 | 5.0 | 81.4% | 7.5ms |
| r=0.5 | 8.0 | 81.4% | 9.5ms |
| r=0.5 | 12.0 | 81.4% | 12.1ms |
| r=0.1 | 3.2 | 77.0% | 34.7ms |
| r=0.1 | 12.0 | 77.0% | 39.5ms |
| r=0.01 | 3.2 | 59.4% | 96.8ms |
| r=0.01 | 12.0 | 59.4% | 100ms |

**结论：nprobe=32 时 rescore_amp 从 3.2 到 12.0 recall 完全不变，仅 RT 增加。**

---

## 3. nprobe 调参（固定 rescore_amp=1.6）

### r=0.5 (50%)

| nprobe | Recall | Avg RT | P95 | P99 |
|--------|--------|--------|-----|-----|
| 50 | 86.8% | 5.8ms | 14.5ms | 26.5ms |
| 100 | 93.0% | 5.7ms | 10.4ms | 16.8ms |
| **150** | **95.4%** | **7.5ms** | **17.6ms** | **42.6ms** |
| 200 | 96.8% | 9.2ms | 18.3ms | 31.7ms |
| 300 | 98.2% | 11.0ms | 20.7ms | 49.2ms |

### r=0.1 (10%)

| nprobe | Recall | Avg RT | P95 | P99 |
|--------|--------|--------|-----|-----|
| 50 | 76.2% | 7.4ms | 12.6ms | 18.1ms |
| 100 | 91.1% | 12.0ms | 21.0ms | 42.7ms |
| 150 | 94.4% | 15.7ms | 32.5ms | 52.1ms |
| **180** | **95.6%** | **18.6ms** | **41.7ms** | **67.3ms** |
| 200 | 96.2% | 18.6ms | 35.6ms | 108ms |
| 300 | 97.9% | 19.6ms | 43.4ms | 63.2ms |

### r=0.01 (1%)

| nprobe | Recall | Avg RT | P95 | P99 |
|--------|--------|--------|-----|-----|
| 50 | 66.6% | 36.4ms | 66.9ms | 111ms |
| 100 | 82.7% | 59.1ms | 105ms | 149ms |
| 200 | 92.6% | 67.7ms | 114ms | 152ms |
| 250 | 94.8% | 82.1ms | 143ms | 193ms |
| **280** | **95.8%** | **73.6ms** | **122ms** | **195ms** |
| 300 | 96.3% | 73.2ms | 122ms | 165ms |

### r=0.001 (0.1%)

| nprobe | Recall | Avg RT | P95 | P99 |
|--------|--------|--------|-----|-----|
| 200 | 30.2% | 126ms | 187ms | 253ms |
| 500 | 30.3% | 159ms | 236ms | 303ms |
| 1024 | 30.5% | 179ms | 279ms | 326ms |

**nprobe=1024（全扫）仍只有 30.5%，不可用。**

### r=0.0001 (0.01%)

| nprobe | Recall | Avg RT | P95 | P99 |
|--------|--------|--------|-----|-----|
| 500 | 3.0% | 150ms | 230ms | 294ms |

**完全不可用。**

---

## 4. 各选择率 ~95% Recall 最优配置汇总

| 选择率 | nprobe | rescore_amp | Recall | Avg RT | P95 | P99 | 达标(≥95%) |
|--------|--------|-------------|--------|--------|-----|-----|-----------|
| r=0.5 (50%) | 150 | 1.6 | **95.4%** | 7.5ms | 17.6ms | 42.6ms | ✓ |
| r=0.1 (10%) | 180 | 1.6 | **95.6%** | 18.6ms | 41.7ms | 67.3ms | ✓ |
| r=0.01 (1%) | 280 | 1.6 | **95.8%** | 73.6ms | 122ms | 195ms | ✓ |
| r=0.001 (0.1%) | 1024 | 1.6 | 30.5%（不可用） | 179ms | 279ms | 326ms | ✗ |
| r=0.0001 (0.01%) | 500 | 1.6 | 3.0%（不可用） | 150ms | 230ms | 294ms | ✗ |

---

## 5. 与 novamr 对比（~95% Recall）

| 选择率 | novad In-Filter | novamr Bitmap Push-Down | novamr GIN+Sort |
|--------|----------------|------------------------|-----------------|
| r=0.5 | **7.5ms / 95.4%** | 540ms / 94.5% | N/A |
| r=0.1 | **18.6ms / 95.6%** | 408ms / 95.4% | N/A |
| r=0.01 | **73.6ms / 95.8%** | 234ms / 95.9% | 257ms / 100% |
| r=0.001 | 不可用(30.5%) | 183ms / 95.1% | **172ms / 100%** |
| r=0.0001 | 不可用(3.0%) | 不可用(39.6%) | **163ms / 100%** |

---

## 关键发现

1. **novad in-filter 在高选择率(r≥0.01)场景下 RT 远优于 novamr：** r=0.5 快 72 倍（7.5ms vs 540ms），r=0.1 快 22 倍（18.6ms vs 408ms）。

2. **rescore_amp 是 novad 的关键调参旋钮：** 从 0 到 1.6 可提升 ~4% recall，但受限于 nprobe 覆盖率，超过甜点值后不再提升。

3. **nprobe 决定 recall 上限，rescore_amp 决定精度：** 两者需要配合调参。低 nprobe + 高 rescore_amp 无效。

4. **r≤0.001 场景 novad in-filter 完全不可用（recall=30%/3%）：** 1000 行/100 行候选分布在 1024 个 cluster 中太稀疏，即使全扫所有 cluster 也找不到。必须走 GIN+Sort。

5. **novad 适用范围：r≥0.01 的高/中选择率混合检索**，在这些场景下提供了远优于 novamr 的 RT 和相当的 recall。

---

## 复现命令

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate vectordb_adb
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset

COMMON="--host gp-2zes1gv12rb579521-master.gpdb.rds.aliyuncs.com --port 5432 \
  --user-name tester --password Nova@test --db-name tester_novad \
  --case-type HybridArrayPerformanceCase \
  --custom-case-name bioasq_1024dim_1M_hybrid_array \
  --custom-dataset-name bioasq_hybrid_medium_1m \
  --custom-dataset-dir . --custom-dataset-size 1000000 \
  --custom-dataset-dim 1024 --custom-dataset-file-count 1 \
  --custom-dataset-use-shuffled --hybrid-mode array \
  --algorithm novad --rabitq-bits 3 --nlist 1024 \
  --skip-drop-old --skip-load --skip-search-concurrent \
  --enable-bitmap-pushdown off --hybrid-brute-force-threshold 0"

# 示例: r=0.5, nprobe=150, rescore_amp=1.6
vectordbbench adbpgnova $COMMON --hybrid-rate 0.5 \
  --nprobe 150 --quantize-rescore-amp 1.6 \
  --db-label novad_r05_np150_rs16
```
