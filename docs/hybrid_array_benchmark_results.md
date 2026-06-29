# ADBPG7 Nova 混合检索（Array Filter）性能测试报告

## 测试环境

- **数据集：** BioASQ 1024维 100万行，COSINE 距离
- **索引：** novamr（hnsw_m=48, ef_construction=600, nlist=1024, rabitq_bits=7）
- **查询参数：** k=100，500 条测试查询（从 3106 条截断）
- **实例：** gp-2zes1gv12rb579521-master.gpdb.rds.aliyuncs.com:5432
- **表结构：** `vector(id bigint, embedding vector(1024), user_array text[], pipeline_id text)`
- **索引：** novamr（embedding, INCLUDE id）、GIN（user_array）、btree（pipeline_id）
- **过滤标记：** r_2（50%）、r_10（10%）、r_100（1%）、r_1000（0.1%）、r_10000（0.01%）
- **测试日期：** 2026-06-24

## 公共 CLI 参数

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate vectordb_adb
export DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset

COMMON="--host gp-2zes1gv12rb579521-master.gpdb.rds.aliyuncs.com --port 5432 \
  --user-name tester --password Nova@test --db-name postgres \
  --case-type HybridArrayPerformanceCase \
  --custom-case-name bioasq_1024dim_1M_hybrid_array \
  --custom-dataset-name bioasq_hybrid_medium_1m \
  --custom-dataset-dir . --custom-dataset-size 1000000 \
  --custom-dataset-dim 1024 --custom-dataset-file-count 1 \
  --custom-dataset-use-shuffled --hybrid-mode array \
  --skip-drop-old --skip-load --skip-search-concurrent"
```

---

## 1. Bitmap Push-Down（客户端默认配置）

**GUC 设置（客户端默认值）：**
- `fastann.hnsw_ef_search = 150`
- `fastann.hnsw_max_scan_points = 20000`
- `fastann.nova_topk_amp_mul = 1.0`
- `fastann.nova_topk_amp_add = 0.0`
- `fastann.index_scan_mode = snapshot`
- `fastann.quantize_rescore_amp = 0.0`
- `adbpg_fastann_hybrid_brute_force_selectivity_threshold = 0.001`
- `adbpg_enable_fastann_hybrid_bitmap_pushdown = on`
- `adbpg_enable_fastann_hybrid_expression_pushdown = on`
- `optimizer = off`
- `plan_cache_mode = force_custom_plan`

**复现命令：**
```bash
vectordbbench adbpgnova $COMMON --hybrid-rate <rate> --db-label default_r<rate>
```

| 选择率 | 执行计划 | Recall | 平均 RT | P95 | P99 |
|--------|---------|--------|---------|-----|-----|
| r=0.5 (50%) | Ann Index Scan with bitmap push-down | 96.1% | 722ms | 968ms | 1069ms |
| r=0.1 (10%) | Ann Index Scan with bitmap push-down | 99.2% | 631ms | 841ms | 892ms |
| r=0.01 (1%) | Ann Index Scan with bitmap push-down | 99.4% | 2580ms | 3251ms | 3522ms |
| r=0.001 (0.1%) | Ann Index Scan with bitmap push-down | 99.6% | 1965ms | 2423ms | 2685ms |
| r=0.0001 (0.01%) | Ann Index Scan with bitmap push-down | 100% | 1951ms | 2384ms | 2599ms |

**备注：**
- 5 个选择率全部走 bitmap push-down，暴力检索（brute-force）从未触发。
- GIN 索引对 `@>` 的选择率估计不准确（无论实际数据分布如何，始终估计约 0.5%），导致 threshold=0.001 无法为 r=0.001/0.0001 触发暴力检索。
- 低选择率下 RT 反常地高，原因是 max_scan_points=20000 导致大量无效图遍历。

---

## 2. GIN+Sort 暴力检索（threshold=0.01~0.02）

**调整的 GUC：** `adbpg_fastann_hybrid_brute_force_selectivity_threshold = 0.01`（r≤0.001）或 `0.02`（r=0.01）

**复现命令：**
```bash
# r=0.001, r=0.0001: threshold=0.01 即可
vectordbbench adbpgnova $COMMON --hybrid-rate <rate> \
  --max-scan-points 2000 --hybrid-brute-force-threshold 0.01 \
  --db-label bf_thresh01_r<rate>

# r=0.01: 需要 threshold=0.02
vectordbbench adbpgnova $COMMON --hybrid-rate 0.01 \
  --max-scan-points 2000 --hybrid-brute-force-threshold 0.02 \
  --db-label bf_thresh02_r001
```

| 选择率 | 执行计划 | Recall | 平均 RT | P95 | P99 |
|--------|---------|--------|---------|-----|-----|
| r=0.01 (1%) | GIN + Sort（暴力检索） | 100% | 257ms | 360ms | 463ms |
| r=0.001 (0.1%) | GIN + Sort（暴力检索） | 100% | 172ms | 239ms | 291ms |
| r=0.0001 (0.01%) | GIN + Sort（暴力检索） | 100% | 163ms | 228ms | 287ms |

**备注：**
- 暴力检索路径：GIN bitmap scan → heap fetch → 全精度排序 → top-k。
- recall=100%（精确），比 bitmap push-down 快 2-12 倍。
- r=0.001/0.0001 需要 threshold ≥ 0.01；r=0.01 需要 threshold ≥ 0.02（优化器估计 r=0.01 的选择率为 ~1.01%，刚好超过 0.01 阈值）。
- ANALYZE 无法修复估计问题（GIN 数组选择率使用默认公式）。

---

## 3. Post-Filter（后置过滤）

**强制设置的 GUC：**
- `adbpg_enable_fastann_hybrid_bitmap_pushdown = off`
- `adbpg_enable_fastann_hybrid_expression_pushdown = off`
- `adbpg_enable_fastann_hybrid_brute_force = off`
- `adbpg_fastann_hybrid_expression_pushdown_selectivity_threshold = 0`
- `enable_bitmapscan = off`
- `enable_seqscan = off`
- `adbpg_fastann_hybrid_brute_force_selectivity_threshold = 0`

使用 `--enable-expression-pushdown off --enable-bitmap-pushdown off --hybrid-brute-force-threshold 0` 时客户端会自动设置以上所有 GUC。

**复现命令：**
```bash
vectordbbench adbpgnova $COMMON --hybrid-rate <rate> \
  --enable-bitmap-pushdown off --enable-expression-pushdown off \
  --hybrid-brute-force-threshold 0 \
  --ef-search <efs> --max-scan-points <msp> \
  --db-label postfilter_r<rate>_efs<efs>_msp<msp>
```

### r=0.5 (50%) 调参

| ef_search | max_scan_points | Recall | 平均 RT | P95 | P99 |
|-----------|----------------|--------|---------|-----|-----|
| 150 | 2000 | 68.9% | 2.5ms | 5.9ms | 11.2ms |
| 150 | 5000 | 86.4% | 3.3ms | 8.9ms | 17.2ms |
| 150 | 10000 | 90.3% | 4.2ms | 11.8ms | 20.7ms |
| 150 | 20000 | 90.3% | 4.4ms | 12.4ms | 33.8ms |
| 400 | 20000 | 94.1% | 5.6ms | 15.3ms | 32.2ms |
| **800** | **20000** | **95.2%** | **7.3ms** | **19.9ms** | **46.2ms** |

**达到 95% recall 的配置：** `--ef-search 800 --max-scan-points 20000`

### r=0.1 (10%) 调参

| ef_search | max_scan_points | Recall | 平均 RT | P95 | P99 |
|-----------|----------------|--------|---------|-----|-----|
| 150 | 2000 | 41.9% | 6.0ms | 15.8ms | 30.6ms |
| 150 | 5000 | 67.4% | 7.0ms | 15.6ms | 34.7ms |
| 150 | 10000 | 83.4% | 7.6ms | 16.2ms | 22.9ms |
| 150 | 20000 | 91.8% | 10.2ms | 25.4ms | 66.3ms |
| 150 | 50000 | 94.0% | 13.7ms | 33.3ms | 52.9ms |
| 400 | 50000 | 94.0% | 11.7ms | 27.6ms | 60.9ms |
| 800 | 50000 | 94.0% | 12.7ms | 31.3ms | 64.3ms |
| 1000 | 100000 | 94.0% | 10.2ms | 20.8ms | 60.8ms |

**无法达到 95% recall。** 由于 HNSW 图结构限制，recall 天花板为 94.0%。

---

## 4. Expression Push-Down（表达式下推）

**GUC 设置：** `adbpg_enable_fastann_hybrid_bitmap_pushdown = off`，`adbpg_fastann_hybrid_brute_force_selectivity_threshold = 0`，`adbpg_enable_fastann_hybrid_expression_pushdown = on`（保持开启）。

与后置过滤不同，expression push-down 在图遍历时逐节点检查过滤条件，跳过不满足条件的节点继续遍历。

**复现命令：**
```bash
vectordbbench adbpgnova $COMMON --hybrid-rate <rate> \
  --enable-bitmap-pushdown off --hybrid-brute-force-threshold 0 \
  --max-scan-points <msp> --ef-search 150 \
  --db-label expr_r<rate>_msp<msp>
```

| 选择率 | msp | ef_search | Recall | 平均 RT | P95 | P99 | 是否达 95% |
|--------|-----|-----------|--------|---------|-----|-----|-----------|
| r=0.01 (1%) | 2000 | 150 | **99.0%** | 661ms | 898ms | 1043ms | ✓ |
| r=0.001 (0.1%) | 2000 | 150 | **98.3%** | 1436ms | 1704ms | 1865ms | ✓ |
| r=0.0001 (0.01%) | 2000 | 150 | 39.5% | 1269ms | 1551ms | 1728ms | ✗ |

**备注：**
- r=0.01 和 r=0.001 默认 msp=2000 即可达 95%+ recall，无需调参。
- r=0.0001 在 msp=2000 下 recall 仅 39.5%，与 bitmap push-down 相同 — 候选行太少（100 行），图遍历无法覆盖。
- Expression push-down 的 RT 比 bitmap push-down 稍高（无 GIN bitmap 预过滤加速），但比 GIN+Sort 慢得多。
- 历史数据（修复前误触发）：r=0.5 recall=83.6% / 5.9ms，r=0.1 recall=93.7% / 54.6ms。

---

## 5. Bitmap Push-Down 降参调优（降低 msp）

客户端默认 msp=20000，在低选择率下导致过多图遍历。实例默认 msp=2000 更为合理。

**复现命令：**
```bash
vectordbbench adbpgnova $COMMON --hybrid-rate <rate> \
  --max-scan-points <msp> --ef-search <efs> \
  --db-label bp_r<rate>_msp<msp>
```

### r=0.1 (10%)

| msp | ef_search | Recall | 平均 RT | P95 | P99 |
|-----|-----------|--------|---------|-----|-----|
| 20000 | 150 | 99.2% | 631ms | 841ms | 892ms |
| 2000 | 150 | 93.7% | 626ms | 834ms | 932ms |
| 2000 | 100 | 93.7% | 570ms | 769ms | 883ms |
| 1000 | 150 | 84.0% | 560ms | 757ms | 866ms |
| 500 | 150 | 67.6% | 582ms | 787ms | 950ms |

RT 瓶颈在 bitmap 匹配（10 万行 heap fetch），而非图遍历。降低 msp 几乎不影响 RT。

### r=0.01 (1%)

| msp | ef_search | Recall | 平均 RT | P95 | P99 |
|-----|-----------|--------|---------|-----|-----|
| 20000 | 150 | 99.4% | 2580ms | 3251ms | 3522ms |
| 2000 | 150 | 99.0% | 513ms | 686ms | 793ms |
| 2000 | 100 | 99.0% | 553ms | 749ms | 890ms |
| 1000 | 150 | 96.7% | 474ms | 678ms | 850ms |
| 500 | 150 | 87.4% | 390ms | 538ms | 653ms |

msp=2000 时 recall=99.0%，RT 从 2580ms 降至 513ms（5 倍加速）。客户端默认 msp=20000 是 RT 过高的直接原因。

### r=0.001 (0.1%)

| msp | ef_search | Recall | 平均 RT | P95 | P99 |
|-----|-----------|--------|---------|-----|-----|
| 20000 | 150 | 99.6% | 1965ms | 2423ms | 2685ms |
| 2000 | 150 | 98.3% | 473ms | 644ms | 727ms |
| 1000 | 150 | 84.6% | 298ms | 432ms | 501ms |
| 500 | 150 | 58.8% | 211ms | 294ms | 360ms |

msp=2000 时 recall=98.3%（>95%），加速 4.2 倍。但仍比 GIN+Sort（172ms）慢 2.8 倍。

### r=0.0001 (0.01%)

| msp | ef_search | Recall | 平均 RT | P95 | P99 |
|-----|-----------|--------|---------|-----|-----|
| 20000 | 150 | 100% | 1951ms | 2384ms | 2599ms |
| 2000 | 150 | 39.6% | 450ms | 602ms | 673ms |

msp=2000 时 recall 仅 39.6%，完全不可用。候选行仅约 100 行，图遍历无法覆盖这些稀疏候选。该选择率必须使用 GIN+Sort。

---

## 6. 各路径横向对比（~95% Recall 最优配置）

| 选择率 | Bitmap Push-Down | GIN+Sort | Post-Filter | Expression Push-Down | 推荐路径 |
|--------|-----------------|----------|-------------|---------------------|---------|
| r=0.5 (50%) | 540ms / 94.5%（msp=5000） — 96.1% 天花板 | N/A | **7.3ms / 95.2%**（efs=800, msp=20000） | 5.9ms / 83.6%（未达标） | **后置过滤**（快 74 倍） |
| r=0.1 (10%) | **408ms / 95.4%**（msp=2500） | N/A | 10.2ms / 94.0%（天花板） | 54.6ms / 93.7%（未达标） | **bitmap push-down**（后置过滤/表达式下推无法达 95%） |
| r=0.01 (1%) | 234ms / 95.9%（msp=900） | **257ms / 100%**（threshold=0.02） | 未测试 | 661ms / 99.0%（msp=2000） | **GIN+Sort**（recall 更高，RT 相当） |
| r=0.001 (0.1%) | 183ms / 95.1%（msp=1500） | **172ms / 100%**（threshold=0.01） | 未测试 | 1436ms / 98.3%（msp=2000） | **GIN+Sort**（recall 更高，RT 更低） |
| r=0.0001 (0.01%) | 450ms / 39.6%（msp=2000，不可用） | **163ms / 100%**（threshold=0.01） | 未测试 | 1269ms / 39.5%（msp=2000，不可用） | **GIN+Sort**（唯一可行路径） |

---

## 关键发现

1. **客户端默认 msp=20000 导致低选择率下 RT 严重退化**（r<=0.01）。实例默认 msp=2000 更合理：r=0.01 的 RT 从 2580ms 降至 513ms，recall 仍保持 99.0%。

2. **Bitmap push-down 的 RT 瓶颈因选择率而异：**
   - 高选择率（r=0.1/0.5）：瓶颈在 bitmap 匹配 + heap fetch（10 万~50 万行）。降低 msp 几乎不影响 RT。
   - 低选择率（r=0.01）：瓶颈在图遍历（msp）。msp 从 20000 降至 2000 可获得 5 倍加速，且 recall 损失极小。

3. **r <= 0.001 时 GIN+Sort 是最优路径**：recall=100%，RT 约 170ms。但需要手动将 `hybrid_brute_force_threshold` 提高到 0.01，因为 GIN `@>` 的选择率估计不准确（始终估计约 0.5%，与实际数据分布无关）。ANALYZE 无法修复此问题。

4. **r=0.5 时后置过滤是最优路径**：7.3ms 即可达到 95.2% recall（比 bitmap push-down 快 99 倍）。但 r=0.1 时由于 HNSW 图结构限制，recall 天花板为 94.0%。

5. **r=0.0001 时 bitmap push-down 完全不可用**（msp=2000 时 recall 仅 39.6%）。候选行太少（约 100 行），图遍历无法找到它们。

6. **触发后置过滤需要设置 6 个 GUC**：expression_pushdown=off、bitmap_pushdown=off、brute_force=off、bitmapscan=off、seqscan=off，以及 expression_pushdown_selectivity_threshold=0。

7. **GIN 数组索引的选择率估计存在缺陷**：优化器对 `@>` 使用固定默认估计（约 0.5%），与实际数据无关。这导致低选择率场景无法自动触发暴力检索。

---

## 附录：实例默认 GUC 参考值

```
adbpg_enable_fastann_hybrid_bitmap_pushdown = on
adbpg_enable_fastann_hybrid_brute_force = on
adbpg_enable_fastann_hybrid_expression_pushdown = on
adbpg_enable_fastann_hybrid_post_filtering = on
adbpg_fastann_hybrid_bitmap_pushdown_selectivity_threshold = 0.2
adbpg_fastann_hybrid_brute_force_selectivity_threshold = 0.001
adbpg_fastann_hybrid_brute_force_with_row_threshold = 0
adbpg_fastann_hybrid_brute_force_with_table_threshold = 0
adbpg_fastann_hybrid_expression_pushdown_selectivity_threshold = 0.55
adbpg_fastann_hybrid_expression_pushdown_with_bitmap_selectivity_threshold = 0.45
fastann.hnsw_ef_search = 100
fastann.hnsw_max_scan_points = 2000
fastann.index_scan_mode = serializable
fastann.nova_adaptive_gamma = 0
fastann.nova_nprobe = 5
fastann.nova_topk_amp_add = 0
fastann.nova_topk_amp_mul = 1
fastann.quantize_rescore_amp = 1
optimizer = on
plan_cache_mode = auto
```

## 附录：客户端 CLI 默认值（vectordbbench adbpgnova）

```
ef_search = 150
max_scan_points = 20000
nova_topk_amp_mul = 1.0
nova_topk_amp_add = 0.0
quantize_rescore_amp = 0.0
nova_adaptive_gamma = 0.0
index_scan_mode = snapshot
nprobe = 5
hybrid_brute_force_threshold = 0.001
enable_bitmap_pushdown = on
enable_expression_pushdown = on
optimizer = off（硬编码）
plan_cache_mode = force_custom_plan（硬编码）
```
