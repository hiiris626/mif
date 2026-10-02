# M1 Runbook：S2 先导实验（E-resolution + E2.2 天花板门禁 + 先验预扫）

> 定位：三阶段方案（定稿 v2.0）M1（S2 隔离实验）的**启动执行手册**。
> 目标：用 2–3 个"便宜实验"在投入 S2 主训练（E2.3+）之前回答三个问题：
>   ① 清晰度/染色是不是瓶颈（E-resolution）；
>   ② H&E 对 15-marker 表型的"识别天花板"有多高（E2.2 门禁）；
>   ③ 表达区先验(F2)/病理软先验是否有增量（E2.7 预扫）。
> 依据：方案主文件附录 A（四问答疑与方案影响）；本文档只做"先跑通、先出结论"，主训练照常回 §5。

---

## 0. 总原则

- 全部在 **val 子集（12,402 tile / 1,528,069 细胞）** 上完成（test 10,952 只测一次）。
- 特征/表格先落盘缓存，供 E2.3 主训练复用（避免重复抽取 Virchow2）。
- LR 基线统一：`sklearn LogisticRegression(max_iter=1000)` + StandardScaler；每 marker 独立，
  报 **AUPRC + AUC + F1(val 最优阈值)**（与 oracle/vit v2 同口径）。
- 每步有"判定/出口"，未过即切计划，不无脑加算力。

---

## 1. 前置：统一细胞特征表（`datacore/build_cell_features.py`，新增）

为 val（+预留 test）每个细胞生成一行，落盘 `features/cells.parquet`（+采样子集）：

| 字段 | 内容 | 来源 |
|---|---|---|
| `split` / `slide` / `tile_idx` / `cell_id` | 定位 | dataframe + nuclei mask |
| `gt[15]` | 15-marker 0/1（无 Hoechst） | `csv_nuclei_pos`（表型真值） |
| `mif_feat[16]` | 核区域真实 mIF 平均荧光 | `if/` + nuclei（=现有 cell_classify 特征） |
| `he_feat[D=1280]` | H&E 核 patch（核±halo，224 crop）Virchow2 冻结特征 | Virchow2 encoder 现成 |
| `prior_agg`（F2, GT 版） | S1 软真值在核内/核周 1–2px 环的 mean/max/p99/最近距 | presence 缓存（见 presence runbook P0，先用其 GT 版） |
| `he_feat_512`（E-resolution 用） | 512 上采样 patch 的 Virchow2 特征（可选，见 M1-R1-b） | cache_512 |

要点：
- patch 尺寸：核 bbox 外扩到 ≥64px@333 语义（256 训练按比例），Virchow2 需 224×224 → resize；
- 总量 1.53M(val) + 1.78M(test) 行 × 1300 维 float32 ≈ 十几 GB，可直接 parquet + 子采样；
- 抽样子集：每 marker 保证 ≥2k 阳性，其余随机压到每 marker ≤50k 样本（LR 用）。

---

## 2. M1-R1：E-resolution（半日，证伪"清晰度假设"）

三种 H&E patch 变体，各抽 **20–50k val 细胞**，抽 Virchow2 特征，跑 LR per-marker：

| 变体 | 说明 |
|---|---|
| R1-a | 原生 333 patch → resize 224（当前基线） |
| R1-b | 333→512 上采样 cache 的 patch → resize 224（验证"上采样给没给新信息"） |
| R1-c | R1-a + **染色归一化**（Reinhard 对齐到参考 WSI，或 MACENKO） |

**判定（预判：R1-a ≈ R1-b，染色归一化只小幅波动）**
- 三者的 marker 均值 AUC 差 <0.01 → **卡点在信息而非清晰度/染色**，不再升级分辨率，R1-a 为默认输入；
- 若 R1-c 显著更好 → 后续训练管线加染色归一化（副作用小、可随时开）；
- 若 R1-b 显著更好 → 才考虑更高清重抽（可能性低，需另找源）。
- 产出：`M1_R1_resolution.json` + 一行结论写回本文件出口表。

---

## 3. M1-R2：E2.2 识别天花板门禁（1 天，最关键）

per-marker LR 两列对比（同 val 细胞）：

| 列 | 特征 | 意义 |
|---|---|---|
| Oracle（E2.1 同法） | `mif_feat`（真实 mIF） | 信息上界（预期 ≈0.982/0.813） |
| E2.2（H&E） | `he_feat`（Virchow2 冻结，R1-a） | H&E 可学性（本项目此前从未直接测过） |

**判定与出口（依方案附录 A.2）**
1. 若 E2.2 均值 AUC ≥0.90 → 识别路线畅通，直接进入主训练（E2.3），并把该表作为 E2.3 的对照；
2. 若 0.80–0.90 → 识别可行但有损：按 marker 分组处理——
   - **A 组（可分辩，如 Pan-CK/E-cad/CD45/CD68/SMA 等形态可分）**：继续逐细胞硬判；
   - **B 组（低可分，预期 CD4/CD8a/FOXP3/PD-L1/Ki67 等）**：切"置信度/区域丰度"口径
     （报告 calibrated 概率 + 预警 + 区域阳性占比），不硬做逐细胞 0/1；
3. 若 <0.80 → 先查对齐/核框（M1-R1 已排除分辨率）→ 加 F2/放大 LoRA（小试 E2.4 变体）→
   仍低则按方案附录 A.2 全面转"生成辅助识别 + 区域口径"，S2 只保留 A 组逐细胞。
- 产出：`M1_R2_ceiling.json`（per-marker oracle AUC/AUPRC/F1 vs H&E）＋ `M1_R2_verdict.md`
  （A/B 组划分，作为后续所有 S2 实验的分组依据）。

---

## 4. M1-R3：先验预扫（半天，E2.7 的便宜预演）

- LR 三列对比（同 val 子集，每 marker）：`he_feat` vs `he_feat+prior_agg(F2,GT)` vs `he_feat + 区域类型`
  （区域类型来自 S1 GT soft 真值阈值化后在核上的众数：上皮/间质/淋巴密集…粗 3–4 类即可）。
- **判定**：若 F2/区域列使低可分 B 组 AUC 平均提升 ≥0.03 → E2.7 全开（区域条件 + GNN/共现正则，
  进 E2.3 主训练的消融清单）；否则先跳过 E2.7，避免复杂化主链。
- 产出：`M1_R3_prior.json`。

---

## 5. 出口与主路线选择

| # | 条件 | 走向 |
|---|---|---|
| 出口 A | R2-1 达成 | 直接主训练 E2.3+（§5.5），E2.7 按 R3 结果排入消融 |
| 出口 B | R2-2 | A 组主训练 + B 组"不确定率/区域丰度"交付（决策 #7），主指标并列报告 |
| 出口 C | R2-3 | S2 收敛为 A 组分类器 + 生成辅助（S3 先行），方案 §9 门槛按 A/B 组改写 |

- 主训练若走出口 A/B：E2.3 从 `cells.parquet` 取特征训练 15 头（Virchow2+LoRA 微调版由
  `vit_matte/train_heads.py` 完成，LoRA 参数从 E2.2 冻结版热启动，见 §5.5）。

---

## 6. 成本与依赖

| 步骤 | 内容 | 估算力 | 依赖 |
|---|---|---|---|
| R1 | 3 变体 × 20–50k 细胞 Virchow2 特征 + LR | 1–2 GPU·h | `build_cell_features.py` |
| R2 | 2 列 × 15 marker LR + 报告 | <1 GPU·h | R1-a |
| R3 | 2 列 × LR | <1 GPU·h | R1-a + presence GT 聚合 |
| 合计 | ≈1 工作日（含编写/冒烟） | ≤3 GPU·h | 1 卡即可，与 vit256 v3 错峰 |

新增代码：`datacore/build_cell_features.py`、`eval/m1_probes.py`（一次跑完 R1–R3 并出 json/verdict）。
输出统一放 `results/m1_probes/`。

---

## 7. 记录与验收

- 记录文件：`results/m1_probes/{M1_R1_resolution,M1_R2_ceiling,M1_R2_verdict,M1_R3_prior}.json/md`
- M1 里程碑"通过"定义：R2 出口已定 + 特征缓存已建 → 进入 E2.3 主训练；同时把 A/B 组划分
  回填 §5.5/§9 与定稿决策 #7。
