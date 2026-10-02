# vit_matte `--task presence`（S1 表达区语义）具体实现方案

> 目标：把现有 vit512/vit_matte（Virchow2 冻结 + LoRA + ViTMatte 解码）从「H&E→16ch 连续强度回归」
> 扩展出 **presence 模式**：H&E → 16ch「表达区 soft 先验(0–1) + 边界」，作为三阶段方案（定稿 v2.0 §4）的 S1 模块。
> 原则：**换任务不改架构**——同一骨干/解码器，只换输出目标、损失与微调策略；回归 ckpt 作热启动。
> 本文档是定稿方案 M2 的落地执行规格；外层计划/门槛见 `../PLAN_定位-表型-重染色方案.md`。

---

## 0. 范围与边界

- **做**：S1 = 逐 marker 表达区语义（哪里会染），输出 16ch soft 概率图（含边界监督）。
- **不做**：核实例分割（交给 StarDist/HoVerNet）；染色强度/质感（交给 S3）；I2SB 不参与本任务。
- 兼容：现有 reg 模式（v2/v3 训练与采样）**不动**，通过 `--task {reg,presence}` 切换，默认 `reg`。

---

## 1. 数据与真值：presence soft 目标怎么造

### 1.1 输入源（全部现成）
- `csv_nuclei_pos/{slide}*.csv`：逐 marker 阳性核 id/坐标（真值来源，15 个非 Hoechst marker）；
- `nuclei/*.tiff`：核实例 mask（int32，label=核 id）；
- `he/*.jpeg`：输入 H&E；
- train/val/test dataframe：tile 级对应关系。

### 1.2 soft 目标公式（与定稿 §3.2 一致）
对每个 tile、每个 marker $c$：

$$
P^{\text{GT}}_c(p)=
\begin{cases}
1,& p \text{ 落在 } c\text{-阳性核内}\\
\exp\!\big(-d^2(p,\text{阳性核集合})/(2\sigma^2)\big),& 0<d<\kappa\sigma\\
0,& \text{其他}
\end{cases},\qquad
\sigma=3\,\mathrm{px}@512 \ (\text{其他分辨率按比例}),
$$

- $d(\cdot,\cdot)$：到最近阳性核的距离（距离变换，`cv2.distanceTransform` 或 `scipy.ndimage`）；
- 硬 0/1 版同时导出，供消融（E1 对比 soft/hard）；
- 核 mask 上采样到与 tile 分辨率一致（256/512）；marker 通道顺序 = `CHANNELS`（Hoechst 用核 mask 全 1 填核即可，但**不进 S2 头**，仅作可视化/核先验）。

> 口径提醒（吸取 I2SB/vit 归一化教训）：presence 目标 = 0–1 soft mask，**不再有** log/线性
> [-0.9,0.9] 与 exp 反变换陷阱，**也没有** reg 口径的 fg_thresh(-0.9/-1) 概念——所有阈值只在
> **最后二值化**用 val 最优阈值，训练/评估全程是 0–1 soft。
> 真值局限声明：软真值是**核锚 + halo 外推**（csv_nuclei_pos 只有核级阳性）。对 CD31/SMA 等
> 组织级 marker，空间还原上限受 halo σ 限制。故 **P0 需统计各 marker“核外阳性像素占比”**
> （用真实 mIF 或 csv 核周证据）：占比高 → 分通道放大 σ，或叠加组织级辅助 GT，否则 S1 对这些
> marker 只能力度不足地外推。
> **凡"用 mIF 强度做统计/GT"的地方必须在对数域**（q 口径，`marker_q.json`）：核外阳性占比判定、
> 组织级辅助 GT 的强度阈值都应在 **log 域**统计并做敏感性分析——禁止在线性域直接看弱信号
> （vit v2"预测仅 GT 的 2–16%"教训：弱表达在线性域不可见、易误判为阴性）。这与 presence 训练
> 目标本身 0–1（不涉强度）不冲突：presence 只判存在性，凡要读亮度的地方一律走 log 域。

### 1.3 落盘格式（避免训练期逐 tile 现算）
新建 `datacore/build_presence_cache.py`（并行，workers=64）：

```
/data/weiyh/orioncrc_presence_{tile}/
  train_prior.raw    # uint8 [N,16,H,W]（0-255 ⇔ 0-1）
  val_prior.raw  / test_prior.raw
  D5k_prior.raw      # 5k 子集（train 免疫富集抽样，与 D5k he 子集同一份索引）
  presence_meta.json # {tile_size, sigma, counts, index}
```

约 0.3 GB/万 tile（256，uint8）量级，磁盘无压力；训练数据流：H&E 从现有 he 缓存读、prior 从 presence 缓存读。

---

## 2. 代码改动清单（精确到函数/参数）

### 2.1 `vit_matte/vitmatte_unet.py`（小改）
- `ViTMatteUNet.__init__(..., task="reg", presence_sigma=3.0)`：
  - `task="presence"` 时 heads 仍为 16×`Conv2d(c64f,1,1)`，但**初始化换成 xavier_uniform + bias 0**；
  - `forward()` 末尾：`return torch.tanh(y)`（reg，保持原样）**vs** `return y`（presence：**裸 logits**，0–1 由损失里的 sigmoid 给出，避免在模型里夹 sigmoid 影响 Dice 的数值稳定性）；
  - 训练/推理前可用 `torch.manual_seed` 固定，保证可复现。

### 2.2 `datacore/orioncrc_dataset.py`（加 presence 分支）
- `OrionCRCDataset.__init__(..., task="reg")`：
  - `task="presence"` 时改用 presence 缓存目录 `orioncrc_presence_{tile_size}`，`__getitem__` 返回
    `(he, prior)`，prior 归一到 [0,1] float32，**不做** log/[-0.9,0.9] 归一化；
  - `return_nuclei=True` 时额外返回 `nuclei`（供 S1 输入消融 E1.4 与后续 S2/S3 复用）；
  - reg 路径一行不改。

### 2.3 `vit_matte/train.py`（主体改动）
1. 参数：`--task {reg,presence}`（默认 reg）；presence 专用：
   `--lambda_bce 1.0 --lambda_bnd 0.5 --fg_weight 10.0 --sigma 3.0 --use_boundary --no-use_boundary`
   `--presence_metric {mdice}`；沿用 `--tile_size/--vit_size/--vit_layers/--lora_*` 等。
   **逐通道权重**（吸取 I2SB w_sparse / vit 1-std 教训）：自动读 presence 缓存里的
   `pos_rate_c`（各 marker 阳性像素占比），`w_c = (1/pos_rate_c)` 归一化到均值 1——
   PD-L1/FOXP3 等极稀疏通道不再被统一 fg 淹没；可用 `--marker_w uniform|inverse_pos` 切换。
2. 模型：`ViTMatteUNet(..., task=args.task)`。
3. **presence 损失**（在 `train.py` 内新增 `presence_loss(logits, prior, sigma, lam)`）：

```python
def presence_loss(logits, prior, w_c=None, lam_dice=1.0, lam_bce=1.0, lam_bnd=0.5, fg=10.0):
    """w_c: [16] per-marker 权重（默认 1/pos_rate_c 归一化；可 uniform）。"""
    p = torch.sigmoid(logits)                       # [B,16,H,W] in (0,1)
    # BCE：先逐像素 fg 加权，再逐通道求和（w_c 归一化到均值 1，不改变整体尺度）
    bce = F.binary_cross_entropy_with_logits(logits, prior, reduction="none")
    bce = (bce * (1.0 + fg * prior)).mean(dim=(0, 2, 3))          # [C]
    bce = (bce * w_c).sum() if w_c is not None else bce.mean()
    # soft Dice（逐通道）: 2|p∧g| / (|p|+|g|+eps)
    num = 2*(p*prior).sum(dim=(2,3)) + 1e-6
    den = p.sum(dim=(2,3)) + prior.sum(dim=(2,3)) + 1e-6
    dice = 1.0 - ((num/den) * w_c).mean() if w_c is not None else 1.0 - (num/den).mean()
    # 边界监督：GT 与预测的边缘用 3x3 Sobel（在 GPU 上现算）
    gx = sobel(prior); gx_hat = sobel(p)            # sobel 用 F.conv2d 实现
    bnd = F.smooth_l1_loss(gx_hat, gx)
    return lam_dice*dice + lam_bce*bce + lam_bnd*bnd
```

4. `evaluate()` 增加 presence 分支：val 上算 **mDice**（16ch 均值）+ 前景 Dice + 边界 F1 + 细胞召回，
   替代 PSNR/SSIM；`best.pt` 的判据从 `psnr` 改为 `mdice`。
5. **指标口径纪律（吸取 "train.py val PSNR 虚高(*0.5+0.5)" 教训）**：Dice/IoU/边界 F/召回
   只实现一次于 `eval/eval_mask.py`，train 的 `evaluate()`、sample 后处理、最终 json **全部调用
   同一函数**；阈值唯一来源 = val 最优阈值（存 ckpt `best_thresh`），test 只跑一次。
6. **热启动（换头微调）**：`--resume <reg_ckpt>` + `--task presence` 时：
   - 加载回归权重（`strict=False`），**剔除不匹配层**：presence heads 的 key 与 reg 相同
     （同是 `heads.*`），但 shape 一致（都 16×1×1×1 输出）——**通道相同、可直接加载**，
     唯一差异是 reg 训练把输出过 tanh 而 presence 不夹——因此权重语义不同，
     **heads 必须随机重置**（见 2.1），其余（encoder LoRA、neck、detail、decoder）原样加载；
   - 两段式微调（推荐）：
     - Phase-A（快速对齐新目标）：冻结 decoder 除 heads 外的全部层？否——保留解码器可训，
       用**小 lr（5e-5）+ 短 warmup(200)**跑 D5k×2 epoch，观察 train loss/mDice 是否降；
     - Phase-B（正式微调）：恢复 lr 2e-4、warmup 400、cosine，D5k 全量 → val mDice 门槛 →
       Dfull；
   - 实现提示：`state = {k:v for k,v in ckpt["model"].items() if not k.startswith("heads.")}`
     + 手动把 heads 重初始化，避免 shape 陷阱。
6. 采样可视化 `save_viz_snapshot` presence 分支：三栏改为 [H&E | GT prior（多色） | 预测概率]，
   预测与 GT 用同一套 multicolor_composite（prior 0–1 ×255）。

### 2.4 `vit_matte/sample.py`（presence 推理）
- `--ckpt` 内 `config["task"]="presence"` 自动识别（现有 config 自动读取逻辑已就绪）：
  - 输出 `pred_{name}_prob.tiff`（16ch，0–255 = 0–1×255 概率）与 `pred_{name}_mask.tiff`
    （按 val 最优阈值二值化，阈值存 ckpt `best_thresh`）；
  - RGB 复合预览直接基于 prob×255。

### 2.5 `eval/`（新增）
- `eval/eval_mask.py`：per-marker Dice/IoU、边界 F1（Sobel 边缘 IoU）、细胞召回
  （预测阳性区 ∩ GT 阳性核 ≥ 阈值 → 判为该核被召回）——DP1 数值唯一来源；
- `eval/eval_mask.py` 输出与 `cell_f1.json` 同风格 json：`{marker: {dice, iou, bnd_f, recall}, mean_*}`。

---

## 3. 数据流（一张图）

```
he缓存(256/512) ─┐
csv_nuclei_pos ──┤ build_presence_cache.py ─► presence_{tile}/{split,D5k}_prior.raw
nuclei ──────────┘
OrionCRCDataset(task="presence") ─► (he, prior) ─► ViTMatteUNet(task="presence")
   loss = dice + λbce·bce(fg加权) + λbnd·sobel边界        （evaluate → mDice/边界F/召回）
   best.pt  ← val mDice 最高；step 快照 + viz 每 1000
sample.py(presence) ─► prob.tiff + mask.tiff(最优阈值) ─► eval_mask.py(DP1) ─► 供 S2 的 F2 / S3 的 P̂
```

---

## 4. 训练配方（presence）

| 项 | 值/说明 |
|---|---|
| 分辨率 | D5k 消融在 256（与 vit256 v3 同尺寸、迭代快）；终版在 512（vit_size 448） |
| batch | 单卡 batch4+grad_accum2（等效8，512）；256 可 batch8–16 |
| 优化 | Adam lr 2e-4（Phase-A 5e-5），wd 1e-5，clip 1.0，warmup 400，cosine |
| 损失权重 | λ_dice=1, λ_bce=1, λ_bnd=0.5（E1.3 消融 0/0.25/0.5/1.0），fg=10（E1.2 消融 1/10/30） |
| 目标 | soft（σ=3，E1 可对比 hard 0/1） |
| 输入 | H&E（E1.4 消融 ± nuclei 通道） |
| 初始化 | `virchow2_vitmatte_v2_512_epoch15.pt`（512）或 v3 best（256，训练结束取），heads 重置 |
| epochs | Phase-A 2（D5k）→ Phase-B 5–8（D5k，达标后 Dfull 同配方） |
| 对照 | E1.5：UNet++/SegFormer-B0 1k 样本（证明大骨干必要性，可选） |
| 验证 | 每 2000 iter mDice/边界F/召回；每 1000 步 ckpt+viz；best=val mDice |

示例命令（256 消融基线）：
```bash
PY=/home/weiyh/.conda/envs/virchow_env/bin/python
$PY -m vit_matte.train \
  --name virchow2_vitmatte_presence_256 \
  --task presence --gpu 1 --tile_size 256 --vit_size 224 \
  --vit_layers 8,16,24,32 --lora_r 32 --lora_alpha 16 \
  --batch_size 8 --epochs 8 --lr 2e-4 --warmup_iters 400 \
  --lambda_bce 1.0 --lambda_bnd 0.5 --fg_weight 10.0 \
  --resume /data/weiyh/weights/vit_matte/virchow2_vitmatte_v3_256_best.pt \
  --eval_every 2000 --save_every 1000 --ckpt_keep 5 \
  --ckpt_dir /data/weiyh/weights/vit_matte
```

---

## 5. 验收与门槛（DP1，test 只测一次）

| 指标 | 门槛 |
|---|---|
| 结构 marker Dice（Hoechst/E-cad/Pan-CK/SMA/CD45 等） | ≥ 0.85 |
| 稀疏 marker Dice（FOXP3/PD-L1/CD8a/CD163） | ≥ 0.55 |
| 边界 F1 | 显著高于"纯 MSE 强度模型阈值化"基线 |
| 细胞召回 | ≥ 0.90（预测阳性区命中 GT 阳性核比例） |
| 对照 E1.5 | 记录轻量骨干结果，决定是否保留 vit512 主干 |

未过 → 先查 soft 真值口径 → 补核实例（StarDist/HoVerNet 输出作核先验输入 E1.4）→ 放大骨干/加数据。

---

## 6. 时间与排期（并入定稿 M2）

| 步骤 | 内容 | 耗时 | 前置 |
|---|---|---|---|
| P0 | `build_presence_cache.py`（train/val/test + D5k）+ 口径校验（含 **σ 合理性：各 marker 核外阳性占比统计**） | 2–3 天 | M0 的 soft 真值已定 |
| P1 | 代码改动 2.1–2.3 + 冒烟（presence 1 epoch D5k，跑通 loss/viz/eval） | 1–2 天 | P0 |
| P2 | D5k 微调基线（256）→ mDice/边界/召回 首报 | 0.5–1 天(GPU) | P1 |
| P3 | 消融 E1.2/E1.3/E1.4（256, D5k）；E1.5 轻量对照 | 1–2 天 | P2 |
| P4 | 入选配置 → 全量（256，与 vit256 v3 错峰；512 终版另排） | 2–4 天 | DP1(D5k) 通过 |
| P5 | 512 重训 + sample + eval_mask → DP1 正式验收 | 2–3 天 | P4 |
| — | 合计（并入 M2） | ≈1.5–2 周 | 与核实例并行 |

> 与正在跑的 vit256 v3 关系：presence 是**独立实验名**，可用 1–2 张空闲卡跑 D5k（P2/P3）
> 不受 v3 影响；v3 结束后其 best.pt 即为 256 presence 的最佳热启动源（其 log 归一化+fg 训练
> 已让特征更偏"阳性结构"，比 v2 更适合起步）。

---

## 7. 风险与兜底

| 风险 | 兜底 |
|---|---|
| 回归权重热启动慢/不稳 | Phase-A 小 lr 先对齐；观察 train loss 若不降先查 heads 是否真被重置 |
| soft 目标与 cell_classify 口径不一致 | P0 加一致性校验（§3.3）：抽样 tile，投影回细胞 ≥99% 与 csv 一致 |
| 边界监督抖动 | λ_bnd 从 0 起消融；Sobel 用归一化梯度避免尺度影响 |
| 稀疏 marker 召回不足 | fg 加权 + σ 调大消融 + 免疫过采样 D5k 抽样保阳性 tile |
| 输出概率校准差 | val 最优阈值存 ckpt；sample 直接输出 prob+mask 两版 |

---

## 8. 交付与记录

- ckpt：`{name}_best.pt`（config 含 `task:"presence"` + `best_thresh`）与 `step*.pt`；
- viz：`viz_{name}/viz_{name}_step{it}.png`（H&E | GT prior | 预测概率，多色）；
- 指标：`eval_mask` 输出 json + 汇报用图（patch/WSI 对照，复用 `reports/` 模板）；
- 记录：本文件即执行手册；结果回填定稿 §4/§9 的 DP1 行与 `PLAN.md`。

---

## 9. 既往教训对照表（presence 是否已防回退）

| # | 此前踩过的坑（来源） | 在 presence 中的防回退设计 | 位置 |
|---|---|---|---|
| 1 | 全通道对数+exp 反变换放大误差（I2SB v2） | 目标=0–1 soft mask，全程无 log/exp/强度反变换 | §1.2 口径提醒 |
| 2 | 线性 MSE 把稀疏峰值压向 0（vit v2 偏暗 2–16%） | 只判有/无不回归强度；Dice 对类别平衡不敏感 | §2.3 loss |
| 3 | 弱表达 marker 被淹没（PD-L1/FOXP3 F1≈0.05） | **per-marker w_c∝1/pos_rate + fg 加权** | §2.3(1)、loss |
| 4 | train 日志 val PSNR 虚高（*0.5+0.5） | 指标**单一实现**（eval_mask），train/eval/sample 同函数同阈值 | §2.3(5) |
| 5 | log vs linear 两套口径对不上 | presence 无强度口径；soft(0–1)→val 最优阈值二值化 | §1.2、§2.5 |
| 6 | batch 受显存限制训练过慢（vit v2 单卡 6.9 天） | 256 大 batch、512 batch4+acc2、4 卡 DDP；先 D5k 后全量 | §4、§6 |
| 7 | 只存 latest 丢最优模型 | best=mDice + step 快照 + keep5 清理（沿用 v3 设施） | §4、§8 |
| 8 | 收敛末期震荡选错 ckpt（I2SB 9000 步尖峰） | best 按 val mDice 独立保存，viz 每 1000 步肉眼复检 | §4 |
| 9 | 免疫富集 tile 占比低、欠采样 | D5k 按免疫富集抽样、保阳性 tile；fg 加权 | §1.3、§4 |
| 10 | （新）csv 只有核级阳性，组织级 marker 空间受限 | P0 σ 合理性校验（核外阳性占比），必要时分通道 σ/组织级辅助 GT | §1.2、P0 |
| 11 | （新）统一 fg 不足以覆盖极稀疏通道 | per-marker w_c（见 #3），默认 inverse_pos | §2.3 |
| 12 | （提醒）mIF 强度统计/GT 必须在对数域（弱信号线性域不可见，vit v2 2–16% 教训） | presence 训练目标仍 0–1（不涉强度）；凡 σ 校验/组织级辅助 GT 等强度统计一律 q 口径 log 域 | §1.2 |
