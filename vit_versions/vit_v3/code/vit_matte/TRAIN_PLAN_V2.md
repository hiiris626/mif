# vit_matte V2 训练计划（多尺度 + 高分辨率 ViT 特征）

> 版本：V2（相对基线 `virchow2_vitmatte_512_iter140000` 的架构增强版）
> 日期：2026-08-29
> 硬件：4× RTX 4090（每卡 49GB，当前被 I2SB 占用 ~27.5GB，可用 ~21.6GB）

## 0. 本版本改动回顾（相对基线）

| 改动 | 基线 | V2 | 影响 |
|---|---|---|---|
| ViT 输入分辨率 | 224（resize） | **448**（32×14，patch grid 32×32） | 特征分辨率 1/32→1/16，信息量 4×；**注意力计算 16×（256→1024 tokens）** |
| ViT 特征层 | 仅最后一层 | **4 层**（8/16/24/32）+ `ViTNeck` 融合 | 多语义层级融合 |
| DetailCapture c1/c2 | 3×3 | **多尺度膨胀卷积**（c1 感受野 3/5/7/11，c2 3/5/7） | 捕捉稀疏小斑点+大块结构 |
| 解码器首层 conv16 | 3×3 | **多尺度膨胀卷积**(3/5/7) | 增强 ViT+CNN 融合 |
| 可训练参数 | 13.42M | **24.89M** | — |

## 1. 目标与基线

**基线**（旧 vit_matte_512 iter140000，test 10,952 tiles）：

| PSNR↑ | SSIM↑ | Pearson↑ | 细胞 AUC↑ | 细胞 F1↑ |
|---|---|---|---|---|
| 39.25 | 0.814 | 0.217 | 0.831 | 0.383 |

**目标**：V2 在保持/提升 PSNR/SSIM 的同时，**重点提升稀有免疫 marker 的细胞级指标**（PD-L1 当前 AUC 0.499 最差、FOXP3/CD8a F1<0.12），这正是多尺度 + 高分辨率的发力点。

## 2. 资源准备（Stage 0）

### 2.1 重建 512 缓存（当前为空）
- 现状：`/data/weiyh/orioncrc_cache`（256）存在（370G）；`/data/weiyh/orioncrc_cache_512` **为空**。
- 原始 tile 本身即 512×512，`tile_size=512` 时 build_cache 直接解码、不 resize。
- 命令：
```bash
python scripts/build_cache.py --data_root /data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x \
    --tile_size 512 --workers 64
```
- 大小估算：train 295,096 × 19ch × 512² ≈ 1.47 TB；加 val/test 共 ~1.55 TB。磁盘 `/data` 可用 5.3T，足够。
- 耗时：318k tile 解码，多进程 64 workers，预计数小时（可后台 `nohup` 跑，用 `.done` 标记监控）。

### 2.2 确认就绪项
- ✅ 权重 `/data/weiyh/weights/virchow2/model.safetensors`
- ✅ `marker_std.json`（`/data/weiyh/orioncrc_miphei/marker_std.json`）
- ✅ 划分 CSV（train/val/test_dataframe.csv）
- ⚠️ GPU 仅剩 ~21.6GB/卡（I2SB 在跑），**batch size 必须实测，不能沿用 16**

## 3. Stage 1：Smoke Test（先跑通 + 测速度/显存）

**目的**：确定 448 输入下真实 it/s、显存峰值、可用 batch。

```bash
cd /home/weiyh/h2e_mif_benchmark
/home/weiyh/.conda/envs/virchow_env/bin/python -m vit_matte.train \
    --name virchow2_vitmatte_v2_smoke \
    --gpu 0 \
    --epochs 1 \
    --batch_size 4 \
    --tile_size 512 \
    --vit_size 448 \
    --vit_layers 8,16,24,32 \
    --lora_r 32 --lora_alpha 16 \
    --eval_every 1000 --save_every 1000 \
    2>&1 | tee /data/weiyh/logs/vitmatte_v2_smoke.log
```

**观察点**：
1. 是否 OOM（448 输入注意力 16×，batch 4 起步，逐档试 4→8）。
2. 记录 `s/it`（预期显著慢于基线的 0.63s/it，可能 2–10×）。
3. 确认 4 层中间特征、neck、多尺度前向无报错。

### ✅ 实测结果（2026-08-29，真实权重 + 真实训练循环，随机输入排除 IO）

> 注意：测试时 4×4090 被 I2SB 占满（每卡 27.5GB，剩 21.6GB），速度为此共享状态下的值，**独占 GPU 会更快**；显存是硬数据、可靠。

| batch | s/it | it/s | 峰值显存 | 备注 |
|---|---|---|---|---|
| 2 | 0.26 | 3.90 | 8.4 GB | OK |
| 4 | 0.48 | 2.09 | 12.8 GB | OK（当前共享 GPU 下的安全上限） |
| 8 | — | — | OOM | 需 ~21.6GB，加上 I2SB 的 27.5GB 超 49GB |

**结论**：
1. **显存**：batch=4 需 12.8GB，当前共享 GPU（剩 21.6GB）下 batch=4 安全、batch=8 贴边 OOM；**等 I2SB 释放后独占 49GB 可跑 batch=8~16**。
2. **速度**：448 输入未导致灾难性变慢（batch=4 共享下 0.48s/it，前向本身可控）。独占 GPU 后 it/s 预计更高。
3. **时间预估（独占 GPU，粗估 it/s 翻倍）**：15 epoch = 295096/batch×15 iters，batch=4 ≈ 3.2 天、batch=8 ≈ 2.5 天。

**决策**：根据实测 it/s 估算 15 epoch 总时长（总 iters = 295096/batch × 15 ≈ 553k（batch 8）），决定：
- 单卡慢 → 改 **4 卡 DDP**（需给 train.py 加 DDP，或写简单启动脚本）；
- 时间不可接受 → 下调 `vit_size` 到 224（保留多尺度/neck，仅放弃高分辨率）作对照。

## 4. Stage 2：完整训练

```bash
/home/weiyh/.conda/envs/virchow_env/bin/python -m vit_matte.train \
    --name virchow2_vitmatte_v2_512 \
    --gpu 0 \
    --epochs 15 \
    --batch_size 8 \
    --tile_size 512 \
    --vit_size 448 \
    --vit_layers 8,16,24,32 \
    --lora_r 32 --lora_alpha 16 \
    --lr 2e-4 --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --warmup_iters 400 \
    --eval_every 2000 --save_every 5000 \
    2>&1 | tee /data/weiyh/logs/vitmatte_v2_512.log
```

**超参说明**：
- lr 保持 2e-4（基线同款）；新增的 neck/多尺度/conv16 随机初始化层靠 warmup + 梯度裁剪稳定。
- LoRA 秩保持 32（1280 维的 2.5%，先不动，后续可消融 64）。
- 断点续训：`--resume <ckpt>` 已支持（沿用基线机制）。

**监控**：每 50 iter 打印 loss；`val PSNR/SSIM` 每 2000 iter；loss 收敛目标 ~0.003（加权 MSE，同基线）。

## 5. Stage 3：评估（与基线同脚本，保证可比）

```bash
# 像素级（PSNR/SSIM/Pearson）
python eval/metrics.py --pred_dir ... --checkpoint virchow2_vitmatte_v2_512_epoch15.pt ...

# 细胞级（AUC/F1，重点看稀有 marker）
python eval/cell_classify.py --checkpoint ... --pred_dir ...
```

**评估口径不变**：`(out+0.9)/1.8*255` 还原（`eval/metrics.py` 正确口径，勿用 train.py 内 val 评估的旧还原）。

## 6. 对比与消融

| 实验 | 目的 |
|---|---|
| V2 完整版 vs 基线 iter140000 | 主对比（PSNR/SSIM/Pearson/AUC/F1） |
| `--no_multi_scale` | 多尺度卷积消融 |
| `--vit_size 224`（V2 其余不变） | 高分辨率 vs 多尺度/neck 各自贡献 |
| （可选）`--lora_r 64` | LoRA 秩消融 |

## 7. 风险与注意

1. **速度未知**：448 输入注意力 16× 是最大变量，务必先 Stage 1 实测再定计划。
2. **显存**：GPU 仅剩 21.6GB，batch 需实测；若不够，用梯度累积补 batch 等效。
3. **checkpoint 体积**：基线每个 ckpt 2.6GB（含冻结 632M 骨干 state_dict），15 epoch + 每 5000 iter 会占大量磁盘（磁盘 5.3T 可用，但仍建议只保留 epoch 级 + 末尾 ckpt）。
4. **512 缓存重建**：约 1.55TB、数小时，需在训练前完成，否则实时解码拖慢吞吐。

## 8. 时间预估（依赖 Stage 1 实测）

| 项目 | 预估 |
|---|---|
| 512 缓存重建 | ~数小时 |
| Smoke test | <1h |
| 完整 15 epoch | 待实测 it/s；若 ~2s/it × 553k ≈ 12.8 天（单卡），**需 DDP 或下调 vit_size** |
