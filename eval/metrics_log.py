"""论文口径评估：MIPHEI-ViT §3.6.3 的 log 归一化域 PSNR/SSIM/Pearson。

论文公式(2)：I_norm = 255 * ln(min(I, q_c) / q_c + 1)
  - q_c：每 marker 训练集前景像素(>0)的 99.9 分位（marker_q.json）
  - 在 log 域上算 PSNR/SSIM/Pearson，与论文 Table 2 的基准对齐：
    MIPHEI-ViT（H-optimus-0 + LoRA, no GAN）：PSNR 31.89 / SSIM 0.951 / Pearson 0.466

聚合口径：同 eval/metrics.py（逐图平均）；2026-09-20 统一重评改为全局 MSE 聚合。

用法：
    python -m eval.metrics_log --pred_dir <pred> --target_dir <ROOT/if> \
        --data_root <ROOT> --split test --q_file /data/weiyh/orioncrc_cache/marker_q.json \
        --out metrics_log.json
"""
import os
import json
import glob
import argparse
import numpy as np
import tifffile
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from datacore.orioncrc_dataset import CHANNELS, MIF_SELECT

DATA_RANGE = 255.0  # 与论文实现惯例一致（log 域理论最大为 255*ln2≈176.8，见注释）


from eval.metrics import load_mif, align_pairs, per_channel_psnr_ssim


def log_normalize(x, q):
    """论文公式(2)。x: [C,H,W] float32 (0-255)，q: [C] float32。"""
    q = np.asarray(q, dtype=np.float32).reshape(-1, 1, 1)
    return 255.0 * np.log(np.minimum(x, q) / q + 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--data_root", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--q_file", default="/data/weiyh/orioncrc_cache/marker_q.json")
    ap.add_argument("--out", default="metrics_log.json")
    args = ap.parse_args()

    with open(args.q_file) as f:
        q = json.load(f)["q"]
    q = np.array(q, dtype=np.float32)
    print(f"[log] q_c(前景99.9分位): {q.round(1).tolist()}")

    preds, targs = align_pairs(args.pred_dir, args.target_dir, args.data_root, args.split)

    all_p, all_s, all_r = [], [], []
    for i, (p, t) in enumerate(zip(preds, targs)):
        pm, tm = load_mif(p), load_mif(t)
        # 目标与预测尺寸不一致时（如预测 512、GT 原生 333），逐通道缩放到预测尺寸再比
        if tm.shape[1] != pm.shape[1] or tm.shape[2] != pm.shape[2]:
            tm = np.stack([cv2.resize(tm[c], (pm.shape[2], pm.shape[1]),
                                      interpolation=cv2.INTER_AREA) for c in range(tm.shape[0])])
        # 论文 log 口径：预测与目标都做 log 变换后再算指标
        pl, tl = log_normalize(pm, q), log_normalize(tm, q)
        ps, ss, rs = per_channel_psnr_ssim(pl, tl)
        all_p.append(ps); all_s.append(ss); all_r.append(rs)
        if (i + 1) % 2000 == 0:
            print(f"  processed {i+1}/{len(preds)}", flush=True)

    P = np.nanmean(all_p, axis=0); S = np.nanmean(all_s, axis=0); R = np.nanmean(all_r, axis=0)
    res = {
        "n_tiles": len(preds),
        "log_norm_formula": "255*ln(min(I,q_c)/q_c+1)",
        "data_range": DATA_RANGE,
        "psnr_per_channel": {c: float(v) for c, v in zip(CHANNELS, P)},
        "ssim_per_channel": {c: float(v) for c, v in zip(CHANNELS, S)},
        "pearson_per_channel": {c: float(v) for c, v in zip(CHANNELS, R)},
        "psnr_mean": float(np.nanmean(P)),
        "ssim_mean": float(np.nanmean(S)),
        "pearson_mean": float(np.nanmean(R)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("psnr_mean", "ssim_mean", "pearson_mean")}, indent=2))
    print(f"已保存 -> {args.out}")


if __name__ == "__main__":
    main()
