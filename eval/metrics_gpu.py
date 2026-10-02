"""GPU 加速评估：PSNR / SSIM / Pearson（torch 批量，GPU），FID（clean-fid GPU）。

目标 mIF 直接从 memmap 缓存（test_mif.raw）读（已 16 通道 256x256），
预测读 tiff。比 CPU 版（skimage 单图）快约两个数量级。

用法：
    python -m eval.metrics_gpu --pred_dir /data/weiyh/results/xxx \
        --data_root /data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x \
        --split test --out metrics.json [--fid] [--gpu 0]
"""
import os
import math
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import tifffile
import pandas as pd

from datacore.orioncrc_dataset import CHANNELS
from datacore.cache import open_cache
from eval.metrics import load_mif, compute_fid_composite

CACHE_DIR = "/data/weiyh/orioncrc_cache"


def gaussian_1d(window_size, sigma):
    g = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
                      for x in range(window_size)], dtype=torch.float32)
    return g / g.sum()


def ssim_batch(img1, img2, window_size=7, data_range=255.0):
    """Match skimage's default SSIM: uniform 7x7, sample covariance, valid area."""
    if img1.shape != img2.shape or min(img1.shape[-2:]) < window_size:
        raise ValueError("SSIM requires matching images at least window_size pixels wide")
    C = img1.shape[1]
    window = torch.full((C, 1, window_size, window_size), 1.0 / window_size**2,
                        dtype=img1.dtype, device=img1.device)
    def mean(x):
        return F.conv2d(x, window, groups=C)
    mu1, mu2 = mean(img1), mean(img2)
    factor = window_size**2 / (window_size**2 - 1)
    s1 = factor * (mean(img1**2) - mu1**2)
    s2 = factor * (mean(img2**2) - mu2**2)
    s12 = factor * (mean(img1 * img2) - mu1 * mu2)
    c1, c2 = (0.01 * data_range)**2, (0.03 * data_range)**2
    score = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (s1 + s2 + c2))
    return score.mean(dim=(2, 3))


def psnr_batch(img1, img2, data_range=255.0):
    """逐通道 PSNR。"""
    mse = ((img1 - img2) ** 2).mean(dim=(2, 3))  # [B,C]
    return 10 * torch.log10(data_range ** 2 / mse)


def pearson_batch(img1, img2):
    """逐通道 Pearson 相关系数。"""
    B, C = img1.shape[:2]
    x = img1.reshape(B, C, -1)
    y = img2.reshape(B, C, -1)
    x = x - x.mean(dim=2, keepdim=True)
    y = y - y.mean(dim=2, keepdim=True)
    num = (x * y).sum(dim=2)
    den = torch.sqrt((x * x).sum(dim=2) * (y * y).sum(dim=2))
    return num / den  # [B,C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--data_root", default="/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="metrics_gpu.json")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--fid", action="store_true", help="额外算 RGB 复合 FID")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--domain", choices=("linear", "log"), default="linear")
    ap.add_argument("--only_psnr", action="store_true",
                    help="只重算全局 MSE PSNR，并保留现有 JSON 中的其他指标")
    ap.add_argument("--q_file", default="/data/weiyh/orioncrc_cache/marker_q.json")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    q = None
    if args.domain == "log":
        with open(args.q_file) as stream:
            q_values = json.load(stream)["q"]
        q = torch.tensor(q_values, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    # 对齐：he basename -> 行号（memmap 按行号索引）
    df = pd.read_csv(os.path.join(args.data_root, f"{args.split}_dataframe.csv"))
    he_col = "image_path" if "image_path" in df.columns else "he_path"
    idx_of = {os.path.splitext(os.path.basename(p))[0]: i for i, p in enumerate(df[he_col])}

    # 按 memmap 行号排序，保证 mif_mm 顺序读（避免大文件随机跳读）
    preds = sorted(glob.glob(os.path.join(args.pred_dir, "*.tif*")),
                   key=lambda p: idx_of[os.path.splitext(os.path.basename(p))[0]])
    rows = [idx_of[os.path.splitext(os.path.basename(p))[0]] for p in preds]
    N = len(preds)
    if N == 0:
        raise ValueError("No prediction TIFFs found")
    print(f"对齐 {N} 对预测/目标")

    # 使用与预测分辨率一致的目标缓存。512 模型不能用 256 缓存上采样代替，
    # 因为原始 target 并非规则的 512 图（部分为 333px），两次 resize 会改变指标。
    first_shape = load_mif(preds[0]).shape[-2:]
    cache_size = first_shape[0] if first_shape[0] == first_shape[1] else 256
    cache_dir = CACHE_DIR if cache_size == 256 else f"{CACHE_DIR}_{cache_size}"
    mif_path = os.path.join(cache_dir, f"{args.split}_mif.raw")
    if not os.path.exists(mif_path):
        cache_size, cache_dir = 256, CACHE_DIR
        mif_path = os.path.join(cache_dir, f"{args.split}_mif.raw")
    if not os.path.exists(mif_path):
        raise FileNotFoundError(f"缺少缓存 {mif_path}（先跑 build_cache.py）")
    mif_mm = open_cache(mif_path, 16, cache_size, len(df))
    print(f"目标缓存: {cache_size}x{cache_size} ({mif_path})")

    C = len(CHANNELS)
    sum_squared_error = torch.zeros(C, device=device, dtype=torch.float64)
    pixel_count = 0
    sum_ssim = torch.zeros(C, device=device)
    sum_pear = torch.zeros(C, device=device)
    count_pear = torch.zeros(C, device=device)

    def load_pred_batch(ps):
        arrs = [load_mif(p) for p in ps]  # 不用 maxworkers，避免线程池累积卡死
        return torch.from_numpy(np.stack(arrs).astype(np.float32)).to(device)

    for s in range(0, N, args.batch):
        e = min(s + args.batch, N)
        pred = load_pred_batch(preds[s:e])                     # [B,16,H,W]
        tgt = torch.from_numpy(mif_mm[rows[s:e]].astype(np.float32)).to(device)  # rows 已排序，顺序读
        if tgt.shape[-2:] != pred.shape[-2:]:
            tgt = F.interpolate(tgt, size=pred.shape[-2:], mode="area")
        if q is not None:
            pred = 255.0 * torch.log(torch.minimum(pred, q) / q + 1.0)
            tgt = 255.0 * torch.log(torch.minimum(tgt, q) / q + 1.0)
        sum_squared_error += ((pred - tgt).double() ** 2).sum(dim=(0, 2, 3))
        pixel_count += pred.shape[0] * pred.shape[2] * pred.shape[3]
        if not args.only_psnr:
            sum_ssim += ssim_batch(pred, tgt).sum(0)
            pear = pearson_batch(pred, tgt)
            sum_pear += torch.nan_to_num(pear).sum(0)
            count_pear += torch.isfinite(pear).sum(0)

        if (e) % (args.batch * 10) == 0 or e == N:
            print(f"  {e}/{N}", flush=True)

    mse = sum_squared_error / pixel_count
    psnr = (10 * torch.log10(255.0 ** 2 / mse)).cpu().numpy()
    res = {}
    if args.only_psnr:
        if not os.path.exists(args.out):
            raise FileNotFoundError(f"--only_psnr 需要已有结果文件: {args.out}")
        with open(args.out) as stream:
            res = json.load(stream)
    else:
        ssim = (sum_ssim / N).cpu().numpy()
        pear = (sum_pear / count_pear).cpu().numpy()
        res.update({
            "ssim_per_channel": {c: float(v) for c, v in zip(CHANNELS, ssim)},
            "pearson_per_channel": {c: float(v) for c, v in zip(CHANNELS, pear)},
            "ssim_mean": float(ssim.mean()),
            "pearson_mean": float(np.nanmean(pear)),
        })
    res.update({
        "n_tiles": N,
        "domain": args.domain,
        "target_cache_size": cache_size,
        "psnr_aggregation": "global_channel_mse",
        "psnr_per_channel": {c: float(v) for c, v in zip(CHANNELS, psnr)},
        "psnr_mean": float(psnr.mean()),
    })
    if args.domain == "log":
        res["log_norm_formula"] = "255*ln(min(I,q_c)/q_c+1)"

    if args.fid:
        tc = "target_path" if "target_path" in df.columns else "if_path"
        targets = [os.path.join(args.data_root, df.iloc[r][tc]) for r in rows]
        res["fid_rgb_composite"] = compute_fid_composite(
            args.pred_dir, "", device, (preds, targets))
    elif os.path.exists(args.out):
        # Allow a cheap metric-only correction without discarding an already
        # computed FID value.
        try:
            with open(args.out) as stream:
                previous = json.load(stream)
            if "fid_rgb_composite" in previous:
                res["fid_rgb_composite"] = previous["fid_rgb_composite"]
        except (OSError, ValueError):
            pass

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"已保存 -> {args.out}")


if __name__ == "__main__":
    main()
