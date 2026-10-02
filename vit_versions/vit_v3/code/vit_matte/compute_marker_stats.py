"""预算训练集 mIF 各通道标准差，用于加权 MSE（MIPHEI §4.2 损失）。

统计口径：随机抽 max_tiles 张训练 tile，把所有像素汇总后按通道计算全局 std；
训练侧用 1/σ（再归一化到均值 1）作为逐通道损失权重（仅 v1/v2 线性域使用；
v3 的 log 域改用均匀权重 + 正样本加权）。

输出 <data_root 父目录>/marker_std.json：
    {"std": [...16 个值...], "n_tiles": N}

用法（数据就绪后运行）：
    python -m vit_matte.compute_marker_stats --data_root /data/weiyh/orioncrc_miphei/...
"""
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from datacore.orioncrc_dataset import OrionCRCDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x")
    ap.add_argument("--max_tiles", type=int, default=20000, help="抽样数量，足够估算 std")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.max_tiles < 1:
        ap.error("max_tiles must be positive")

    ds = OrionCRCDataset("train", root=args.data_root, aug=False, tile_size=args.tile_size)
    n = min(len(ds), args.max_tiles)
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        num_workers=args.num_workers, sampler=torch.utils.data.SubsetRandomSampler(np.random.default_rng(args.seed).choice(len(ds), n, replace=False).tolist()))
    # 增量统计一阶/二阶矩（所有采样像素汇总后按通道计算 std）
    sums = np.zeros(16)
    sqs = np.zeros(16)
    cnt = 0
    for he, mif in loader:
        m = mif.numpy().astype(np.float64)  # 线性目标域，已在 [-0.9,0.9]
        sums += m.sum(axis=(0, 2, 3))
        sqs += (m ** 2).sum(axis=(0, 2, 3))
        cnt += m.shape[0] * m.shape[2] * m.shape[3]
    mean = sums / cnt
    var = np.clip(sqs / cnt - mean ** 2, 0, None)   # std² = E[x²] - E[x]²（夹掉数值负值）
    std = np.sqrt(var)
    out = {"std": std.tolist(), "n_tiles": n, "mean": mean.tolist(), "seed": args.seed, "tile_size": args.tile_size}
    dst = os.path.join(os.path.dirname(args.data_root.rstrip("/")), "marker_std.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print("std:", np.round(std, 4))
    print(f"已保存 -> {dst}")


if __name__ == "__main__":
    main()
