"""一次性预处理缓存：把全部 mIF/H&E tile 解码为 256x256 memmap 连续文件。

背景：318k 个小 tiff/jpeg 的随机读取是训练瓶颈（~33 样本/s）。本脚本用
多进程并行把每个 split 的 tile 一次性解码成连续的内存映射数组，之后
DataLoader 按行索引顺序读 memmap，吞吐可提升一个量级。

产物（缓存目录按 tile_size 自动区分）：
  tile_size=256 -> /data/weiyh/orioncrc_cache/     （训练 256 版用）
  tile_size=512 -> /data/weiyh/orioncrc_cache_512/（v2 用；原始 tile 即 512，无需 resize）
  train_he.raw   train_mif.raw   形状 (N,3/16,tile,tile) uint8
  val_he.raw     val_mif.raw
  test_he.raw    test_mif.raw
  cache_meta.json（tile 数/通道信息） + 每个 split 的 {split}.done 完成标记

用法：
    python scripts/build_cache.py --data_root ... --tile_size 256 --workers 64
    python scripts/build_cache.py --data_root ... --tile_size 512 --workers 64  # v2 前置步骤
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datacore.cache import open_cache
import json
import time
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool

import tifffile
import cv2

SEL = [i for i in range(17) if i != 13]  # 17 -> 16，剔除 PD-1


def resolve(root, p):
    return p if os.path.isabs(p) else os.path.join(root, p)


def process_one(args):
    """读取一个 tile 的 H&E(3ch) 与 mIF(16ch)，缩放为 256，返回两个 uint8 数组。"""
    root, he_path, mif_path, tile_size = args
    he = cv2.imread(resolve(root, he_path), cv2.IMREAD_COLOR)
    he = cv2.cvtColor(he, cv2.COLOR_BGR2RGB)
    if he.shape[:2] != (tile_size, tile_size):
        he = cv2.resize(he, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    he = np.ascontiguousarray(he.transpose(2, 0, 1))          # (3,H,W) uint8

    a = tifffile.imread(resolve(root, mif_path), maxworkers=8)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim == 3 and a.shape[2] not in (16, 17):
        a = a.transpose(1, 2, 0)
    if a.shape[2] == 17:
        a = a[:, :, SEL]
    if a.shape[0] != tile_size or a.shape[1] != tile_size:
        out = np.empty((tile_size, tile_size, a.shape[2]), dtype=np.uint8)
        for c in range(a.shape[2]):
            out[:, :, c] = cv2.resize(a[:, :, c], (tile_size, tile_size),
                                      interpolation=cv2.INTER_AREA)
        a = out
    mif = np.ascontiguousarray(a.transpose(2, 0, 1))          # (16,H,W) uint8
    return he, mif


# 每个 worker 通过 initializer 拿到共享的 dataframe（只读，避免逐任务传大对象）
_G = {}


def _init_worker(df, he_col, mif_col, root, tile_size):
    _G["df"] = df
    _G["he_col"] = he_col
    _G["mif_col"] = mif_col
    _G["root"] = root
    _G["tile_size"] = tile_size


def process_chunk(he_path, mif_path, idxs):
    """直接写自己负责的 memmap 区间，不走结果队列。返回处理的 tile 数。"""
    df, he_col, mif_col, root, ts = _G["df"], _G["he_col"], _G["mif_col"], _G["root"], _G["tile_size"]
    he_mm = np.lib.format.open_memmap(he_path, mode="r+", dtype=np.uint8)
    mif_mm = np.lib.format.open_memmap(mif_path, mode="r+", dtype=np.uint8)
    n = 0
    for i in idxs:
        r = df.iloc[i]
        he, mif = process_one((root, r[he_col], r[mif_col], ts))
        he_mm[i] = he
        mif_mm[i] = mif
        n += 1
    he_mm.flush(); mif_mm.flush()
    return n


def _chunk_wrapper(task):
    """模块级包装，供 Pool 序列化（lambda 不可 pickle）。"""
    return process_chunk(*task)


def build_split(root, split, cache_dir, tile_size, workers):
    df = pd.read_csv(os.path.join(root, f"{split}_dataframe.csv"))
    he_col = "image_path" if "image_path" in df.columns else "he_path"
    mif_col = "target_path" if "target_path" in df.columns else "if_path"
    N = len(df)
    he_path = os.path.join(cache_dir, f"{split}_he.raw")
    mif_path = os.path.join(cache_dir, f"{split}_mif.raw")
    if N == 0 or tile_size <= 0 or workers <= 0:
        raise ValueError("Cache requires nonempty data and positive tile_size/workers")
    done_marker = os.path.join(cache_dir, f"{split}.done")
    if os.path.exists(done_marker):
        try:
            open_cache(he_path, 3, tile_size, N)
            open_cache(mif_path, 16, tile_size, N)
        except (OSError, ValueError):
            pass
        else:
            print(f"[cache] {split} 已存在且完整，跳过 ({N} tiles)")
            return N
        os.remove(done_marker)

    # 预分配精确大小并 flush，确保文件尺寸固定
    he_mm = np.lib.format.open_memmap(he_path, mode="w+", dtype=np.uint8,
                                      shape=(N, 3, tile_size, tile_size))
    mif_mm = np.lib.format.open_memmap(mif_path, mode="w+", dtype=np.uint8,
                                       shape=(N, 16, tile_size, tile_size))
    he_mm.flush(); mif_mm.flush()
    del he_mm, mif_mm

    chunk = max(1, N // (workers * 4))   # 任务块大小：让每个 worker 约分到 4 个块，负载均衡
    tasks = [(he_path, mif_path, list(range(s, min(s + chunk, N))))
             for s in range(0, N, chunk)]
    t0 = time.time()
    done = 0
    with Pool(workers, initializer=_init_worker,
              initargs=(df, he_col, mif_col, root, tile_size)) as pool:
        for n in pool.imap_unordered(_chunk_wrapper, tasks):
            done += n
            if done % 50000 < chunk or done >= N:
                el = time.time() - t0
                print(f"[cache] {split} {done}/{N}  {el:.0f}s  {done/el:.0f} tiles/s  "
                      f"剩余约 {(N-done)/(done/el)/60:.1f}min", flush=True)
    with open(done_marker, "w") as f:
        f.write(str(N))
    print(f"[cache] {split} 完成: {N} tiles, 用时 {(time.time()-t0)/60:.1f} min")
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--tile_size", type=int, default=256, help="缓存分辨率：256/512（512 供 v2）")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--splits", "--split", default="train,val,test")
    args = ap.parse_args()
    # 缓存目录按 tile_size 命名：256→orioncrc_cache；512→orioncrc_cache_512
    args.cache_dir = args.cache_dir or ("/data/weiyh/orioncrc_cache" + (f"_{args.tile_size}" if args.tile_size != 256 else ""))
    os.makedirs(args.cache_dir, exist_ok=True)
    counts = {}
    for s in args.splits.split(","):
        counts[s] = build_split(args.data_root, s, args.cache_dir, args.tile_size, args.workers)
    meta = {"tile_size": args.tile_size, "channels_mif": 16, "counts": counts}
    with open(os.path.join(args.cache_dir, "cache_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("缓存构建完成:", meta)


if __name__ == "__main__":
    main()
