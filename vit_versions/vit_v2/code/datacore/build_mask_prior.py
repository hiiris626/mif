"""M0′：从 nuclei 实例 + csv_nuclei_pos 表型构建 16 通道分割监督真值（软先验 / 硬掩膜）。

口径（PLAN_S1_unetpp_seghead.md §3.1，与 eval/cell_classify.py 同源）：
  channel 0            Hoechst = nuclei union 二值（不做 σ 外扩）
  channel 1..15        15 个 marker（CSV 的 _pos 列去掉 PD-1），顺序与 datacore.CHANNELS 一致

软先验：阳性核内 = 1；核外 0 < d ≤ κσ 用 exp(-d²/2σ²) 衰减；其余 = 0
硬掩膜：阳性核内 = 1，其余 = 0

CSV 列顺序（实测）：label,x,y,CD31_pos,CD45_pos,CD68_pos,CD4_pos,FOXP3_pos,CD8a_pos,
                      CD45RO_pos,CD20_pos,PD-L1_pos,CD3e_pos,CD163_pos,E-cadherin_pos,
                      PD-1_pos,Ki67_pos,Pan-CK_pos,SMA_pos
→ 去掉 PD-1 后与 mIF 通道 1..15 一一对应。

用法：
    python -m datacore.build_mask_prior --split train --tile_size 256 \
        --out /data1/weiyh/orioncrc_cache_mask256 --workers 16
    python -m datacore.build_mask_prior --split test --tile_size 256 --limit 200 ...   # 冒烟
"""
import os
import sys
import json
import time
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
import cv2
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datacore.orioncrc_dataset import CHANNELS, MIF_SELECT  # noqa: E402

ROOT = "/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x"
CSV_DIR = "csv_nuclei_pos"

# CSV 的 16 个 marker 列（含 PD-1），顺序即 mIF 通道 1..15 的来源
CSV_MARKERS = ["CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO", "CD20",
               "PD-L1", "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67", "Pan-CK", "SMA"]
DROP = {"PD-1"}                                     # 与 MIF_SELECT 一致，剔除
KEEP = [m for m in CSV_MARKERS if m not in DROP]    # 15 个，顺序 = CHANNELS[1:]
assert len(KEEP) == 15 and list(CHANNELS[1:])[:3] == ["CD31", "CD45", "CD68"]

# 全局（fork 后子进程共享）
_G = {}


def slide_prefix(path):
    """tile 路径 -> slide 前缀（去掉最后 5 段坐标）。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.rsplit("_", 5)[0]


def build_luts(root, if_path, cache=None):
    """-> lut (16,max_label+1) uint8；lut[c,label]=1 表示该核在 c 通道阳性。

    channel 0 = Hoechst（所有核都阳性）；channel 1..15 = KEEP 顺序的 marker。
    slide 级 CSV 有百万行，必须按 slide 缓存（cache 为 per-worker dict）。
    """
    pref = slide_prefix(if_path)
    if cache is not None and pref in cache:
        return cache[pref]
    csv_path = os.path.join(root, CSV_DIR, pref + ".csv")
    if not os.path.exists(csv_path):
        if cache is not None:
            cache[pref] = None
        return None
    df = pd.read_csv(csv_path, usecols=["label"] + [f"{m}_pos" for m in KEEP])
    labels = df["label"].to_numpy(np.int64)
    pos = df[[f"{m}_pos" for m in KEEP]].to_numpy(bool)            # (N,15)
    max_label = int(labels.max())
    lut = np.zeros((16, max_label + 1), dtype=np.uint8)
    lut[0, labels] = 1                                             # Hoechst
    lut[1:, labels] = pos.T.astype(np.uint8)
    if cache is not None:
        cache[pref] = lut
    return lut


def soft_prior(mask_bin, sigma, kappa):
    """二值核掩膜 -> exp(-d²/2σ²) 软先验（仅 κσ 范围内非零）。σ=0 直接返回二值。"""
    if sigma <= 0:
        return mask_bin.astype(np.float32)
    d = cv2.distanceTransform(1 - mask_bin.astype(np.uint8), cv2.DIST_L2, 3)
    d = d.astype(np.float32)
    out = np.exp(-np.square(d) / np.float32(2 * sigma ** 2)).astype(np.float32)
    out[d > kappa * sigma] = 0.0
    out[mask_bin > 0] = 1.0
    return out


def process_one(i):
    """处理第 i 个 tile，直接写入继承的 memmap。"""
    row = _G["df"].iloc[i]
    nuc_path = os.path.join(_G["root"], row["nuclei_path"])
    if_path = row["target_path"]
    lut = build_luts(_G["root"], if_path, _G["lut_cache"])
    T = _G["tile_size"]
    if lut is None:
        return i, "no_csv"
    try:
        n = tifffile.imread(nuc_path).astype(np.int64)
        if n.ndim == 3:
            n = n[..., 0]
        if n.shape[0] != T or n.shape[1] != T:
            n = cv2.resize(n.astype(np.float32), (T, T), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        if np.any(n < 0) or np.any(n >= lut.shape[1]):
            raise ValueError("Nuclei label missing from phenotype LUT")
        m = lut[:, n]                                              # (16,T,T) uint8
        if _G["soft"]:
            for c in range(1, 16):
                m[c] = (soft_prior(m[c], _G["sigma"], _G["kappa"]) * 255).astype(np.uint8)
            m[0] = (m[0] > 0).astype(np.uint8) * 255
        else:
            # 硬掩膜：所有通道统一缩放到 0/255，否则训练 target≈0 等于全背景
            m = (m > 0).astype(np.uint8) * 255
        _G["mm"][i] = m
    except Exception as e:                                          # noqa: BLE001
        return i, f"err:{type(e).__name__}:{e}"
    return i, "ok"


def _init(mm, df, root, tile_size, sigma, kappa, soft, scale255):
    _G.update(mm=mm, df=df, root=root, tile_size=tile_size, sigma=sigma,
              kappa=kappa, soft=soft, scale255=scale255, lut_cache={})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--split", default="train")
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=3.0, help="σ(px)，0=硬掩膜")
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--hard", action="store_true", help="输出硬 0/1 而非 σ 软先验")
    ap.add_argument("--scale255", action="store_true", default=True)
    ap.add_argument("--no_scale255", dest="scale255", action="store_false")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--df", "--index_csv", default="", help="自定义划分表（如 D5k 子集）")
    ap.add_argument("--name", default="", help="缓存文件名前缀（默认=split）")
    args = ap.parse_args()

    if not args.scale255:
        ap.error("Masks must use 0..255; --no_scale255 is unsupported")
    os.makedirs(args.out, exist_ok=True)
    df_path = args.df or os.path.join(args.root, f"{args.split}_dataframe.csv")
    df = pd.read_csv(df_path)
    if args.limit:
        df = df.iloc[:args.limit].copy()
    n = len(df)
    name = args.name or args.split
    raw = os.path.join(args.out, f"{name}_mask.raw")
    done = os.path.join(args.out, f"{name}.done")
    if os.path.exists(done):
        os.remove(done)

    mm = np.memmap(raw, dtype=np.uint8, mode="w+", shape=(n, 16, args.tile_size, args.tile_size))
    soft = not args.hard
    print(f"[M0′] split={name} tiles={n} tile={args.tile_size} σ={args.sigma if soft else 'hard'} "
          f"workers={args.workers} -> {raw}  ({n*16*args.tile_size**2/2**30:.1f} GB)", flush=True)

    t0 = time.time()
    bad = {"no_csv": 0, "err": 0}
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(mm, df, args.root, args.tile_size, args.sigma, args.kappa,
                           soft, args.scale255)) as pool:
        for k, (i, status) in enumerate(pool.imap_unordered(process_one, range(n), chunksize=64)):
            if status == "no_csv":
                bad["no_csv"] += 1
            elif status.startswith("err"):
                bad["err"] += 1
                if bad["err"] <= 5:
                    print(f"  [warn] tile {i}: {status}", flush=True)
            if (k + 1) % 5000 == 0 or k + 1 == n:
                el = time.time() - t0
                print(f"  {k+1}/{n}  {el:.0f}s  {k+1/max(el,1e-9):.0f} tile/s  "
                      f"eta {(el/(k+1)*(n-k-1))/60:.1f} min  bad={bad}", flush=True)
    mm.flush()
    del mm
    meta = dict(split=name, n=n, tile_size=args.tile_size, sigma=args.sigma,
                kappa=args.kappa, soft=soft, scale255=args.scale255,
                channels=CHANNELS, csv_markers=CSV_MARKERS, dropped=list(DROP),
                df=os.path.basename(df_path), bad=bad, elapsed_sec=time.time() - t0)
    mp_ = os.path.join(args.out, "mask_meta.json")
    old = json.load(open(mp_)) if os.path.exists(mp_) else {}
    old[name] = meta
    json.dump(old, open(mp_, "w"), indent=2, ensure_ascii=False)
    if any(bad.values()):
        raise RuntimeError(f"Mask cache incomplete: {bad}; completion marker not written")
    with open(done, "w") as f:
        f.write(f"ok {n}\n")
    print(f"[done] {name}: {n} tiles / {(time.time()-t0)/60:.1f} min -> {raw}")


if __name__ == "__main__":
    main()
