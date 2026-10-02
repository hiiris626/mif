"""统一评估脚本：PSNR / SSIM / FID（+ Pearson 参考）。

口径（对齐 MIPHEI-ViT 论文 §5.2）：
  - 逐通道 PSNR、SSIM，跨通道取平均
  - FID 用 RGB 复合图（16 通道 mIF 按标准配色合成），基于 clean-fid (legacy_pytorch)
  - 另附逐通道 Pearson 相关系数（稀疏 mIF 下更稳健，论文推荐）

口径说明（重要）：
  - 本脚本按“逐 tile 计算 → 跨 tile 平均”聚合（当时口径，v1/v2/v3 各自评估都用它）；
  - 2026-09-20 七模型统一重评改用“全数据全局 MSE 聚合”（避免稀疏通道逐图
    PSNR 出现极端值），同一模型的 PSNR 数值会与本脚本略有差异，引用时须注明口径。

用法：
    python metrics.py --pred_dir /data/weiyh/results/virchow2/pred \
                      --target_dir /data/weiyh/orioncrc_miphei/test \
                      --composite_fid --out results_metrics.json
"""
import os
import json
import glob
import argparse
import numpy as np
import torch
import tifffile
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# 16 通道定义与选择（与 datacore 保持一致，单一来源）
from datacore.orioncrc_dataset import CHANNELS, MIF_SELECT
NUM_CHANNELS = len(CHANNELS)

# RGB 复合标准配色（R,G,B 各自的通道权重）：Hoechst=蓝, Pan-CK=红, SMA=绿, CD3e=黄...
RGB_COMPOSITE_MIX = {
    "R": ["Pan-CK", "ECadherin", "CD3e"],   # 红：上皮 + T 细胞
    "G": ["SMA", "CD68", "CD163"],          # 绿：基质 + 巨噬
    "B": ["Hoechst", "CD20"],                # 蓝：核 + B 细胞
}


def load_mif(path):
    """读取 mIF TIFF（17 或 16 通道）-> [16, H, W] float32 (0-255)，剔除 PD-1。"""
    a = tifffile.imread(path, maxworkers=8)
    if a.ndim == 2:  # 单通道
        a = a[None]
    if a.ndim == 3 and a.shape[0] not in (NUM_CHANNELS, len(MIF_SELECT) + 1):  # 转成 C,H,W
        a = a.transpose(2, 0, 1)
    a = a.astype(np.float32)
    if a.shape[0] == len(MIF_SELECT) + 1:  # 17 -> 16
        a = a[MIF_SELECT]
    if a.shape[0] != NUM_CHANNELS:
        raise ValueError(f"{path}: 通道数 {a.shape[0]} != {NUM_CHANNELS}")
    return a

def resize_mif(a, h, w):
    """逐通道 cv2 缩放到 (h, w)，与预测对齐。a: [C,H,W] float32。"""
    if a.shape[1] == h and a.shape[2] == w:
        return a
    C = a.shape[0]
    out = np.empty((C, h, w), dtype=np.float32)
    for c in range(C):
        out[c] = cv2.resize(a[c], (w, h), interpolation=cv2.INTER_AREA)
    return out

def rgb_composite(mif):
    """16 通道 -> RGB 复合图 [H,W,3] (0-255)。"""
    C = len(CHANNELS)
    rgb = np.zeros((mif.shape[1], mif.shape[2], 3), dtype=np.float32)
    idx = {c: i for i, c in enumerate(CHANNELS)}
    for k, chans in RGB_COMPOSITE_MIX.items():
        j = "RGB".index(k)
        for c in chans:
            if c in idx:
                rgb[..., j] += mif[idx[c]]
    return np.clip(rgb, 0, 255)


def per_channel_psnr_ssim(pred, target):
    """pred/target: [16,H,W] float32(0-255) -> (psnr_list, ssim_list, pearson_list)。"""
    psnr_l, ssim_l, pear_l = [], [], []
    for c in range(pred.shape[0]):
        p, t = pred[c], target[c]
        mse = np.mean((t.astype(np.float64) - p) ** 2)
        psnr_l.append(float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12))))
        ssim_l.append(structural_similarity(t, p, data_range=255.0))
        pear_l.append(np.nan if p.max() == p.min() or t.max() == t.min()
                      else np.corrcoef(p.ravel(), t.ravel())[0, 1])
    return psnr_l, ssim_l, pear_l


def align_pairs(pred_dir, target_dir, data_root="", split="test"):
    """对齐预测与目标，返回按同序配对的路径列表。

    - 给了 data_root：按 {split}_dataframe.csv 的 H&E 文件名对齐（推荐，防漏/防重）；
    - 否则：pred_dir 与 target_dir 内按 basename 直接配对，并做重复/缺失校验。
    """
    preds = sorted(glob.glob(os.path.join(pred_dir, "*.tif*")))
    def key(path):
        return os.path.splitext(os.path.basename(path))[0]
    if data_root:
        import pandas as pd
        df = pd.read_csv(os.path.join(data_root, f"{split}_dataframe.csv"))
        hc = "image_path" if "image_path" in df.columns else "he_path"
        tc = "target_path" if "target_path" in df.columns else "if_path"
        keys = [key(p) for p in df[hc]]
        targets = [os.path.join(data_root, p) for p in df[tc]]
    else:
        targets = sorted(glob.glob(os.path.join(target_dir, "*.tif*")))
        keys = [key(p) for p in targets]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate target/sample basenames; pairing is ambiguous")
    mapping = dict(zip(keys, targets))
    if len({key(p) for p in preds}) != len(preds):
        raise ValueError("Duplicate prediction basenames")
    missing = [p for p in preds if key(p) not in mapping]
    if missing:
        raise ValueError(f"Predictions without paired targets: {missing[:3]}")
    if data_root:
        absent = set(keys) - {key(p) for p in preds}
        if absent:
            raise ValueError(f"Missing predictions for {len(absent)} split samples: {sorted(absent)[:3]}")
    if not preds:
        raise ValueError("No prediction TIFFs found")
    return preds, [mapping[key(p)] for p in preds]


def compute_fid_composite(pred_dir, target_dir, device="cuda", pairs=None):
    """在“实际参与评估的那批配对”上计算 RGB 复合图 FID（clean-fid legacy_pytorch）。

    预测与目标各自先合成 RGB 图写临时目录，再调 cleanfid 计算（两边严格同配对）。
    """
    import tempfile
    from PIL import Image
    from cleanfid import fid as cleanfid
    preds, targets = pairs or align_pairs(pred_dir, target_dir)
    with tempfile.TemporaryDirectory(prefix="mif_fid_") as tmp:
        pp, tp = os.path.join(tmp, "pred"), os.path.join(tmp, "target")
        os.makedirs(pp); os.makedirs(tp)
        for i, (p, t) in enumerate(zip(preds, targets)):
            pm = load_mif(p)
            tm = resize_mif(load_mif(t), *pm.shape[1:])
            Image.fromarray(rgb_composite(pm).astype(np.uint8)).save(os.path.join(pp, f"{i}.png"))
            Image.fromarray(rgb_composite(tm).astype(np.uint8)).save(os.path.join(tp, f"{i}.png"))
        return float(cleanfid.compute_fid(pp, tp, mode="legacy_pytorch", device=torch.device(device)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="预测 mIF 输出目录（16通道 tiff）")
    ap.add_argument("--target_dir", required=True, help="目标 mIF 目录")
    ap.add_argument("--data_root", default="", help="OrionCRC 数据根；提供后按 {split}_dataframe.csv 对齐 pred(target H&E名) 与 target")
    ap.add_argument("--split", default="test")
    ap.add_argument("--composite_fid", action="store_true", help="额外计算 RGB 复合图 FID")
    ap.add_argument("--out", default="metrics_result.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    preds, targs = align_pairs(args.pred_dir, args.target_dir, args.data_root, args.split)

    all_p, all_s, all_r = [], [], []
    for i, (p, t) in enumerate(zip(preds, targs)):
        pm, tm = load_mif(p), load_mif(t)
        tm = resize_mif(tm, pm.shape[1], pm.shape[2])  # 目标缩放到预测尺寸
        ps, ss, rs = per_channel_psnr_ssim(pm, tm)
        all_p.append(ps); all_s.append(ss); all_r.append(rs)
        if (i + 1) % 2000 == 0:
            print(f"  processed {i+1}/{len(preds)}", flush=True)

    # 聚合口径：先每张 tile 的逐通道指标 → 再跨 tile 求均值（nan 安全）
    P = np.nanmean(all_p, axis=0); S = np.nanmean(all_s, axis=0); R = np.nanmean(all_r, axis=0)
    res = {
        "n_tiles": len(preds),
        "psnr_per_channel": {c: float(v) for c, v in zip(CHANNELS, P)},
        "ssim_per_channel": {c: float(v) for c, v in zip(CHANNELS, S)},
        "pearson_per_channel": {c: float(v) for c, v in zip(CHANNELS, R)},
        "psnr_mean": float(np.nanmean(P)),
        "ssim_mean": float(np.nanmean(S)),
        "pearson_mean": float(np.nanmean(R)),
    }
    if args.composite_fid:
        res["fid_rgb_composite"] = compute_fid_composite(args.pred_dir, args.target_dir, args.device, (preds, targs))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"已保存 -> {args.out}")


if __name__ == "__main__":
    main()
