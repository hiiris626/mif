"""虚拟免疫染色可视化：patch 级 + WSI 级多色荧光对比。

对每个模型（vit512/pix2pixHD/pix2pix）生成：
  [H&E 原图] [GT 真实 mIF 多色荧光] [模型预测多色荧光]

- Patch 级：单 tile 高清对比
- WSI 级：同一 slide 相邻 tiles 拼接成全景对比
每个 marker 用独立颜色标记免疫细胞（多色荧光）。

用法：
    python scripts/visualize_virtual_stain.py --tile 18459_..._0_27648_0_512_512 \
        --out /data/weiyh/results/visualization --wsi

说明：默认模型清单见下方 MODELS（vit512=v1、vit256v3=v3）；可用 --model NAME=DIR
重复传入来覆盖/追加模型。输出：patch_<tile>.png / wsi_<slide>.png。
"""
import os
import sys
import json
import argparse
import numpy as np
import tifffile
import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datacore.orioncrc_dataset import CHANNELS
from eval.metrics import load_mif

DATA_ROOT = "/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x"
# 默认预测目录：vit512=v1（iter140000）、vit256v3=v3（best）、再加 CNN/GAN/扩散基线
MODELS = {
    "vit512": "/data/weiyh/results/vit_matte/virchow2_vitmatte_512_iter140000",
    "vit256v3": "/data/weiyh/results/vit_matte/virchow2_vitmatte_v3_256_best",
    "pix2pixhd": "/data/weiyh/results/pix2pixhd/pix2pixhd_16ch_in_epoch20",
    "pix2pix": "/data/weiyh/results/pix2pix/pix2pix_16ch_best",
    "i2sb": "/data/weiyh/results/i2sb/i2sb_virtual_stain",
}
# 各模型预测尺寸（vit=512，pix2pixHD/pix2pix/I2SB=256）
MODEL_SIZE = {"vit512": 512, "vit256v3": 256, "pix2pixhd": 256, "pix2pix": 256, "i2sb": 256}

# 多色荧光配色：marker -> RGB（标记各类免疫细胞）
MARKER_COLORS = {
    "Hoechst": (0, 0, 255),       # 蓝：细胞核
    "CD4": (0, 255, 255),         # 青：CD4+ T 辅助
    "CD8a": (255, 255, 0),        # 黄：CD8+ T 杀伤
    "FOXP3": (255, 0, 255),       # 洋红：Treg
    "CD20": (255, 0, 0),          # 红：B 细胞
    "CD68": (0, 255, 0),          # 绿：巨噬细胞
    "CD163": (0, 200, 100),       # 深绿：M2 巨噬
    "Pan-CK": (255, 120, 180),    # 粉：肿瘤/上皮
    "SMA": (255, 165, 0),         # 橙：基质
    "CD3e": (0, 128, 255),        # 天蓝：泛 T 细胞
}

# 信号增益（稀疏 mIF 增强，让阳性信号可见）
GAIN = {"Hoechst": 1.2, "CD4": 4.0, "CD8a": 5.0, "FOXP3": 8.0, "CD20": 5.0,
        "CD68": 4.0, "CD163": 5.0, "Pan-CK": 3.0, "SMA": 3.0, "CD3e": 4.0}


def load_he(path):
    b = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(b, cv2.COLOR_BGR2RGB)


def multicolor_composite(mif):
    """mif: [C,H,W] float(0-255) -> RGB 多色荧光 [H,W,3]。"""
    H, W = mif.shape[1], mif.shape[2]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    idx = {c: i for i, c in enumerate(CHANNELS)}
    for marker, color in MARKER_COLORS.items():
        if marker not in idx:
            continue
        c = idx[marker]
        gain = GAIN.get(marker, 1.0)
        intensity = np.clip(mif[c] / 255.0 * gain, 0, 1)
        rgb += intensity[..., None] * np.array(color, dtype=np.float32)[None, None, :]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def resize_to(img, size):
    """cv2 resize 到 (size,size)。"""
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def find_if_path(tile):
    """从 test_dataframe 找 tile 的真实 mIF 路径。"""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA_ROOT, "test_dataframe.csv"))
    row = df[df["image_path"].str.contains(tile, na=False)]
    if len(row) == 0:
        return None, None
    he = os.path.join(DATA_ROOT, row.iloc[0]["image_path"])
    if_path = row.iloc[0]["target_path"]
    if not os.path.isabs(if_path):
        if_path = os.path.join(DATA_ROOT, if_path)
    return he, if_path


def make_panel(he, gt_mif, pred_mif, size=512, labels=("H&E", "GT 真实", "预测")):
    """并排 [H&E | GT | 预测]，统一尺寸。"""
    he_rgb = resize_to(he, size)
    gt_rgb = resize_to(multicolor_composite(gt_mif), size)
    pred_rgb = resize_to(multicolor_composite(pred_mif), size)
    panels = [he_rgb, gt_rgb, pred_rgb]
    # 加标题
    titled = []
    for img, lab in zip(panels, labels):
        im = Image.fromarray(img)
        w, h = im.size
        canvas = Image.new("RGB", (w, h + 40), (0, 0, 0))
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((10, h + 10), lab, fill=(255, 255, 255))
        titled.append(np.array(canvas))
    return np.hstack(titled)


def make_patch_panel(tile, size=512):
    """patch 级：单 tile 的 [H&E | GT | 各模型预测]。"""
    he, if_path = find_if_path(tile)
    if he is None:
        print(f"[warn] tile 不在 test 集: {tile}")
        return None
    he_img = load_he(he)
    gt_mif = load_mif(if_path)  # [16,H,W]
    panels = [resize_to(he_img, size)]
    # GT
    gt_rgb = resize_to(multicolor_composite(gt_mif), size)
    gt_titled = np.hstack([gt_rgb])  # 占位，稍后组装标题
    rows = [he_img, gt_mif]
    titles = ["H&E 原图", "GT 真实 mIF"]
    base = os.path.basename(he)
    base = os.path.splitext(base)[0]  # 去扩展名（tile 可能带 .jpeg）
    for name, pred_dir in MODELS.items():
        cand = os.path.join(pred_dir, base + ".tiff")
        if not os.path.exists(cand):
            cand = os.path.join(pred_dir, base + ".tif")
        if os.path.exists(cand):
            pm = load_mif(cand)
            rows.append(pm)
            titles.append(f"{name} 预测")
        else:
            print(f"[warn] 无预测: {name}/{tile}")
    # 统一尺寸并组装
    images = []
    for i, r in enumerate(rows):
        if r.ndim == 3 and r.shape[0] in (16, 17):  # mIF [C,H,W] -> 多色荧光
            r = multicolor_composite(r)
        images.append((r, titles[i]))
    parts = []
    for img, lab in images:
        img = resize_to(img, size)
        im = Image.fromarray(img)
        w, h = im.size
        canvas = Image.new("RGB", (w, h + 40), (0, 0, 0))
        canvas.paste(im, (0, 0))
        ImageDraw.Draw(canvas).text((10, h + 10), lab, fill=(255, 255, 255))
        parts.append(np.array(canvas))
    return np.hstack(parts)


def make_wsi_panel(slide_prefix, anchor_tile, n_cols=3, n_rows=2, size=256):
    """WSI 级：从 anchor tile 找相邻 tiles 拼接成全景对比。"""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA_ROOT, "test_dataframe.csv"))
    # 找同一 slide 的 tiles
    mask = df["image_path"].str.contains(slide_prefix, na=False)
    sub = df[mask].copy()
    if len(sub) == 0:
        print(f"[warn] slide {slide_prefix} 无 tiles")
        return None
    # 从文件名提取坐标 x,y（格式 _<x>_<y>_0_512_512，slide 名可含下划线）
    import re
    def coord(p):
        m = re.search(r"_(\d+)_(\d+)_0_512_512\.", os.path.basename(p))
        return int(m.group(1)), int(m.group(2))
    sub["xy"] = sub["image_path"].map(coord)
    sub = sub.sort_values(["xy"])
    anchor_xy = coord(anchor_tile if anchor_tile.endswith(".jpeg") else anchor_tile + ".jpeg")
    lookup = {xy: row for xy, (_, row) in zip(sub["xy"], sub.iterrows())}
    spatial_rows = []
    for rr in range(n_rows):
        for cc in range(n_cols):
            spatial_rows.append(lookup.get((anchor_xy[0]+cc*512, anchor_xy[1]+rr*512)))
    # 组装 WSI 网格：每格 [H&E | GT | 预测]
    grid_rows = []
    for r in range(n_rows):
        row_tiles = []
        for ccol in range(n_cols):
            row = spatial_rows[r * n_cols + ccol]
            if row is None:
                row_tiles.append(np.zeros((2*size, size, 3), dtype=np.uint8))
                continue
            he_img = load_he(os.path.join(DATA_ROOT, row["image_path"]))
            if_path = row["target_path"]
            if not os.path.isabs(if_path):
                if_path = os.path.join(DATA_ROOT, if_path)
            gt_mif = load_mif(if_path)
            # 单格：H&E 上 + GT 下 两行堆叠（此处不含预测；预测全景由 wsi_panorama.py 生成）
            he_s = resize_to(he_img, size)
            gt_s = resize_to(multicolor_composite(gt_mif), size)
            cell = np.vstack([he_s, gt_s])
            row_tiles.append(cell)
        grid_rows.append(np.hstack(row_tiles))
    return np.vstack(grid_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="18459_LSP10364_US_SCAN_OR_001__092347-registered.ome_0_27648_0_512_512")
    ap.add_argument("--out", default="/data/weiyh/results/visualization")
    ap.add_argument("--wsi", action="store_true")
    ap.add_argument("--model", action="append", default=[],
                    help="覆盖/追加预测目录，格式 NAME=DIR；可重复")
    args = ap.parse_args()
    if args.model:
        MODELS.clear()
    for spec in args.model:
        if "=" not in spec:
            ap.error(f"--model 需要 NAME=DIR，收到 {spec}")
        name, path = spec.split("=", 1)
        MODELS[name] = path
    os.makedirs(args.out, exist_ok=True)

    # Patch 级
    panel = make_patch_panel(args.tile)
    if panel is not None:
        fname = os.path.join(args.out, f"patch_{args.tile[:40]}.png")
        Image.fromarray(panel).save(fname)
        print(f"[patch] 已保存 {fname}")

    # WSI 级
    if args.wsi:
        slide = args.tile.split("_0_")[0]
        wsi = make_wsi_panel(slide, args.tile)
        if wsi is not None:
            fname = os.path.join(args.out, f"wsi_{slide[:40]}.png")
            Image.fromarray(wsi).save(fname)
            print(f"[wsi] 已保存 {fname}")


if __name__ == "__main__":
    main()
