"""整张 WSI 虚拟 mIF 全景生成。

对指定 slide 的所有 H&E tiles 用训练好的模型推理，按坐标 (x,y) 拼接成整张 WSI：
  - H&E 全景（原始 H&E 拼图）
  - GT 真实 mIF 多色荧光全景（if/ 目录，经 train/val/test dataframe 映射）
  - 模型预测 mIF 多色荧光全景

用法：
    python scripts/wsi_panorama.py --slide 18459_LSP10353 --model vit512 \
        --tile_out 128 --gpu 0
    python scripts/wsi_panorama.py --slide 18459_LSP10353 \
        --model vit512 --model pix2pixhd --model pix2pix --tile_out 128

输出：wsi_<slide>_{HE,GT,<model>_pred}.png（原生分辨率大图 + 另行缩放缩略图）。
"""
import os
import re
import sys
import glob
import argparse
import numpy as np
import cv2
import torch
import pandas as pd
import tifffile
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from datacore.orioncrc_dataset import (CHANNELS, ENCODER_STATS, TARGET_RANGE,
                                       MIF_FULL_CHANNELS, MIF_SELECT,
                                       denormalize_mif_np, load_marker_q)
from vit_matte.vitmatte_unet import ViTMatteUNet
from vit_matte.encoder import DEFAULT_WEIGHTS

DATA_ROOT = "/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x"

# 多色荧光配色（与 visualize_virtual_stain.py 一致）
MARKER_COLORS = {
    "Hoechst": (0, 0, 255), "CD4": (0, 255, 255), "CD8a": (255, 255, 0),
    "FOXP3": (255, 0, 255), "CD20": (255, 0, 0), "CD68": (0, 255, 0),
    "CD163": (0, 200, 100), "Pan-CK": (255, 120, 180), "SMA": (255, 165, 0),
    "CD3e": (0, 128, 255),
}
GAIN = {"Hoechst": 1.2, "CD4": 4.0, "CD8a": 5.0, "FOXP3": 8.0, "CD20": 5.0,
        "CD68": 4.0, "CD163": 5.0, "Pan-CK": 3.0, "SMA": 3.0, "CD3e": 4.0}

# 各模型的最终 checkpoint 路径（vit512 = v1；vit256v3 = v3-best）
MODEL_CKPT = {
    "vit512": "/data/weiyh/weights/vit_matte/virchow2_vitmatte_512_iter140000.pt",
    "vit256v3": "/data/weiyh/weights/vit_matte/virchow2_vitmatte_v3_256_best.pt",
    "pix2pixhd": "/data/weiyh/weights/pix2pixhd/pix2pixhd_16ch_in_epoch20.pt",
    "pix2pix": "/data/weiyh/weights/pix2pix/pix2pix_16ch_best.pt",
}
MODEL_SIZE = {"vit512": 512, "vit256v3": 256, "pix2pixhd": 256, "pix2pix": 256}
MODEL_STATS = {"vit512": "virchow2", "vit256v3": "virchow2",
               "pix2pixhd": "pix2pix", "pix2pix": "pix2pix"}
MODEL_LOG_NORM = {"vit256v3": True}  # log 域目标模型（采样输出用 denormalize_mif_np 还原）


def multicolor_composite(mif, gain_scale=1.0):
    """mif: [C,H,W] float(0-255) -> RGB 多色荧光 [H,W,3]。"""
    H, W = mif.shape[1], mif.shape[2]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    idx = {c: i for i, c in enumerate(CHANNELS)}
    for marker, color in MARKER_COLORS.items():
        if marker not in idx:
            continue
        c = idx[marker]
        gain = GAIN.get(marker, 1.0) * gain_scale
        intensity = np.clip(mif[c] / 255.0 * gain, 0, 1)
        rgb += intensity[..., None] * np.array(color, dtype=np.float32)[None, None, :]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def load_mif(path):
    """读 mIF TIFF -> [16,H,W] uint8(0-255)，剔除 PD-1(17->16)。"""
    a = tifffile.imread(path, maxworkers=8)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim == 3 and a.shape[2] not in (16, 17):
        a = a.transpose(1, 2, 0)
    if a.shape[2] == 17:
        a = a[:, :, MIF_SELECT]
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def collect_tiles(slide):
    """scandir 收集 slide 所有 he tiles -> [(basename, x, y)]。"""
    he_dir = os.path.join(DATA_ROOT, "he")
    tiles = []
    with os.scandir(he_dir) as it:
        for e in it:
            if not e.name.endswith(".jpeg") or slide not in e.name:
                continue
            m = re.search(r"_(\d+)_(\d+)_0_512_512\.", e.name)
            if not m:
                continue
            tiles.append((e.name, int(m.group(1)), int(m.group(2))))
    tiles.sort(key=lambda t: (t[2], t[1]))  # 按 y,x 排序
    return tiles


def build_gt_map(slide):
    """读 train/val/test dataframe -> {he_basename_noext: target_path}。"""
    mp = {}
    for sp in ("train", "val", "test"):
        dfp = os.path.join(DATA_ROOT, f"{sp}_dataframe.csv")
        if not os.path.exists(dfp):
            continue
        df = pd.read_csv(dfp)
        sub = df[df["image_path"].str.contains(slide, na=False)]
        for _, r in sub.iterrows():
            key = os.path.splitext(os.path.basename(r["image_path"]))[0]
            tgt = r["target_path"]
            if not os.path.isabs(tgt):
                tgt = os.path.join(DATA_ROOT, tgt)
            mp[key] = tgt
    return mp


class HeTileDataset(Dataset):
    """按文件名列表加载 he jpeg -> 归一化 tensor。"""

    def __init__(self, he_paths, in_size, stats_key):
        self.paths = he_paths
        self.in_size = in_size
        st = ENCODER_STATS[stats_key]
        self.mean = torch.tensor(st["mean"]).view(3, 1, 1)
        self.std = torch.tensor(st["std"]).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        b = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)  # BGR
        if b is None:
            raise FileNotFoundError(self.paths[i])
        b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        if b.shape[:2] != (self.in_size, self.in_size):
            b = cv2.resize(b, (self.in_size, self.in_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(b.transpose(2, 0, 1).astype(np.float32).copy())
        return (x / 255.0 - self.mean) / self.std


def load_model(name, device):
    if name == "vit512":
        model = ViTMatteUNet(num_markers=16, input_size=512, vit_size=224, vit_layers=(32,), multi_scale=False, legacy_v1=True,
                             weights_path=DEFAULT_WEIGHTS, device=device,
                             lora_r=32, lora_alpha=16.0)
        ck = torch.load(MODEL_CKPT[name], map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
    elif name == "vit256v3":
        # 从 ckpt config 读取训练架构（256 / vit_size 224 / log_norm），与采样一致
        ck = torch.load(MODEL_CKPT[name], map_location="cpu", weights_only=False)
        cfg = ck.get("config") or {}
        vit_layers = tuple(int(s) for s in cfg.get("vit_layers", "8,16,24,32").split(",")
                           if s.strip())
        model = ViTMatteUNet(num_markers=16,
                             input_size=cfg.get("tile_size", 256),
                             vit_size=cfg.get("vit_size", 224),
                             vit_layers=vit_layers, multi_scale=True,
                             weights_path=DEFAULT_WEIGHTS, device=device,
                             lora_r=cfg.get("lora_r", 32),
                             lora_alpha=cfg.get("lora_alpha", 16.0))
        model.load_state_dict(ck["model"])
    else:
        from pix2pixhd16.models import GlobalGenerator
        from pix2pix.models import UNet256
        if name == "pix2pixhd":
            model = GlobalGenerator(3, 16, ngf=64)
        else:
            model = UNet256(3, 16, ngf=64)
        ck = torch.load(MODEL_CKPT[name], map_location="cpu", weights_only=False)
        model.load_state_dict(ck["G"])
    model.eval().to(device)
    return model


@torch.no_grad()
def infer_all(model, he_paths, in_size, stats_key, batch_size, num_workers, device,
              log_norm=False):
    """推理所有 he -> [N,16,512,512] uint8(0-255)，统一还原到 512。"""
    q_np = load_marker_q() if log_norm else None
    if log_norm and q_np is None:
        raise FileNotFoundError("Log inference requires marker_q.json")
    ds = HeTileDataset(he_paths, in_size, stats_key)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out_all = []
    for i, x in enumerate(loader):
        pred = model(x.to(device)).cpu().clamp(-1, 1)
        if log_norm and q_np is not None:
            p = denormalize_mif_np(pred.numpy(), q_np)  # log 域反变换 -> 0-255
        else:
            p = ((pred + 0.9) / 1.8).numpy() * 255.0  # 目标 [-0.9,0.9] -> 0-255
        p = np.clip(p, 0, 255).astype(np.uint8)
        if p.shape[-1] != 512 or p.shape[-2] != 512:
            # 小模型输出 resize 回 512，与 GT 对齐
            up = np.empty((p.shape[0], 16, 512, 512), dtype=np.uint8)
            for j in range(p.shape[0]):
                for c in range(16):
                    up[j, c] = cv2.resize(p[j, c], (512, 512), interpolation=cv2.INTER_CUBIC)
            p = up
        out_all.append(p)
        print(f"  推理 {min((i + 1) * batch_size, len(ds))}/{len(ds)}", flush=True)
    return np.concatenate(out_all, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", default="18459_LSP10353")
    ap.add_argument("--model", action="append", default=None,
                    choices=["vit512", "vit256v3", "pix2pixhd", "pix2pix", "i2sb"])
    ap.add_argument("--tile_out", type=int, default=128, help="全景拼块边长(px)")
    ap.add_argument("--out", default="/data/weiyh/results/wsi_panorama")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--gain", type=float, default=1.0, help="多色荧光整体增益")
    ap.add_argument("--no_gt", action="store_true", help="不拼 GT 全景")
    ap.add_argument("--he_gt_only", action="store_true", help="只拼 H&E+GT 全景（不需要 GPU）")
    ap.add_argument("--save_tiff", action="store_true", help="同时保存每 tile 预测 tiff")
    ap.add_argument("--pred_dir", default=None, help="从已有预测目录拼接(不现场推理)，与 --model 配合",
                    nargs="*", action="append")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个 tiles(调试用，0=全部)")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.model:
        args.model = ["vit512"]
    print(f"[wsi] device={device}, slide={args.slide}, models={args.model}")

    # 1. 收集 tiles
    tiles = collect_tiles(args.slide)
    if not tiles:
        print(f"[err] slide {args.slide} 无 tiles")
        return
    if args.limit > 0:
        tiles = tiles[:args.limit]
    xs = [t[1] for t in tiles]
    ys = [t[2] for t in tiles]
    x_min, y_min = min(xs), min(ys)
    nx = (max(xs) - x_min) // 512 + 1
    ny = (max(ys) - y_min) // 512 + 1
    print(f"[wsi] tiles={len(tiles)}, 网格 {nx}x{ny}, "
          f"坐标 x[{x_min},{max(xs)}] y[{y_min},{max(ys)}]")

    # 2. GT 映射
    gt_map = build_gt_map(args.slide) if not args.no_gt else {}
    print(f"[wsi] GT 映射 {len(gt_map)}/{len(tiles)}")

    # 3. 全景画布
    W, H = nx * args.tile_out, ny * args.tile_out
    print(f"[wsi] 全景尺寸 {W}x{H} px")

    def place(canvas, img, x, y):
        px = (x - x_min) // 512 * args.tile_out
        py = (y - y_min) // 512 * args.tile_out
        canvas[py:py + args.tile_out, px:px + args.tile_out] = img

    # H&E 全景
    he_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    gt_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    os.makedirs(args.out, exist_ok=True)
    slide_short = re.sub(r"[^\w]+", "_", args.slide)[:50]

    he_paths = []
    for fname, x, y in tiles:
        p = os.path.join(DATA_ROOT, "he", fname)
        he_paths.append(p)
        he_img = cv2.imread(p, cv2.IMREAD_COLOR)
        if he_img is None:
            continue
        he_rgb = cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB)
        small = cv2.resize(he_rgb, (args.tile_out, args.tile_out),
                           interpolation=cv2.INTER_AREA)
        place(he_canvas, small, x, y)
        # GT
        key = os.path.splitext(fname)[0]
        if key in gt_map:
            try:
                mif = load_mif(gt_map[key])
                small = cv2.resize(multicolor_composite(mif, args.gain),
                                   (args.tile_out, args.tile_out),
                                   interpolation=cv2.INTER_AREA)
                place(gt_canvas, small, x, y)
            except Exception as ex:
                print(f"[warn] GT 加载失败 {key}: {ex}")

    Image.fromarray(he_canvas).save(os.path.join(args.out, f"wsi_{slide_short}_HE.png"))
    print(f"[wsi] 已保存 H&E 全景")
    if not args.no_gt:
        Image.fromarray(gt_canvas).save(os.path.join(args.out, f"wsi_{slide_short}_GT.png"))
        print(f"[wsi] 已保存 GT 全景")

    if args.he_gt_only:
        print(f"[wsi] he_gt_only 模式完成，输出目录: {args.out}")
        return

    # 4. 模型推理 / 已有预测拼接 -> 预测全景
    if args.pred_dir and len(args.model) != 1:
        raise ValueError("--pred_dir requires exactly one --model to avoid relabeling the same predictions")
    pred_dirs = args.pred_dir[0] if args.pred_dir else None
    for name in args.model:
        pred_canvas = np.zeros((H, W, 3), dtype=np.uint8)
        if pred_dirs is not None:
            # 从已有预测目录读取（如 I2SB 采样结果），不现场推理
            tdir = pred_dirs if isinstance(pred_dirs, str) else pred_dirs[0]
            print(f"[wsi] 从 {tdir} 拼接 {name} 预测 ...", flush=True)
            n_hit = 0
            for i, (fname, x, y) in enumerate(tiles):
                key = os.path.splitext(fname)[0]
                cand = os.path.join(tdir, key + ".tiff")
                if not os.path.exists(cand):
                    cand = os.path.join(tdir, key + ".tif")
                if not os.path.exists(cand):
                    continue
                pm = load_mif(cand)
                small = cv2.resize(multicolor_composite(pm, args.gain),
                                   (args.tile_out, args.tile_out),
                                   interpolation=cv2.INTER_AREA)
                place(pred_canvas, small, x, y)
                n_hit += 1
            print(f"[wsi] {name} 命中 {n_hit}/{len(tiles)}")
        else:
            if not os.path.exists(MODEL_CKPT[name]):
                print(f"[warn] {name} ckpt 不存在: {MODEL_CKPT[name]}，跳过")
                continue
            print(f"[wsi] 加载 {name} ...")
            model = load_model(name, device)
            preds = infer_all(model, he_paths, MODEL_SIZE[name], MODEL_STATS[name],
                              args.batch_size, args.num_workers, device,
                              log_norm=MODEL_LOG_NORM.get(name, False))
            for i, (fname, x, y) in enumerate(tiles):
                small = cv2.resize(multicolor_composite(preds[i], args.gain),
                                   (args.tile_out, args.tile_out),
                                   interpolation=cv2.INTER_AREA)
                place(pred_canvas, small, x, y)
            del model, preds
            torch.cuda.empty_cache()
        Image.fromarray(pred_canvas).save(
            os.path.join(args.out, f"wsi_{slide_short}_{name}_pred.png"))
        print(f"[wsi] 已保存 {name} 预测全景")

    print(f"[wsi] 完成，输出目录: {args.out}")


if __name__ == "__main__":
    main()
