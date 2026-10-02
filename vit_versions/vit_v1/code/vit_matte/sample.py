"""Virchow2 + ViTMatte 推理（采样）：对指定 split（默认 test）生成 16 通道 mIF 预测。

特点：
  - 自动读取 checkpoint 内保存的 config（tile_size / vit_size / vit_layers / lora / log_norm），
    无需手动传架构参数（v1/v2/v3 三种配置都能直接加载）；
  - 目标域反变换：log 域用 denormalize_mif_np（配合 marker_q.json）；线性域用 (x+0.9)/1.8*255；
  - 输出到 <out_dir>/<ckpt 文件名>/：每 tile 一个 16ch uint8 TIFF + 一个 RGB 复合 PNG。

输出：
  /data/weiyh/results/vit_matte/<name>/<tile>.tiff           16 通道预测（按 H&E basename 命名）
  /data/weiyh/results/vit_matte/<name>/<tile>_composite.png  RGB 复合预览

用法：
    python -m vit_matte.sample --ckpt /data/weiyh/weights/vit_matte/xxx.pt --gpu 0
"""
import os
import glob
import argparse
import numpy as np
import torch
import tifffile
from PIL import Image

from .vitmatte_unet import ViTMatteUNet
from .encoder import DEFAULT_WEIGHTS
from datacore.orioncrc_dataset import OrionCRCDataset, CHANNELS, denormalize_mif_np, load_marker_q  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="训练好的 checkpoint .pt")
    ap.add_argument("--data_root", default="/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_dir", default="/data/weiyh/results/vit_matte")
    ap.add_argument("--tile_size", type=int, default=512, help="与训练一致的输入分辨率")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--gpu", type=str, default="0")
    # log_norm：None=自动从 ckpt 的 config 读取；显式覆盖用 --log_norm（--no-log_norm 强制线性）
    ap.add_argument("--log_norm", action=argparse.BooleanOptionalAction, default=None,
                    help="预测目标域是否为 per-marker log（默认自动读 ckpt config）")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    legacy_v1 = "vit_proj.weight" in ckpt["model"]
    if legacy_v1:
        cfg.update(vit_size=224, vit_layers="32", multi_scale=False, no_multi_scale=True)
    # 架构参数从 ckpt config 读取（vit256 训练时 vit_size=224、tile_size=256，避免写死 512 配置）
    tile_size = cfg.get("tile_size", args.tile_size)           # 训练时的输入边长（256/512）
    vit_size = cfg.get("vit_size", 448)                        # ViT 输入（448=v2/v3；224=v1）
    vit_layers = tuple(int(s) for s in cfg.get("vit_layers", "8,16,24,32").split(",") if s.strip())
    lora_r = cfg.get("lora_r", 32)
    lora_alpha = cfg.get("lora_alpha", 16.0)
    log_norm = cfg.get("log_norm", False) if args.log_norm is None else args.log_norm  # 自动识别目标域
    q_np = (np.asarray(cfg["marker_q"], dtype=np.float32) if cfg.get("marker_q") is not None else load_marker_q()) if log_norm else None               # log 反变换所需的分位 q（marker_q.json）
    if log_norm and q_np is None:
        raise FileNotFoundError("Log checkpoint requires marker_q.json")

    # 按 ckpt 内 config 重建同构模型（含 LoRA 注入点），再加载权重
    model = ViTMatteUNet(num_markers=16, weights_path=DEFAULT_WEIGHTS, device=device,
                         input_size=tile_size, vit_size=vit_size,
                         vit_layers=vit_layers, multi_scale=cfg.get("multi_scale", True) and not cfg.get("no_multi_scale", False),
                         lora_r=lora_r, lora_alpha=lora_alpha, legacy_v1=legacy_v1)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[sample] 加载 {args.ckpt} (iter {ckpt.get('iter')}, log_norm={log_norm}, "
          f"tile={tile_size}, vit={vit_size})")

    ds = OrionCRCDataset(args.split, root=args.data_root, tile_size=tile_size, encoder_stats=cfg.get("encoder_stats", "virchow2"), aug=False, return_target=False)  # inference needs no ground truth
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)

    # 输出目录 = out_dir/<checkpoint 文件名>/；输出文件名沿用 H&E basename（评估时按 dataframe 对齐）
    out = os.path.join(args.out_dir, os.path.basename(args.ckpt).replace(".pt", ""))
    os.makedirs(out, exist_ok=True)
    paths = ds.df[ds.he_col].tolist()

    # RGB 复合预览配色：红/绿/蓝三组 marker 强度分别相加混合（与 eval/metrics.py 的复合图口径一致）
    r_mix = ["Pan-CK", "ECadherin", "CD3e"]
    g_mix = ["SMA", "CD68", "CD163"]
    b_mix = ["Hoechst", "CD20"]
    cidx = {c: i for i, c in enumerate(CHANNELS)}   # 通道名 → 通道下标

    with torch.no_grad():
        idx = 0
        for he in loader:
            pred = model(he.to(device)).cpu().clamp(-1, 1)
            if log_norm and q_np is not None:
                # log 域反变换到 0-255（与训练 normalize_mif_log 互逆）
                p = denormalize_mif_np(pred.numpy(), q_np)
            else:
                # 还原到 0-255：训练目标在 [-0.9,0.9]，故 (out+0.9)/1.8*255
                p = ((pred + 0.9) / 1.8).numpy() * 255.0
            p = np.clip(p, 0, 255)
            # 逐 tile 写出：16ch uint8 TIFF（供评估）+ RGB 复合 PNG（供预览）
            for i in range(p.shape[0]):
                fname = os.path.splitext(os.path.basename(paths[idx]))[0] + ".tiff"
                tifffile.imwrite(os.path.join(out, fname), p[i].astype(np.uint8))
                # RGB 复合预览
                rgb = np.zeros((p.shape[2], p.shape[3], 3), dtype=np.float32)
                for ch in r_mix:
                    rgb[..., 0] += p[i, cidx[ch]]
                for ch in g_mix:
                    rgb[..., 1] += p[i, cidx[ch]]
                for ch in b_mix:
                    rgb[..., 2] += p[i, cidx[ch]]
                Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(
                    os.path.join(out, fname.replace(".tiff", "_composite.png")))
                idx += 1
            print(f"  已生成 {idx}/{len(ds)}")
    print(f"完成，输出目录: {out}")


if __name__ == "__main__":
    main()
