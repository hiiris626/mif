"""Virchow2 + ViTMatte U-Net + LoRA —— H&E→mIF 虚拟染色训练脚本（vit v1/v2/v3 三版共用入口）。

整体流程：
  1) 数据：OrionCRCDataset 逐 tile 读 H&E(3ch) 与 mIF(16ch)；
     mIF 目标域可选「线性 [-0.9,0.9]」(v1/v2) 或「per-marker log [-1,1]」(v3，默认)；
  2) 模型：冻结的 Virchow2 编码器 + LoRA + ViTMatte 式解码器（见 vitmatte_unet.py）；
  3) 损失：逐通道加权 MSE（1/std，权重由 compute_marker_stats.py 预算）
     × 正样本（阳性像素）加权 w = 1 + fg_weight·fg（阳性像素约 8%，默认 fg_weight=10 近似类平衡）；
  4) 优化：Adam lr=2e-4 / wd=1e-5 / grad_clip 1.0；前 400 iter 线性 warmup，之后 cosine 衰减到 0；
  5) 训练设施：单卡或 4 卡 DDP（--ddp，torchrun 启动）；
     每 eval_every 步在 val 上评估（log/线性口径与 eval/metrics*.py 一致，训练日志数可直接比对）；
     每 save_every 步存 step 快照 + [H&E|GT|预测] 可视化；val PSNR 最优存 best.pt；
     支持平台期早停（EarlyStopper）与断点续训（--resume）。

参照 MIPHEI-ViT §5.1：
  - 损失：逐 marker 加权 MSE（权重 = 1/std_j）
  - 优化：Adam, lr=2e-4, wd=1e-5, grad clip max_norm=1, dropout=0.1
  - 调度：前 400 iter 线性 warmup，之后 cosine 衰减到 0
  - fp16 混合精度，断点续训

用法：
    # 单卡（v1/v2 风格）
    python -m vit_matte.train --data_root /data/weiyh/orioncrc_miphei/... \
        --epochs 15 --batch_size 16 --gpu 0
    # 四卡 DDP（v3 风格，用 torchrun 启动）
    torchrun --nproc_per_node=4 -m vit_matte.train --ddp --name xxx --tile_size 256 ...
"""
import os
import re
import json
import glob
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from .vitmatte_unet import ViTMatteUNet
from .encoder import DEFAULT_WEIGHTS
from datacore.orioncrc_dataset import OrionCRCDataset, load_marker_q
from datacore.training_monitor import EarlyStopper, save_training_artifacts

# log 域指标换算常数：训练 log 域 x0=2*log2(min(I,q)/q+1)-1，
# 论文口径 I_log = 255*ln(min(I,q)/q+1) = 255*ln2*(x0+1)/2，与 eval/metrics_log.py 完全一致
LN2_255 = 255.0 * math.log(2.0)

DATA_DEFAULT = "/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x"
CKPT_DEFAULT = "/data/weiyh/weights/vit_matte"


def parse_args():
    """命令行参数解析。参数按用途分组：数据 / 训练规模 / 优化 / 模型 / 目标域与损失 / 训练设施。"""
    ap = argparse.ArgumentParser()
    # ---- 数据与输出位置 ----
    ap.add_argument("--data_root", default=DATA_DEFAULT)
    ap.add_argument("--ckpt_dir", default=CKPT_DEFAULT, help="checkpoint 输出目录")
    ap.add_argument("--name", default="virchow2_vitmatte", help="实验名（保存文件名前缀）")
    # ---- 训练规模 ----
    ap.add_argument("--tile_size", type=int, default=512, help="输入 tile 边长（256/512，须与数据/缓存一致）")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=8, help="单卡 batch（DDP 时指每卡）")
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="梯度累积步数（等效 batch = batch_size * grad_accum）")
    # ---- 优化器与梯度 ----
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--legacy_v1", action="store_true")
    ap.add_argument("--q_file", default=None)
    ap.add_argument("--val_tiles", type=int, default=400)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_iters", type=int, default=400)
    # ---- 模型结构（Virchow2 编码器 / LoRA / 解码器） ----
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--vit_size", type=int, default=448,
                    help="ViT 输入分辨率（须为 patch_size=14 的整数倍，448→32x32 grid）")
    ap.add_argument("--vit_layers", default="8,16,24,32",
                    help="要提取的 ViT 中间层号（逗号分隔，1-indexed）")
    ap.add_argument("--multi_scale", action="store_true", default=True,
                    help="DetailCapture 高分辨率层启用多尺度膨胀卷积（默认开）")
    ap.add_argument("--no_multi_scale", action="store_true",
                    help="关闭多尺度卷积（消融用）")
    ap.add_argument("--encoder_stats", default="virchow2", help="H&E 输入归一化统计来源")
    # ---- 目标域与损失 ----
    ap.add_argument("--log_norm", action=argparse.BooleanOptionalAction, default=True,
                    help="mIF 目标用 per-marker log 归一化（marker_q.json，默认开；--no-log_norm 回线性域做消融）")
    ap.add_argument("--fg_weight", type=float, default=10.0,
                    help="正样本（阳性像素）损失权重：w = 1 + fg_weight * fg，"
                         "默认 10 近似类平衡（阳性像素仅约 8%%）")
    # ---- 训练设施：数据加载 / 验证 / 快照 ----
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=1000, help="每 N iter 在 val 上评估一次")
    ap.add_argument("--save_every", type=int, default=1000,
                    help="每 N iter 保存 ckpt_step{it}.pt 快照 + viz_step{it}.png")
    ap.add_argument("--ckpt_keep", type=int, default=5,
                    help="保留最近 N 个 ckpt_step*.pt 快照，更旧的自动删除")
    ap.add_argument("--viz_tiles", type=int, default=2,
                    help="验证可视化快照的 tile 数（[H&E | GT | 预测] 三栏）")
    ap.add_argument("--early_stop_patience", type=int, default=8,
                    help="连续多少次验证无显著提升后停止；0=关闭")
    ap.add_argument("--early_stop_min_delta", type=float, default=0.02,
                    help="val PSNR 提升小于该值视为“无提升”")
    ap.add_argument("--early_stop_warmup", type=int, default=3,
                    help="前 N 次验证不触发早停（等待收敛）")
    # ---- 续训 / 并行 ----
    ap.add_argument("--resume", default=None, help="checkpoint 路径（含 optimizer/iter/早停状态）")
    ap.add_argument("--gpu", type=str, default="0", help="单卡模式下的 CUDA_VISIBLE_DEVICES")
    ap.add_argument("--ddp", action="store_true", help="用 torchrun 启动的多卡 DDP 模式")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def build_marker_weights(data_root, device):
    """构造逐 marker 损失权重（= 1/std，归一化到均值 1）。

    背景：线性目标域下各 marker 强度动态范围差异很大，弱 marker（免疫信号稀疏）
    的 loss 会被强 marker 淹没；用 1/σ 加权可让弱通道获得更大梯度权重。
    v1/v2 使用该加权；v3 的 log 域已平衡动态范围，改用均匀权重（见 main 内说明）。
    权重来自 compute_marker_stats.py 预统计的 marker_std.json；缺失时退化为均匀权重。
    """
    cand = os.path.join(os.path.dirname(data_root), "marker_std.json")
    if os.path.exists(cand):
        with open(cand) as f:
            stds = json.load(f)["std"]
        w = torch.tensor([1.0 / max(s, 1e-6) for s in stds], device=device, dtype=torch.float32)
        w = w / w.mean()  # 归一化到均值 1，避免改变整体损失尺度
        print("[marker] 使用加权 MSE，权重(归一化):", w.tolist())
        return w
    print("[marker] 未找到 marker_std.json，使用均匀权重（先运行 compute_marker_stats.py 更佳）")
    return torch.ones(16, device=device)


def get_lr_schedule(it, warmup, total, decay_half=0.5):
    """学习率调度：返回缩放系数 ∈[0,1]（实际 lr = --lr × 该系数）。

    - 前 warmup 步：线性从 0 升到峰值 1；
    - 之后：cosine 从 1 平滑衰减到 0（到 total 步归零）。
    """
    if it < warmup:
        return (it + 1) / warmup
    progress = (it - warmup) / max(1.0, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, marker_w=None, log_norm=False, fg_weight=10.0, max_tiles=400):
    """在 val 子集上评估：加权 MSE loss + 逐通道 PSNR/SSIM。返回 (val_loss, psnr, ssim)。

    - loss 与训练同口径：正样本加权（w = 1 + fg_weight·fg） + 逐通道权重 marker_w；
    - PSNR/SSIM 口径：
        · log 域（log_norm=True，默认，v3）：训练目标 x0∈[-1,1] 与论文指标
          I_log = 255·ln(min(I,q)/q+1) 只差线性缩放（I_log = LN2_255·(x0+1)/2），
          直接在该尺度上计算，与 eval/metrics_log.py 的最终 json 数值一致；
        · 线性域（--no-log_norm，v1/v2 旧口径）：映射回 0–255 后计算。
    - 注意：速度考虑只统计前 ~400 张 tile 的 PSNR/SSIM（训练内观察用），
      test 集正式指标以 eval/metrics*.py 为准。
    """
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    model.eval()
    psnr_list, ssim_list = [], []
    loss_sum, n_batch = 0.0, 0
    n_tiles = 0
    fg_thresh = -0.99 if log_norm else -0.89  # 背景恰为 -1(log)/-0.9(线性)
    for he, mif in loader:
        if max_tiles:
            he, mif = he[:max_tiles-n_tiles], mif[:max_tiles-n_tiles]
        n_tiles += len(he)
        he, mif = he.to(device), mif.to(device)
        pred = model(he)
        if marker_w is not None:
            # 与训练一致的加权损失（仅记录趋势，不参与反传）
            se = (pred - mif) ** 2                          # 逐像素平方误差 [B,16,H,W]
            fg = (mif > fg_thresh).to(se.dtype)             # 阳性像素 mask
            se_w = se * (1.0 + fg_weight * fg)              # 阳性像素加权
            loss_sum += (se_w.mean(dim=(0, 2, 3)) * marker_w).mean().item()
            n_batch += 1
        if log_norm:
            # log 域口径（论文公式(2) 的线性等价形式）：I_log = 255*ln(min(I,q)/q+1)
            # 训练目标域 x0 与 I_log 仅差线性缩放：I_log = LN2_255*(x0+1)/2
            # pred/target 同变换，PSNR data_range 取 255（与 eval/metrics_log.py 的 DATA_RANGE 一致）
            pl = ((pred.clamp(-1, 1) + 1.0) / 2.0 * LN2_255).cpu().numpy()
            tl = ((mif.clamp(-1, 1) + 1.0) / 2.0 * LN2_255).cpu().numpy()
        else:
            # 线性域旧口径：目标在 [-0.9,0.9]，(x+0.9)/1.8*255 映射到 [0,255]
            p = ((pred.clamp(-0.9, 0.9) + 0.9) / 1.8).cpu().numpy() * 255.0
            t = ((mif + 0.9) / 1.8).cpu().numpy() * 255.0
            pl, tl = p, t
        for i in range(pl.shape[0]):          # 逐 tile
            for c in range(pl.shape[1]):      # 逐通道（16 marker）
                a, b = pl[i, c], tl[i, c]
                mse = float(np.mean((a.astype(np.float64) - b) ** 2))
                psnr_list.append(10 * math.log10(255.0 ** 2 / max(mse, 1e-12)))
                ssim_list.append(structural_similarity(b, a, data_range=255.0))
        if max_tiles and n_tiles >= max_tiles:  # 采样即可（速度考虑）
            break
    model.train()
    val_loss = loss_sum / max(1, n_batch)
    return val_loss, float(np.mean(psnr_list)), float(np.mean(ssim_list))


# ---- 多色荧光可视化配色与亮度增益 ----
# 与 scripts/visualize_virtual_stain.py / wsi_panorama.py 使用同一套配色与增益（保证跨图“控制变量”）
# MARKER_COLORS: 每个 marker 的 RGB 颜色（0-255）；GAIN: 显示放大系数（弱 marker 放大更多，便于肉眼观察）
MARKER_COLORS = {
    "Hoechst": (0, 0, 255), "CD4": (0, 255, 255), "CD8a": (255, 255, 0),
    "FOXP3": (255, 0, 255), "CD20": (255, 0, 0), "CD68": (0, 255, 0),
    "CD163": (0, 200, 100), "Pan-CK": (255, 120, 180), "SMA": (255, 165, 0),
    "CD3e": (0, 128, 255),
}
GAIN = {"Hoechst": 1.2, "CD4": 4.0, "CD8a": 5.0, "FOXP3": 8.0, "CD20": 5.0,
        "CD68": 4.0, "CD163": 5.0, "Pan-CK": 3.0, "SMA": 3.0, "CD3e": 4.0}


def multicolor_composite(mif):
    """把 16 通道 mIF 强度图渲染为多色荧光 RGB 预览图。

    参数：mif [C,H,W] float(0-255)；返回：RGB [H,W,3] uint8。
    原理：每个 marker 按其配色权重叠加（加法混色），只渲染 MARKER_COLORS 里
    列出的 10 个代表性 marker（弱 marker 乘 GAIN 放大后夹到 [0,1]）。
    """
    from datacore.orioncrc_dataset import CHANNELS
    H, W = mif.shape[1], mif.shape[2]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    idx = {c: i for i, c in enumerate(CHANNELS)}   # 通道名 → 通道下标
    for marker, color in MARKER_COLORS.items():
        if marker not in idx:
            continue
        gain = GAIN.get(marker, 1.0)
        intensity = np.clip(mif[idx[marker]] / 255.0 * gain, 0, 1)   # 归一化 + 增益
        rgb += intensity[..., None] * np.array(color, dtype=np.float32)[None, None, :]
    return np.clip(rgb, 0, 255).astype(np.uint8)


@torch.no_grad()
def save_viz_snapshot(model, val_loader, ds, it, out_dir, name, log_norm, q_np, device, n=2):
    """取 val 前 n 个 tile 推理一次，保存 [H&E | GT | 预测] 三栏对比图（训练过程监控用）。"""
    from PIL import Image
    from datacore.orioncrc_dataset import denormalize_mif_np
    try:
        he_b, mif_b = next(iter(val_loader))
    except StopIteration:
        return
    he_b = he_b[:n].to(device)
    mif_b = mif_b[:n].to(device)
    was_training = model.training
    model.eval()
    pred = model(he_b)
    model.train(was_training)
    he_rgb = ((he_b.cpu() * ds.he_std + ds.he_mean) * 255.0).clamp(0, 255)  # H&E 反归一化回 0-255
    if log_norm and q_np is not None:
        # log 域：用与训练 normalize 互逆的 denormalize（含 q_c 分位反变换）
        p = denormalize_mif_np(pred.cpu().clamp(-1, 1).numpy(), q_np)
        t = denormalize_mif_np(mif_b.cpu().numpy(), q_np)
    else:
        # 线性域：[-0.9,0.9] → [0,255]
        p = np.clip((pred.cpu().numpy() + 0.9) / 1.8 * 255.0, 0, 255)
        t = np.clip((mif_b.cpu().numpy() + 0.9) / 1.8 * 255.0, 0, 255)
    panels = []
    for i in range(he_b.shape[0]):
        h = he_rgb[i].permute(1, 2, 0).numpy().astype(np.uint8)
        g = multicolor_composite(t[i])
        pr = multicolor_composite(p[i])
        panels.append(np.hstack([h, g, pr]))
    rows = [panels[k] for k in range(min(n, len(panels)))]
    img = np.vstack(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"viz_{name}_step{it}.png")
    Image.fromarray(img).save(path)
    print(f"[viz] 已保存验证可视化 {path}")


def prune_ckpt_snapshots(ckpt_dir, name, keep):
    """仅保留最近 keep 个 ckpt_step*.pt 快照。"""
    if keep < 1:
        raise ValueError("ckpt_keep must be positive")
    snapshots = sorted(
        glob.glob(os.path.join(ckpt_dir, f"{name}_step*.pt")),
        key=lambda p: int(re.search(r"step(\d+)", os.path.basename(p)).group(1)))
    for p in snapshots[:-keep] if len(snapshots) > keep else []:
        os.remove(p)
        print(f"[ckpt] 删除旧快照 {os.path.basename(p)}")


def main():
    args = parse_args()
    if args.grad_accum < 1:
        raise ValueError("grad_accum must be positive")
    torch.manual_seed(args.seed)  # 固定随机种子（增广/采样等）

    # ---- DDP 初始化（torchrun 启动时自动注入 RANK / LOCAL_RANK / WORLD_SIZE） ----
    rank, world_size, local_rank = 0, 1, 0
    if args.ddp:
        dist.init_process_group(backend="nccl")       # NCCL 通信组
        rank = dist.get_rank()                        # 全局进程号
        world_size = dist.get_world_size()            # 总卡数
        local_rank = int(os.environ["LOCAL_RANK"])    # 本机第几张卡
        torch.cuda.set_device(local_rank)             # 每进程独占一张卡
        device = f"cuda:{local_rank}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = "cuda" if torch.cuda.is_available() else "cpu"
    is_main = (rank == 0)  # rank0 负责：打印日志 / 评估 / 存 ckpt（避免重复写盘）
    if is_main:
        print(f"device={device}, world={world_size}, ddp={args.ddp}, 数据={args.data_root}", flush=True)

    # 数据集：训练集开增广（aug=True），验证集关
    train_ds = OrionCRCDataset("train", root=args.data_root, tile_size=args.tile_size,
                               encoder_stats=args.encoder_stats, aug=True, log_norm=args.log_norm, q_file=args.q_file)
    val_ds = OrionCRCDataset("val", root=args.data_root, tile_size=args.tile_size,
                             encoder_stats=args.encoder_stats, aug=False, log_norm=args.log_norm, q_file=args.q_file)
    # DDP：DistributedSampler 把训练集均分到各卡（每 epoch 需 set_epoch 重新打乱）
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                       shuffle=True, seed=args.seed) if args.ddp else None
    val_sampler = None
    # drop_last=True：丢掉尾批，保证各卡批数一致（DDP 同步要求）；pin_memory 加速 CPU→GPU 拷贝
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
    if is_main:
        print(f"train tiles={len(train_ds)}, val tiles={len(val_ds)}, "
              f"全局 batch={args.batch_size * world_size * args.grad_accum}", flush=True)

    # 解析要融合的 ViT 中间层号（默认 8/16/24/32 四层；v1 只用最后一层 32）
    vit_layers = tuple(int(s) for s in args.vit_layers.split(",") if s.strip())
    multi_scale = args.multi_scale and not args.no_multi_scale
    # 构建模型：Virchow2（冻结）+ LoRA（可训练）+ ViTMatte 式解码器
    model = ViTMatteUNet(num_markers=16, input_size=args.tile_size, vit_size=args.vit_size,
                         vit_layers=vit_layers, multi_scale=multi_scale, legacy_v1=args.legacy_v1, dropout=args.dropout,
                         weights_path=DEFAULT_WEIGHTS, lora_r=args.lora_r,
                         lora_alpha=args.lora_alpha, device=device)
    if is_main:
        print(f"可训练参数: {model.trainable_parameters()/1e6:.3f}M", flush=True)

    if args.log_norm:
        # log 域各通道动态范围已统一到 [-1,1]，1/std 的补偿动机（通道强度差异）已由归一化解决，
        # 再叠加 1/std 会重复补偿并压制 log 域增益；改为均匀权重，靠 fg_weight 处理正负样本不平衡。
        marker_w = torch.ones(16, device=device)
        if is_main:
            print("[marker] log_norm 模式：逐通道权重置 1（动态范围已由 log 归一化平衡），"
                  f"正样本加权 fg_weight={args.fg_weight}", flush=True)
    else:
        marker_w = build_marker_weights(args.data_root, device)
    q_np = load_marker_q(args.q_file) if args.log_norm else None
    args.marker_q = q_np.tolist() if q_np is not None else None  # 每 marker 前景 99.9 分位 q（log 反变换用）
    # 只优化需要梯度的参数（LoRA 注入点 + 解码器；Virchow2 主体冻结不训练）
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device != "cpu")  # fp16 混合精度梯度缩放

    start_iter, start_epoch = 0, 0
    best_psnr = float("-inf")
    history = []
    stopper = EarlyStopper(args.early_stop_patience, args.early_stop_min_delta,
                           mode="max", warmup=args.early_stop_warmup)
    if args.resume and not os.path.isfile(args.resume):
        raise FileNotFoundError(args.resume)
    if args.resume:
        # 断点续训：兼容 DDP 保存的 module. 前缀；恢复 iter / 历史 / 早停计数 / fp16 scaler
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = {k.replace("module.", "") if k.startswith("module.") else k: v
                 for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iter", 0)
        best_psnr = ckpt.get("best_psnr", float("-inf"))
        history = ckpt.get("history", [])
        stopper.load_state_dict(ckpt.get("early_stopper"))
        if stopper.best is None and math.isfinite(best_psnr):
            stopper.best = best_psnr
        if start_iter % len(train_loader) % args.grad_accum:
            raise ValueError("Cannot resume inside an unfinished accumulation group")
        previous = ckpt.get("config", {})
        for key in ("batch_size", "grad_accum", "tile_size", "log_norm"):
            if key in previous and previous[key] != getattr(args, key):
                raise ValueError(f"Resume configuration changed: {key}")
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = start_iter // len(train_loader)
        if is_main:
            print(f"[resume] 从 iter {start_iter} (epoch {start_epoch}) 继续", flush=True)

    if args.ddp:
        # DDP 包裹（find_unused_parameters=True 兼容部分分支未用到的参数；实测无 unused）
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model.train()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    viz_dir = os.path.join(args.ckpt_dir, f"viz_{args.name}")   # 可视化快照目录
    global_step = start_iter                                    # 全局步计数（含续训起点）
    total_iters = len(train_loader) * args.epochs               # cosine 调度的总步数
    t0 = time.time()
    stopped_early = False

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # DDP：每 epoch 重设随机种子（各卡打乱不同且可复现）
        for it, (he, mif) in enumerate(train_loader):
            if epoch == start_epoch and it < start_iter % len(train_loader):
                continue  # 续训：跳过本 epoch 已训练过的批次
            he, mif = he.to(device), mif.to(device)
            # 每步更新学习率：warmup 线性升温 + 之后 cosine 衰减（乘到基准 lr 上）
            lr_scale = get_lr_schedule(global_step, args.warmup_iters, total_iters)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * lr_scale

            if it % args.grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)
            group_size = min(args.grad_accum, len(train_loader) - (it // args.grad_accum) * args.grad_accum)
            step_done = (it + 1) % args.grad_accum == 0 or it + 1 == len(train_loader)
            # 每个 microbatch 同步梯度，累积周期结束后更新参数。
            sync_ctx = nullcontext()
            with sync_ctx, torch.amp.autocast("cuda", enabled=device != "cpu"):
                pred = model(he)  # [-1,1]
                # 逐通道加权 MSE + 正样本（阳性像素）加权
                se = (pred - mif) ** 2                       # [B,16,H,W]
                fg_thresh = -0.99 if args.log_norm else -0.89
                fg = (mif > fg_thresh).to(se.dtype)          # 阳性像素 mask
                se_w = se * (1.0 + args.fg_weight * fg)      # 背景 w=1，阳性 w=1+fg_weight
                loss = (se_w.mean(dim=(0, 2, 3)) * marker_w).mean()

            # 梯度累积：loss 除以累积步数，每 grad_accum 步才更新一次参数
            with sync_ctx:
                scaler.scale(loss / group_size).backward()
            if step_done:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            global_step += 1

            if is_main and global_step % 50 == 0:
                el = time.time() - t0
                print(f"[iter {global_step}/{total_iters}] ep{epoch} "
                      f"train_loss={loss.item():.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e} "
                      f"t={el:.0f}s ({el/global_step:.2f}s/it)", flush=True)

            # ---- 定期验证：val 加权 loss + PSNR/SSIM，更新 best.pt 与早停状态（仅 rank0）----
            should_stop = False
            if is_main and step_done and global_step % args.eval_every == 0:
                val_loss, psnr, ssim = evaluate(model.module if args.ddp else model, val_loader, device, marker_w,
                                                log_norm=args.log_norm,
                                                fg_weight=args.fg_weight, max_tiles=args.val_tiles)
                improved, should_stop = stopper.update(psnr)
                history.append(dict(step=global_step, epoch=epoch + 1,
                                    train_loss=float(loss.detach()), val_loss=val_loss,
                                    val_psnr=psnr, val_ssim=ssim,
                                    lr=optimizer.param_groups[0]["lr"]))
                save_training_artifacts(history, args.ckpt_dir, args.name)
                print(f"  >>> [iter {global_step}] val_loss={val_loss:.4f} "
                      f"PSNR={psnr:.2f} SSIM={ssim:.4f} "
                      f"plateau={stopper.bad_checks}/{stopper.patience}", flush=True)
                if psnr > best_psnr:
                    best_psnr = psnr
                    ckpt = {"model": (model.module if args.ddp else model).state_dict(),
                            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                            "best_psnr": best_psnr, "history": history,
                            "early_stopper": stopper.state_dict(),
                            "iter": global_step, "epoch": epoch,
                            "config": vars(args)}
                    torch.save(ckpt, os.path.join(args.ckpt_dir, f"{args.name}_best.pt"))
                    print(f"  [best] val PSNR={psnr:.2f} 已保存 {args.name}_best.pt", flush=True)

            if args.ddp and step_done:
                # DDP：把 rank0 的早停决定广播给所有卡（否则其余卡会一直等待集合通信）
                stop_tensor = torch.tensor(int(should_stop), device=device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
            if should_stop:
                stopped_early = True
                if is_main:
                    print(f"[early-stop] PSNR 平台期已达 {stopper.bad_checks} 次验证，停止训练并输出曲线",
                          flush=True)
                break

            # ---- 定期保存：step 快照（含 scaler/history/早停状态）+ 可视化，并清理旧快照（仅 rank0）----
            if is_main and step_done and global_step % args.save_every == 0:
                ckpt = {"model": (model.module if args.ddp else model).state_dict(),
                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "best_psnr": best_psnr, "history": history,
                        "early_stopper": stopper.state_dict(),
                        "iter": global_step, "epoch": epoch, "config": vars(args)}
                torch.save(ckpt, os.path.join(args.ckpt_dir, f"{args.name}_step{global_step}.pt"))
                prune_ckpt_snapshots(args.ckpt_dir, args.name, args.ckpt_keep)
                save_viz_snapshot(model.module if args.ddp else model, val_loader, val_ds, global_step, viz_dir,
                                  args.name, args.log_norm, q_np, device, args.viz_tiles)
                print(f"  [save] {args.name}_step{global_step}.pt (best_psnr={best_psnr:.2f})", flush=True)

        # epoch 末尾保存：早停时命名 {name}_stopped_step{N}.pt，否则 {name}_epoch{N}.pt
        if is_main:
            suffix = f"stopped_step{global_step}" if stopped_early else f"epoch{epoch+1}"
            ckpt = {"model": (model.module if args.ddp else model).state_dict(),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "best_psnr": best_psnr, "history": history,
                    "early_stopper": stopper.state_dict(),
                    "iter": global_step, "epoch": epoch, "config": vars(args)}
            torch.save(ckpt, os.path.join(args.ckpt_dir, f"{args.name}_{suffix}.pt"))
            print(f"[save] {suffix}", flush=True)
        if stopped_early:
            break

    if is_main:
        save_training_artifacts(history, args.ckpt_dir, args.name)
        print(f"训练完成。最佳 val PSNR: {best_psnr:.2f} ({args.name}_best.pt)", flush=True)
    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
