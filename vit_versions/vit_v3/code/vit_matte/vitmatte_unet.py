"""Virchow2 (ViT, LoRA) + ViTMatte 风格 U-Net 解码器 —— H&E → 16 通道 mIF。

三种版本的架构差异（本文件同时兼容）：
  - v1（架构基线）：vit_layers=(32,) 只用最后一层、multi_scale=False（普通 3x3 卷积），
    ViT 输入 224；
  - v2/v3（增强）：vit_layers=(8,16,24,32) 四层 + ViTNeck 融合、multi_scale=True
    （多尺度膨胀卷积），ViT 输入 448。

架构参照 MIPHEI-ViT §4.1（ViTMatte 变体）：
  - 编码器：Virchow2（ViT-H/14，冻结 + LoRA 在 Q/V），输出 patch 特征；
  - Detail Capture Module：轻量卷积，从 H&E 提取多尺度金字塔特征，作为解码器跳连；
  - 解码器：双线性上采样 + 3x3 卷积 + BN + ReLU + 跳连；
  - 输出：每个 marker 一个独立输出头（1x1 卷积），16 通道 + Tanh。

输入：H&E [B,3,512,512]（已按编码器 mean/std 归一化；ViT 分支 resize 到 vit_size=448）
输出：mIF [B,16,512,512]（缩放到 [-1,1]，配合 Tanh；训练目标在 [-0.9,0.9] 或 log 域 [-1,1]）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Virchow2Encoder, DEFAULT_WEIGHTS, DEFAULT_CONFIG


class ConvBlock(nn.Module):
    """3x3 卷积 + BN + ReLU（论文解码器基本单元）。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleConv(nn.Module):
    """多尺度卷积单元：不同膨胀率的 3x3 并行 + 1x1 融合。

    3x3(dilation=1/2/3) 等效 3x3 / 5x5 / 7x7 感受野，参数约为真正大核的 1/4~1/9。
    同时捕捉稀疏小斑点（小感受野）与大块结构（大感受野），针对 mIF 的
    "免疫 marker 稀疏小信号 + 上皮/间质大块结构" 双尺度特性。
    """

    def __init__(self, in_ch, out_ch, dilations=(1, 2, 3)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            )
            for d in dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * len(dilations), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


class ViTNeck(nn.Module):
    """ViT 多层级特征融合颈部（neck）。

    ViT 各中间层的输出分辨率相同（都是 patch grid），差异只在语义深浅，
    因此这里是"层级语义融合"（layer fusion），而非 FPN 式的分辨率融合：
      - 每层用 1x1 卷积投影到 neck_dim；
      - concat 后 1x1 卷积投影到 out_ch。
    融合后的深特征再与 CNN Detail Capture 的 s16 特征在解码器入口处 concat。
    """

    def __init__(self, in_dim=1280, neck_dim=256, out_ch=512, n_layers=4):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(in_dim, neck_dim, 1) for _ in range(n_layers)])
        self.fuse = nn.Sequential(nn.Conv2d(neck_dim * n_layers, out_ch, 1), nn.ReLU(True))

    def forward(self, feats):
        """feats: 按层号排序的特征列表，每项 [B, in_dim, H, W]（分辨率一致）。"""
        xs = [p(f) for p, f in zip(self.projs, feats)]
        return self.fuse(torch.cat(xs, dim=1))


class DetailCapture(nn.Module):
    """轻量卷积金字塔：从 H&E 提取 stride 16/8/4/2/1 的多尺度特征，供解码器跳连。

    multi_scale=True 时，最高分辨率层 c1/c2 用多尺度膨胀卷积（等效 3/5/7 感受野），
    以同时捕捉稀疏小斑点与大块结构；深层 c3-c5 保持 3x3（已通过下采样获得大感受野）。
    """

    def __init__(self, base=32, multi_scale=True):
        super().__init__()
        if multi_scale:
            self.c1 = MultiScaleConv(3, base, dilations=(1, 2, 3, 5))   # 256, ch=32 (s1)，更宽感受野
            self.c2 = MultiScaleConv(base, base * 2)                    # 128, ch=64 (s2)
        else:
            self.c1 = ConvBlock(3, base)                      # 256, ch=32 (s1)
            self.c2 = ConvBlock(base, base * 2)               # 128, ch=64 (s2)
        self.c3 = ConvBlock(base * 2, base * 4)      # 64,   ch=128  (s4)
        self.c4 = ConvBlock(base * 4, base * 4)      # 32,   ch=128  (s8)
        self.c5 = ConvBlock(base * 4, base * 2)      # 16,   ch=64   (s16)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.c1(x)                  # 256
        f2 = self.c2(self.pool(f1))      # 128
        f3 = self.c3(self.pool(f2))      # 64
        f4 = self.c4(self.pool(f3))      # 32
        f5 = self.c5(self.pool(f4))      # 16
        return {"s16": f5, "s8": f4, "s4": f3, "s2": f2, "s1": f1}


def up2x(x):
    """双线性上采样 2 倍（解码器逐级放大用）。"""
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class ViTMatteUNet(nn.Module):
    """ViTMatte 风格 U-Net：ViT 高层语义 + CNN 多尺度细节 → 16 通道 mIF 预测。

    输入：H&E [B,3,H,W]（已归一化）；输出：mIF [B,16,H,W] ∈[-1,1]（Tanh）。
    前向流程：
      1) ViT 分支：H&E resize 到 vit_size → 冻结 Virchow2 + LoRA 提多层特征 → ViTNeck 融合；
      2) 细节分支：DetailCapture 得到 s16/s8/s4/s2/s1 五级 CNN 特征；
      3) 解码：从 s16 出发逐级“上采样 + concat 跳连 + 卷积”，共 5 级到全分辨率；
      4) 输出头：16 个独立 1x1 卷积（逐 marker 独立预测）→ Tanh。

    版本差异：v1 用单层 (vit_layers=(32,)) + multi_scale=False；
             v2/v3 用四层 (8,16,24,32) + ViTNeck + 多尺度膨胀卷积。
    """

    def __init__(self, num_markers=16, input_size=256, vit_size=448,
                 vit_layers=(8, 16, 24, 32), neck_dim=256,
                 decoder_channels=(512, 256, 128, 64, 64), base=32,
                 multi_scale=True, legacy_v1=False, output_activation="tanh", dropout=0.0,
                 weights_path=DEFAULT_WEIGHTS, config_path=DEFAULT_CONFIG,
                 lora_r=8, lora_alpha=1.0, device="cuda"):
        super().__init__()
        self.input_size = input_size
        self.vit_size = vit_size
        self.vit_layers = tuple(sorted(set(vit_layers)))
        self.num_markers = num_markers
        self.legacy_v1 = legacy_v1
        self.output_activation = output_activation
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        if output_activation not in ("tanh", "logits"):
            raise ValueError("output_activation must be tanh or logits")
        if vit_size % 14 or input_size % 16:
            raise ValueError("vit_size must be divisible by 14; input_size by 16")
        if not self.vit_layers or min(self.vit_layers) < 1 or max(self.vit_layers) > 32:
            raise ValueError("vit_layers must be within 1..32")
        if legacy_v1 and (self.vit_layers != (32,) or multi_scale):
            raise ValueError("legacy_v1 requires layers=(32,) and multi_scale=False")

        # 编码器（冻结 + LoRA）
        self.encoder = Virchow2Encoder(weights_path, config_path, lora_r, lora_alpha, device)
        self.encoder.eval()  # 冻结骨干：LoRA 为线性层，不受 train/eval 影响

        c512, c256, c128, c64, c64f = decoder_channels
        # Detail Capture
        self.detail = DetailCapture(base, multi_scale)

        # ViT 多层级特征融合 neck（替代原单层 vit_proj：多层语义 + 投影 + 融合）
        if legacy_v1:
            self.vit_proj = nn.Conv2d(1280, c512, 1)
        else:
            self.vit_neck = ViTNeck(in_dim=1280, neck_dim=neck_dim, out_ch=c512,
                                    n_layers=len(self.vit_layers))
        # 各尺度的跳连投影（把 detail capture 特征通道对齐到解码器输入）
        self.proj_s16 = nn.Conv2d(base * 2, c512, 1)
        self.proj_s8 = nn.Conv2d(base * 4, c256, 1)
        self.proj_s4 = nn.Conv2d(base * 4, c128, 1)
        self.proj_s2 = nn.Conv2d(base * 2, c64, 1)
        self.proj_s1 = nn.Conv2d(base, c64f, 1)

        # 解码器各阶段（首层用多尺度膨胀卷积增强 ViT+CNN 融合，其余保持 3x3）
        self.conv16 = (MultiScaleConv(c512 + c512, c512) if multi_scale else ConvBlock(c512 + c512, c512))      # 16（首层多尺度融合）
        self.conv8 = ConvBlock(c512 + c256, c256)       # 32
        self.conv4 = ConvBlock(c256 + c128, c128)       # 64
        self.conv2 = ConvBlock(c128 + c64, c64)         # 128
        self.conv1 = ConvBlock(c64 + c64f, c64f)        # 256

        # 每个 marker 独立输出头
        self.heads = nn.ModuleList([nn.Conv2d(c64f, 1, 1) for _ in range(num_markers)])
        self.to(device)  # 编码器已在 device，解码器/DetailCapture/heads 统一移动

    def forward(self, he):
        # 1) ViT 分支：resize 到 vit_size（默认 448 = 32x14，patch grid 32x32）后
        #    直接在高分辨率上提取多层特征（仅位置编码插值，非特征插值）
        if he.shape[-1] != self.vit_size or he.shape[-2] != self.vit_size:
            he_vit = F.interpolate(he, size=(self.vit_size, self.vit_size),
                                   mode="bilinear", align_corners=False)
        else:
            he_vit = he
        vit_feats = self.encoder.forward_multiscale(he_vit, self.vit_layers)
        v = self.vit_proj(vit_feats[32]) if self.legacy_v1 else self.vit_neck([vit_feats[l] for l in self.vit_layers])  # [B,512,H,W]

        # 2) Detail Capture（CNN 多尺度特征）
        f = self.detail(he)

        # 3) 对齐：vit_size/14 == tile/16 == 32 时天然对齐，无需特征插值；否则兜底对齐
        if v.shape[-2:] != f["s16"].shape[-2:]:
            v = F.interpolate(v, size=f["s16"].shape[-2:], mode="bilinear", align_corners=False)

        # 4) 解码：16 → 32 → 64 → 128 → 256（逐级 2x 上采样 + 跳连 concat + 卷积）
        d = torch.cat([v, self.proj_s16(f["s16"])], 1)   # 512+512
        d = self.conv16(d)                                # 16 -> 512
        d = up2x(d)                                       # 32
        d = torch.cat([d, self.proj_s8(f["s8"])], 1)      # +256
        d = self.conv8(d)                                 # 32 -> 256
        d = up2x(d)                                       # 64
        d = torch.cat([d, self.proj_s4(f["s4"])], 1)      # +128
        d = self.conv4(d)                                 # 64 -> 128
        d = up2x(d)                                       # 128
        d = torch.cat([d, self.proj_s2(f["s2"])], 1)      # +64
        d = self.conv2(d)                                 # 128 -> 64
        d = up2x(d)                                       # 256
        d = torch.cat([d, self.proj_s1(f["s1"])], 1)      # +64
        d = self.conv1(d)                                 # 256 -> 64

        d = self.dropout(d)
        outs = [h(d) for h in self.heads]                 # 每个 marker 一张单通道图
        y = torch.cat(outs, dim=1)                        # [B,16,H,W]（H=input_size）
        return y if self.output_activation == "logits" else torch.tanh(y)                              # 夹到 [-1,1] 作为训练目标域

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 自测：随机输入前向一遍，打印输出形状与可训练参数量
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTMatteUNet(weights_path=DEFAULT_WEIGHTS if __import__("os").path.exists(DEFAULT_WEIGHTS) else None,
                         input_size=512, vit_size=448, device=dev)
    x = torch.randn(1, 3, 512, 512, device=dev)
    with torch.no_grad():
        y = model(x)
    print("输出:", tuple(y.shape))
    print(f"可训练参数: {model.trainable_parameters()/1e6:.3f}M")
