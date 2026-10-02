"""Virchow2 编码器（冻结骨干 + LoRA 微调）。

- 权重：本地 safetensors（/data/weiyh/weights/virchow2/model.safetensors）+ timm 显式构建
  （reg_tokens=4, SwiGLUPacked MLP, SiLU, LayerScale）；
- 微调方式：只给注意力层的 Q/V 注入 LoRA（见 lora.py），骨干参数全部冻结；
- 高分辨率：启用 dynamic_img_size，位置编码按实际 grid 自动插值，
  因此可直接输入 224（v1）或 448（v2/v3，patch grid 32x32）而无需把图缩回 224；
- token 布局：1 cls + 4 register + N patch（224→16x16=256，448→32x32=1024）。

本项目将其做成自包含模块，不依赖 virchow2_pannuke_decoder 目录。
"""
import json
import os
import torch
import torch.nn as nn
from timm.layers import SwiGLUPacked

from .lora import inject_lora

# 本文件目录
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)

DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "virchow2_config.json")
# Virchow2 权重本地路径（大数据统一放 /data/weiyh/weights/，用软链/真实文件）
DEFAULT_WEIGHTS = os.path.join("/data/weiyh/weights/virchow2", "model.safetensors")

# token 布局：1 cls + 4 register + 256 patch = 261
PATCH_SIZE = 14          # patch size（Virchow2 ViT-H/14）
PATCH_GRID = 16          # 224 / 14（默认 224 输入时的 grid；448 输入时为 32）
NUM_PREFIX_TOKENS = 5    # 1 cls + 4 register（前 5 个 token 不是图像 patch）
NUM_PATCH_TOKENS = 256   # 16x16（224 输入；448 输入为 1024）
FEAT_DIM = 1280          # ViT-H 隐藏维度


class Virchow2Encoder(nn.Module):
    """Virchow2 ViT-H/14 编码器。forward 返回 patch tokens 重排后的 2D 特征。

    - forward(x): x [B,3,224,224] → [B, 1280, 16, 16]（单层，v1 用）；
    - forward_multiscale(x, layers): 一次前向抽取多个中间层（v2/v3 用，448 输入时
      输出 [B, 1280, 32, 32]）。
    """

    def __init__(self, weights_path=DEFAULT_WEIGHTS, config_path=DEFAULT_CONFIG,
                 lora_r=8, lora_alpha=1.0, device="cuda"):
        super().__init__()
        self.model = self._build(weights_path, config_path, device)
        inject_lora(self.model, r=lora_r, alpha=lora_alpha, freeze_all=True)
        self.model.eval()
        self.num_layers = len(self.model.blocks)
        self.num_prefix_tokens = self.model.num_prefix_tokens

    @staticmethod
    def _build(weights_path, config_path, device):
        """按 config 构建 timm 模型并加载本地 safetensors 权重（weights_path=None 则随机初始化）。"""
        with open(config_path) as f:
            cfg = json.load(f)
        model = timm_create(cfg)
        if weights_path and os.path.exists(weights_path):
            from safetensors.torch import load_file
            sd = load_file(weights_path)
            model.load_state_dict(sd)
            print(f"[Virchow2] 已加载本地权重 {weights_path}")
        elif weights_path is not None:
            raise FileNotFoundError(f"Missing Virchow2 pretrained weights: {weights_path}")
        else:
            print("[Virchow2] weights_path=None: 随机初始化，仅架构验证用")
        return model.to(device)

    def _forward_tokens(self, x):
        """patch embed + pos embed + drop + pre-norm，返回 token 序列 [B, N+prefix, D]。

        Virchow2 开启 dynamic_img_size，patch_embed 输出 NHWC，_pos_embed 会按实际
        H/W 用 resample_abs_pos_embed 自动插值位置编码（仅插值位置编码，非特征），
        因此可直接在高分辨率（14 的整数倍）输入上提取特征，无需 resize 到 224。
        """
        x = self.model.patch_embed(x)   # [B, H, W, D]（NHWC）
        x = self.model._pos_embed(x)    # [B, N+prefix, D]
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        return x

    def forward_multiscale(self, x, layers=(8, 16, 24, 32), reshape=True):
        """提取多个中间层特征（层号为 1-indexed）。

        参数:
            x: [B, 3, H, W]，H/W 须为 patch_size(14) 的整数倍。
            layers: 要取出的层号（1..num_layers）。
        返回:
            dict {layer: 特征}。reshape=True 时特征为 [B, D, H/14, W/14]，
            否则为 patch token 序列 [B, N, D]（不含 cls/reg）。
        """
        B = x.shape[0]
        g = (x.shape[-2] // PATCH_SIZE, x.shape[-1] // PATCH_SIZE)
        tokens = self._forward_tokens(x)
        n_prefix = self.num_prefix_tokens
        layers = sorted(set(layers))
        out = {}
        next_i = 0
        # 逐层前向；命中目标层号时抽取其 patch token（层号 1-indexed）
        for i, blk in enumerate(self.model.blocks):
            tokens = blk(tokens)
            li = i + 1
            if next_i < len(layers) and li == layers[next_i]:
                f = tokens[:, n_prefix:]  # 去掉 cls + register tokens
                if reshape:
                    f = f.reshape(B, g[0], g[1], -1).permute(0, 3, 1, 2).contiguous()
                out[li] = f
                next_i += 1
        return out

    def forward(self, x):
        """返回最后一层 patch 特征 [B, D, H/14, W/14]（向后兼容）。"""
        return self.forward_multiscale(x, layers=(self.num_layers,))[self.num_layers]


def timm_create(cfg):
    """按 configs/virchow2_config.json 创建 Virchow2 结构（不加载预训练权重）。"""
    import timm
    return timm.create_model(
        cfg["architecture"],
        pretrained=False,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        **cfg["model_args"],
    )


def build_virchow2_transform():
    """与 Virchow2 预训练一致的预处理（mean/std）。"""
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    import json

    with open(DEFAULT_CONFIG) as f:
        cfg = json.load(f)
    model = timm.create_model(
        cfg["architecture"], pretrained=False,
        mlp_layer=SwiGLUPacked, act_layer=nn.SiLU, **cfg["model_args"])
    dc = resolve_data_config(model.pretrained_cfg, model=model)
    return create_transform(**dc)


if __name__ == "__main__":
    # 架构验证（随机权重即可跑通）
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Virchow2Encoder(weights_path=DEFAULT_WEIGHTS if os.path.exists(DEFAULT_WEIGHTS) else None,
                          device=dev)
    x = torch.randn(1, 3, 224, 224, device=dev)
    with torch.no_grad():
        out = enc(x)
    print("输出:", tuple(out.shape))
    n_all = sum(p.numel() for p in enc.parameters())
    n_tr = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"总参数 {n_all/1e6:.1f}M，可训练(LoRA) {n_tr/1e6:.3f}M")
