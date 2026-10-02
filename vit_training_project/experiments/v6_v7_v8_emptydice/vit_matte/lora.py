"""LoRA 注入：为 timm ViT 注意力层的 Q/V 投影添加低秩适配。

参照 MIPHEI-ViT §4.3：LoRA rank=8, alpha=1，只加在 Attention 的 Q 与 V 上。
timm ViT 的 Attention 结构：self.qkv = nn.Linear(dim, 3*dim)（timm 0.9.x）。
我们用 LoRAQKVLinear 包装它：保留原 qkv 权重并冻结，附加 A/B 低秩矩阵。
"""
import torch
import torch.nn as nn


class LoRAQKVLinear(nn.Module):
    """包装 timm Attention 的 qkv 线性层，在 Q 和 V 上做低秩适配（LoRA）。

    原 qkv: [B, N, dim] -> [B, N, 3*dim]（q, k, v 拼接）
    LoRA 更新：q += (x @ A_q^T) @ B_q^T * (alpha/r)，v 同理。
    A 用 kaiming 初始化、B 初始为 0 → 训练起点输出与原模型完全一致（稳定起步）。
    """

    def __init__(self, qkv: nn.Linear, r: int = 8, alpha: float = 1.0):
        super().__init__()
        assert isinstance(qkv, nn.Linear), "qkv 必须是 nn.Linear"
        self.qkv = qkv
        self.r = int(r)
        self.scaling = alpha / max(r, 1)
        dim = qkv.out_features // 3  # 每个头的总 dim（q/k/v 各占 1/3）
        dev, dt = qkv.weight.device, qkv.weight.dtype

        # 低秩参数：A 用 kaiming 均匀初始化，B 全零（Hu et al. LoRA 原版）
        self.lora_A_q = nn.Parameter(torch.zeros(self.r, dim, device=dev, dtype=dt))
        self.lora_B_q = nn.Parameter(torch.zeros(dim, self.r, device=dev, dtype=dt))
        self.lora_A_v = nn.Parameter(torch.zeros(self.r, dim, device=dev, dtype=dt))
        self.lora_B_v = nn.Parameter(torch.zeros(dim, self.r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A_q, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.lora_A_v, a=5 ** 0.5)
        # B 保持零

    def forward(self, x):
        y = self.qkv(x)  # [B, N, 3*dim]
        dim = y.shape[-1] // 3
        q, k, v = y[..., :dim], y[..., dim:2 * dim], y[..., 2 * dim:]

        # (x @ A^T) @ B^T
        q_delta = (x @ self.lora_A_q.t()) @ self.lora_B_q.t() * self.scaling
        v_delta = (x @ self.lora_A_v.t()) @ self.lora_B_v.t() * self.scaling
        q = q + q_delta
        v = v + v_delta
        return torch.cat([q, k, v], dim=-1)


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 1.0,
                freeze_all: bool = True):
    """把 LoRA 注入到 model.blocks[*].attn.qkv，并冻结非 LoRA 参数。

    返回: 可训练参数数量（仅 LoRA）。
    """
    n_injected = 0
    # 先冻结全部，再注入 LoRA（新参数默认 requires_grad=True）
    if freeze_all:
        for p in model.parameters():
            p.requires_grad_(False)
    # 遍历 ViT 所有 block，把其注意力 qkv 线性层替换为 LoRA 包装版
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
            attn.qkv = LoRAQKVLinear(attn.qkv, r=r, alpha=alpha)
            n_injected += 1

    n_trainable = 0
    for p in model.parameters():
        if p.requires_grad:
            n_trainable += p.numel()
    print(f"[LoRA] 注入 {n_injected} 个注意力层 (rank={r}, alpha={alpha})，可训练参数 {n_trainable/1e6:.3f}M")
    return n_trainable
