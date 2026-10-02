"""Classification head on the existing ViT/CNN decoder; fusion design unchanged."""
from vit_matte.vitmatte_unet import ViTMatteUNet
from vit_matte.encoder import DEFAULT_WEIGHTS
from vit_matte.lora import LoRAQKVLinear


def build_model(config, device="cpu"):
    model = ViTMatteUNet(num_markers=16, input_size=config.get("tile_size", 256),
                         vit_size=config.get("vit_size", 224), vit_layers=tuple(config.get("vit_layers", [8,16,24,32])),
                         multi_scale=True, lora_r=config.get("lora_r", 32), lora_alpha=config.get("lora_alpha", 16),
                         dropout=config.get("dropout", 0), output_activation="logits",
                         weights_path=config.get("weights_path", DEFAULT_WEIGHTS), device=device)
    audit_lora(model)
    return model


def audit_lora(model):
    blocks = model.encoder.model.blocks
    present = [i+1 for i,b in enumerate(blocks) if isinstance(b.attn.qkv, LoRAQKVLinear)]
    if len(present) != len(blocks):
        raise RuntimeError(f"Missing Q/V LoRA layers: {set(range(1,len(blocks)+1))-set(present)}")
    return dict(block_count=len(blocks), qv_lora_layers=present, k_lora=False, mlp_lora=False)
