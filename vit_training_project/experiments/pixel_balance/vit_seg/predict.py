"""HE-only multilabel inference. Argmax is only a lossy display."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import tifffile
from datacore.orioncrc_dataset import CHANNELS, IMAGENET_STATS
from .data import read_he, tissue_mask
from .model import build_model
from .thresholds import load_thresholds
from .artifacts import file_hash
from .display import PALETTE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--input-dir", required=True)
    p.add_argument("--out", required=True); p.add_argument("--device", default="cuda")
    p.add_argument("--thresholds", help="Validation-selected thresholds; auto-discovered beside results/model")
    args = p.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("task") != "pixel_multilabel":
        raise ValueError("Expected classification checkpoint")
    cfg = ck["config"]; size = cfg["tile_size"]
    threshold_path = Path(args.thresholds) if args.thresholds else Path(args.checkpoint).resolve().parent.parent/"thresholds.json"
    selected_thresholds = (load_thresholds(threshold_path, args.checkpoint, ck.get("data_signature"))
                           if threshold_path.exists() else cfg.get("probability_threshold", .5))
    if args.thresholds and not threshold_path.is_file():
        raise FileNotFoundError(threshold_path)
    model = build_model(cfg, args.device); model.load_state_dict(ck["model"]); model.eval()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    if len({p.stem for p in paths}) != len(paths) or not paths:
        raise ValueError("Input list is empty or contains duplicate basenames")
    palette = PALETTE
    with torch.no_grad():
        for path in paths:
            native = read_he(path)
            native_mask = tissue_mask(native)
            native[~native_mask] = 0
            he = cv2.resize(native, (size, size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(native_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
            x = (he.astype(np.float32)/255-np.array(IMAGENET_STATS["mean"])) / np.array(IMAGENET_STATS["std"])
            x[~mask] = 0
            if mask.any():
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=args.device.startswith("cuda") and cfg.get("precision") == "bf16"):
                    logits = model(torch.from_numpy(x.transpose(2, 0, 1).copy()).float()[None].to(args.device))
                probabilities = logits.float().sigmoid()[0].cpu().numpy()
                probabilities[:, ~mask] = 0
                label = (probabilities.argmax(0)+1).astype(np.uint8)
                label[~mask] = 0
            else:
                probabilities = np.zeros((16, size, size), np.float32); label = np.zeros((size, size), np.uint8)
            threshold = np.broadcast_to(np.asarray(selected_thresholds), (16,))[:,None,None]
            binary = (probabilities >= threshold).astype(np.uint8)
            binary[:, ~mask] = 0
            # With channel-specific thresholds, the largest raw probability
            # may be negative. Display only among channels that actually pass.
            positive_label = (np.where(binary.astype(bool), probabilities, -1).argmax(0)+1).astype(np.uint8)
            positive_label[~binary.any(0)] = 0
            tifffile.imwrite(out/f"{path.stem}_multilabel.tiff", binary, metadata={"axes":"CYX"})
            tifffile.imwrite(out/f"{path.stem}_argmax_display.tiff", label)
            tifffile.imwrite(out/f"{path.stem}_positive_argmax_display.tiff", positive_label)
            np.savez_compressed(out/f"{path.stem}_probabilities.npz", probabilities=probabilities, tissue=mask)
            panel = np.hstack([he, palette[label]])
            cv2.imwrite(str(out/f"{path.stem}_argmax_display.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    (out/"prediction_definition.json").write_text(json.dumps(dict(task="pixel_multilabel", channels=CHANNELS,
        probability_threshold=selected_thresholds, threshold_source="validation_selected" if threshold_path.exists() else "default_0.5",
        thresholds_sha256=file_hash(threshold_path) if threshold_path.exists() else None,
        argmax="display_only_raw_probability_argmax_within_HE_tissue; zero=excluded_background",
        positive_argmax="additional_threshold_gated_display; zero=excluded or no predicted positive", ground_truth_used=False), indent=2))
    (out/"classes.json").write_text(json.dumps({0:"excluded_background", **{i+1:c for i,c in enumerate(CHANNELS)}}, indent=2))


if __name__ == "__main__":
    main()
