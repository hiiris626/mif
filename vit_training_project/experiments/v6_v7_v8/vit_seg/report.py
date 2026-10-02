"""Final data/augmentation figures and reproducible held-out multilabel exports."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import CHANNELS, IGNORE, read_he, read_mif, tissue_mask, labels_from_mif, augment


def data_figures(data, results):
    stats = json.loads((data/"statistics.json").read_text())
    root = Path(stats["source_spec"]["root"])
    counts = pd.read_csv(data/"channel_patch_counts.csv")
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), constrained_layout=True)
    for ax, split in zip(axes, ("train", "val", "test")):
        frame = counts[counts.split == split].set_index("channel").loc[CHANNELS]
        bottom = np.zeros(16)
        for name, color in (("false", "#8197ad"), ("true1", "#e39051"), ("true2", "#51a891"), ("unavailable", "#c3c3c3"), ("untrusted_zero", "#d6b2cb")):
            ax.bar(CHANNELS, frame[name], bottom=bottom, label=name, color=color)
            bottom += frame[name].to_numpy()
        ax.set_title(f"{split}: natural patch distribution (before balancing)")
        ax.set_ylabel("Patches"); ax.tick_params(axis="x", rotation=35)
    axes[0].legend(ncol=5)
    fig.savefig(results/"channel_patch_counts.png", dpi=180); plt.close(fig)
    train = pd.read_csv(data/"train.csv")
    # Deterministic cases, no visual cherry-picking for the augmentation figure.
    rows = train.groupby("orion_slide_id", sort=True).head(1).head(3)
    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 3*len(rows)), squeeze=False)
    for row_number, (_, row) in enumerate(rows.iterrows()):
        he, mif = read_he(root/row.image_path), read_mif(root/row.target_path)
        tissue = tissue_mask(he)
        labels, _ = labels_from_mif(mif, tissue, stats["q"], policy="multilabel")
        he[~tissue] = 0
        for col in range(4):
            image = he if col == 0 else augment(he.copy(), labels.copy(), tissue.copy(), np.random.default_rng(42+row_number*4+col))[0]
            axes[row_number, col].imshow(image); axes[row_number, col].axis("off")
            axes[row_number, col].set_title(f"{row.orion_slide_id}: " + ("original, masked" if col == 0 else f"mild augmentation {col}"))
    fig.tight_layout(); fig.savefig(results/"augmentation_preview.png", dpi=180); plt.close(fig)
    rows[["patch_id", "orion_slide_id", "image_path"]].to_csv(results/"augmentation_preview_patches.csv", index=False)


def trained_figures(data, results, device):
    metrics = json.loads((results/"test_metrics.json").read_text())
    frame = pd.DataFrame(metrics["per_class"]).T
    frame[["precision", "recall", "f1", "iou"]].to_csv(results/"test_class_metrics.csv")
    ax = frame[["precision", "recall", "f1", "iou"]].plot.bar(figsize=(15, 5), ylim=(0, 1))
    ax.set_title("Held-out patient test: multilabel classification"); ax.set_ylabel("Score")
    plt.tight_layout(); plt.savefig(results/"test_class_metrics.png", dpi=180); plt.close()
    thresholds = json.loads((results/"thresholds.json").read_text())
    pd.DataFrame([{k:v for k,v in row.items() if k not in ("grid", "val_f1")}
                  for row in thresholds["per_channel"]]).to_csv(results/"thresholds.csv", index=False)
    fig, axes = plt.subplots(4, 4, figsize=(14, 10), constrained_layout=True)
    for ax, row in zip(axes.flat, thresholds["per_channel"]):
        ax.plot(row["grid"], row["val_f1"])
        ax.axvline(.5, color="gray", linestyle="--", label="0.5")
        ax.axvline(row["threshold"], color="red", label="selected")
        ax.set_title(row["channel"]); ax.set_xlabel("Threshold"); ax.set_ylabel("Validation F1")
    axes[0,0].legend()
    fig.savefig(results/"validation_threshold_curves.png", dpi=160); plt.close(fig)
    baseline = json.loads((results/"test_metrics_default05.json").read_text())
    comparison = pd.DataFrame([dict(channel=name, threshold=thresholds["thresholds"][i],
        f1_default05=baseline["per_class"][name]["f1"], f1_selected=metrics["per_class"][name]["f1"],
        iou_default05=baseline["per_class"][name]["iou"], iou_selected=metrics["per_class"][name]["iou"])
        for i,name in enumerate(CHANNELS)])
    comparison.to_csv(results/"test_threshold_comparison.csv", index=False)
    stats = json.loads((data/"statistics.json").read_text())
    root = Path(stats["source_spec"]["root"])
    rows = pd.read_csv(data/"test.csv").groupby("orion_slide_id", sort=True).head(2)
    rows.to_csv(results/"prediction_preview_patches.csv", index=False)
    from . import predict
    with tempfile.TemporaryDirectory(prefix="vit_prediction_inputs_") as tmp:
        for _, row in rows.iterrows():
            (Path(tmp)/f"patch_{row.patch_id}.jpeg").symlink_to((root/row.image_path).resolve())
        original_argv = sys.argv
        try:
            sys.argv = ["predict", "--checkpoint", str(results/"model/best.pt"), "--input-dir", tmp,
                        "--out", str(results/"predictions"), "--device", device,
                        "--thresholds", str(results/"thresholds.json")]
            predict.main()
        finally:
            sys.argv = original_argv
    for _, row in rows.groupby("orion_slide_id", sort=True).head(1).iterrows():
        probabilities = np.load(results/f"predictions/patch_{row.patch_id}_probabilities.npz")["probabilities"]
        he, mif = read_he(root/row.image_path), read_mif(root/row.target_path)
        labels, _ = labels_from_mif(mif, tissue_mask(he), stats["q"], policy="multilabel")
        fig, axes = plt.subplots(4, 4, figsize=(16, 9))
        for c, ax in enumerate(axes.flat):
            target = cv2.resize((labels[c] == 1).astype(np.float32), probabilities.shape[1:][::-1], interpolation=cv2.INTER_NEAREST)
            ax.imshow(np.hstack((target, probabilities[c])), vmin=0, vmax=1, cmap="magma")
            ax.set_title(f"{CHANNELS[c]}: GT | probability"); ax.axis("off")
        fig.suptitle(f"{row.orion_slide_id} / patch {row.patch_id}; independent channels")
        fig.tight_layout(); fig.savefig(results/f"predictions/patch_{row.patch_id}_all_channels.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True); parser.add_argument("--results", required=True)
    parser.add_argument("--device", default="cuda:2"); parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    data, results = Path(args.data), Path(args.results)
    if not json.loads((data/"COMPLETE.json").read_text())["complete_dataset"]:
        raise ValueError("Final figures require complete data statistics")
    results.mkdir(parents=True, exist_ok=True)
    data_figures(data, results)
    if not args.data_only:
        trained_figures(data, results, args.device)


if __name__ == "__main__":
    main()
