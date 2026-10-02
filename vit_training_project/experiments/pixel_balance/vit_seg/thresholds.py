"""Post-selection per-channel F1 thresholds, fitted on natural validation only."""
import json
from pathlib import Path
import numpy as np
from .artifacts import file_hash
from .data import CHANNELS


def select_thresholds(metrics, positive_patients, negative_patients, config):
    """Histogram bin edges give exact >= decisions on this discrete grid.

    This selects a decision boundary, not a probability calibration transform.
    Pixel counts alone are not independent observations; require patient support.
    """
    bins = metrics.bins
    indices = np.arange(int(np.ceil(config["min"]*bins)), int(np.floor(config["max"]*bins))+1)
    values = indices/bins
    if not len(values) or not np.any(values == .5):
        raise ValueError("Threshold grid must contain 0.5")
    rows, thresholds = [], []
    for c, name in enumerate(CHANNELS):
        pos, neg = metrics.pos[c], metrics.neg[c]
        tp = pos[::-1].cumsum()[::-1][indices]
        fp = neg[::-1].cumsum()[::-1][indices]
        fn = pos.sum()-tp
        denominator = 2*tp+fp+fn
        f1 = np.divide(2*tp, denominator, out=np.zeros(len(indices), dtype=float), where=denominator>0)
        supported = (pos.sum() >= config["min_positive_pixels"] and neg.sum() >= config["min_negative_pixels"]
                     and positive_patients[c] >= config["min_patients_per_label"]
                     and negative_patients[c] >= config["min_patients_per_label"])
        # Equal maxima: choose nearest to 0.5, then the lower threshold.
        candidates = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
        chosen = int(candidates[np.argmin(np.abs(values[candidates]-.5))]) if supported else int(np.flatnonzero(values == .5)[0])
        thresholds.append(float(values[chosen]))
        rows.append(dict(channel=name, threshold=float(values[chosen]), tuned=bool(supported),
                         reason="validation_F1_maximum" if supported else "insufficient_validation_support_fallback_0.5",
                         positive_pixels=int(pos.sum()), negative_pixels=int(neg.sum()),
                         positive_patients=int(positive_patients[c]), negative_patients=int(negative_patients[c]),
                         val_f1_at_selected=float(f1[chosen]), val_f1_at_05=float(f1[values == .5][0]),
                         grid=values.tolist(), val_f1=f1.tolist()))
    return dict(channels=CHANNELS, thresholds=thresholds, fitted_split="val", objective="per_channel_pixel_F1",
                model_selection_threshold=.5, grid_bins=bins, support_config=config, per_channel=rows,
                note="Validation scores are selection scores, not unbiased performance; test is untouched")


def load_thresholds(path, checkpoint, signature=None):
    value = json.loads(Path(path).read_text())
    thresholds = np.asarray(value["thresholds"], dtype=float)
    if value.get("channels") != CHANNELS or thresholds.shape != (16,) or not np.isfinite(thresholds).all() or not ((thresholds > 0)&(thresholds < 1)).all():
        raise ValueError("Invalid thresholds or channel order")
    if value.get("fitted_split") != "val" or value.get("checkpoint_sha256") != file_hash(checkpoint):
        raise ValueError("Thresholds were not fitted on validation for this checkpoint")
    if signature is not None and value.get("data_signature") != signature:
        raise ValueError("Threshold calibration data mismatch")
    return thresholds.tolist()
