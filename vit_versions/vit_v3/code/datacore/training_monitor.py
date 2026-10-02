"""Early stopping and training-curve artifacts shared by all trainers."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass


@dataclass
class EarlyStopper:
    """Stop after ``patience`` validation checks without a material improvement."""

    patience: int = 0
    min_delta: float = 0.0
    mode: str = "max"
    warmup: int = 0
    best: float | None = None
    bad_checks: int = 0
    checks: int = 0

    def __post_init__(self):
        if self.patience < 0 or self.min_delta < 0 or self.warmup < 0:
            raise ValueError("patience, min_delta and warmup must be non-negative")
        if self.mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")

    def update(self, value: float) -> tuple[bool, bool]:
        """Return ``(improved, should_stop)`` for one validation value."""
        if not math.isfinite(value):
            raise ValueError(f"early-stop metric must be finite, got {value}")
        self.checks += 1
        improved = self.best is None
        if self.best is not None:
            improved = (value > self.best + self.min_delta) if self.mode == "max" \
                else (value < self.best - self.min_delta)
        if improved:
            self.best = float(value)
            self.bad_checks = 0
        elif self.checks > self.warmup:
            self.bad_checks += 1
        should_stop = self.patience > 0 and self.checks > self.warmup \
            and self.bad_checks >= self.patience
        return improved, should_stop

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state: dict | None):
        if not state:
            return
        for key in ("best", "bad_checks", "checks"):
            if key in state:
                setattr(self, key, state[key])


def save_training_artifacts(history: list[dict], out_dir: str, name: str,
                            x_key: str = "step") -> tuple[str, str | None]:
    """Write JSON history and a PNG containing every numeric series."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{name}_history.json")
    with open(json_path, "w") as stream:
        json.dump(history, stream, indent=2, ensure_ascii=False)
    if not history:
        return json_path, None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = []
    for key in history[0]:
        if key == x_key:
            continue
        values = [row.get(key) for row in history]
        if any(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values):
            keys.append(key)
    if not keys:
        return json_path, None

    x = [row.get(x_key, i + 1) for i, row in enumerate(history)]
    ncols = min(2, len(keys))
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.8 * nrows), squeeze=False)
    for axis, key in zip(axes.flat, keys):
        y = [row.get(key, float("nan")) for row in history]
        axis.plot(x, y, marker="o", markersize=3, linewidth=1.4)
        axis.set_title(key)
        axis.set_xlabel(x_key)
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(keys):]:
        axis.axis("off")
    fig.suptitle(name)
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{name}_history.png")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return json_path, png_path
