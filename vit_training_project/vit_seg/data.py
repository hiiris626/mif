"""Shared label semantics. No GT-derived information is used to gate inference."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset
from datacore.orioncrc_dataset import CHANNELS, MIF_SELECT, IMAGENET_STATS

IGNORE = 255


def valid_flag(value):
    """CSV missing values mean unknown, never implicitly True."""
    if pd.isna(value):
        return False
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("true", "1"):
            return True
        if value in ("false", "0", ""):
            return False
        raise ValueError(f"Invalid channel availability flag: {value}")
    if value not in (True, False, 0, 1):
        raise ValueError(f"Invalid channel availability flag: {value}")
    return bool(value)


def read_he(path):
    x = cv2.imread(str(path))
    if x is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


def read_mif(path):
    a = tifffile.imread(path, maxworkers=1)
    if a.ndim != 3:
        raise ValueError(f"Invalid mIF shape {a.shape}: {path}")
    if a.shape[-1] in (16, 17):
        a = a.transpose(2, 0, 1)
    if a.shape[0] == 17:
        a = a[MIF_SELECT]
    if a.shape[0] != 16 or a.dtype != np.uint8:
        raise ValueError(f"Expected 16-channel uint8: {path}, {a.shape}, {a.dtype}")
    return a


def tissue_mask(he):
    """Conservative H&E-only white-glass exclusion, identical at train/inference."""
    hsv = cv2.cvtColor(he, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] > 12) | (he.mean(-1) < 220)).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)






def labels_from_mif(mif, tissue, q, policy="multilabel", thresholds=None,
                    channel_valid=None):
    """Independent binary expression labels; 255 excludes unknown/background."""
    if policy != "multilabel":
        raise ValueError("This project preserves coexpression with multilabel targets")
    q = np.asarray(q, np.float32)[:, None, None]
    if q.shape[0] != 16 or np.any(q <= 0):
        raise ValueError("q must contain 16 positive training-derived scales")
    thresholds = np.asarray(thresholds if thresholds is not None else [0]*16)[:, None, None]
    positive = (mif > thresholds) & tissue[None]
    channel_valid = np.ones(16, bool) if channel_valid is None else np.asarray(channel_valid, bool)
    if channel_valid.shape != (16,):
        raise ValueError("channel_valid must contain 16 booleans")
    if policy == "multilabel":
        # A zero in one observed channel is a useful negative whenever another
        # observed marker is positive. Missing/unreliable channels are ignored.
        valid = tissue & positive[channel_valid].any(0)
        target = positive.astype(np.uint8)
        target[:, ~valid] = IGNORE
        target[~channel_valid] = IGNORE
        return target, positive


def augment(he, label, tissue, rng):
    """Joint D4 + mild affine; H&E-only photometric variation. No elastic warps."""
    multilabel = label.ndim == 3
    if multilabel:
        label = label.transpose(1, 2, 0)
    k = int(rng.integers(4))
    he, label, tissue = [np.rot90(a, k).copy() for a in (he, label, tissue)]
    if rng.random() < .5:
        he, label, tissue = [np.flip(a, 1).copy() for a in (he, label, tissue)]
    h, w = label.shape[:2]
    if rng.random() < .35:
        matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), rng.uniform(-10, 10), rng.uniform(.95, 1.05))
        he = cv2.warpAffine(he, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        if multilabel:
            label = np.stack([cv2.warpAffine(label[..., c], matrix, (w,h), flags=cv2.INTER_NEAREST, borderValue=IGNORE) for c in range(label.shape[-1])], -1)
        else:
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=IGNORE)
        tissue = cv2.warpAffine(tissue.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
    if rng.random() < .5:
        contrast, brightness = rng.uniform(.95, 1.05), rng.uniform(-5, 5)
        channels = rng.uniform(.97, 1.03, size=(1, 1, 3))
        he = np.clip(((he.astype(float)-127.5)*contrast+127.5+brightness)*channels, 0, 255).astype(np.uint8)
    label[~tissue] = IGNORE
    he[~tissue] = 0
    return he, label.transpose(2, 0, 1).copy() if multilabel else label, tissue


class PixelDataset(Dataset):
    def __init__(self, manifest, stats, root, size=256, training=False, policy="multilabel", seed=0, cache_dir=None):
        columns = {"patch_id", "image_path", "target_path", *(f"{name}_valid" for name in CHANNELS)}
        self.df = pd.read_csv(manifest, usecols=lambda name: name in columns)
        if not len(self.df):
            raise ValueError(f"Empty split: {manifest}")
        self.stats = json.loads(Path(stats).read_text()) if isinstance(stats, (str, Path)) else stats
        self.root, self.size, self.training, self.policy = Path(root), size, training, policy
        self.seed = seed
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.cache = None
        if cache_dir is not None:
            from .cache import NativeCache
            self.cache = NativeCache(cache_dir, Path(manifest).parent)

    @property
    def epoch(self):
        return int(self._epoch.item())

    @epoch.setter
    def epoch(self, value):
        self._epoch.fill_(value)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        r = self.df.iloc[index]
        if self.cache is not None:
            he, label, tissue = self.cache.read(int(r.patch_id))
        else:
            he = read_he(self.root / r.image_path)
            mif = read_mif(self.root / r.target_path)
            tissue = tissue_mask(he)
            channel_valid = np.array([valid_flag(r.get(f"{name}_valid", True)) for name in CHANNELS])
            label, _ = labels_from_mif(mif, tissue, self.stats["q"], self.policy,
                                       channel_valid=channel_valid)
        he[~tissue] = 0
        if self.training:
            he, label, tissue = augment(he, label, tissue, np.random.default_rng(self.seed + self.epoch*1000003 + index))
        he = cv2.resize(he, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if label.ndim == 3:
            label = np.stack([cv2.resize(y, (self.size, self.size), interpolation=cv2.INTER_NEAREST) for y in label])
        else:
            label = cv2.resize(label, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        tissue = cv2.resize(tissue.astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST).astype(bool)
        x = he.astype(np.float32) / 255
        x = (x - np.array(IMAGENET_STATS["mean"], np.float32)) / np.array(IMAGENET_STATS["std"], np.float32)
        x[~tissue] = 0  # also zero after normalization
        if label.ndim == 3:
            label[:, ~tissue] = IGNORE
        else:
            label[~tissue] = IGNORE
        return {"image": torch.from_numpy(x.transpose(2, 0, 1).copy()),
                "label": torch.from_numpy(label.astype(np.uint8, copy=False)),
                "tissue": torch.from_numpy(tissue), "index": index}
