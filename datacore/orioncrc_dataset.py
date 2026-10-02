"""OrionCRC 统一数据集加载器（读取 MIPHEI 处理版 /data/weiyh/orioncrc_miphei）。

MIPHEI Zenodo 处理版解压后的结构：
  ORIONCRC_dataset_tile_20x/
  ├── he/                H&E tile (JPEG)
  ├── if/                16 通道 mIF tile (8-bit TIFF, 已去AF+归一化)
  ├── nuclei/            核掩码 (label TIFF)
  ├── csv_nuclei_pos/    单细胞表型 CSV
  ├── slide_dataframe.csv
  ├── train_dataframe.csv / val_dataframe.csv / test_dataframe.csv
      每行: slide_name, image_path, target_path, nuclei_path

本 loader 按官方划分（train/val/test_dataframe.csv）返回配对 H&E(3ch)+mIF(16ch)。
归一化：
  - H&E：按编码器官方 mean/std（用 encoder_stats 参数切换，默认 Virchow2/ImageNet）
  - mIF（线性域，v1/v2）：缩放到 [-0.9, 0.9]（配合 Tanh，MIPHEI 设定）
  - mIF（log 域，v3）：per-marker log 归一化 x0 = 2·log2(min(y,q_c)/q_c+1)-1 ∈ [-1,1]，
    分位 q 来自 marker_q.json（前景 99.9 分位，与 I2SB v2 同口径）

数据读取优先级：memmap 缓存（build_cache.py 生成，须有 {split}.done 标记）
  > 直接解码（cv2 读 JPEG + tifffile 读 17ch TIFF → 剔除 PD-1 → resize 到 tile_size）。
"""
import os
from datacore.cache import open_cache
import json
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
import cv2
from PIL import Image

DEFAULT_ROOT = "/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x"
CACHE_DIR = "/data/weiyh/orioncrc_cache"

IMAGENET_STATS = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# Virchow2 与 ImageNet 相同（见 virchow2_config.json）
ENCODER_STATS = {
    "imagenet": IMAGENET_STATS,
    "virchow2": IMAGENET_STATS,
    "hoptimus": dict(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
    "pix2pix": dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # [-1,1]
}

# 16 通道 marker 名称（MIPHEI 论文口径；从 17 通道 TIFF 中剔除 PD-1 后得到）
CHANNELS = [
    "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO",
    "CD20", "PDL1", "CD3e", "CD163", "ECadherin", "Ki67", "Pan-CK", "SMA",
]

# 原始 TIFF 的 17 通道（OME 元数据顺序），PD-1 在 index 13，信号差弃用（同 MIPHEI）
MIF_FULL_CHANNELS = [
    "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO",
    "CD20", "PDL1", "CD3e", "CD163", "ECadherin", "PD-1", "Ki67", "Pan-CK", "SMA",
]
MIF_SELECT = [i for i in range(len(MIF_FULL_CHANNELS)) if i != 13]  # 去掉 PD-1
NUM_MIF_CHANNELS = len(CHANNELS)
TARGET_RANGE = 0.9  # mIF 目标缩放到 [-0.9, 0.9]（线性域）
LOG2 = float(np.log(2.0))


def validate_marker_q(q):
    q = np.asarray(q, dtype=np.float32)
    if q.shape != (NUM_MIF_CHANNELS,) or not np.all(np.isfinite(q) & (q > 0)):
        raise ValueError("marker_q must contain 16 finite, positive values")
    return q


def load_marker_q(q_file=None):
    """per-marker log 归一化分位参数 q_c（uint8 尺度，前景 99.9 分位）。

    来自 compute_q.py 生成于 256 缓存目录；强度分位与分辨率无关，
    512 训练同样复用（与 I2SB v2 / pix2pixHD v2 同一口径）。
    """
    p = q_file or os.path.join(CACHE_DIR, "marker_q.json")
    if os.path.exists(p):
        with open(p) as f:
            return validate_marker_q(json.load(f)["q"])
    return None


def normalize_mif_log(y, q):
    """per-marker log 归一化（论文公式(2) 的 log2 变体）：

        x0 = 2 * log2(min(y, q_c) / q_c + 1) - 1   ∈ [-1, 1]

    y: [C,H,W] float32 (0-255)；q: [C] float32 (uint8 尺度)。
    背景 0 → -1，y=q_c → +1（与 Tanh 输出范围完全匹配）。
    log 域把稀疏免疫 marker 的弱阳性信号从背景附近"拉开"，
    避免线性域 MSE 回归把预测峰值压向 0。
    """
    q = torch.as_tensor(q, dtype=y.dtype, device=y.device).view(-1, 1, 1)
    return torch.clamp(torch.log2(torch.minimum(y, q) / q + 1.0) * 2.0 - 1.0, -1.0, 1.0)


def denormalize_mif_np(x0, q):
    """log 域 [-1,1] -> 0-255 的 numpy 反变换（与 normalize_mif_log 互逆）。

    mif = q_c * (exp((x0 + 1)/2 * ln2) - 1)，x0=1 时 mif=q_c。
    """
    x0 = np.asarray(x0, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32).reshape(-1, 1, 1)
    mif = q * (np.exp((x0 + 1.0) / 2.0 * LOG2) - 1.0)
    return np.clip(mif, 0.0, 255.0)


class OrionCRCDataset(Dataset):
    """OrionCRC tile 级数据集：返回配对的 H&E(3ch) 与 mIF(16ch) 张量。

    参数：
        split: train/val/test（按官方 *_dataframe.csv 划分，行数 295096/12402/10952）
        root: ORIONCRC_dataset_tile_20x 根目录
        tile_size: 输出边长（256/512；原生 333px tile 会 resize，INTER_AREA）
        encoder_stats: H&E 归一化统计（virchow2/imagenet/hoptimus/pix2pix）
        aug: 是否随机翻转增广（仅训练集用）
        return_nuclei: 是否额外返回核 label mask（int64，最近邻取样保持整数 id）
        use_cache: 是否尝试读 memmap 缓存（命中且带 .done 标记才启用）
        log_norm: mIF 目标域是否用 per-marker log 归一化（v3 起；需 marker_q.json）
        cache_dir: 手动指定缓存目录（默认按 tile_size 自动选：256→orioncrc_cache，512→_512）
        return_target: 是否读 mIF 目标；纯推理时可关掉省 I/O

    __getitem__ 返回：
        (H&E, mIF) 或 (H&E, mIF, nuclei)；return_target=False 时只返回 H&E。
        H&E: [3,H,W] float32（编码器 mean/std 归一化）；
        mIF: [16,H,W] float32（线性 [-0.9,0.9] 或 log 域 [-1,1]）。
    """

    def __init__(self, split="train", root=DEFAULT_ROOT, tile_size=256,
                 encoder_stats="imagenet", aug=False, return_nuclei=False, use_cache=True,
                 log_norm=False, cache_dir=None, return_target=True, q_file=None):
        assert split in ("train", "val", "test")
        self.split = split
        self.root = root
        self.tile_size = tile_size
        self.aug = aug
        self.return_nuclei = return_nuclei
        self.return_target = return_target
        self.log_norm = log_norm
        self.q = None
        if log_norm:
            self.q = load_marker_q(q_file)
            if self.q is None:
                raise FileNotFoundError("log_norm=True requires marker_q.json")
        st = ENCODER_STATS[encoder_stats]
        self.he_mean = torch.tensor(st["mean"]).view(3, 1, 1)
        self.he_std = torch.tensor(st["std"]).view(3, 1, 1)

        df_path = os.path.join(root, f"{split}_dataframe.csv")
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"缺少划分文件 {df_path}，不能用全量数据替代 {split} 集")
        else:
            self.df = pd.read_csv(df_path)
        # 路径列名兼容：image_path/target_path 或 he_path/if_path
        self.he_col = "image_path" if "image_path" in self.df.columns else "he_path"
        self.if_col = "target_path" if "target_path" in self.df.columns else "if_path"

        # memmap 缓存（build_cache.py 构建）：命中则不再逐 tile 解码
        # 缓存目录按 tile_size 区分（256 用默认，512 用 _512），避免不同分辨率互相覆盖
        self.cache_dir = cache_dir or (CACHE_DIR if tile_size == 256 else f"{CACHE_DIR}_{tile_size}")
        self.he_cache = self.mif_cache = None
        if use_cache and (cache_dir is not None or os.path.realpath(root) == os.path.realpath(DEFAULT_ROOT)):
            self._try_load_cache()

    def _build_from_scan(self):
        he = sorted(glob.glob(os.path.join(self.root, "he", "*")))
        base = [os.path.basename(x) for x in he]
        # IF 与 H&E 同名前缀（.jpeg <-> .tiff），用 basename 去掉扩展名匹配
        rows = []
        for h in he:
            key = os.path.splitext(os.path.basename(h))[0]
            cand = os.path.join(self.root, "if", key + ".tiff")
            if os.path.exists(cand):
                rows.append({"image_path": h, "target_path": cand})
        return pd.DataFrame(rows)

    def __len__(self):
        return len(self.df)

    def _resolve(self, path):
        """划分文件里的路径可能是相对根目录的（如 he/xxx.jpeg），解析为绝对路径。"""
        if os.path.isabs(path):
            return path
        return os.path.join(self.root, path)

    def _load_he(self, path):
        """cv2 快速读取 + 缩放（比 PIL 快一个量级）。"""
        b = cv2.imread(self._resolve(path), cv2.IMREAD_COLOR)  # BGR
        if b is None:
            raise FileNotFoundError(path)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        if b.shape[:2] != (self.tile_size, self.tile_size):
            b = cv2.resize(b, (self.tile_size, self.tile_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(b.transpose(2, 0, 1).astype(np.float32).copy())
        return (x / 255.0 - self.he_mean) / self.he_std

    def _normalize_target(self, y):
        """mIF uint8 尺度 -> 训练目标域（线性 [-0.9,0.9] 或 log [-1,1]）。"""
        if self.log_norm and self.q is not None:
            return normalize_mif_log(y, torch.from_numpy(self.q))
        return y / 255.0 * (2 * TARGET_RANGE) - TARGET_RANGE

    def _load_mif(self, path):
        """tifffile(maxworkers) 读取 + 逐通道 cv2 缩放 + 剔除 PD-1。"""
        a = tifffile.imread(self._resolve(path), maxworkers=8)
        if a.ndim == 2:
            a = a[:, :, None]
        if a.ndim == 3 and a.shape[2] not in (NUM_MIF_CHANNELS, len(MIF_FULL_CHANNELS)):
            a = a.transpose(1, 2, 0)  # C,H,W -> H,W,C
        if a.shape[2] == len(MIF_FULL_CHANNELS):  # 17 -> 16，剔除 PD-1
            a = a[:, :, MIF_SELECT]
        if a.ndim != 3 or a.shape[2] != 16:
            raise ValueError(f"Expected 16/17-channel mIF TIFF: {path}, shape={a.shape}")
        if a.shape[0] != self.tile_size or a.shape[1] != self.tile_size:
            out = np.empty((self.tile_size, self.tile_size, a.shape[2]), dtype=np.uint8)
            for c in range(a.shape[2]):
                out[:, :, c] = cv2.resize(a[:, :, c], (self.tile_size, self.tile_size),
                                          interpolation=cv2.INTER_AREA)
            a = out
        y = torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32).copy())
        return self._normalize_target(y)

    def _try_load_cache(self):
        """命中 build_cache.py 生成的 memmap 且带 .done 标记才启用缓存。"""
        he_path = os.path.join(self.cache_dir, f"{self.split}_he.raw")
        mif_path = os.path.join(self.cache_dir, f"{self.split}_mif.raw")
        done_marker = os.path.join(self.cache_dir, f"{self.split}.done")
        n = len(self.df)
        if not os.path.exists(done_marker):
            return False
        try:
            he = open_cache(he_path, 3, self.tile_size, n)
            mif = open_cache(mif_path, 16, self.tile_size, n)
        except (OSError, ValueError) as exc:
            print(f"[cache] 忽略无效缓存: {exc}")
            return False
        self.he_cache, self.mif_cache = he, mif
        print(f"[cache] {self.split}: 使用 memmap 缓存 ({n} tiles)")
        return True

    def _he_from_cache(self, i):
        x = torch.from_numpy(self.he_cache[i].astype(np.float32))  # (3,H,W)
        return (x / 255.0 - self.he_mean) / self.he_std

    def _mif_from_cache(self, i):
        y = torch.from_numpy(self.mif_cache[i].astype(np.float32))  # (16,H,W)
        return self._normalize_target(y)

    def __getitem__(self, i):
        if self.he_cache is not None:
            x = self._he_from_cache(i)
            y = self._mif_from_cache(i) if self.return_target else None
        else:
            row = self.df.iloc[i]
            x = self._load_he(row[self.he_col])
            y = self._load_mif(row[self.if_col]) if self.return_target else None
        n = None
        if self.return_nuclei:
            if "nuclei_path" not in self.df.columns:
                raise ValueError("return_nuclei requires nuclei_path")
            labels = tifffile.imread(self._resolve(self.df.iloc[i]["nuclei_path"])).squeeze()
            if labels.ndim != 2:
                raise ValueError("Nuclei labels must be a 2D image")
            # Index selection preserves integer IDs without interpolation/float rounding.
            rows = np.minimum(np.arange(self.tile_size) * labels.shape[0] // self.tile_size, labels.shape[0] - 1)
            cols = np.minimum(np.arange(self.tile_size) * labels.shape[1] // self.tile_size, labels.shape[1] - 1)
            n = torch.from_numpy(labels[np.ix_(rows, cols)].astype(np.int64))
        if self.aug:
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [2])
                if y is not None:
                    y = torch.flip(y, [2])
                if n is not None:
                    n = torch.flip(n, [1])
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [1])
                if y is not None:
                    y = torch.flip(y, [1])
                if n is not None:
                    n = torch.flip(n, [0])
        if n is not None:
            return x, y, n.numpy()
        if y is None:
            return x
        return x, y


def build_dataloaders(root=DEFAULT_ROOT, batch_size=16, tile_size=256, encoder_stats="imagenet",
                      num_workers=8, aug_train=True, log_norm=False):
    """一次性构建 train/val/test 三个 DataLoader（训练集 shuffle+增广，验证/测试不 shuffle）。"""
    from torch.utils.data import DataLoader
    tr = OrionCRCDataset("train", root, tile_size, encoder_stats, aug=aug_train, log_norm=log_norm)
    va = OrionCRCDataset("val", root, tile_size, encoder_stats, aug=False, log_norm=log_norm)
    te = OrionCRCDataset("test", root, tile_size, encoder_stats, aug=False, log_norm=log_norm)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(tr, shuffle=True, **kw),
        DataLoader(va, shuffle=False, **kw),
        DataLoader(te, shuffle=False, **kw),
    )


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    ds = OrionCRCDataset("train", root=root, aug=False)
    x, y = ds[0]
    print(f"train 样本数: {len(ds)}")
    print(f"H&E:  {tuple(x.shape)} 范围[{x.min():.2f},{x.max():.2f}]")
    print(f"mIF:  {tuple(y.shape)} 范围[{y.min():.2f},{y.max():.2f}]")
    print("列:", list(ds.df.columns))
