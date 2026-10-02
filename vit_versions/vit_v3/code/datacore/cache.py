"""缓存读取器：兼容两种历史格式的 *.raw memmap 缓存。

背景：build_cache.py 生成 train/val/test_{he,mif}.raw。
早期文件是裸 memmap（无文件头），后加过 NPY 头（np.save 写出）；
本模块自动探测（读前 6 字节是否为 NPY magic），并做形状/dtype/大小校验，
避免误读或读到截断文件。使用方式：open_cache(path, channels, tile_size, count)。
"""
import os

import numpy as np


def open_cache(path, channels, tile_size, count=None, mode="r"):
    """打开并校验缓存文件，返回可索引的 [N, C, tile_size, tile_size] uint8 数组（memmap）。

    - 自动识别 NPY 格式（带文件头）与裸 raw 格式（无头，按文件大小推算行数）；
    - count 给定时校验行数与 dataframe 一致（防呆）。
    """
    with open(path, "rb") as stream:
        is_npy = stream.read(6) == b"\x93NUMPY"
    if is_npy:
        array = np.lib.format.open_memmap(path, mode=mode)
        if array.dtype != np.uint8 or array.ndim != 4 or array.shape[1:] != (channels, tile_size, tile_size):
            raise ValueError(f"Invalid cache shape/dtype: {path}: {array.shape}, {array.dtype}")
        if os.path.getsize(path) != array.offset + array.nbytes:
            raise ValueError(f"Invalid cache file size: {path}")
    else:
        row_bytes = channels * tile_size * tile_size
        size = os.path.getsize(path)
        if size == 0 or size % row_bytes:
            raise ValueError(f"Invalid raw cache file size: {path}")
        array = np.memmap(path, dtype=np.uint8, mode=mode,
                          shape=(size // row_bytes, channels, tile_size, tile_size))
    if count is not None and len(array) != count:
        raise ValueError(f"Cache row count mismatch: {path}: {len(array)} != {count}")
    return array
