"""Lossless native-resolution training cache, built once on the run filesystem.

RGB bytes, 16 packed expression bits, tissue/supervision bits and channel
availability preserve the original label and augmentation semantics exactly.
Each fixed-size record has an identity and CRC. No TIFF is read during training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import zlib
import cv2
import numpy as np
import pandas as pd
from .artifacts import file_hash
from .data import CHANNELS, IGNORE, read_he, read_mif, tissue_mask, labels_from_mif, valid_flag


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def identity(data):
    return {name: file_hash(Path(data)/name) for name in ('patch_manifest.csv', 'statistics.json')}


def encode(patch_id, he, labels, tissue, channel_valid):
    eligible = (labels != IGNORE).any(0)
    he = he.copy(); he[~tissue] = 0
    record = np.concatenate((np.asarray([patch_id], dtype='<i8').view(np.uint8),
        np.packbits(channel_valid), he.ravel(), np.packbits(labels == 1, axis=0).ravel(),
        np.packbits(np.stack((tissue, eligible)).ravel())))
    crc = np.asarray([zlib.crc32(record)], dtype='<u4').view(np.uint8)
    return np.concatenate((record, crc))


class NativeCache:
    def __init__(self, folder, data=None):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder/'COMPLETE.json').read_text())
        if self.meta['format'] != 'native_packed_v1':
            raise ValueError('Unsupported training cache format')
        if data is not None and self.meta['source_identity'] != identity(data):
            raise ValueError('Training cache source/label statistics changed')
        self.shape = (self.meta['count'], self.meta['record_bytes'])
        if (self.folder/'records.raw').stat().st_size != int(np.prod(self.shape)):
            raise ValueError('Truncated training cache')
        self.records = None

    def __getstate__(self):
        state = self.__dict__.copy(); state['records'] = None
        return state

    def read(self, patch_id):
        if not 0 <= patch_id < self.shape[0]:
            raise ValueError('Patch ID outside training cache')
        if self.records is None:
            self.records = np.memmap(self.folder/'records.raw', mode='r', dtype=np.uint8, shape=self.shape)
        record = self.records[patch_id]
        stored_id = int(np.frombuffer(record[:8], dtype='<i8')[0])
        crc = int(np.frombuffer(record[-4:], dtype='<u4')[0])
        if stored_id != patch_id or zlib.crc32(record[:-4]) != crc:
            raise ValueError(f'Corrupt training cache record: {patch_id}')
        h, w = self.meta['height'], self.meta['width']; pixels = h*w
        channels = np.unpackbits(record[8:10], count=16).astype(bool)
        he = record[10:10+3*pixels].reshape(h, w, 3).copy()
        labels = np.unpackbits(record[10+3*pixels:10+5*pixels].reshape(2,h,w), axis=0)
        masks = np.unpackbits(record[10+5*pixels:-4], count=2*pixels).reshape(2,h,w).astype(bool)
        tissue, eligible = masks
        labels[:, ~eligible] = IGNORE
        labels[~channels] = IGNORE
        return he, labels, tissue


def build(data, output, workers=8, reserve_gib=15.):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'COMPLETE.json').exists():
            NativeCache(output, data)
            print('Verified complete native training cache; reusing it.', flush=True)
            return
        cv2.setNumThreads(1)
        columns = ['patch_id','image_path','target_path',*[f'{c}_valid' for c in CHANNELS]]
        frame = pd.read_csv(data/'patch_manifest.csv', usecols=lambda c: c in columns).sort_values('patch_id')
        if not np.array_equal(frame.patch_id.to_numpy(), np.arange(len(frame))):
            raise ValueError('Cache requires complete contiguous global patch IDs')
        stats = json.loads((data/'statistics.json').read_text())
        root = Path(stats['source_spec']['root'])
        first = read_he(root/frame.iloc[0].image_path)
        h, w = first.shape[:2]; pixels = h*w
        width = 8+2+5*pixels+(2*pixels+7)//8+4
        spec = dict(format='native_packed_v1', count=len(frame), height=h, width=w,
                    record_bytes=width, source_identity=identity(data), source_root=str(root.resolve()),
                    labels='native_positive_gt_observed_channels_HE_tissue_v1')
        spec_path = output/'metadata.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError('Cache source or geometry changed; use a new cache directory')
        total_bytes = len(frame)*width
        raw = output/'records.raw'; done_path = output/'completed.raw'
        allocated = raw.stat().st_blocks*512 if raw.exists() else 0
        needed = max(0, total_bytes-allocated)+reserve_gib*2**30
        if shutil.disk_usage(output).free < needed:
            raise OSError(f'Cache needs {needed/2**30:.1f} GiB free including reserve')
        if not raw.exists():
            with raw.open('wb') as stream: stream.truncate(total_bytes)
        elif raw.stat().st_size != total_bytes:
            raise ValueError('Invalid partial cache length')
        if not done_path.exists():
            with done_path.open('wb') as stream: stream.truncate(len(frame))
        if done_path.stat().st_size != len(frame):
            raise ValueError('Invalid cache completion bitmap')
        atomic_json(spec_path, spec)
        records = np.memmap(raw, dtype=np.uint8, mode='r+', shape=(len(frame),width))
        done = np.memmap(done_path, dtype=np.uint8, mode='r+', shape=(len(frame),))
        pending = frame.loc[np.asarray(done) != 1].to_dict('records')
        initial = int((done == 1).sum()); started = time.monotonic()
        print(json.dumps(dict(cache_bytes=total_bytes, total=len(frame), completed=initial, workers=workers)), flush=True)
        def load(row):
            he = read_he(root/row['image_path']); mif = read_mif(root/row['target_path'])
            if he.shape[:2] != (h,w) or mif.shape[1:] != (h,w):
                raise ValueError(f'Unexpected native geometry: {row["patch_id"]}')
            tissue = tissue_mask(he)
            available = np.array([valid_flag(row.get(f'{c}_valid', True)) for c in CHANNELS])
            labels, _ = labels_from_mif(mif, tissue, stats['q'], channel_valid=available)
            return row['patch_id'], encode(row['patch_id'], he, labels, tissue, available)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0,len(pending),256):
                ids = []
                for patch_id, record in pool.map(load, pending[offset:offset+256]):
                    records[patch_id] = record; ids.append(patch_id)
                # Record writes must be durable before marking this chunk complete.
                records.flush(); done[ids] = 1; done.flush()
                count = initial+min(offset+256,len(pending)); elapsed = time.monotonic()-started
                speed = (count-initial)/max(elapsed,1e-6)
                progress = dict(completed=count,total=len(frame),patches_per_second=speed,
                                elapsed_seconds=elapsed,remaining_seconds=(len(frame)-count)/max(speed,1e-6))
                atomic_json(output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
        if not (done == 1).all():
            raise RuntimeError('Incomplete native cache')
        records.flush(); done.flush()
        atomic_json(output/'COMPLETE.json', spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True); parser.add_argument('--out',required=True)
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    if args.workers < 1: parser.error('workers must be positive')
    build(args.data,args.out,args.workers)
