"""Materialize all binary labels with separate masks; never modifies source data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zlib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vit_seg.binary_targets import BinaryTargetStore


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--shard-size',type=int,default=2048)
    a=p.parse_args();source=Path(a.source).resolve();out=Path(a.out).resolve();targets=out/'targets'
    if a.workers<1 or a.shard_size<1:raise ValueError('workers/shard-size must be positive')
    targets.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'export.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache=source/'cache';meta=json.loads((cache/'COMPLETE.json').read_text())
    n,h,w=meta['count'],meta['height'],meta['width'];pixels=h*w
    spec=dict(format='binary_expression_zlib_v1',count=n,height=h,width=w,channels=16,
        shard_size=a.shard_size,source_cache=str(cache),source_cache_metadata_sha256=sha(cache/'COMPLETE.json'),
        source_manifest_sha256=sha(source/'data/patch_manifest.csv'),
        target_values=[0,1],validity='separate observed-channel and eligible-pixel masks',
        threshold='processed GT > 0; existing source semantics preserved; no new StarDist filtering')
    spec_path=targets/'metadata.json'
    if spec_path.exists() and json.loads(spec_path.read_text())!=spec:raise ValueError('Existing export differs from source')
    write_json(spec_path,spec)
    if (targets/'COMPLETE.json').exists() and (out/'BINARY_DATASET_COMPLETE.json').exists():
        BinaryTargetStore(targets);print('Already complete; preserving dataset');return
    raw=np.memmap(cache/'records.raw',mode='r',dtype=np.uint8,shape=(n,meta['record_bytes']))
    index=np.zeros((n,2),np.uint64);started=time.monotonic();completed=0
    def convert(i):
        row=raw[i]
        if int(np.frombuffer(row[:8],'<i8')[0])!=i or zlib.crc32(row[:-4])!=int(np.frombuffer(row[-4:],'<u4')[0]):
            raise ValueError(f'Source cache identity/CRC failed: {i}')
        # Cache already encodes GT>0 as bits. Extract independent binary targets;
        # never apply >0 to a decoded 255 ignore sentinel.
        payload=row[:10].tobytes()+row[10+3*pixels:-4].tobytes()
        compressed=zlib.compress(payload,1)
        if zlib.decompress(compressed)!=payload:raise ValueError('Compression roundtrip failed')
        return i,compressed
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for shard,start in enumerate(range(0,n,a.shard_size)):
            end=min(n,start+a.shard_size);dest=targets/f'shard_{shard:05d}.bin'
            receipt=dest.with_suffix('.json');ix=dest.with_suffix('.npy')
            if receipt.exists():
                info=json.loads(receipt.read_text())
                if info['start']!=start or info['end']!=end or sha(dest)!=info['sha256']:
                    raise ValueError('Existing shard corrupted')
                index[start:end]=np.load(ix);completed=end;continue
            if shutil.disk_usage(out).free<5*2**30:raise OSError('Keep at least 5 GiB free; export is resumable')
            temporary=dest.with_suffix('.tmp');offset=0;hasher=hashlib.sha256()
            with temporary.open('wb') as f:
                # Bound memory: submit only 64 native targets at a time.
                for block in range(start,end,64):
                    for i,blob in pool.map(convert,range(block,min(end,block+64))):
                        f.write(blob);hasher.update(blob);index[i]=[offset,len(blob)];offset+=len(blob)
                f.flush();os.fsync(f.fileno())
            temporary.replace(dest);np.save(ix,index[start:end])
            write_json(receipt,dict(start=start,end=end,sha256=hasher.hexdigest(),bytes=offset))
            completed=end;elapsed=time.monotonic()-started
            progress=dict(completed=completed,total=n,elapsed_seconds=elapsed,records_per_second=completed/max(elapsed,.001),
                stage='exporting_binary_targets',original_data_unchanged=True)
            write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    np.save(targets/'index.npy',index)
    all_shards=[json.loads(p.read_text()) for p in sorted(targets.glob('shard_*.json'))]
    complete=dict(spec,shards=all_shards,index_sha256=sha(targets/'index.npy'),
        target_bytes=sum(x['bytes'] for x in all_shards),all_records_source_crc_verified=True,
        all_records_compression_roundtrip_verified=True)
    write_json(targets/'COMPLETE.json',complete)
    # Decode widely spaced records through the public reader and compare every
    # target/mask pixel with the original packed representation.
    store=BinaryTargetStore(targets)
    for i in np.linspace(0,n-1,min(512,n),dtype=int):
        target,valid,tissue=store.read(i);row=raw[i]
        original=np.unpackbits(row[10+3*pixels:10+5*pixels].reshape(2,h,w),axis=0)
        assert np.array_equal(target,original) and set(np.unique(target))<={0,1}
        assert not target[~valid].any()
    for path in (source/'data').rglob('*'):
        relative=path.relative_to(source/'data');dest=out/relative
        if path.is_dir():dest.mkdir(exist_ok=True,parents=True)
        elif path.suffix in ('.csv','.json') and not dest.exists():
            # Copy metadata so future validation cannot write into the source run.
            dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(path,dest)
    write_json(out/'binary_targets.json',dict(folder=str(targets),manifest_sha256=sha(targets/'COMPLETE.json'),
        labels='uint8 0/1, lossless packed storage; use BinaryTargetStore.read',valid_mask='separate bool mask',
        original_target_path_column='provenance only; binary loader overrides intensity targets'))
    write_json(out/'BINARY_DATASET_COMPLETE.json',dict(patches=n,binary_target_values=[0,1],ignore_is_separate=True,
        native_shape=[h,w],source_run=str(source),no_patch_removed=True,patient_splits_unchanged=True,
        independent_binary_target_storage=True,decoded_records_verified=min(512,n),seconds=time.monotonic()-started,
        target_manifest_sha256=sha(targets/'COMPLETE.json')))
    write_json(out/'progress.json',dict(stage='complete',completed=n,total=n,seconds=time.monotonic()-started))
    print('BINARY DATASET COMPLETE',out,flush=True)


if __name__=='__main__':main()
