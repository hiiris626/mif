"""Migrate cached inventory after the CRC33 patient-identity correction.

Reuses all original patch labels. Only rereads removed-training patches to
subtract their intensity histograms; no StarDist or new data cleaning.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from vit_seg.prepare import source_table, finalize
from vit_seg.data import CHANNELS, read_mif
from vit_seg.cache import NativeCache, identity
from vit_seg.validate import validate


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',default='runs/ddp_baseline')
    p.add_argument('--out',default='runs/ddp_positive_dice');args=p.parse_args()
    base,out=Path(args.baseline).resolve(),Path(args.out).resolve()
    if out.exists():raise ValueError('Choose a new run directory; migration does not overwrite runs')
    (out/'data').mkdir(parents=True)
    cfg=json.loads((PROJECT/'configs/train.json').read_text());cfg['dice_scope']='positive_only'
    cfg['patient_identity']='CRC33_sections_grouped_v2'
    (PROJECT/'configs/train_positive_dice.json').write_text(json.dumps(cfg,indent=2))
    new=source_table(cfg['data_root'],cfg['split_policy'],cfg['split_seed'])
    old=pd.read_csv(base/'data/source_manifest.csv')
    assert np.array_equal(old.patch_id,new.patch_id)
    changed=old[old.split.to_numpy()!=new.split.to_numpy()].copy()
    assert set(changed.orion_slide_id)=={'CRC33_01'} and len(changed)==3065
    assert set(changed.split)=={'train'}
    assert set(new.set_index('patch_id').loc[changed.patch_id,'split'])=={'test'}
    old_cache=NativeCache(base/'cache',base/'data')
    hist_delta=np.zeros((8,16,256),np.int64)
    start=time.monotonic()
    def histogram(row):
        _,_,tissue=old_cache.read(int(row['patch_id']))
        mif=read_mif(Path(cfg['data_root'])/row['target_path'])
        h=np.stack([np.bincount(a[(a>0)&tissue],minlength=256) for a in mif])
        return int(row['patch_id'])%8,h
    # Initialize memmap before sharing its read-only records among threads.
    old_cache.read(0)
    with ThreadPoolExecutor(4) as pool:
        for i,(shard,h) in enumerate(pool.map(histogram,changed.to_dict('records'))):
            hist_delta[shard]+=h
            if i%100==0:print(json.dumps(dict(histograms=i+1,total=len(changed),seconds=time.monotonic()-start)),flush=True)
    mapping=new.set_index('patch_id')[['split','patient_id']].to_dict('index')
    for shard in range(8):
        db=sqlite3.connect(f'file:{base}/data/shard_{shard}.sqlite?mode=ro',uri=True)
        target=sqlite3.connect(out/f'data/shard_{shard}.sqlite');db.backup(target);db.close()
        hist=np.frombuffer(target.execute("SELECT value FROM metadata WHERE key='hist'").fetchone()[0],np.int64).reshape(16,256).copy()
        hist-=hist_delta[shard]
        if (hist<0).any():raise ValueError('Histogram subtraction underflow')
        spec=json.loads(target.execute("SELECT value FROM metadata WHERE key='spec'").fetchone()[0])
        spec['signature']=hashlib.sha256(new[new.patch_id%8==shard].to_csv(index=False).encode()).hexdigest()
        spec['patient_identity']='CRC33_sections_grouped_v2'
        records=[]
        for patch,record in target.execute('SELECT id,record FROM patches'):
            row=json.loads(record);row.update(mapping[patch]);records.append((json.dumps(row),patch))
        with target:
            target.executemany('UPDATE patches SET record=? WHERE id=?',records)
            target.execute("UPDATE metadata SET value=? WHERE key='hist'",(hist.tobytes(),))
            target.execute("UPDATE metadata SET value=? WHERE key='spec'",(json.dumps(spec),))
        target.close()
        print('migrated shard',shard,flush=True)
    new.to_csv(out/'data/source_manifest.csv',index=False)
    finalize(argparse.Namespace(out=str(out/'data'),shards=8,seed=42))
    receipt=validate(out/'data')
    # Verify every field that determines cached inputs/targets before reusing bytes.
    columns=['patch_id','image_path','target_path','tissue_pixels',*[f'{c}_{suffix}' for c in CHANNELS for suffix in ['valid','pixels']]]
    previous=pd.read_csv(base/'data/patch_manifest.csv',usecols=columns).sort_values('patch_id')
    current=pd.read_csv(out/'data/patch_manifest.csv',usecols=columns).sort_values('patch_id')
    pd.testing.assert_frame_equal(previous,current)
    (out/'cache').mkdir()
    os.link(base/'cache/records.raw',out/'cache/records.raw')
    meta=dict(old_cache.meta,source_identity=identity(out/'data'))
    (out/'cache/COMPLETE.json').write_text(json.dumps(meta,indent=2))
    check=NativeCache(out/'cache',out/'data')
    for patch in np.linspace(0,len(new)-1,64,dtype=int):
        a,b=old_cache.read(int(patch)),check.read(int(patch))
        assert all(np.array_equal(x,y) for x,y in zip(a,b))
    (out/'migration.json').write_text(json.dumps(dict(baseline=str(base),moved_patches=len(changed),
        moved_case='CRC33_01 train -> test, joining CRC33_02',patient_counts=receipt['patient_counts'],
        labels_unchanged=True,cache_hardlink=True,histogram_subtraction_verified=True,
        validation=receipt,seconds=time.monotonic()-start),indent=2))
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
