"""Count actual 256-grid supervision from immutable packed cache, training only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vit_seg.data import CHANNELS
from vit_seg.cache import NativeCache
from vit_seg.artifacts import file_hash
from vit_seg.prepare import patient_id


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();source=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    columns=['patch_id','orion_slide_id','general_qc_status',*[c+'_pixels' for c in CHANNELS]]
    df=pd.read_csv(source/'data/train.csv',usecols=columns)
    cache=NativeCache(source/'cache',source/'data');cache.read(0)
    h,w=cache.meta['height'],cache.meta['width'];pixels=h*w
    yy=(np.arange(256)*h//256);xx=(np.arange(256)*w//256)
    bits=np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1).astype(np.int64)
    def count(patch):
        record=cache.records[int(patch)]
        if int(np.frombuffer(record[:8],dtype='<i8')[0])!=int(patch):raise ValueError('Bad record ID')
        observed=np.unpackbits(record[8:10],count=16).astype(bool)
        packed=record[10+3*pixels:10+5*pixels].reshape(2,h,w)[:,yy][:,:,xx]
        masks=np.unpackbits(record[10+5*pixels:-4],count=2*pixels).reshape(2,h,w)[:,yy][:,:,xx].astype(bool)
        valid=masks[0]&masks[1]
        hist=np.stack([np.bincount(a[valid],minlength=256) for a in packed])
        positive=(hist@bits).reshape(16).astype(np.uint32)
        positive[~observed]=0
        return positive,(int(valid.sum())*observed).astype(np.uint32),int(masks[0].sum())
    positives=np.zeros((len(df),16),np.uint32);valids=np.zeros_like(positives);tissues=np.zeros(len(df),np.uint32)
    started=time.monotonic()
    with ThreadPoolExecutor(4) as pool:
        for start in range(0,len(df),1024):
            for offset,(pos,valid,tissue) in enumerate(pool.map(count,df.patch_id.iloc[start:start+1024])):
                positives[start+offset]=pos;valids[start+offset]=valid;tissues[start+offset]=tissue
            if start%10240==0:print(json.dumps(dict(done=min(start+1024,len(df)),total=len(df),seconds=time.monotonic()-started)),flush=True)
    # Cross-check packed counting against the ordinary reader, CRC, and resizing.
    import cv2
    for i in np.linspace(0,len(df)-1,64,dtype=int):
        _,label,_=cache.read(int(df.iloc[i].patch_id))
        label=np.stack([cv2.resize(a,(256,256),interpolation=cv2.INTER_NEAREST) for a in label])
        np.testing.assert_array_equal(positives[i],(label==1).sum((1,2)))
        np.testing.assert_array_equal(valids[i],(label!=255).sum((1,2)))
    np.savez_compressed(out/'training_pixels.npz',patch_ids=df.patch_id.to_numpy(),
        patients=df.orion_slide_id.map(patient_id).to_numpy(dtype=str),positive=positives,valid=valids,
        tissue=tissues,native_positive=df[[c+'_pixels' for c in CHANNELS]].to_numpy(np.uint32),
        usable=df.general_qc_status.eq('usable').to_numpy())
    receipt=dict(fitted_split='train',n_patches=len(df),n_patients=df.orion_slide_id.map(patient_id).nunique(),
        grid=256,geometry='before random augmentation; same nearest-neighbor resize as training',
        train_manifest_sha256=file_hash(source/'data/train.csv'),cache_identity=cache.meta['source_identity'],
        ordinary_reader_and_crc_checks=64,seconds=time.monotonic()-started)
    (out/'pixel_inventory.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
