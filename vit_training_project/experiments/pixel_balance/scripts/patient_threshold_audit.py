"""Bounded validation-only threshold-transfer diagnostic; leaves training intact."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vit_seg.data import PixelDataset, CHANNELS
from vit_seg.metrics import DeviceMultilabelMetrics
from vit_seg.model import build_model
from vit_seg.artifacts import file_hash


def curves(pos,neg):
    tp=pos[...,::-1].cumsum(-1)[...,::-1];fp=neg[...,::-1].cumsum(-1)[...,::-1]
    fn=pos.sum(-1,keepdims=True)-tp;tn=neg.sum(-1,keepdims=True)-fp
    return tp,fp,fn,tn


def fit(pos,neg,grid,objective):
    tp,fp,fn,_=curves(pos,neg)
    if objective=='patient_equal':
        n=(pos+neg).sum(-1,keepdims=True).clip(1)
        tp,fp,fn=tp/n,fp/n,fn/n
    tp,fp,fn=tp.sum(0),fp.sum(0),fn.sum(0)
    f1=2*tp/np.maximum(2*tp+fp+fn,1e-12)
    output=[]
    for c in range(pos.shape[1]):
        supported=((pos[:,c].sum(1)>=100).sum()>=2 and (neg[:,c].sum(1)>=100).sum()>=2)
        candidates=grid[np.isclose(f1[c,grid],f1[c,grid].max(),atol=1e-12,rtol=0)]
        output.append(int(candidates[np.argmin(abs(candidates-pos.shape[-1]/2))]) if supported else pos.shape[-1]//2)
    return np.array(output)


def summarize(out,pos,neg,patients,bins):
    # Keep the current workflow's validated threshold search range.
    grid=np.arange(int(np.ceil(.05*bins)),int(np.floor(.95*bins))+1)
    records=[];full={}
    tp,fp,fn,tn=curves(pos,neg)
    ratio=lambda a,b:float(a/b) if b else None
    for objective in ['pooled_pixels','patient_equal']:
        full[objective]=(fit(pos,neg,grid,objective)/bins).tolist()
        for held,patient in enumerate(patients):
            train=np.arange(len(patients))!=held
            chosen=fit(pos[train],neg[train],grid,objective)
            for c,name in enumerate(CHANNELS):
                k=chosen[c];a,b,d,e=tp[held,c,k],fp[held,c,k],fn[held,c,k],tn[held,c,k]
                j=bins//2
                records.append(dict(objective=objective,patient=patient,channel=name,threshold=k/bins,
                    precision=ratio(a,a+b),recall=ratio(a,a+d),f1=ratio(2*a,2*a+b+d),fpr=ratio(b,b+e),
                    f1_at_05=ratio(2*tp[held,c,j],2*tp[held,c,j]+fp[held,c,j]+fn[held,c,j]),
                    positive_pixels=int(pos[held,c].sum()),negative_pixels=int(neg[held,c].sum()),
                    predicted_positive_pixels=int(a+b)))
    frame=pd.DataFrame(records);frame.to_csv(out/'leave_one_patient_out.csv',index=False)
    summaries=[]
    for (objective,channel),g in frame.groupby(['objective','channel'],sort=False):
        values=g.threshold.to_numpy()
        summaries.append(dict(objective=objective,channel=channel,threshold_min=values.min(),threshold_max=values.max(),
            threshold_median=float(np.median(values)),threshold_iqr=float(np.quantile(values,.75)-np.quantile(values,.25)),
            threshold_range=float(np.ptp(values)),boundary_folds=int(((values==grid[0]/bins)|(values==grid[-1]/bins)).sum()),
            mean_heldout_f1=g.f1.mean(),mean_heldout_f1_at_05=g.f1_at_05.mean(),
            mean_heldout_precision=g.precision.mean(),mean_heldout_recall=g.recall.mean(),mean_heldout_fpr=g.fpr.mean()))
    pd.DataFrame(summaries).to_csv(out/'stability_summary.csv',index=False)
    (out/'full_sample_thresholds_diagnostic_only.json').write_text(json.dumps(full,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:3');p.add_argument('--patches-per-patient',type=int,default=64)
    args=p.parse_args();source=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    gpu=args.device.startswith('cuda')
    torch.set_num_threads(1 if gpu else 8)
    if gpu:
        torch.cuda.set_device(args.device)
        free,total=torch.cuda.mem_get_info()
        cap=min(3.0*2**30,free-.8*2**30)
        if cap<2.85*2**30:raise RuntimeError('Not enough spare memory for a bounded audit; current training is left intact')
        torch.cuda.set_per_process_memory_fraction(cap/total,args.device)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=dict(ck['config'],weights_path=None);model=build_model(cfg,'cpu')
    model.load_state_dict(ck['model']);del ck
    model.requires_grad_(False).to(args.device).eval()
    frame=pd.read_csv(source/'data/val.csv',usecols=['patch_id','orion_slide_id'])
    patients=sorted(frame.orion_slide_id.unique());selected=[]
    for i,patient in enumerate(patients):
        rows=frame[frame.orion_slide_id==patient]
        selected.extend(rows.sample(min(args.patches_per_patient,len(rows)),random_state=20260929+i).index)
    selected=np.array(selected);frame.iloc[selected].to_csv(out/'validation_patch_manifest.csv',index=False)
    stats=json.loads((source/'data/statistics.json').read_text())
    ds=PixelDataset(source/'data/val.csv',stats,cfg['data_root'],256,False,cache_dir=source/'cache')
    ds.df=ds.df.iloc[selected].reset_index(drop=True)
    labels=frame.iloc[selected].orion_slide_id.tolist()
    metrics={patient:DeviceMultilabelMetrics(args.device,bins=1024) for patient in patients}
    started=time.monotonic()
    with torch.inference_mode():
        for i in range(len(ds)):
            sample=ds[i]
            # Disable cached BF16 copies of all encoder weights to keep the
            # audit below 3 GiB allocator memory while DDP continues running.
            with torch.autocast('cuda' if gpu else 'cpu',dtype=torch.bfloat16,cache_enabled=not gpu):
                prob=model(sample['image'][None].to(args.device)).float().sigmoid()
            metrics[labels[i]].update(prob,sample['label'][None])
            del prob,sample
            if i%32==0:print(json.dumps(dict(done=i+1,total=len(ds),seconds=time.monotonic()-started,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if gpu else 0)),flush=True)
            time.sleep(.01)
    pos=np.stack([metrics[v].as_numpy().pos for v in patients]);neg=np.stack([metrics[v].as_numpy().neg for v in patients])
    np.savez_compressed(out/'patient_histograms.npz',positive=pos,negative=neg,patients=np.array(patients))
    summarize(out,pos,neg,patients,1024)
    (out/'receipt.json').write_text(json.dumps(dict(scope='validation-only pilot; not final calibration; not nested model-selection CV',
        patients=patients,patches=len(ds),patches_per_patient=args.patches_per_patient,bins=1024,
        checkpoint=args.checkpoint,checkpoint_sha256=file_hash(args.checkpoint),
        source_val_sha256=file_hash(source/'data/val.csv'),threshold_range=[.05,.95],
        precision=f'FP32 parameters, BF16 autocast on {args.device}; CPU/GPU kernels may differ slightly',
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if gpu else 0,seconds=time.monotonic()-started,
        current_training_unchanged=True,threshold_files_used_by_training_unchanged=True),indent=2))
    print('COMPLETE threshold pilot',flush=True)


if __name__=='__main__':main()
