"""Re-evaluate archived models on shared, unseen CRC02; never fit on test."""
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = Path(__file__).resolve().parents[2]
# Freeze the baseline implementation independently of subsequent loss changes.
sys.path.insert(0, str(PROJECT/'runs/ddp_baseline/source_at_completion'))
from vit_seg.cache import NativeCache
from vit_seg.data import CHANNELS, IMAGENET_STATS, read_he
from vit_seg.metrics import DeviceMultilabelMetrics
from vit_seg.thresholds import select_thresholds
from vit_seg.model import build_model
from vit_matte.vitmatte_unet import ViTMatteUNet

BASE = PROJECT/'runs/ddp_baseline'
OUT = Path(__file__).resolve().parent


class Inputs(Dataset):
    def __init__(self, frame, size, legacy):
        self.frame = frame.reset_index(drop=True)
        self.size, self.legacy = size, legacy
        self.cache = NativeCache(BASE/'cache')
        self.root = Path(json.loads((BASE/'data/statistics.json').read_text())['source_spec']['root'])

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        he, labels, tissue = self.cache.read(int(row.patch_id))
        # Preserve each historical checkpoint's trained H&E preprocessing.
        if self.legacy: he = read_he(self.root/row.image_path)
        image = cv2.resize(he, (self.size,self.size), interpolation=cv2.INTER_AREA)
        x = (image.astype(np.float32)/255-np.array(IMAGENET_STATS['mean'],np.float32))/np.array(IMAGENET_STATS['std'],np.float32)
        if not self.legacy:
            mask = cv2.resize(tissue.astype(np.uint8),(self.size,self.size),interpolation=cv2.INTER_NEAREST).astype(bool)
            x[~mask] = 0
        target = np.stack([cv2.resize(y,(256,256),interpolation=cv2.INTER_NEAREST) for y in labels])
        mask = cv2.resize(tissue.astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(target),int(row.patch_id),str(row.orion_slide_id),torch.from_numpy(mask)


def prepare():
    cols = ['patch_id','orion_slide_id','split','original_split','image_path','target_path']
    frame = pd.read_csv(BASE/'data/source_manifest.csv',usecols=cols)
    for name, selected in [('common_test',frame[(frame.split=='test')&(frame.original_split=='test')]),
                           ('legacy_val',frame[frame.original_split=='val']),
                           ('baseline_val',frame[frame.split=='val']),
                           ('baseline_test',frame[frame.split=='test'])]:
        selected.to_csv(OUT/f'{name}.csv',index=False)
    previews = frame[frame.split=='test'].groupby('orion_slide_id',group_keys=False).apply(
        lambda x:x.sort_values('patch_id').iloc[np.linspace(0,len(x)-1,4,dtype=int)],include_groups=True)
    previews.to_csv(OUT/'preview_manifest.csv',index=False)
    audit = dict(shared_test_patients=['CRC02'],shared_test_patches=6638,
        contaminated_by_legacy_training=int(((frame.split=='test')&(frame.original_split=='train')).sum()),
        baseline_patient_identity_issue='CRC33_01 train / CRC33_02 test: exclude CRC33 for clean baseline test',
        calibration='Each checkpoint uses its own unseen validation patients; validation populations differ.',
        common_protocol='16 GT>0 binary channels; native labels nearest to 256; unknown and all-marker-zero pixels ignored; '
            'same 256-bin ROC/AP; per-channel validation F1 thresholds on grid 1/256..255/256; no test fitting.',
        legacy_scores='Inverse-transform regression output, clip intensity to [0,255], divide by 255. NOT expression probabilities.',
        inference='Native trained input geometry/preprocessing per checkpoint; outputs bilinear to common 256 grid before scoring.',
        limitations='Only one common independent case. Historical selected checkpoints use different selection criteria. '
            'This is a matched held-out checkpoint comparison, not a controlled architecture experiment.')
    (OUT/'protocol.json').write_text(json.dumps(audit,indent=2))


def run(name, device, extra_previews=False):
    torch.set_num_threads(1); cv2.setNumThreads(1)
    dest=OUT/name; dest.mkdir(exist_ok=True)
    legacy=name!='baseline'
    if legacy:
        archive=json.loads((PROJECT.parent/f'vit_versions/vit_{name}/config.json').read_text())
        path=archive['checkpoint']
    else: path=str(BASE/'results/model/best.pt')
    ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True); cfg=ck['config']
    if legacy:
        v1='vit_proj.weight' in ck['model']
        layers=(32,) if v1 else tuple(map(int,cfg.get('vit_layers','8,16,24,32').split(',')))
        model=ViTMatteUNet(input_size=cfg['tile_size'],vit_size=224 if v1 else cfg['vit_size'],
            vit_layers=layers,multi_scale=not v1,legacy_v1=v1,lora_r=cfg['lora_r'],
            lora_alpha=cfg['lora_alpha'],weights_path=None,device=device)
    else:
        cfg=dict(cfg,weights_path=None); model=build_model(cfg,device)
    model.load_state_dict(ck['model'],strict=True); model.to(device).eval()
    q=None
    if cfg.get('log_norm'):
        q=cfg.get('marker_q')
        if q is None:q=json.loads(Path('/data/weiyh/orioncrc_cache/marker_q.json').read_text())['q']
        q=torch.tensor(q,device=device,dtype=torch.float32)[None,:,None,None]
    del ck
    batches=8 if name=='v2' else (16 if name=='v1' else 64)
    preview=set(pd.read_csv(OUT/'preview_manifest.csv').patch_id)
    threshold_config=dict(min=1/256,max=255/256,min_positive_pixels=100,min_negative_pixels=100,min_patients_per_label=2)
    started=time.monotonic()
    def loader(frame):
        return DataLoader(Inputs(frame,cfg['tile_size'],legacy),batch_size=batches,num_workers=4,
                          pin_memory=True,prefetch_factor=2,timeout=180)
    def scores(x):
        with torch.autocast('cuda',dtype=torch.bfloat16):raw=model(x.to(device,non_blocking=True))
        if not legacy: value=raw.float().sigmoid()
        elif q is not None:value=(q*(torch.exp((raw.float().clamp(-1,1)+1)*.5*np.log(2))-1)).clamp(0,255)/255
        else:value=((raw.float()+.9)/1.8).clamp(0,1)
        return F.interpolate(value,size=(256,256),mode='bilinear',align_corners=False) if value.shape[-1]!=256 else value
    with torch.inference_mode():
        if extra_previews:
            frame=pd.read_csv(OUT/'preview_manifest.csv')
            previews=dest/'previews';previews.mkdir(exist_ok=True)
            for x,y,ids,patients,mask in loader(frame):
                p=scores(x)
                for j,patch in enumerate(ids.tolist()):
                    np.savez_compressed(previews/f'patch_{patch}.npz',scores=p[j].cpu().numpy(),
                        labels=y[j].numpy(),tissue=mask[j].numpy())
            print('EXTRA PREVIEWS COMPLETE',name,flush=True)
            return
        val=pd.read_csv(OUT/('legacy_val.csv' if legacy else 'baseline_val.csv'))
        metrics=DeviceMultilabelMetrics(device)
        support={p:torch.zeros((2,16),device=device,dtype=torch.bool) for p in val.orion_slide_id.unique()}
        for i,(x,y,ids,patients,mask) in enumerate(loader(val)):
            p=scores(x);y=y.to(device,non_blocking=True); metrics.update(p,y)
            for patient in set(patients):
                ix=torch.tensor([j for j,v in enumerate(patients) if v==patient],device=device)
                target=y[ix];support[patient][0]|=(target==1).any((0,2,3));support[patient][1]|=(target==0).any((0,2,3))
            if i%50==0:print(json.dumps(dict(model=name,phase='val',batch=i,total=len(val),seconds=time.monotonic()-started)),flush=True)
        support=torch.stack(list(support.values())).sum(0).cpu().numpy()
        thresholds=select_thresholds(metrics.as_numpy(),support[0],support[1],threshold_config)
        thresholds.update(model=name,validation_patients=sorted(val.orion_slide_id.unique()))
        (dest/'thresholds.json').write_text(json.dumps(thresholds,indent=2))
        frame=pd.read_csv(OUT/('common_test.csv' if legacy else 'baseline_test.csv'))
        all_metrics=DeviceMultilabelMetrics(device,threshold=thresholds['thresholds'])
        common=DeviceMultilabelMetrics(device,threshold=thresholds['thresholds'])
        clean=DeviceMultilabelMetrics(device,threshold=thresholds['thresholds'])
        per_patient={p:DeviceMultilabelMetrics(device,threshold=thresholds['thresholds']) for p in frame.orion_slide_id.unique()}
        previews=dest/'previews';previews.mkdir(exist_ok=True)
        for i,(x,y,ids,patients,mask) in enumerate(loader(frame)):
            p=scores(x);y=y.to(device,non_blocking=True);all_metrics.update(p,y)
            for patient in set(patients):
                ix=torch.tensor([j for j,v in enumerate(patients) if v==patient],device=device)
                per_patient[patient].update(p[ix],y[ix])
                if patient=='CRC02':common.update(p[ix],y[ix])
                if not patient.startswith('CRC33'):clean.update(p[ix],y[ix])
            for j,patch in enumerate(ids.tolist()):
                if patch in preview:
                    np.savez_compressed(previews/f'patch_{patch}.npz',scores=p[j].cpu().numpy(),
                        labels=y[j].cpu().numpy(),tissue=mask[j].numpy())
            if i%50==0:print(json.dumps(dict(model=name,phase='test',batch=i,total=len(frame),seconds=time.monotonic()-started)),flush=True)
        for title,m in [('all_test',all_metrics),('common_test',common),('clean_test',clean)]:
            result=m.as_numpy().result(CHANNELS)
            if legacy:
                result.pop('ece',None);result.pop('brier',None)
                for row in result['per_class'].values():row.pop('ece',None);row.pop('brier',None)
            result.update(model=name,score_kind='normalized_intensity_not_probability' if legacy else 'sigmoid_probability',
                calibration_source='own_heldout_validation',checkpoint=path)
            (dest/f'{title}_metrics.json').write_text(json.dumps(result,indent=2))
        (dest/'per_patient.json').write_text(json.dumps({k:v.as_numpy().result(CHANNELS) for k,v in per_patient.items()},indent=2))
    (dest/'COMPLETE.json').write_text(json.dumps(dict(seconds=time.monotonic()-started,model=name,
        calibration_patches=len(val),test_patches=len(frame),checkpoint=path),indent=2))
    print('COMPLETE',name,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--model',choices=['v1','v2','v3','baseline']);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--extra-previews',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:run(args.model,args.device,args.extra_previews)
