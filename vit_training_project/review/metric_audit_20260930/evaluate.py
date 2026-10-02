"""Read-only matched checkpoint audit; expanded zero-label scope is sensitivity only."""
import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
PROJECT=Path(__file__).resolve().parents[2]
NEW=PROJECT/'experiments/binary_bce'
OLD=PROJECT/'review/unified_eval_20260929'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(NEW))
from vit_seg.data import CHANNELS, IMAGENET_STATS, read_he
from vit_seg.cache import NativeCache
from vit_seg.binary_targets import BinaryTargetStore
from vit_seg.metrics import DeviceMultilabelMetrics
from vit_seg.model import build_model
from vit_matte.vitmatte_unet import ViTMatteUNet
from vit_seg.artifacts import file_hash

class Inputs(Dataset):
 def __init__(self,frame,size,legacy,binary):
  self.frame=frame.reset_index(drop=True);self.size=size;self.legacy=legacy
  self.cache=NativeCache(PROJECT/'runs/ddp_baseline/cache')
  self.store=BinaryTargetStore(PROJECT/'datasets/binary_expression_v1/targets') if binary else None
  self.root=Path('/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x')
 def __len__(self):return len(self.frame)
 def __getitem__(self,i):
  row=self.frame.iloc[i];he,y,tissue=self.cache.read(int(row.patch_id))
  if self.store:
   by,bv,bt=self.store.read(int(row.patch_id));by[~bv]=255
   np.testing.assert_array_equal(y,by);np.testing.assert_array_equal(tissue,bt)
   y=by
  if self.legacy:he=read_he(self.root/row.image_path)
  x=cv2.resize(he,(self.size,self.size),interpolation=cv2.INTER_AREA).astype(np.float32)/255
  x=(x-np.array(IMAGENET_STATS['mean'],np.float32))/np.array(IMAGENET_STATS['std'],np.float32)
  if not self.legacy:x[~cv2.resize(tissue.astype(np.uint8),(self.size,self.size),interpolation=cv2.INTER_NEAREST).astype(bool)]=0
  y=np.stack([cv2.resize(c,(256,256),interpolation=cv2.INTER_NEAREST) for c in y])
  tissue=cv2.resize(tissue.astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST).astype(bool)
  available=np.array([bool(row[c+'_valid']) for c in CHANNELS])
  return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(y),torch.from_numpy(tissue),torch.from_numpy(available),str(row.patient_id),int(row.patch_id)

def main():
 a=argparse.ArgumentParser();a.add_argument('--model',required=True,choices=['v1','v2','v3','baseline','binary_bce']);args=a.parse_args();name=args.model
 torch.set_num_threads(1);cv2.setNumThreads(1);device='cuda:0';legacy=name in ['v1','v2','v3']
 dest=OUT/name;dest.mkdir(exist_ok=True);started=time.monotonic()
 if legacy:path=json.loads((PROJECT.parent/f'vit_versions/vit_{name}/config.json').read_text())['checkpoint']
 else:path=str((NEW/'runs/prepared' if name=='binary_bce' else PROJECT/'runs/ddp_baseline')/'results/model/best.pt')
 checkpoint_sha=file_hash(path)
 ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True);cfg=ck['config'];epoch=ck.get('epoch')
 if legacy:
  v1='vit_proj.weight' in ck['model'];layers=(32,) if v1 else tuple(map(int,cfg.get('vit_layers','8,16,24,32').split(',')))
  model=ViTMatteUNet(input_size=cfg['tile_size'],vit_size=224 if v1 else cfg['vit_size'],vit_layers=layers,multi_scale=not v1,legacy_v1=v1,lora_r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],weights_path=None,device=device)
 else:model=build_model(dict(cfg,weights_path=None),device)
 model.load_state_dict(ck['model'],strict=True);model.to(device).eval();del ck
 q=None
 if cfg.get('log_norm'):
  q=cfg.get('marker_q')
  if q is None:q=json.loads(Path('/data/weiyh/orioncrc_cache/marker_q.json').read_text())['q']
  q=torch.tensor(q,device=device,dtype=torch.float32)[None,:,None,None]
 threshold_path=NEW/'runs/prepared/results/thresholds.json' if name=='binary_bce' else OLD/name/'thresholds.json'
 threshold_spec=json.loads(threshold_path.read_text());thresholds=threshold_spec['thresholds']
 frame=pd.read_csv(NEW/'runs/prepared/data/test.csv',usecols=['patch_id','image_path','patient_id',*[c+'_valid' for c in CHANNELS]])
 common_ids=set(pd.read_csv(OLD/'common_test.csv').patch_id)
 assert set(frame[frame.patient_id=='CRC02'].patch_id)==common_ids and len(common_ids)==6638
 if name!='binary_bce':frame=frame[frame.patch_id.isin(common_ids)]
 assert frame[[c+'_valid' for c in CHANNELS]].notna().all().all()
 frame.to_csv(dest/'test_manifest.csv',index=False)
 batch=8 if name=='v2' else (16 if name=='v1' else 64)
 loader=DataLoader(Inputs(frame,cfg['tile_size'],legacy,name=='binary_bce'),batch_size=batch,num_workers=4,pin_memory=True,prefetch_factor=2,timeout=180)
 scopes=['common']+(['full'] if name=='binary_bce' else [])
 metrics={s:{k:DeviceMultilabelMetrics(device,threshold=.5 if k=='original_05' else thresholds) for k in ['original','expanded','original_05']} for s in scopes}
 # total tissue, ignored tissue, any predicted marker in ignored tissue, per-channel ignored predictions
 extra={s:torch.zeros(3+16,dtype=torch.int64,device=device) for s in scopes}
 per_patient={p:DeviceMultilabelMetrics(device,threshold=thresholds) for p in frame.patient_id.unique()}
 thr=torch.tensor(thresholds,device=device)[None,:,None,None]
 preview_ids=set(pd.read_csv(OLD/'preview_manifest.csv').patch_id)
 with torch.inference_mode():
  for i,(x,y,tissue,available,patients,ids) in enumerate(loader):
   with torch.autocast('cuda',dtype=torch.bfloat16):raw=model(x.to(device,non_blocking=True))
   if not legacy:p=raw.float().sigmoid()
   elif q is not None:p=(q*(torch.exp((raw.float().clamp(-1,1)+1)*.5*np.log(2))-1)).clamp(0,255)/255
   else:p=((raw.float()+.9)/1.8).clamp(0,1)
   if p.shape[-1]!=256:p=F.interpolate(p,size=(256,256),mode='bilinear',align_corners=False)
   y=y.to(device);tissue=tissue.to(device);available=available.to(device)
   expanded=torch.where(tissue[:,None]&available[:,:,None,None],(y==1).to(torch.uint8),255)
   common=torch.tensor([v=='CRC02' for v in patients],device=device)
   for s in scopes:
    ix=common if s=='common' else torch.ones(len(p),dtype=torch.bool,device=device)
    if not ix.any():continue
    pp,yy,tt=p[ix],y[ix],tissue[ix];ignored=tt & (yy==255).all(1)
    metrics[s]['original'].update(pp,yy);metrics[s]['original_05'].update(pp,yy);metrics[s]['expanded'].update(pp,expanded[ix])
    pred=(pp>=thr)&available[ix,:,None,None]
    extra[s][0]+=tt.sum();extra[s][1]+=ignored.sum();extra[s][2]+=(pred.any(1)&ignored).sum();extra[s][3:]+=(pred&ignored[:,None]).sum((0,2,3))
   for patient in set(patients):
    ix=torch.tensor([v==patient for v in patients],device=device);per_patient[patient].update(p[ix],y[ix])
   if name=='binary_bce':
    for j,pid in enumerate(ids.tolist()):
     if pid in preview_ids:
      (dest/'previews').mkdir(exist_ok=True);np.savez_compressed(dest/'previews'/f'patch_{pid}.npz',scores=p[j].cpu().numpy(),labels=y[j].cpu().numpy(),tissue=tissue[j].cpu().numpy())
   if i%25==0:print(json.dumps(dict(model=name,batch=i+1,batches=len(loader),seconds=round(time.monotonic()-started,1))),flush=True)
 for s in scopes:
  for kind,m in metrics[s].items():
   ref=m.as_numpy();result=ref.result(CHANNELS)
   if legacy:
    result.pop('ece',None);result.pop('brier',None)
    for row in result['per_class'].values():row.pop('ece',None);row.pop('brier',None)
   result.update(model=name,scope=s,mask_policy=kind,expanded_is_provisional=True if kind=='expanded' else False)
   (dest/f'{s}_{kind}.json').write_text(json.dumps(result,indent=2))
   np.savez_compressed(dest/f'{s}_{kind}_sufficient_stats.npz',confusion=ref.confusion,pos=ref.pos,neg=ref.neg)
  e=extra[s].cpu().tolist();n=e[1]
  (dest/f'{s}_ignored_tissue.json').write_text(json.dumps(dict(tissue_pixels=e[0],ignored_tissue_pixels=n,any_marker_positive_pixels=e[2],any_marker_positive_fraction=e[2]/n if n else None,per_channel={c:dict(positive_pixels=k,fraction=k/n if n else None) for c,k in zip(CHANNELS,e[3:])}),indent=2))
 (dest/'per_patient.json').write_text(json.dumps({k:v.as_numpy().result(CHANNELS) for k,v in per_patient.items()},indent=2))
 (dest/'COMPLETE.json').write_text(json.dumps(dict(model=name,checkpoint=path,checkpoint_sha256=checkpoint_sha,checkpoint_epoch_zero_based=epoch,threshold_path=str(threshold_path),threshold_sha256=file_hash(threshold_path),test_patches=len(frame),seconds=time.monotonic()-started,new_binary_matches_old_labels_all_records=name=='binary_bce',source_code_sha256=file_hash(__file__)),indent=2))
 print('COMPLETE',name,flush=True)
if __name__=='__main__':main()
