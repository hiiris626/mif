"""Matched read-only evaluation of every usable trained ViT experiment."""
import argparse, json, sys, time, hashlib
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT=Path(__file__).resolve().parents[2]
SOURCE=PROJECT/'experiments/binary_bce'
OLD=PROJECT/'review/unified_eval_20260929'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE))
from vit_seg.data import CHANNELS,IMAGENET_STATS,read_he
from vit_seg.cache import NativeCache
from vit_seg.binary_targets import BinaryTargetStore
from vit_seg.metrics import DeviceMultilabelMetrics
from vit_seg.model import build_model
from vit_matte.vitmatte_unet import ViTMatteUNet
from vit_seg.artifacts import file_hash

MODELS={
 'v1':('legacy',Path(json.loads((PROJECT.parent/'vit_versions/vit_v1/config.json').read_text())['checkpoint']),OLD/'v1/thresholds.json'),
 'v2':('legacy',Path(json.loads((PROJECT.parent/'vit_versions/vit_v2/config.json').read_text())['checkpoint']),OLD/'v2/thresholds.json'),
 'v3':('legacy',Path(json.loads((PROJECT.parent/'vit_versions/vit_v3/config.json').read_text())['checkpoint']),OLD/'v3/thresholds.json'),
 'v4':('classification',PROJECT/'runs/ddp_baseline/results/model/best.pt',OLD/'baseline/thresholds.json'),
 'v5':('classification',PROJECT/'experiments/binary_bce/runs/prepared/results/model/best.pt',PROJECT/'experiments/binary_bce/runs/prepared/results/thresholds.json'),
 'v6':('classification',PROJECT/'experiments/v6_v7_v8/runs/v6/results/model/best.pt',PROJECT/'experiments/v6_v7_v8/runs/v6/results/thresholds.json'),
 'v9':('classification',PROJECT/'experiments/v6_v7_v8_emptydice/runs/v9/results/model/best.pt',PROJECT/'experiments/v6_v7_v8_emptydice/runs/v9/results/thresholds.json'),
 'positive_dice_partial':('classification',PROJECT/'runs/ddp_positive_dice/results/model/best.pt',None),
}

class Inputs(Dataset):
 def __init__(self,frame,size,legacy):
  self.frame=frame.reset_index(drop=True);self.size=size;self.legacy=legacy
  self.cache=NativeCache(PROJECT/'runs/ddp_baseline/cache')
  self.store=BinaryTargetStore(PROJECT/'datasets/binary_expression_v1/targets')
  self.root=Path('/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x')
 def __len__(self):return len(self.frame)
 def __getitem__(self,i):
  row=self.frame.iloc[i];he,old_y,tissue=self.cache.read(int(row.patch_id))
  y,old_valid,binary_tissue=self.store.read(int(row.patch_id));y[~old_valid]=255
  np.testing.assert_array_equal(old_y,y);np.testing.assert_array_equal(tissue,binary_tissue)
  if self.legacy:he=read_he(self.root/row.image_path)
  x=cv2.resize(he,(self.size,self.size),interpolation=cv2.INTER_AREA).astype(np.float32)/255
  x=(x-np.array(IMAGENET_STATS['mean'],np.float32))/np.array(IMAGENET_STATS['std'],np.float32)
  resized_tissue=cv2.resize(tissue.astype(np.uint8),(self.size,self.size),interpolation=cv2.INTER_NEAREST).astype(bool)
  if not self.legacy:x[~resized_tissue]=0
  y=np.stack([cv2.resize(c,(256,256),interpolation=cv2.INTER_NEAREST) for c in y])
  tissue=cv2.resize(tissue.astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST).astype(bool)
  available=np.array([bool(row[c+'_valid']) for c in CHANNELS])
  return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(y),torch.from_numpy(tissue),torch.from_numpy(available),int(row.patch_id)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODELS,required=True);parser.add_argument('--previews',action='store_true');a=parser.parse_args();name=a.model
 torch.set_num_threads(1);cv2.setNumThreads(1);device='cuda:0';kind,checkpoint,threshold_path=MODELS[name];legacy=kind=='legacy'
 dest=OUT/name;dest.mkdir(parents=True,exist_ok=True);started=time.monotonic()
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);cfg=ck['config'];epoch=ck.get('epoch')
 if legacy:
  is_v1='vit_proj.weight' in ck['model'];layers=(32,) if is_v1 else tuple(map(int,cfg.get('vit_layers','8,16,24,32').split(',')))
  model=ViTMatteUNet(input_size=cfg['tile_size'],vit_size=224 if is_v1 else cfg['vit_size'],vit_layers=layers,multi_scale=not is_v1,legacy_v1=is_v1,lora_r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],weights_path=None,device=device)
 else:model=build_model(dict(cfg,weights_path=None),device)
 model.load_state_dict(ck['model'],strict=True);model.to(device).eval();del ck
 q=None
 if cfg.get('log_norm'):
  q=cfg.get('marker_q') or json.loads(Path('/data/weiyh/orioncrc_cache/marker_q.json').read_text())['q']
  q=torch.tensor(q,device=device,dtype=torch.float32)[None,:,None,None]
 own_thresholds=json.loads(threshold_path.read_text())['thresholds'] if threshold_path else None
 frame=pd.read_csv(PROJECT/'experiments/binary_bce/runs/prepared/data/test.csv',usecols=['patch_id','image_path','patient_id',*[c+'_valid' for c in CHANNELS]])
 common_ids=set(pd.read_csv(OLD/'common_test.csv').patch_id);frame=frame[frame.patch_id.isin(common_ids)].copy()
 assert len(frame)==6638 and set(frame.patient_id)=={'CRC02'} and set(frame.patch_id)==common_ids
 if a.previews:
  preview_ids=set(pd.read_csv(OUT/'preview_manifest.csv').patch_id);frame=frame[frame.patch_id.isin(preview_ids)].copy()
  assert len(frame)==len(preview_ids)
 else:frame.to_csv(dest/'test_manifest.csv',index=False)
 loader=DataLoader(Inputs(frame,cfg['tile_size'],legacy),batch_size=8 if name=='v2' else (16 if name=='v1' else 64),num_workers=4,pin_memory=True,prefetch_factor=2,timeout=180)
 policies={'fixed_05':.5}
 if own_thresholds is not None:policies['own_validation']=own_thresholds
 metrics={scope:{policy:DeviceMultilabelMetrics(device,threshold=threshold) for policy,threshold in policies.items()} for scope in ['original','expanded']}
 with torch.inference_mode():
  for i,(x,y,tissue,available,ids) in enumerate(loader):
   with torch.autocast('cuda',dtype=torch.bfloat16):raw=model(x.to(device,non_blocking=True))
   if not legacy:p=raw.float().sigmoid()
   elif q is not None:p=(q*(torch.exp((raw.float().clamp(-1,1)+1)*.5*np.log(2))-1)).clamp(0,255)/255
   else:p=((raw.float()+.9)/1.8).clamp(0,1)
   if p.shape[-1]!=256:p=F.interpolate(p,size=(256,256),mode='bilinear',align_corners=False)
   y=y.to(device);tissue=tissue.to(device);available=available.to(device)
   expanded=torch.where(tissue[:,None]&available[:,:,None,None],(y==1).to(torch.uint8),255)
   for policy in policies:
    metrics['original'][policy].update(p,y);metrics['expanded'][policy].update(p,expanded)
   if a.previews:
    folder=dest/'previews';folder.mkdir(exist_ok=True)
    for j,pid in enumerate(ids.tolist()):
     np.savez_compressed(folder/f'patch_{pid}.npz',scores=p[j].cpu().numpy(),labels=y[j].cpu().numpy(),
                        expanded=expanded[j].cpu().numpy(),tissue=tissue[j].cpu().numpy(),available=available[j].cpu().numpy())
   if i%25==0:print(json.dumps(dict(model=name,batch=i+1,batches=len(loader),seconds=round(time.monotonic()-started,1))),flush=True)
 if a.previews:
  print('PREVIEWS COMPLETE',name,flush=True);return
 for scope in metrics:
  for policy,m in metrics[scope].items():
   result=m.as_numpy().result(CHANNELS)
   if legacy:
    result.pop('ece',None);result.pop('brier',None)
    for row in result['per_class'].values():row.pop('ece',None);row.pop('brier',None)
   result.update(model=name,scope=scope,threshold_policy=policy,score_kind='normalized_intensity' if legacy else 'sigmoid_probability',n_patches=len(frame),patient='CRC02')
   (dest/f'{scope}_{policy}.json').write_text(json.dumps(result,indent=2))
 complete=dict(model=name,status='completed_training' if name!='positive_dice_partial' else 'interrupted_after_epoch_4',checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),checkpoint_epoch_one_based=epoch+1 if epoch is not None else None,threshold_path=str(threshold_path) if threshold_path else None,threshold_sha256=file_hash(threshold_path) if threshold_path else None,test_manifest_sha256=hashlib.sha256((dest/'test_manifest.csv').read_bytes()).hexdigest(),test_patches=len(frame),seconds=time.monotonic()-started,source_code_sha256=file_hash(__file__))
 (dest/'COMPLETE.json').write_text(json.dumps(complete,indent=2));print('COMPLETE',name,flush=True)
if __name__=='__main__':main()
