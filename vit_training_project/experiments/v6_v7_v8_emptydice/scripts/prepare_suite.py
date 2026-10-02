"""Prepare three patient-matched experiments; do not launch training."""
import json,sys,os,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vit_seg.artifacts import file_hash
from vit_seg.data import CHANNELS
from vit_seg.binary_targets import BinaryTargetStore
from vit_seg.validate import validate
from vit_seg.pixel_sampling import fit_probabilities,DistributedWeightedPatchSampler
P=Path(__file__).resolve().parents[1];BASE=P.parents[1]
SOURCE=P.parent/'binary_bce/runs/prepared/data';DATA=BASE/'datasets/binary_expression_tissue_v2'

def attach(source,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():os.link(source,dest)

def main():
 DATA.mkdir(exist_ok=True)
 for src in SOURCE.iterdir():
  if src.suffix in ('.csv','.json') and src.name not in ('binary_targets.json','VALIDATED.json','training_policy.json','bce_pixel_weights.json','BINARY_DATASET_COMPLETE.json'):
   attach(src,DATA/src.name)
 spec=json.loads((SOURCE/'binary_targets.json').read_text());spec.update(validity_policy='available_channel_and_tissue',
  reliable_channel_definition='Use published-QC channel availability flags; unavailable channels ignored, no claim of new biological staining QC',
  semantics='0/1 target planes reused losslessly; validity recomputed from recorded channel availability AND HE tissue; all-marker-zero tissue retained as negative')
 (DATA/'binary_targets.json').write_text(json.dumps(spec,indent=2))
 (DATA/'training_policy.json').write_text(json.dumps(dict(sampling_policy='shuffle_patches')))
 receipt=json.loads((P/'review/all_patch_tissue_counts.json').read_text())
 assert receipt['source_manifest_sha256']==spec['manifest_sha256']
 with np.load(P/'review/all_patch_tissue_counts.npz') as z:tissue=z['tissue'];available=z['available']
 train=pd.read_csv(DATA/'train.csv',usecols=['patch_id','patient_id',*[c+'_valid' for c in CHANNELS]])
 ids=train.patch_id.to_numpy();np.testing.assert_array_equal(available[ids],train[[c+'_valid' for c in CHANNELS]].to_numpy(bool))
 inv=P.parent/'pixel_balance/runs/prepared/data';invmeta=json.loads((inv/'pixel_inventory.json').read_text())
 assert invmeta['train_manifest_sha256']==file_hash(DATA/'train.csv') and invmeta['grid']==256
 with np.load(inv/'training_pixels.npz') as z:
  np.testing.assert_array_equal(z['patch_ids'],ids);positive=z['positive'].astype(np.int64);old_valid=z['valid'].astype(np.int64)
 valid=tissue[ids,None]*available[ids];assert np.all(valid>=old_valid) and np.all(positive<=valid)
 store=BinaryTargetStore(spec['folder'],spec['validity_policy'])
 for i in np.linspace(0,len(ids)-1,256,dtype=int):
  y,m,t=store.read(ids[i]);y=np.stack([cv2.resize(c,(256,256),interpolation=cv2.INTER_NEAREST) for c in y]);m=np.stack([cv2.resize(c.astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST) for c in m])
  np.testing.assert_array_equal(y.sum((1,2)),positive[i]);np.testing.assert_array_equal(m.sum((1,2)),valid[i])
 np.savez_compressed(DATA/'training_pixels.npz',patch_ids=ids,positive=positive.astype(np.int32),valid=valid.astype(np.int32))
 (DATA/'BINARY_DATASET_COMPLETE.json').write_text(json.dumps(dict(patches=len(tissue),binary_target_values=[0,1],ignore_is_separate=True,validity_policy=spec['validity_policy'],storage='versioned logical view over unchanged binary shards',old_dataset_unchanged=True,all_native_mask_records_counted=True,training_patches=len(train),new_training_valid_pixels=valid.sum(0).tolist(),old_training_valid_pixels=old_valid.sum(0).tolist(),model_grid=256),indent=2))
 checked=validate(DATA)
 print('New validity prepared; added negative channel-pixels',int((valid-old_valid).sum()),flush=True)
 q,fit=fit_probabilities(positive,valid)
 print('V8 fit',fit,flush=True)
 sampler=DistributedWeightedPatchSampler(q,4,0,42);draw=sampler.global_indices()
 fit.update(first_epoch_unique_patches=int(np.unique(draw).size),first_epoch_positive_fraction=(positive[draw].sum(0)/valid[draw].sum(0)).tolist())
 (P/'review/v8_sampling_fit.json').write_text(json.dumps(fit,indent=2))
 v5=json.loads((SOURCE/'bce_pixel_weights.json').read_text())
 table=pd.DataFrame(dict(channel=CHANNELS,positive_pixels=positive.sum(0),old_negative_pixels=(old_valid-positive).sum(0),new_negative_pixels=(valid-positive).sum(0),natural_positive_fraction=positive.sum(0)/valid.sum(0),v6_positive_weight=1.,v7_positive_weight=np.sqrt(v5['positive_weights']),v8_positive_weight=1.,v8_expected_positive_fraction=fit['expected_positive_fraction'],v8_epoch1_positive_fraction=fit['first_epoch_positive_fraction']))
 table.to_csv(P/'review/experiment_comparison.csv',index=False)
 template=json.loads((P.parent/'binary_bce/configs/train_binary_bce.json').read_text())
 for name in ['v6','v7','v8']:
  run=P/'runs'/name;data=run/'data';data.mkdir(parents=True,exist_ok=True)
  for src in DATA.iterdir():
   if src.suffix in ('.csv','.json') and src.name not in ('VALIDATED.json','training_policy.json'):attach(src,data/src.name)
  (run/'cache').mkdir(exist_ok=True)
  for filename in ['records.raw','COMPLETE.json']:attach(P.parent/'binary_bce/runs/prepared/cache'/filename,run/'cache'/filename)
  mode='sqrt_v5' if name=='v7' else 'none';wp=np.sqrt(v5['positive_weights']) if name=='v7' else np.ones(16)
  weights=dict(mode=mode,fitted_split='train',channels=CHANNELS,grid=256,validity_policy=spec['validity_policy'],
   train_manifest_sha256=file_hash(data/'train.csv'),binary_manifest_sha256=spec['manifest_sha256'],
   positive_counts=positive.sum(0).tolist(),valid_counts=valid.sum(0).tolist(),positive_weights=wp.tolist(),negative_weights=[1.]*16,
   v5_source_weights=v5 if name=='v7' else None,v5_source_sha256=file_hash(SOURCE/'bce_pixel_weights.json') if name=='v7' else None)
  (data/'bce_pixel_weights.json').write_text(json.dumps(weights,indent=2))
  policy='pixel_balanced_patches' if name=='v8' else 'shuffle_patches'
  (data/'training_policy.json').write_text(json.dumps(dict(sampling_policy=policy,patches_per_epoch=len(train),replacement=name=='v8',drop_last=False,pixel_balance_is_approximate=name=='v8'),indent=2))
  if name=='v8':np.savez_compressed(data/'patch_sampling.npz',patch_ids=ids,probabilities=q)
  cfg=dict(template,experiment=name,channel_weighting='uniform',sampling_policy=policy,
   bce_pixel_weights_file=str(data/'bce_pixel_weights.json'),validity_policy=spec['validity_policy'],bce_weight=1.,overlap_weight=1.,dice_scope='all_valid')
  (P/'configs'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
  validation=validate(data)
  (run/'status.json').write_text(json.dumps(dict(stage='prepared_not_started',experiment=name,validation=validation,loss='BCE + Dice',validity_policy=spec['validity_policy']),indent=2))
 print(table.to_string(index=False),flush=True)
 (P/'review/PREPARED.json').write_text(json.dumps(dict(experiments=['v6','v7','v8'],validity_policy=spec['validity_policy'],patient_split_unchanged=True,bce_dice_coefficients=[1,1],channel_weighting='uniform',new_training_executed=False),indent=2))
if __name__=='__main__':main()
