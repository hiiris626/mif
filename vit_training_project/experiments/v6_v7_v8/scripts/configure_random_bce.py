"""Configure shuffled unique training patches and dataset-level equal-label BCE."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vit_seg.artifacts import file_hash
from vit_seg.data import CHANNELS
from vit_seg.binary_targets import BinaryTargetStore
from vit_seg.bce_balance import balanced_pixel_weights
from vit_seg.validate import validate


def main():
    project=Path(__file__).resolve().parents[1];data=project/'runs/prepared/data'
    inventory=project.parent/'pixel_balance/runs/prepared/data'
    receipt=json.loads((inventory/'pixel_inventory.json').read_text())
    train=pd.read_csv(data/'train.csv',usecols=['patch_id','patient_id'])
    binary=json.loads((data/'binary_targets.json').read_text());store=BinaryTargetStore(binary['folder'])
    native=json.loads((project/'runs/prepared/cache/COMPLETE.json').read_text())
    if receipt['fitted_split']!='train' or receipt['grid']!=256 or receipt['train_manifest_sha256']!=file_hash(data/'train.csv'):
        raise ValueError('Pixel inventory does not match current training split/grid')
    if receipt['cache_identity']!=native['source_identity'] or store.meta['source_cache_metadata_sha256']!=file_hash(project/'runs/prepared/cache/COMPLETE.json'):
        raise ValueError('Binary targets and full-training pixel inventory have different sources')
    if not store.meta['all_records_compression_roundtrip_verified']:raise ValueError('Binary export not verified')
    with np.load(inventory/'training_pixels.npz') as z:
        if not np.array_equal(z['patch_ids'],train.patch_id.to_numpy()):raise ValueError('Pixel inventory row mismatch')
        pos=z['positive'];valid=z['valid']
        # Reuse existing complete train counts, not a subset estimate. Verify
        # the newly materialized binary labels against the cached counts too.
        for i in np.linspace(0,len(train)-1,256,dtype=int):
            target,mask,_=store.read(int(train.patch_id.iloc[i]))
            target=np.stack([cv2.resize(c,(256,256),interpolation=cv2.INTER_NEAREST) for c in target])
            mask=np.stack([cv2.resize(c.astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST) for c in mask]).astype(bool)
            np.testing.assert_array_equal((target*mask).sum((1,2)),pos[i])
            np.testing.assert_array_equal(mask.sum((1,2)),valid[i])
        positive=pos.sum(0,dtype=np.int64);total=valid.sum(0,dtype=np.int64)
    wp,wn=balanced_pixel_weights(positive,total);negative=total-positive
    np.testing.assert_allclose(wp*positive,wn*negative,rtol=1e-12)
    weights=dict(fitted_split='train',channels=CHANNELS,grid=256,n_patches=len(train),n_patients=int(train.patient_id.nunique()),
        train_manifest_sha256=file_hash(data/'train.csv'),binary_manifest_sha256=binary['manifest_sha256'],
        inventory_sha256=file_hash(inventory/'training_pixels.npz'),new_binary_count_crosschecks=256,
        positive_counts=positive.tolist(),negative_counts=negative.tolist(),valid_counts=total.tolist(),
        positive_weights=wp.tolist(),negative_weights=wn.tolist(),
        rule='w_positive=N_negative/N_positive; w_negative=1; no clipping',
        denominator='unweighted valid pixel count per channel; DDP global reduction',
        scope='all unique training patches, model 256 grid before random augmentation; val/test excluded',
        balance='equal total class weight in fitted training data, not equal actual loss/gradient per batch')
    (data/'bce_pixel_weights.json').write_text(json.dumps(weights,indent=2))
    policy=dict(sampling_policy='shuffle_patches',training_manifest='train.csv',
        patches_per_epoch=len(train),replacement=False,full_coverage=True,drop_last=False,
        world_size=4,per_rank_patches=len(train)//4,per_rank_batch=64,steps_per_epoch=int(np.ceil(len(train)/256)),
        last_batch_per_rank=(len(train)//4)%64,patient_split_unchanged=True)
    (data/'training_policy.json').write_text(json.dumps(policy,indent=2))
    for name in ['train.json','train_binary_bce.json']:
        path=project/'configs'/name;cfg=json.loads(path.read_text())
        cfg.update(sampling_policy='shuffle_patches',balanced_training=False,
                   bce_pixel_weights_file=str(data/'bce_pixel_weights.json'))
        path.write_text(json.dumps(cfg,indent=2))
    table=pd.DataFrame(dict(channel=CHANNELS,positive_pixels=positive,negative_pixels=negative,
        positive_fraction=positive/total,positive_weight=wp,negative_weight=wn,
        positive_weight_mass=wp*positive,negative_weight_mass=wn*negative))
    table.to_csv(project/'review/bce_pixel_weights.csv',index=False)
    checked=validate(data)
    status=dict(stage='prepared_not_started',new_training_executed=False,pixel_loss='weighted_BCEWithLogitsLoss',
        dice_scope='all_valid',sampling=policy,weights_file=str(data/'bce_pixel_weights.json'),
        positive_and_negative_class_mass_equal=True,validation=checked)
    (project/'runs/prepared/status.json').write_text(json.dumps(status,indent=2))
    print(table.to_string(index=False));print('PREPARED ONLY; original live training unchanged')


if __name__=='__main__':main()
