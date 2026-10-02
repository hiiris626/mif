"""Prepare a reviewable experiment; no model/GPU training is launched."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vit_seg.pixel_sampling import make_schedule, fitted_positive_weights, strata
from vit_seg.artifacts import file_hash
from vit_seg.validate import validate
from vit_seg.data import CHANNELS


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);args=p.parse_args()
    source=Path(args.source).resolve();project=Path(__file__).resolve().parents[1]
    run=project/'runs/prepared';data=run/'data';review=project/'review'
    # Immutable source CSVs can share storage; files written by validation are copied.
    for path in (source/'data').rglob('*'):
        relative=path.relative_to(source/'data');dest=data/relative
        if path.is_dir():dest.mkdir(exist_ok=True,parents=True)
        elif not dest.exists():
            dest.parent.mkdir(exist_ok=True,parents=True)
            if path.name=='VALIDATED.json':shutil.copy2(path,dest)
            else:os.link(path,dest)
    for name in ['records.raw','COMPLETE.json']:
        dest=run/'cache'/name
        if not dest.exists():os.link(source/'cache'/name,dest)
    cfg=json.loads((source/'submitted_config.json').read_text())
    receipt=json.loads((source/'data/VALIDATED.json').read_text())
    epoch_size=receipt['balanced_train_rows']//256*256
    with np.load(data/'training_pixels.npz') as packed:inventory={k:packed[k] for k in packed.files}
    pools,cutoff=strata(inventory);cohorts=np.full(inventory['positive'].shape,255,np.uint8)
    cohort_rows=[]
    for c,(false,high,low) in enumerate(pools):
        for tag,ids in enumerate((false,high,low)):cohorts[ids,c]=tag
        cohort_rows.append(dict(channel=CHANNELS[c],false=len(false),true1=len(high),true2=len(low),
            excluded_anchor=len(cohorts)-len(false)-len(high)-len(low),threshold=cutoff[c]))
    np.savez_compressed(data/'channel_cohorts.npz',patch_ids=inventory['patch_ids'],stratum=cohorts)
    pd.DataFrame(cohort_rows).to_csv(review/'channel_cohort_counts.csv',index=False)
    schedules=[];exposures=[]
    for epoch in range(3):
        ids,summary=make_schedule(inventory,epoch_size,cfg['seed']+epoch*1000003)
        (review/f'sampling_epoch_{epoch+1:03d}.json').write_text(json.dumps(summary,indent=2))
        pd.DataFrame(summary['anchors']).to_csv(review/f'anchor_counts_epoch_{epoch+1:03d}.csv',index=False)
        schedules.append(summary);exposures.append(np.bincount(ids,minlength=len(inventory['patch_ids'])))
    # Fit on anticipated TRAINING exposures, never on validation/test prevalence.
    averaged=dict(schedules[0])
    for key in ['positive_pixel_counts','valid_pixel_counts']:
        averaged[key]=np.array([s[key] for s in schedules],np.float64).mean(0).tolist()
    alpha=fitted_positive_weights(averaged)
    fit=dict(fitted_split='train',train_manifest_sha256=file_hash(data/'train.csv'),
        inventory_sha256=file_hash(data/'training_pixels.npz'),channels=CHANNELS,
        positive_weights=alpha,negative_weights=[1.]*16,channel_weights=[1.]*16,
        rule='clip(1 + 0.25*log(N_negative/N_positive), 0.5, 2); unsupported (<100 either label) -> 1',
        normalization='sum(weight*pixel_error)/sum(weight*valid_pixels) per channel',
        positive_pixel_counts=averaged['positive_pixel_counts'],valid_pixel_counts=averaged['valid_pixel_counts'],
        exposure_fit='mean of deterministic training schedules for epochs 1-3, before random augmentation',
        parameters_status='conservative candidate, not established optimum',training_initialization='fresh pretrained encoder + fresh adapters/decoder')
    (data/'pixel_balance.json').write_text(json.dumps(fit,indent=2))
    original_stats=json.loads((data/'statistics.json').read_text())
    old_draws=pd.read_csv(source/'data/train_balanced.csv',usecols=['patch_id'])
    positions=pd.Index(inventory['patch_ids']).get_indexer(old_draws.patch_id)
    if (positions<0).any():raise ValueError('Previous sampling contains patches outside this fixed training split')
    old_used=np.bincount(positions,minlength=len(inventory['patch_ids']))
    old_pos=(inventory['positive'].astype(np.int64)*old_used[:,None]).sum(0)
    old_valid=(inventory['valid'].astype(np.int64)*old_used[:,None]).sum(0)
    pos=np.array(fit['positive_pixel_counts']);valid=np.array(fit['valid_pixel_counts'])
    pd.DataFrame(dict(channel=CHANNELS,old_inverse_sigma_weight=original_stats['channel_weights'],
        proposed_channel_weight=1.,positive_fraction=pos/valid,positive_pixel_weight=alpha,negative_pixel_weight=1.,
        natural_positive_fraction=inventory['positive'].sum(0)/inventory['valid'].sum(0),
        previous_sampling_positive_fraction=old_pos/old_valid)).to_csv(review/'pixel_weight_table.csv',index=False)
    cfg.update(channel_weight_rule='uniform',pixel_weight_file=str(data/'pixel_balance.json'),
        pixel_sampling=dict(inventory=str(data/'training_pixels.npz'),epoch_size=epoch_size,max_repeats=4,natural_fraction=.5),
        overlap_weight=.5,batch_min=64,batch_max=64)
    for name in ['train_pixel_balance.json','train.json']:(project/'configs'/name).write_text(json.dumps(cfg,indent=2))
    control=dict(cfg,overlap_weight=1.)
    (project/'configs/train_pixel_balance_dice1_control.json').write_text(json.dumps(control,indent=2))
    no_boost=dict(cfg);no_boost.pop('pixel_weight_file')
    (project/'configs/train_pixel_balance_no_positive_boost.json').write_text(json.dumps(no_boost,indent=2))
    checked=validate(data)
    (run/'status.json').write_text(json.dumps(dict(stage='prepared_not_started',current_training_untouched=True,
        sampling_checks_epochs=3,new_training_executed=False,validation=checked),indent=2))
    print(pd.read_csv(review/'pixel_weight_table.csv').to_string(index=False),flush=True)
    print('PREPARED ONLY: no training launched',flush=True)


if __name__=='__main__':main()
