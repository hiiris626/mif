"""Count valid positive/negative patches before augmentation, replaying completed draws."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
from vit_seg.pixel_sampling import DistributedWeightedPatchSampler
run=P/'runs/v11'
with np.load(P.parents[1]/'datasets/binary_expression_tissue_v2/training_pixels.npz') as z:
    ids=z['patch_ids'];positive=z['positive'];valid=z['valid']
with np.load(run/'data/patch_sampling.npz') as z:
    np.testing.assert_array_equal(ids,z['patch_ids']);q=z['probabilities']
np.testing.assert_array_equal(ids,pd.read_csv(run/'data/train.csv',usecols=['patch_id']).patch_id.to_numpy())
cfg=json.loads((P/'configs/v11.json').read_text())
channels=json.loads((run/'data/bce_pixel_weights.json').read_text())['channels']
pos=(positive>0)&(valid>0);neg=(positive==0)&(valid>0);ignored=valid==0
assert np.all(pos.astype(int)+neg.astype(int)+ignored.astype(int)==1)
epochs=[]
for line in (run/'train.log').read_text().splitlines():
    try:r=json.loads(line)
    except ValueError:continue
    if 'epoch_seconds' in r:epochs.append(int(r['epoch'])-1)
assert len(epochs)==len(set(epochs))
counts=np.zeros((3,len(channels)),dtype=np.int64);details=[]
sampler=DistributedWeightedPatchSampler(q,4,0,cfg['seed'])
for epoch in epochs:
    sampler.set_epoch(epoch);draw=sampler.global_indices()
    totals=np.array([a[draw].sum(0) for a in [pos,neg,ignored]])
    counts+=totals
    for c,name in enumerate(channels):
        details.append(dict(epoch=epoch+1,channel=name,positive_draws=int(totals[0,c]),negative_draws=int(totals[1,c]),ignored_draws=int(totals[2,c])))
rows=[]
for c,name in enumerate(channels):
    npos=int(pos[:,c].sum());nneg=int(neg[:,c].sum());sp,sn,si=map(int,counts[:,c])
    expected_pos=float(q@pos[:,c]);expected_neg=float(q@neg[:,c])
    rows.append(dict(channel=name,natural_positive_patches=npos,natural_negative_patches=nneg,natural_ignored_patches=int(ignored[:,c].sum()),
        natural_positive_percent=100*npos/(npos+nneg),natural_negative_percent=100*nneg/(npos+nneg),
        expected_sampled_positive_percent=100*expected_pos/(expected_pos+expected_neg),
        sampled_positive_draws=sp,sampled_negative_draws=sn,sampled_ignored_draws=si,
        sampled_positive_percent=100*sp/(sp+sn),sampled_negative_percent=100*sn/(sp+sn)))
out=run/'results'
pd.DataFrame(rows).to_csv(out/'patch_balance.csv',index=False)
pd.DataFrame(details).to_csv(out/'patch_balance_per_epoch.csv',index=False)
(out/'patch_balance_definition.json').write_text(json.dumps(dict(epochs=[e+1 for e in epochs],patches_per_epoch=len(ids),
    positive='at least one positive target pixel and valid tissue in this channel',negative='valid tissue exists, zero positive pixels',
    ignored='zero valid pixels',grid=256,augmentation='before random augmentation',draws='reconstructed from recorded seed and sampler; repeated patches counted per draw'),indent=2))
print(pd.DataFrame(rows).to_string(index=False))
print('epochs',len(epochs),'draws',len(epochs)*len(ids))
