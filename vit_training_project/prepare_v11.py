"""Prepare v10 objective with the frozen v8 patch sampling distribution."""
import json, os, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
src=P/'experiments/v10'; dst=P/'experiments/v11'; v8=P/'experiments/v6_v7_v8/runs/v8/data'
if dst.exists():raise SystemExit('v11 exists; refusing overwrite')
dst.mkdir()
for name in ['vit_seg','vit_matte','datacore','configs']:
    shutil.copytree(src/name,dst/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(dst/'scripts').mkdir();shutil.copy2(src/'scripts/run.sh',dst/'scripts/run.sh')
run=dst/'runs/v11';data=run/'data';data.mkdir(parents=True)
for f in (src/'runs/v10/data').iterdir():
    if f.is_file() and f.name not in ['VALIDATED.json','training_policy.json']:
        if f.suffix=='.csv':os.link(f,data/f.name)
        else:shutil.copy2(f,data/f.name)
for name in ['training_policy.json','patch_sampling.npz']:shutil.copy2(v8/name,data/name)
(run/'cache').mkdir()
for name in ['records.raw','COMPLETE.json']:os.link(src/'runs/v10/cache'/name,run/'cache'/name)
cfg=json.loads((src/'configs/v10.json').read_text())
cfg.update(experiment='v11',sampling_policy='pixel_balanced_patches',bce_pixel_weights_file=str(data/'bce_pixel_weights.json'))
(dst/'configs/v11.json').write_text(json.dumps(cfg,indent=2))
sys.path.insert(0,str(dst))
from vit_seg.bce_balance import load_pixel_weights
from vit_seg.pixel_sampling import DistributedWeightedPatchSampler
from vit_seg.validate import validate
wp,wn=load_pixel_weights(cfg,data)
with np.load(data/'patch_sampling.npz') as z: ids=z['patch_ids'];q=z['probabilities']
np.testing.assert_array_equal(ids,pd.read_csv(data/'train.csv',usecols=['patch_id']).patch_id.to_numpy())
assert np.isfinite(q).all() and (q>0).all() and np.isclose(q.sum(),1)
with np.load(P/'datasets/binary_expression_tissue_v2/training_pixels.npz') as z:
    np.testing.assert_array_equal(ids,z['patch_ids']);positive=z['positive'].astype(np.int64);valid=z['valid'].astype(np.int64)
sampler=DistributedWeightedPatchSampler(q,4,0,42);draw=sampler.global_indices()
expected=(positive*q[:,None]).sum(0)/(valid*q[:,None]).sum(0)
actual=positive[draw].sum(0)/valid[draw].sum(0)
weights=json.loads((data/'bce_pixel_weights.json').read_text())
assert weights==json.loads((src/'runs/v10/data/bce_pixel_weights.json').read_text())
pd.DataFrame({'channel':weights['channels'],'natural_positive_fraction':positive.sum(0)/valid.sum(0),
    'expected_sampled_positive_fraction':expected,'epoch1_sampled_positive_fraction':actual,'positive_weight':wp}).to_csv(run/'sampling_audit.csv',index=False)
check=validate(data)
(run/'prelaunch_check.json').write_text(json.dumps({'passed':True,'validation':check,'sampling_source':str(v8/'patch_sampling.npz'),
    'first_epoch_draws':len(draw),'first_epoch_unique_patches':len(np.unique(draw)),
    'weight_source':'v10 unchanged, natural train counts before sampling','expected_positive_fraction':expected.tolist()},indent=2))
(run/'status.json').write_text(json.dumps({'stage':'waiting_for_v10','experiment':'v11'},indent=2))
print('PASS: v11 data, weights, sampling IDs and first epoch draw; unique patches',len(np.unique(draw)))
