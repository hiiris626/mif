"""Prepare isolated v10 from v7 using current tissue-valid training counts."""
import json, os, shutil
from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
src = P/'experiments/v6_v7_v8'
dst = P/'experiments/v10'
if dst.exists():
    raise SystemExit('v10 already exists; refusing to overwrite')
dst.mkdir()
for name in ['vit_seg','vit_matte','datacore','configs']:
    shutil.copytree(src/name,dst/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(dst/'scripts').mkdir()
shutil.copy2(src/'scripts/run.sh',dst/'scripts/run.sh')
run=dst/'runs/v10';data=run/'data';data.mkdir(parents=True)
for f in (src/'runs/v7/data').iterdir():
    if f.is_file() and f.name not in ['bce_pixel_weights.json','VALIDATED.json']:
        shutil.copy2(f,data/f.name)
(run/'cache').mkdir()
for name in ['records.raw','COMPLETE.json']:
    os.link(src/'runs/v7/cache'/name,run/'cache'/name)
weights=json.loads((src/'runs/v7/data/bce_pixel_weights.json').read_text())
with np.load(P/'datasets/binary_expression_tissue_v2/training_pixels.npz') as z:
    ids=pd.read_csv(data/'train.csv',usecols=['patch_id']).patch_id.to_numpy()
    np.testing.assert_array_equal(ids,z['patch_ids'])
    positive=z['positive'].astype(np.int64).sum(0)
    valid=z['valid'].astype(np.int64).sum(0)
np.testing.assert_array_equal(positive,weights['positive_counts'])
np.testing.assert_array_equal(valid,weights['valid_counts'])
ratio=(valid-positive)/positive
wp=np.minimum(16,np.maximum(1,ratio)**.6)
weights.pop('v5_source_weights');weights.pop('v5_source_sha256')
weights.update(mode='capped_power',power=.6,maximum=16.,minimum=1.,
    negative_counts=(valid-positive).tolist(),ratios=ratio.tolist(),positive_weights=wp.tolist(),
    rule='min(16, max(1, N_negative / N_positive) ** 0.6)',
    denominator='unweighted valid pixel count per channel',
    scope='unique training patches, 256 grid, tissue AND available channel, before augmentation')
(data/'bce_pixel_weights.json').write_text(json.dumps(weights,indent=2))
cfg=json.loads((src/'configs/v7.json').read_text())
cfg.update(experiment='v10',bce_pixel_weights_file=str(data/'bce_pixel_weights.json'))
(dst/'configs/v10.json').write_text(json.dumps(cfg,indent=2))
pd.DataFrame({'channel':weights['channels'],'positive_pixels':positive,'negative_pixels':valid-positive,
    'ratio':ratio,'v10_positive_weight':wp}).to_csv(run/'weight_audit.csv',index=False)
print(dst)
print('Verified current counts from all',len(ids),'unique train patches; weights:',wp.tolist())
