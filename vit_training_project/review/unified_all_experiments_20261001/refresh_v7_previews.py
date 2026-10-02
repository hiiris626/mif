"""Reproducible 24-patch PNG-only comparison including v7; no training."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess

P = Path(__file__).resolve().parent
OUT = P.parent / 'v7_visualizations_20261001'

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, P / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    import pandas as pd
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] != '--render-only':
        e = load('evaluation', 'evaluate.py')
        e.OUT = OUT
        e.MODELS['v7'] = ('classification', e.PROJECT/'experiments/v6_v7_v8/runs/v7/results/model/best.pt', e.PROJECT/'experiments/v6_v7_v8/runs/v7/results/thresholds.json')
        sys.argv = [sys.argv[0], '--model', sys.argv[1], '--previews']
        e.main()
        return
    original = pd.read_csv(P/'preview_manifest.csv')
    data = pd.read_csv(P.parents[1]/'experiments/binary_bce/runs/prepared/data/test.csv')
    pool = data[(data.patient_id == 'CRC02') & ~data.patch_id.isin(original.patch_id)]
    extra = pool.sample(n=12, random_state=20261001).copy()
    extra['selection'] = 'fixed_seed_random'
    manifest = pd.concat([original, extra[original.columns.intersection(extra.columns)]], ignore_index=True)
    manifest.to_csv(OUT/'preview_manifest.csv', index=False)
    models = ['v1','v2','v3','v4','v5','v6','v7','v9','positive_dice_partial']
    for model in models:
        if '--render-only' in sys.argv:
            continue
        with (OUT/f'{model}.log').open('w') as log:
            subprocess.run([sys.executable, '-B', __file__, model], stdout=log, stderr=subprocess.STDOUT, check=True)
        print('inference complete', model, flush=True)
    r = load('rendering', 'render_previews.py')
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    figures = OUT/'figures'; figures.mkdir(exist_ok=True)
    legend = [Patch(color=r.MARKER_COLORS[i],label=c) for i,c in enumerate(r.CHANNELS)]
    e = load('registry', 'evaluate.py')
    threshold_paths = {m: e.MODELS[m][2] for m in models if m != 'v7'}
    threshold_paths['v7'] = e.PROJECT/'experiments/v6_v7_v8/runs/v7/results/thresholds.json'
    thresholds = {m: json.loads(p.read_text())['thresholds'] if p else [.5]*16 for m,p in threshold_paths.items()}
    for row in manifest.itertuples():
        pid = int(row.patch_id)
        if all((figures/f'patch_{pid}_{kind}.png').exists() for kind in ['all_models','focus','probabilities']):
            continue
        arrays = {m:np.load(OUT/m/'previews'/f'patch_{pid}.npz') for m in models}
        ref = arrays['v6']; tissue=ref['tissue'].astype(bool); available=ref['available'].astype(bool); binary=ref['expanded']
        for a in arrays.values():
            np.testing.assert_array_equal(a['expanded'], binary)
            assert np.isfinite(a['scores']).all()
        he = r.cv2.resize(r.read_he(r.ROOT/row.image_path),(256,256),interpolation=r.cv2.INTER_AREA)
        mif = r.read_mif(r.ROOT/row.target_path)
        mif = np.stack([r.cv2.resize(c,(256,256),interpolation=r.cv2.INTER_NEAREST) for c in mif])
        _,gt = r.gt_argmax(binary,mif,tissue,available)
        panels={'H&E':he,'GT dominant positive':gt}
        for m in models:
            _,panels[m] = r.argmax_display(arrays[m]['scores'],tissue,available,thresholds[m])
        for name,keys,shape,size in [('all_models',list(panels),(3,4),(16,13)),('focus',['H&E','GT dominant positive','v3','v5','v6','v7'],(2,3),(12,9))]:
            fig,axes=plt.subplots(*shape,figsize=size)
            for ax in axes.flat:ax.axis('off')
            for ax,key in zip(axes.flat,keys):ax.imshow(panels[key]);ax.set_title(key)
            fig.suptitle(f'CRC02 patch {pid} | {row.selection} | validation thresholds')
            fig.legend(handles=legend,loc='lower center',ncol=8,fontsize=8)
            fig.subplots_adjust(bottom=.09,top=.94,wspace=.03,hspace=.12)
            fig.savefig(figures/f'patch_{pid}_{name}.png',dpi=140);plt.close(fig)
        # Binary GT alongside true probabilities: avoids argmax hiding coexpression.
        fig,axes=plt.subplots(4,4,figsize=(20,15))
        for c,ax in enumerate(axes.flat):
            strips=[np.where(binary[c]==255,np.nan,(binary[c]==1).astype(float))]
            strips += [np.where(binary[c]==255,np.nan,arrays[m]['scores'][c]) for m in ['v5','v6','v7','v9']]
            ax.imshow(np.hstack(strips),cmap='magma',vmin=0,vmax=1,interpolation='nearest')
            ax.set_title(r.CHANNELS[c]+' | GT / v5 / v6 / v7 / v9',fontsize=9);ax.axis('off')
        fig.suptitle(f'CRC02 patch {pid} | binary GT and probabilities (0 to 1); ignored pixels blank')
        fig.tight_layout();fig.savefig(figures/f'patch_{pid}_probabilities.png',dpi=150);plt.close(fig)
        for a in arrays.values():a.close()
        print('rendered',pid,flush=True)
    (OUT/'COMPLETE.json').write_text(json.dumps({'patches':len(manifest),'models':models,'new_random_patches':12,'seed':20261001,'gt_coexpression_tie_break':'original intensity','thresholds':'frozen own validation; partial experiment 0.5','training_changed':False},indent=2))

if __name__ == '__main__':
    main()
