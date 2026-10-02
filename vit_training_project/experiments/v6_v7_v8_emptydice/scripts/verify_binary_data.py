"""CPU-only real-data probe: binary reader, worker safety, masking and BCE."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from vit_seg.data import PixelDataset
from vit_seg.distributed import DistributedPixelLoss,EvaluationLossAccumulator
from vit_seg.binary_targets import BinaryTargetStore
from vit_seg.cache import NativeCache
from vit_seg.bce_balance import load_pixel_weights, objective_channel_weights


def main():
    torch.set_num_threads(1);project=Path(__file__).resolve().parents[1];run=project/'runs/prepared';data=run/'data'
    cfg=json.loads((project/'configs/train.json').read_text());stats=json.loads((data/'statistics.json').read_text())
    info=json.loads((data/'binary_targets.json').read_text());store=BinaryTargetStore(info['folder'])
    cache=NativeCache(run/'cache',data)
    ids=[0,35422,60661,139439,180649,245825,314024,318449]
    positive_total=0
    for i in ids:
        target,valid,tissue=store.read(i);_,old,old_tissue=cache.read(i)
        assert set(np.unique(target)) <= {0,1}
        np.testing.assert_array_equal(valid,old!=255);np.testing.assert_array_equal(tissue,old_tissue)
        np.testing.assert_array_equal(target,(old==1).astype(np.uint8));positive_total+=int(target.sum())
    ds=PixelDataset(data/'train.csv',stats,cfg['data_root'],256,False,cache_dir=run/'cache')
    assert ds.binary_store is not None
    wp,wn=load_pixel_weights(cfg,data)
    criterion=DistributedPixelLoss(objective_channel_weights(cfg, stats),bce_weight=cfg['bce_weight'],
                                  overlap_weight=cfg['overlap_weight'],dice_scope=cfg['dice_scope'],
                                  positive_weights=wp,negative_weights=wn)
    observed=0
    with patch('vit_seg.data.read_mif',side_effect=AssertionError('Old intensity target read unexpectedly')):
        ds[0]  # Open target shard in parent before workers fork: exercises PID ownership.
        loader=DataLoader(ds,batch_size=2,num_workers=2,sampler=[0,2000,4000,6000])
        for batch in loader:
            y=batch['label'];logits=torch.zeros(y.shape,requires_grad=True)
            loss=criterion(logits,y);loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
            assert (logits.grad[y==255]==0).all()
            acc=EvaluationLossAccumulator(criterion,'cpu');acc.update(logits,y)
            assert abs(acc.result()-loss.item())<1e-6
            expected=(y!=255).any((2,3)).sum(0).tolist()
            assert acc.components['overlap_image_counts']==expected
            # Fixed population weights need not sum to one in this small batch.
            valid=y!=255;target=y.masked_fill(~valid,0).float()
            pixel_weight=target*torch.tensor(wp)[None,:,None,None]+(1-target)*torch.tensor(wn)[None,:,None,None]
            per_channel=(pixel_weight*valid).sum((0,2,3))/valid.sum((0,2,3)).clamp_min(1)
            cw=torch.tensor(objective_channel_weights(cfg, stats))*(valid.sum((0,2,3))>0)
            expected=float((per_channel*cw).sum()/cw.sum()*np.log(2))
            assert abs(acc.components['bce']-expected)<1e-6
            observed+=len(y)
    # Non-cache route must also read binary targets rather than old intensity TIFFs.
    uncached=PixelDataset(data/'train.csv',stats,cfg['data_root'],256,False)
    with patch('vit_seg.data.read_mif',side_effect=AssertionError('Old intensity target read unexpectedly')):
        np.testing.assert_array_equal(uncached[0]['label'].numpy(),ds[0]['label'].numpy())
    main_project=project.parent.parent;source=main_project/'runs/ddp_positive_dice'
    provenance=json.loads((source/'provenance.json').read_text())
    changed=[name for name,digest in provenance['code_sha256'].items()
             if hashlib.sha256((main_project/name).read_bytes()).hexdigest()!=digest]
    assert not changed,changed
    for name in ['train.csv','val.csv','test.csv','patient_split.csv','train_balanced.csv']:
        assert (data/name).read_bytes()==(source/'data'/name).read_bytes()
    result=dict(real_native_patches_checked=len(ids),positive_pixels_checked=positive_total,
        model_grid_patches_checked=observed,native_target_values=[0,1],validity_separate=True,
        cache_and_uncached_binary_loader=True,old_intensity_target_reads=0,two_worker_after_parent_read=True,
        stable_bce=True,negative_dice_pairs_counted=True,ignored_gradient_zero=True,
        current_training_source_unchanged=True,fixed_patient_splits_unchanged=True,
        original_manifest_files_preserved=True,train_fitted_pixel_weights=True,gpu_executed=False)
    (project/'review/real_data_cpu_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
