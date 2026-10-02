"""Run production orchestration with four CPU ranks and a tiny substituted model.

The data loop, optimizer, stopping, checkpoints, resume, threshold selection,
test evaluation and report/export code are real. CUDA/SyncBN are not validated.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from vit_seg import train_ddp, predict, report
from vit_seg.artifacts import data_signature
from vit_seg.data import CHANNELS
from vit_seg.cache import build as build_cache


def model(config, device):
    return torch.nn.Conv2d(3,16,1).to(device)


def main():
    torch.set_num_threads(1);dist.init_process_group('gloo');rank=dist.get_rank()
    temporary=tempfile.TemporaryDirectory(prefix='vit_production_cpu_') if rank==0 else None
    names=[temporary.name if rank==0 else None];dist.broadcast_object_list(names,src=0)
    root=Path(names[0]);data=root/'data';results=root/'results'
    cfg=json.loads(Path('configs/train.json').read_text())
    cfg.update(data_root=str(root),tile_size=16,batch_size=1,grad_accum=2,num_workers=0,
               epochs=4,patience=1,min_delta=2.,early_stop_warmup=0,warmup_epochs=1,sync_batchnorm=False,use_data_cache=True,dice_scope='positive_only')
    cfg['threshold_selection'].update(min_positive_pixels=1,min_negative_pixels=1,min_patients_per_label=1)
    if rank==0:
        data.mkdir();results.mkdir()
        stats=dict(q=[20]*16,channel_weights=[1]*16,source_spec={'root':str(root)})
        (data/'statistics.json').write_text(json.dumps(stats))
        (data/'COMPLETE.json').write_text(json.dumps(dict(complete_dataset=True,n_source=17,n_retained=17)))
        pd.DataFrame(dict(orion_slide_id=['p0','p1'],split=['train','val'])).to_csv(data/'patient_split.csv',index=False)
        all_rows=[]
        for split,n in [('train_balanced',12),('val',3),('test',2)]:
            rows=[]
            for i in range(n):
                name=f'{split}_{i}';he=np.full((16,16,3),100,np.uint8);he[:2]=255
                mif=np.zeros((16,16,16),np.uint8);mif[2:,2:10,:2]=20
                cv2.imwrite(str(root/f'{name}.jpeg'),he);tifffile.imwrite(root/f'{name}.tiff',mif)
                row=dict(patch_id=len(all_rows),orion_slide_id=f'{split}_p{i}',image_path=f'{name}.jpeg',target_path=f'{name}.tiff')
                rows.append(row); all_rows.append(row)
            pd.DataFrame(rows).to_csv(data/f'{split}.csv',index=False)
        (data/'VALIDATED.json').write_text(json.dumps(dict(signature=data_signature(data))))
        pd.DataFrame(all_rows).to_csv(data/'patch_manifest.csv',index=False)
        build_cache(data,root/'cache',workers=2,reserve_gib=0)
    dist.barrier()
    args=argparse.Namespace(data=str(data),out=str(results/'model'),resume=None,checkpoint=None,
                            evaluate_only=None,calibrate=False,thresholds=None)
    with patch.object(train_ddp,'build_model',model), patch.object(train_ddp,'audit_lora',return_value={'tiny_test':True}), \
         patch.object(train_ddp,'DDP',side_effect=lambda module,**kwargs:DistributedDataParallel(module)), \
         patch.object(torch,'autocast',side_effect=lambda *a,**kw:nullcontext()), \
         patch.object(torch.cuda,'get_rng_state',side_effect=lambda *a:torch.get_rng_state()), \
         patch.object(torch.cuda,'set_rng_state'),patch.object(train_ddp,'save_training_artifacts'):
        train_ddp.run(args,cfg,rank,torch.device('cpu'))
        checkpoint=results/'model/last.pt';before=checkpoint.stat().st_mtime_ns
        args.resume=str(checkpoint)
        train_ddp.run(args,cfg,rank,torch.device('cpu'))
        assert checkpoint.stat().st_mtime_ns==before, 'Early-stopped training incorrectly continued'
        args.resume=None;args.checkpoint=str(results/'model/best.pt');args.out=str(results)
        args.evaluate_only='val';args.calibrate=True
        train_ddp.run(args,cfg,rank,torch.device('cpu'))
        dist.barrier()
        args.evaluate_only='test';args.calibrate=False;args.thresholds=str(results/'thresholds.json')
        train_ddp.run(args,cfg,rank,torch.device('cpu'))
        dist.barrier()
    if rank==0:
        ck=torch.load(checkpoint,weights_only=False)
        assert ck['epoch']==1 and ck['step']==4
        assert all(np.isfinite(row['val_loss']) for row in ck['history']), 'Uneven validation lost samples'
        logs=pd.read_csv(results/'model/train_log.csv')
        assert len(logs)==2 and (logs.epoch_seconds > 0).all()
        for row in ck['history']:
            snapshot=results/'model/snapshots'/row['snapshot']
            assert snapshot.is_file(), 'Missing epoch snapshot'
            saved=np.load(snapshot.with_suffix('.npz'))
            assert (saved['probabilities'][:,:2] == 0).all(), 'Snapshot includes glass background'
        assert (results/'model/epoch_metrics/epoch_002.json').is_file()
        with patch.object(predict,'build_model',model):
            report.trained_figures(data,results,'cpu')
        assert (results/'test_metrics_default05.json').exists()
        assert (results/'test_threshold_comparison.csv').exists()
        assert (results/'predictions/patch_15_probabilities.npz').exists()
        binary=tifffile.imread(results/'predictions/patch_15_multilabel.tiff')
        display=tifffile.imread(results/'predictions/patch_15_positive_argmax_display.tiff')
        yy,xx=np.where(display > 0)
        assert (binary[display[yy,xx]-1,yy,xx] == 1).all()
        probability=np.load(results/'predictions/patch_15_probabilities.npz')
        assert (probability['probabilities'][:,~probability['tissue']] == 0).all()
        raw_display=tifffile.imread(results/'predictions/patch_15_argmax_display.tiff')
        tissue=probability['tissue']
        assert np.array_equal(raw_display[tissue],probability['probabilities'].argmax(0)[tissue]+1)
        assert (raw_display[~tissue]==0).all()
        result=dict(world_size=4,backend='gloo',tiny_model_substituted=True,production_train_loop=True,
                    early_stop_after_two_epochs=True,completed_resume_did_not_retrain=True,
                    val_threshold_selection=True,dual_threshold_test_metrics=True,final_report_and_prediction=True,
                    gradient_accumulation_and_partial_group=True,argmax_only_positive_channels=True,
                    native_cache_enabled=True,gpu_executed=False)
        Path('review/production_cpu_verification.json').write_text(json.dumps(result,indent=2))
        print('PASS: production CPU train/stop/resume/calibrate/test/report/export',flush=True)
    dist.barrier();dist.destroy_process_group()
    if temporary:temporary.cleanup()


if __name__=='__main__':main()
