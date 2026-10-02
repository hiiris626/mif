"""Bounded real-GPU throughput check. Discard all updates; never save a model."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from vit_seg.data import PixelDataset
from vit_seg.distributed import DistributedPixelLoss, optimizer_for
from vit_seg.model import build_model
from vit_seg.train_ddp import loader_for


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--config',required=True)
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=12)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());data=Path(args.data)
    rank=int(os.environ['RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    dist.init_process_group('nccl');torch.manual_seed(42)
    stats=json.loads((data/'statistics.json').read_text())
    model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(build_model(cfg,device))
    ddp=DistributedDataParallel(model,device_ids=[rank],broadcast_buffers=False,gradient_as_bucket_view=True)
    optimizer=optimizer_for(model,cfg)
    alpha=json.loads(Path(cfg['pixel_weight_file']).read_text())['positive_weights'] if cfg.get('pixel_weight_file') else None
    weights=[1.]*16 if cfg.get('channel_weight_rule')=='uniform' else stats['channel_weights']
    loss_fn=DistributedPixelLoss(weights,cfg['overlap_loss'],cfg['mse_weight'],cfg['overlap_weight'],cfg.get('dice_scope','all_valid'),alpha).to(device)
    ds=PixelDataset(data/'train_balanced.csv',stats,cfg['data_root'],cfg['tile_size'],True,seed=42,cache_dir=data.parent/'cache')
    loader=loader_for(ds,64,DistributedSampler(ds,num_replicas=4,rank=rank,shuffle=True),cfg,True)
    waits=[];steps=[];before=time.monotonic();start=before
    for i,batch in enumerate(loader):
        ready=time.monotonic();wait=ready-before
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=loss_fn(ddp(batch['image'].to(device,non_blocking=True)),batch['label'].to(device,non_blocking=True))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite probe loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        torch.cuda.synchronize();before=time.monotonic()
        if i>=3:waits.append(wait);steps.append(before-ready)
        if rank==0:print(json.dumps(dict(probe_batch=i+1,loss=float(loss),data_wait_seconds=wait,step_seconds=before-ready)),flush=True)
        if i+1>=args.steps:break
    report=dict(rank=rank,batch_size=64,warmup_steps=3,measured_steps=len(steps),
                total_seconds=time.monotonic()-start,mean_data_wait_seconds=float(np.mean(waits)),
                mean_step_including_sync_seconds=float(np.mean(steps)),
                mean_iteration_seconds=float(np.mean(waits)+np.mean(steps)),
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                real_model=True,real_native_patches=True,cache_enabled=True,model_saved=False,
                small_hot_subset=True)
    reports=[None]*4 if rank==0 else None;dist.gather_object(report,reports,dst=0)
    if rank==0:
        Path(args.out).write_text(json.dumps(dict(ranks=reports,global_patches_per_second=256/max(r['mean_iteration_seconds'] for r in reports)),indent=2))
    dist.destroy_process_group()


if __name__=='__main__':main()
