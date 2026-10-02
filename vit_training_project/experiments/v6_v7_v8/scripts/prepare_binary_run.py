"""Attach a completed binary dataset to an isolated BCE run; never launch training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vit_seg.validate import validate
from vit_seg.cache import NativeCache
from vit_seg.binary_targets import BinaryTargetStore


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);args=p.parse_args()
    dataset=Path(args.dataset).resolve();project=Path(__file__).resolve().parents[1];run=project/'runs/prepared'
    receipt=json.loads((dataset/'BINARY_DATASET_COMPLETE.json').read_text())
    if receipt['binary_target_values']!=[0,1] or not receipt['ignore_is_separate']:raise ValueError('Wrong label semantics')
    BinaryTargetStore(dataset/'targets');checked=validate(dataset)
    data=run/'data';data.mkdir(exist_ok=True,parents=True)
    for path in dataset.rglob('*'):
        relative=path.relative_to(dataset)
        if relative.parts[0] in ('targets','examples'):continue
        if path.suffix not in ('.csv','.json'):continue
        dest=data/relative;dest.parent.mkdir(exist_ok=True,parents=True)
        if path.name=='VALIDATED.json':shutil.copy2(path,dest)
        elif not dest.exists():os.link(path,dest)
    source=Path(receipt['source_run']);cache=run/'cache';cache.mkdir(exist_ok=True,parents=True)
    for name in ('records.raw','COMPLETE.json'):
        if not (cache/name).exists():os.link(source/'cache'/name,cache/name)
    NativeCache(cache,data)
    configured=(data/'training_policy.json').is_file() and (data/'bce_pixel_weights.json').is_file()
    checked=validate(data)
    # Binary labels override the old packed labels. The large source cache is
    # reused only to supply H&E and geometry, without a second RGB data copy.
    (run/'status.json').write_text(json.dumps(dict(stage='prepared_not_started' if configured else 'data_ready_weights_pending',new_training_executed=False,
        dataset=str(dataset),pixel_loss='BCEWithLogitsLoss',dice_scope='all_valid',
        negative_patch_dice_included=True,
        sampling_policy='shuffle_patches' if configured else 'pending_configuration',
        bce_weights_file=str(data/'bce_pixel_weights.json') if configured else None,
        channel_weights='selected by config: uniform for new runs; legacy sigma for archived checkpoints',validation=checked),indent=2))
    print('DATA READY: run configure_random_bce.py to fit/check frozen class weights; no training launched',flush=True)


if __name__=='__main__':main()
