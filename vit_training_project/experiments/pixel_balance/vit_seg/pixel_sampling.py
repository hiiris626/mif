"""Patient-mixed natural coverage + capped channel-stratified extra draws.

4 false : 1 true1 : 1 true2 holds for each channel's EXTRA draws. All valid
channels remain supervised; their final marginal ratios are measured, not
claimed to be equal to the anchor ratio.
"""
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Sampler
from .data import CHANNELS


def strata(inventory):
    positive,valid=inventory['positive'],inventory['valid']
    coverage=np.divide(positive,valid,out=np.zeros(positive.shape,dtype=np.float64),where=valid>0)
    pools=[];threshold=[]
    for c in range(positive.shape[1]):
        observed=valid[:,c]>0
        present=observed&(positive[:,c]>0)
        cutoff=float(coverage[present,c].mean()) if present.any() else 0.
        absent=observed&(positive[:,c]==0)&(inventory['native_positive'][:,c]==0)&inventory['usable']
        pools.append((np.flatnonzero(absent),np.flatnonzero(present&(coverage[:,c]>cutoff)),
                      np.flatnonzero(present&(coverage[:,c]<=cutoff))))
        threshold.append(cutoff)
    return pools,np.array(threshold)


def patient_mix(indices,patients,rng):
    """Shuffle within patients and interleave across the complete epoch."""
    keys=np.empty(len(indices),np.float64)
    groups=np.unique(patients[indices]);rng.shuffle(groups)
    for patient in groups:
        positions=np.flatnonzero(patients[indices]==patient);rng.shuffle(positions)
        keys[positions]=(np.arange(len(positions))+rng.random(len(positions)))/len(positions)
    return indices[np.argsort(keys,kind='stable')]


def make_schedule(inventory,epoch_size,seed,max_repeats=4,natural_fraction=.5):
    rng=np.random.default_rng(seed);n=len(inventory['patch_ids'])
    if not 0<n<=epoch_size<=n*max_repeats:raise ValueError('Epoch size must cover all patches and fit repeat cap')
    if not 0<natural_fraction<=1:raise ValueError('Invalid natural mixture')
    pools,cutoffs=strata(inventory);used=np.zeros(n,np.int64)
    natural_target=max(n,int(epoch_size*natural_fraction))
    # One complete pass, then uniformly draw remaining natural tickets.
    natural=rng.permutation(n);used+=1
    def draw(pool,count):
        tickets=np.repeat(pool,max_repeats-used[pool])
        if len(tickets)<count:raise ValueError('Repeat cap exceeded')
        picked=tickets[rng.choice(len(tickets),size=count,replace=False)] if count else np.empty(0,np.int64)
        used[:]+=np.bincount(picked,minlength=n)
        return picked
    if natural_target>n:natural=np.concatenate((natural,draw(np.arange(n),natural_target-n)))
    pieces=[natural];anchor_rows=[]
    requested_blocks=(epoch_size-natural_target)//(len(pools)*6)
    # Reserve scarce trustworthy negatives first. Otherwise common anchors
    # can consume their shared patches' repeat budgets before they get a turn.
    feasibility=[min(int((max_repeats-used[p[0]]).sum())//4,
                     int((max_repeats-used[p[1]]).sum()),int((max_repeats-used[p[2]]).sum())) for p in pools]
    tie=rng.random(len(pools))
    order=sorted(range(len(pools)),key=lambda c:(feasibility[c],tie[c]))
    for c in order:
        false,high,low=pools[c]
        capacity=[int((max_repeats-used[p]).sum()) for p in (false,high,low)]
        blocks=min(requested_blocks,capacity[0]//4,capacity[1],capacity[2])
        for pool,count in ((false,4*blocks),(high,blocks),(low,blocks)):pieces.append(draw(pool,count))
        anchor_rows.append(dict(channel=CHANNELS[c],false=4*blocks,true1=blocks,true2=blocks,
            requested_draws=requested_blocks*6,actual_draws=blocks*6,
            limited=blocks<requested_blocks,available_false=len(false),available_true1=len(high),available_true2=len(low),
            coverage_threshold=float(cutoffs[c])))
    fill=epoch_size-sum(map(len,pieces));pieces.append(draw(np.arange(n),fill))
    indices=patient_mix(np.concatenate(pieces),inventory['patients'],rng)
    assert len(indices)==epoch_size and used.max()<=max_repeats and used.min()>=1
    pos=(inventory['positive'].astype(np.int64)*used[:,None]).sum(0)
    valid=(inventory['valid'].astype(np.int64)*used[:,None]).sum(0)
    observed=inventory['valid']>0
    empty=((inventory['positive']==0)&observed)
    negative_patches=(empty*used[:,None]).sum(0)
    observed_patches=(observed*used[:,None]).sum(0)
    patients={p:int(used[inventory['patients']==p].sum()) for p in np.unique(inventory['patients'])}
    summary=dict(epoch_size=epoch_size,unique_patches=n,max_repeats=int(used.max()),
        natural_draws=len(natural)+fill,targeted_draws=sum(a['actual_draws'] for a in anchor_rows),
        anchor_ratio='false:true1:true2=4:1:1; shortages capped, filled by natural draws',
        all_valid_channels_supervised=True,patient_draw_counts=patients,
        positive_pixel_counts=pos.tolist(),valid_pixel_counts=valid.tolist(),
        positive_pixel_fraction=np.divide(pos,valid,out=np.zeros(len(pos),float),where=valid>0).tolist(),
        actual_negative_patch_fraction=np.divide(negative_patches,observed_patches,out=np.zeros(len(pos),float),where=observed_patches>0).tolist(),
        anchors=sorted(anchor_rows,key=lambda row:CHANNELS.index(row['channel'])))
    return indices,summary


class PatientChannelSampler(Sampler):
    def __init__(self,dataset,inventory_path,epoch_size,rank,world,seed=42,max_repeats=4,natural_fraction=.5):
        with np.load(inventory_path) as src:self.inventory={k:src[k] for k in src.files}
        if not np.array_equal(self.inventory['patch_ids'],dataset.df.patch_id.to_numpy()):
            raise ValueError('Pixel inventory and dataset row order differ')
        if epoch_size%world:raise ValueError('Global epoch must divide world size')
        self.epoch_size,self.rank,self.world=epoch_size,rank,world
        self.seed,self.max_repeats,self.natural_fraction=seed,max_repeats,natural_fraction
        self.set_epoch(0)

    def set_epoch(self,epoch):
        self.epoch=epoch
        self.indices,self.summary=make_schedule(self.inventory,self.epoch_size,self.seed+epoch*1000003,
                                               self.max_repeats,self.natural_fraction)

    def __len__(self):return self.epoch_size//self.world

    def __iter__(self):
        for draw in range(self.rank,self.epoch_size,self.world):yield int(self.indices[draw]),draw


def fitted_positive_weights(summary,gain=.25,minimum=.5,maximum=2.):
    """Tempered log-odds, capped; never unbounded inverse prevalence."""
    pos=np.array(summary['positive_pixel_counts'],np.float64)
    valid=np.array(summary['valid_pixel_counts'],np.float64);neg=valid-pos
    weights=np.clip(1+gain*np.log(np.maximum(neg,1)/np.maximum(pos,1)),minimum,maximum)
    weights[(pos<100)|(neg<100)]=1.
    return weights.tolist()
