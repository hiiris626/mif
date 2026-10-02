"""Frozen train-only positive/negative BCE weights, with equal class mass."""
import json
from pathlib import Path
import numpy as np
from .artifacts import file_hash
from .data import CHANNELS


def objective_channel_weights(cfg, stats):
    """Keep completed checkpoint semantics; new configs ignore intensity sigma."""
    rule=cfg.get('channel_weighting', 'legacy_intensity_sigma')
    if rule=='uniform':return [1.]*len(CHANNELS)
    if rule=='legacy_intensity_sigma':return stats['channel_weights']
    raise ValueError('Unknown outer channel weighting rule')


def balanced_pixel_weights(positive, valid):
    positive=np.asarray(positive,np.float64);valid=np.asarray(valid,np.float64)
    negative=valid-positive
    if positive.shape!=valid.shape or not np.isfinite(valid).all() or not np.isfinite(positive).all():
        raise ValueError('Invalid pixel counts')
    if (positive<=0).any() or (negative<=0).any():
        raise ValueError('Equal two-label weighting requires both labels in every training channel')
    return negative/positive,np.ones_like(negative)


def load_pixel_weights(cfg, data=None):
    path=cfg.get('bce_pixel_weights_file')
    if not path:return None,None
    value=json.loads(Path(path).read_text())
    if value['fitted_split']!='train' or value['channels']!=CHANNELS or value['grid']!=cfg['tile_size']:
        raise ValueError('BCE pixel weights have incorrect source or channel/grid order')
    mode=value.get('mode','balanced')
    if mode=='none':positive=np.ones(len(CHANNELS));negative=positive.copy()
    elif mode=='sqrt_v5':
        source=value['v5_source_weights']
        ratio,_=balanced_pixel_weights(source['positive_counts'],source['valid_counts'])
        positive=np.sqrt(ratio);negative=np.ones_like(positive)
    elif mode=='balanced':positive,negative=balanced_pixel_weights(value['positive_counts'],value['valid_counts'])
    else:raise ValueError('Unknown BCE weighting mode')
    if not np.allclose(positive,value['positive_weights']) or not np.allclose(negative,value['negative_weights']):
        raise ValueError('BCE weights do not match equal-class-mass counts')
    if data is not None:
        data=Path(data);binary=json.loads((data/'binary_targets.json').read_text())
        if value.get('validity_policy','legacy_eligible')!=binary.get('validity_policy','legacy_eligible'):
            raise ValueError('BCE weights use another validity policy')
        if value['train_manifest_sha256']!=file_hash(data/'train.csv') or value['binary_manifest_sha256']!=binary['manifest_sha256']:
            raise ValueError('BCE weights belong to a different training split or binary dataset')
    return positive.tolist(),negative.tolist()
