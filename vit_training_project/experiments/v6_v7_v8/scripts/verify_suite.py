"""Check actual loader masks, weights, BCE gradients and all three configs."""
import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vit_seg.data import PixelDataset,CHANNELS
from vit_seg.bce_balance import load_pixel_weights,objective_channel_weights
from vit_seg.distributed import DistributedPixelLoss,EvaluationLossAccumulator
from vit_seg.binary_targets import BinaryTargetStore
from vit_seg.workflow import validate_config
P=Path(__file__).resolve().parents[1]

def main():
 rows=[]
 for name in ['v6','v7','v8']:
  cfg=json.loads((P/'configs'/f'{name}.json').read_text());validate_config(cfg)
  data=P/'runs'/name/'data';stats=json.loads((data/'statistics.json').read_text())
  ds=PixelDataset(data/'test.csv',stats,cfg['data_root'],256,False,cache_dir=data.parent/'cache')
  index=int(np.flatnonzero(ds.df.patch_id.to_numpy()==168299)[0]);sample=ds[index]
  label=sample['label'][None];t=sample['tissue'].bool();assert (label[0,:,t]!=255).all();assert (label[0,:,~t]==255).all()
  wp,wn=load_pixel_weights(cfg,data);assert wn==[1.]*16;assert objective_channel_weights(cfg,stats)==[1.]*16
  criterion=DistributedPixelLoss([1.]*16,positive_weights=wp,negative_weights=wn,dice_scope='all_valid')
  z=torch.zeros(label.shape,requires_grad=True);loss=criterion(z,label);loss.backward()
  assert torch.isfinite(loss) and (z.grad[label==0]>0).all() and (z.grad[label==255]==0).all()
  acc=EvaluationLossAccumulator(criterion,'cpu');acc.update(z,label);assert abs(acc.result()-loss.item())<1e-6
  row=dict(experiment=name,patch=168299,tissue_pixels=int(t.sum()),valid_channel_pixels=int((label!=255).sum()),positive_pixels=int((label==1).sum()),negative_pixels=int((label==0).sum()),weights=wp,loss_at_zero_logits=float(loss),all_tissue_zero_labels_supervised=True,missing_and_background_ignored=True)
  rows.append(row)
 (P/'review/real_data_verification.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
