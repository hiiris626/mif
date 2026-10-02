import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vit_seg.data import PixelDataset,CHANNELS
from vit_seg.binary_targets import BinaryTargetStore
from vit_seg.distributed import DistributedPixelLoss
from vit_seg.bce_balance import load_pixel_weights
P=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
provenance=json.loads((P/'runs/v6/provenance.json').read_text())
assert all(hashlib.sha256((P/f).read_bytes()).hexdigest()==h for f,h in provenance['code_sha256'].items())
rows=[]
for name in ['v6','v7','v8']:
 cfg=json.loads((P/f'configs/{name}.json').read_text());data=P/f'runs/{name}/data'
 ds=PixelDataset(data/'test.csv',data/'statistics.json',cfg['data_root'],256,False,cache_dir=data.parent/'cache')
 idx=int(np.flatnonzero(ds.df.patch_id.to_numpy()==168299)[0]);sample=ds[idx];lab=sample['label'][None];tissue=sample['tissue'].bool()
 allneg=tissue & (lab[0]==0).all(0)
 assert allneg.any() and (lab[0,:,allneg]==0).all()
 wp,wn=load_pixel_weights(cfg,data);results={}
 for component,bw,dw in [('bce',1,0),('dice',0,1)]:
  crit=DistributedPixelLoss([1.]*16,bce_weight=bw,overlap_weight=dw,positive_weights=wp,negative_weights=wn,dice_scope='all_valid')
  z=torch.zeros(lab.shape,requires_grad=True);loss=crit(z,lab);loss.backward()
  g=z.grad[0,:,allneg];assert (g>0).all() and (z.grad[lab==255]==0).all()
  results[component]={'loss':float(loss),'all_negative_region_gradient_min':float(g.min()),'all_negative_region_gradient_max':float(g.max()),'positive_gradients':int((g>0).sum())}
 row={'experiment':name,'patch':168299,'all_channels_negative_tissue_pixels':int(allneg.sum()),'supervised_channel_pixels_in_this_region':int(allneg.sum())*16,'negative_weights':wn,'components':results}
 ds.training=True;aug=[]
 for epoch in range(3):
  ds.epoch=epoch;sample=ds[idx];y=sample['label'];t=sample['tissue'].bool()
  assert (y[:,t]!=255).all() and (y[:,~t]==255).all()
  aug.append({'epoch':epoch,'all_negative_tissue_pixels':int((t&(y==0).all(0)).sum())})
 row['augmentation_checks']=aug;rows.append(row)
# Fully negative image/channel: document Dice gradient limitation explicitly.
empty=torch.zeros((1,1,256,256),dtype=torch.uint8);empty_rows=[]
for component,bw,dw in [('bce',1,0),('dice',0,1)]:
 z=torch.zeros(empty.shape,requires_grad=True)
 loss=DistributedPixelLoss([1.],bce_weight=bw,overlap_weight=dw,dice_scope='all_valid')(z,empty);loss.backward()
 empty_rows.append({'component':component,'loss':float(loss),'gradient_per_pixel':float(z.grad.flatten()[0]),'gradient_sum':float(z.grad.sum())})
report={'running_source_matches_provenance':True,'real_patch_checks':rows,'fully_negative_channel_256x256_at_probability_0_5':empty_rows,'note':'Positive gradient lowers predicted probability under gradient descent. Empty-channel soft Dice is included but has a near-zero gradient with epsilon=1e-6; BCE supplies effective negative supervision.'}
(P/'review/negative_supervision_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
