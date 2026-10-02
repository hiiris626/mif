import json,tempfile,zlib,unittest
from pathlib import Path
import numpy as np
import torch
from vit_seg.binary_targets import BinaryTargetStore
from vit_seg.distributed import DistributedPixelLoss
from vit_seg.pixel_sampling import fit_probabilities,DistributedWeightedPatchSampler
from vit_seg.bce_balance import load_pixel_weights
from vit_seg.data import CHANNELS

class TissueExperimentTests(unittest.TestCase):
 def test_all_zero_tissue_negative_missing_channel_ignored(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);h=w=4;t=np.ones((h,w),bool);t[0]=False;eligible=np.zeros_like(t)
   available=np.ones(16,bool);available[2]=False
   raw=np.concatenate((np.array([0],dtype='<i8').view(np.uint8),np.packbits(available),np.zeros(2*h*w,np.uint8),np.packbits(np.stack((t,eligible)).ravel())))
   record=zlib.compress(raw.tobytes());(p/'shard_00000.bin').write_bytes(record)
   np.save(p/'index.npy',np.array([[0,len(record)]],dtype=np.uint64))
   (p/'COMPLETE.json').write_text(json.dumps(dict(format='binary_expression_zlib_v1',count=1,shard_size=1,height=h,width=w)))
   old_store=BinaryTargetStore(p);y,old,_=old_store.read(0);old_store.stream.close();assert not old.any()
   new_store=BinaryTargetStore(p,'available_channel_and_tissue');y,valid,tissue=new_store.read(0);new_store.stream.close()
   self.assertEqual(int(valid.sum()),15*12);self.assertFalse(valid[2].any());self.assertFalse(valid[:,0].any())
   labels=torch.tensor(np.where(valid,y,255))[None];z=torch.zeros(labels.shape,requires_grad=True)
   loss=DistributedPixelLoss([1.]*16,dice_scope='all_valid')(z,labels);loss.backward()
   self.assertTrue((z.grad[labels==0]>0).all());self.assertTrue((z.grad[labels==255]==0).all())
 def test_sampler_is_reproducible_shared_draw_and_shuffled_epochs(self):
  q=np.arange(1,101,dtype=float);q/=q.sum()
  samplers=[DistributedWeightedPatchSampler(q,4,r) for r in range(4)]
  rank_indices=[list(s) for s in samplers];full=np.array(rank_indices).T.ravel()
  np.testing.assert_array_equal(full,samplers[0].global_indices())
  self.assertEqual(len(full),100);self.assertLess(len(np.unique(full)),100)
  tagged=list(DistributedWeightedPatchSampler(q,4,0,with_draw_ids=True))
  self.assertEqual([item[1] for item in tagged],list(range(0,100,4)))
  self.assertEqual([item[0] for item in tagged],rank_indices[0])
  samplers[0].set_epoch(1);self.assertFalse(np.array_equal(full,samplers[0].global_indices()))
 def test_fit_improves_pixel_balance_with_finite_full_support(self):
  p=np.array([[0,0]]*90+[[100,100]]*10);v=np.full_like(p,100)
  q,report=fit_probabilities(p,v)
  self.assertAlmostEqual(q.sum(),1);self.assertTrue((q>0).all())
  self.assertGreater(report['expected_positive_fraction'][0],.1)
  self.assertLess(abs(report['expected_positive_fraction'][0]-.5),.1)
 def test_v7_uses_sqrt_of_frozen_v5_not_new_counts(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'weights.json'
   base=dict(fitted_split='train',channels=CHANNELS,grid=256,positive_counts=[1]*16,valid_counts=[100]*16,negative_weights=[1]*16)
   for mode,expected in [('none',1),('sqrt_v5',3)]:
    p.write_text(json.dumps(dict(base,mode=mode,positive_weights=[expected]*16,v5_source_weights=dict(positive_counts=[1]*16,valid_counts=[10]*16))))
    positive,negative=load_pixel_weights(dict(bce_pixel_weights_file=str(p),tile_size=256));self.assertEqual(positive,[expected]*16);self.assertEqual(negative,[1]*16)
