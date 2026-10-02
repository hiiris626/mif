import math
import unittest
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from vit_seg.bce_balance import balanced_pixel_weights, objective_channel_weights
from vit_seg.distributed import DistributedPixelLoss,EvaluationLossAccumulator
from vit_seg.train_ddp import loader_for


class RandomBalancedBCETests(unittest.TestCase):
    def test_new_objective_ignores_intensity_sigma(self):
        stats={'channel_weights':[.5,2.]*8}
        self.assertEqual(objective_channel_weights({'channel_weighting':'uniform'},stats),[1.]*16)
        self.assertEqual(objective_channel_weights({},stats),stats['channel_weights'])
        with self.assertRaises(ValueError):objective_channel_weights({'channel_weighting':'typo'},stats)

    def test_equal_population_mass_and_value_and_gradient(self):
        wp,wn=balanced_pixel_weights([1],[4])
        self.assertEqual(wp[0],3);self.assertEqual(wn[0],1)
        z=torch.zeros((1,1,1,5),requires_grad=True)
        y=torch.tensor([[[[1,0,0,0,255]]]],dtype=torch.uint8)
        criterion=DistributedPixelLoss([1],overlap_weight=0,positive_weights=wp,negative_weights=wn)
        value=criterion(z,y);self.assertAlmostEqual(float(value),1.5*math.log(2),places=6)
        value.backward();self.assertAlmostEqual(float(-z.grad[0,0,0,0]),float(z.grad[0,0,0,1:4].sum()),places=6)
        self.assertEqual(float(z.grad[0,0,0,4]),0)
        acc=EvaluationLossAccumulator(criterion,'cpu');acc.update(z,y)
        self.assertAlmostEqual(acc.result(),float(value),places=6)

    def test_sparse_channel_is_not_silently_clipped(self):
        wp,wn=balanced_pixel_weights([1,9],[1000,10])
        self.assertEqual(wp[0],999)
        np.testing.assert_allclose(wp*np.array([1,9]),wn*np.array([999,1]))
        for pos in ([0],[10]):
            with self.assertRaises(ValueError):balanced_pixel_weights(pos,[10])

    def test_class_weights_do_not_modify_dice(self):
        z=torch.randn(2,2,3,3);y=torch.randint(0,2,z.shape);y[1,0]=0
        a=DistributedPixelLoss([1,1],bce_weight=0)
        b=DistributedPixelLoss([1,1],bce_weight=0,positive_weights=[100,2],negative_weights=[.5,.75])
        torch.testing.assert_close(a(z,y),b(z,y))

    def test_four_rank_full_coverage_and_epoch_shuffle_with_partial_batch(self):
        dataset=list(range(223400));ranks=[];next_epoch=[]
        for rank in range(4):
            sampler=DistributedSampler(dataset,num_replicas=4,rank=rank,shuffle=True,seed=42,drop_last=False)
            ranks.append(list(sampler));sampler.set_epoch(1);next_epoch.append(list(sampler))
        self.assertEqual(len(set(sum(ranks,[]))),223400)
        self.assertEqual(sorted(sum(ranks,[])),dataset)
        self.assertNotEqual(ranks,next_epoch)
        self.assertTrue(all(len(ids)==55850 for ids in ranks))
        loader=loader_for(dataset,64,ranks[0],dict(sampling_policy='shuffle_patches',num_workers=0),training=True)
        self.assertFalse(loader.drop_last);self.assertEqual(len(loader),873)
        sizes=[len(batch) for batch in loader];self.assertEqual(sum(sizes),55850);self.assertEqual(sizes[-1],42)


if __name__=='__main__':unittest.main()
