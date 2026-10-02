import unittest
import numpy as np
import pandas as pd
import torch
from vit_seg.pixel_sampling import make_schedule, strata, fitted_positive_weights
from vit_seg.distributed import DistributedPixelLoss, EvaluationLossAccumulator


class PixelBalanceTests(unittest.TestCase):
    def inventory(self):
        pos=np.zeros((120,16),np.uint32)
        for c in range(16):
            group=(np.arange(120)+c)%3;pos[group==1,c]=10;pos[group==2,c]=70
        return dict(patch_ids=np.arange(120),patients=np.array([f'P{i%4}' for i in range(120)]),
                    positive=pos,valid=np.full_like(pos,100),native_positive=pos.copy(),usable=np.ones(120,bool))

    def test_strata_use_pixel_coverage_and_do_not_call_lost_signals_negative(self):
        data=self.inventory();data['positive'][0,0]=0;data['native_positive'][0,0]=1
        pools,threshold=strata(data)
        self.assertNotIn(0,pools[0][0])
        self.assertAlmostEqual(threshold[0],.4)
        self.assertTrue((data['positive'][pools[0][1],0]==70).all())

    def test_sampling_coverage_cap_reproducibility_and_ddp_partition(self):
        data=self.inventory();a,report=make_schedule(data,360,42)
        b,_=make_schedule(data,360,42);c,_=make_schedule(data,360,43)
        np.testing.assert_array_equal(a,b);self.assertFalse(np.array_equal(a,c))
        counts=np.bincount(a,minlength=120);self.assertEqual(counts.sum(),360)
        self.assertGreaterEqual(counts.min(),1);self.assertLessEqual(counts.max(),4)
        for row in report['anchors']:
            self.assertEqual(row['false'],4*row['true1']);self.assertEqual(row['true1'],row['true2'])
        draws=[(int(a[i]),i) for rank in range(4) for i in range(rank,len(a),4)]
        self.assertEqual(len({d for _,d in draws}),360)
        # Repeated patches have unique draw IDs for independent augmentation.
        self.assertGreater(len(draws),len({p for p,_ in draws}))

    def test_adaptive_weights_are_bounded_and_not_uniformly_positive_boosted(self):
        summary=dict(positive_pixel_counts=[100,9000,5000,0],valid_pixel_counts=[10000]*4)
        w=fitted_positive_weights(summary)
        self.assertEqual(w[0],2);self.assertLess(w[1],1);self.assertEqual(w[2],1);self.assertEqual(w[3],1)
        self.assertTrue(all(.5<=v<=2 for v in w))

    def test_weighted_mse_value_gradient_and_mask(self):
        x=torch.full((1,1,1,4),float(np.log(1/3)),requires_grad=True)
        target=torch.tensor([[[[1,0,0,255]]]],dtype=torch.uint8)
        criterion=DistributedPixelLoss([1],overlap_weight=0,positive_weights=[2],dice_scope='positive_only')
        loss=criterion(x,target);self.assertAlmostEqual(float(loss),.3125,places=6)
        loss.backward()
        self.assertAlmostEqual(float(x.grad[0,0,0,0]),-.140625,places=6)
        self.assertAlmostEqual(float(x.grad[0,0,0,1]),.0234375,places=6)
        self.assertEqual(float(x.grad[0,0,0,3]),0)
        accumulator=EvaluationLossAccumulator(criterion,'cpu');accumulator.update(x,target)
        self.assertAlmostEqual(accumulator.result(),.3125,places=6)
        self.assertEqual(accumulator.components['positive_pixel_counts'],[1])
        self.assertEqual(accumulator.components['negative_pixel_counts'],[2])
        self.assertAlmostEqual(accumulator.components['per_channel_negative_mse'][0],.0625,places=6)

    def test_fractional_weight_keeps_single_pixel_normalization(self):
        x=torch.zeros((1,1,1,1),requires_grad=True)
        y=torch.ones_like(x,dtype=torch.uint8)
        criterion=DistributedPixelLoss([1],overlap_weight=0,positive_weights=[.5])
        loss=criterion(x,y);self.assertAlmostEqual(float(loss),.25)
        loss.backward();self.assertAlmostEqual(float(x.grad),-.25)
        accumulator=EvaluationLossAccumulator(criterion,'cpu');accumulator.update(x,y)
        self.assertAlmostEqual(accumulator.result(),.25)


if __name__=='__main__':unittest.main()
