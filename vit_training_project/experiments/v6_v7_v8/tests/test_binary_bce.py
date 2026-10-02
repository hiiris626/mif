import math
import unittest
import torch
from vit_seg.distributed import DistributedPixelLoss,EvaluationLossAccumulator


class BinaryBCETests(unittest.TestCase):
    def test_bce_value_gradient_and_ignore(self):
        z=torch.zeros((1,1,1,3),requires_grad=True)
        y=torch.tensor([[[[1,0,255]]]],dtype=torch.uint8)
        loss=DistributedPixelLoss([1],overlap_weight=0)(z,y)
        self.assertAlmostEqual(float(loss),math.log(2),places=6)
        loss.backward()
        torch.testing.assert_close(z.grad,torch.tensor([[[[-.25,.25,0.]]]]))

    def test_extreme_logits_use_stable_bce(self):
        z=torch.tensor([[[[1000.,-1000.]]]],requires_grad=True)
        y=torch.tensor([[[[0,1]]]],dtype=torch.uint8)
        criterion=DistributedPixelLoss([1],overlap_weight=0)
        loss=criterion(z,y);loss.backward()
        self.assertEqual(float(loss),1000.)
        torch.testing.assert_close(z.grad,torch.tensor([[[[.5,-.5]]]]))
        acc=EvaluationLossAccumulator(criterion,'cpu');acc.update(z,y)
        self.assertEqual(acc.result(),1000.)

    def test_negative_images_contribute_dice_and_unknown_images_do_not(self):
        z=torch.zeros((3,1,2,2),requires_grad=True)
        y=torch.zeros_like(z,dtype=torch.uint8);y[0,0,0,0]=1;y[2]=255
        criterion=DistributedPixelLoss([1],bce_weight=0,dice_scope='all_valid')
        positive=1-(1+1e-6)/(3+1e-6)
        negative=1-1e-6/(2+1e-6)
        loss=criterion(z,y);self.assertAlmostEqual(float(loss),(positive+negative)/2,places=6)
        loss.backward()
        self.assertTrue((z.grad[1]>0).all());self.assertEqual(float(z.grad[2].abs().sum()),0)
        acc=EvaluationLossAccumulator(criterion,'cpu');acc.update(z,y)
        self.assertAlmostEqual(acc.result(),float(loss),places=6)
        self.assertEqual(acc.components['overlap_image_counts'],[2])

    def test_all_ignored_has_zero_loss_and_gradient(self):
        z=torch.randn(2,2,3,3,requires_grad=True);y=torch.full_like(z,255,dtype=torch.uint8)
        loss=DistributedPixelLoss([1,1])(z,y);loss.backward()
        self.assertEqual(float(loss),0);self.assertEqual(float(z.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
