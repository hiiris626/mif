import unittest
from unittest.mock import patch
import torch
from vit_seg.distributed import DistributedPixelLoss,EvaluationLossAccumulator

class EmptyDiceTests(unittest.TestCase):
 def test_empty_analytic_gradients_and_ignore(self):
  y=torch.zeros((1,1,4,4),dtype=torch.uint8);y[:,:,0]=255
  z=torch.full(y.shape,-0.7,requires_grad=True)
  c=DistributedPixelLoss([1.],bce_weight=0,empty_dice_policy='mean_probability')
  loss=c(z,y);loss.backward();p=z.detach().sigmoid();valid=y!=255;n=valid.sum()
  torch.testing.assert_close(loss,p[valid].mean())
  torch.testing.assert_close(z.grad[valid],(p*(1-p)/n)[valid]);self.assertTrue((z.grad[~valid]==0).all())
 def test_sum_and_mean_differ_by_valid_count(self):
  z=torch.zeros((1,1,256,256),requires_grad=True);y=torch.zeros_like(z,dtype=torch.uint8)
  loss=DistributedPixelLoss([1.],bce_weight=0,empty_dice_policy='mean_probability')(z,y);loss.backward()
  self.assertAlmostEqual(float(loss),.5);self.assertAlmostEqual(float(z.grad[0,0,0,0]),.25/65536)
  self.assertAlmostEqual(float(z.grad.sum()),.25)
  z2=z.detach().clone().requires_grad_();z2.sigmoid().sum().backward();torch.testing.assert_close(z2.grad,z.grad*65536)
 def test_nonempty_dice_unchanged_and_gradient_finite_difference(self):
  y=torch.tensor([[[[1,0],[255,0]]]],dtype=torch.uint8)
  z=torch.tensor([[[[.2,-.3],[.7,1.]]]],requires_grad=True)
  old=DistributedPixelLoss([1.],bce_weight=0);new=DistributedPixelLoss([1.],bce_weight=0,empty_dice_policy='mean_probability')
  a=old(z,y);ga=torch.autograd.grad(a,z)[0];b=new(z,y);gb=torch.autograd.grad(b,z)[0]
  torch.testing.assert_close(a,b);torch.testing.assert_close(ga,gb)
  y.zero_();eps=1e-3;z0=z.detach();direction=torch.zeros_like(z0);direction[0,0,0,0]=eps
  fd=(new(z0+direction,y)-new(z0-direction,y))/(2*eps)
  g=torch.autograd.grad(new(z,y),z)[0][0,0,0,0];torch.testing.assert_close(g,fd,atol=2e-5,rtol=1e-3)
 def test_mixed_unknown_and_empty_evaluation_matches(self):
  torch.manual_seed(3);y=torch.zeros((3,2,5,5),dtype=torch.uint8);y[0,0,0,0]=1;y[1,1]=255;y[2]=255
  z=torch.randn(y.shape,requires_grad=True)
  criterion=DistributedPixelLoss([1.,2.],positive_weights=[2.,3.],empty_dice_policy='mean_probability',empty_dice_weight=.7)
  value=criterion(z,y);value.backward();self.assertTrue((z.grad[y==255]==0).all())
  acc=EvaluationLossAccumulator(criterion,'cpu')
  for i in range(3):acc.update(z[i:i+1],y[i:i+1])
  self.assertAlmostEqual(acc.result(),float(value),places=6)
 def test_all_unknown_is_zero_and_invalid_config_rejected(self):
  y=torch.full((1,1,3,3),255,dtype=torch.uint8);z=torch.zeros(y.shape,requires_grad=True)
  loss=DistributedPixelLoss([1.],empty_dice_policy='mean_probability')(z,y);loss.backward()
  self.assertEqual(float(loss),0);self.assertTrue((z.grad==0).all())
  for kw in [{'empty_dice_policy':'wrong'},{'empty_dice_weight':-1},{'empty_dice_weight':float('nan')},{'empty_dice_policy':'mean_probability','dice_scope':'positive_only'}]:
   with self.assertRaises(ValueError):DistributedPixelLoss([1.],**kw)
