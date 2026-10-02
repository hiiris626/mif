import unittest
import torch
from vit_seg.distributed import DistributedPixelLoss, EvaluationLossAccumulator
from vit_seg.prepare import patient_split, patient_id
import pandas as pd


class PositiveDiceTests(unittest.TestCase):
    def test_negative_images_keep_mse_gradient_but_no_dice(self):
        x=torch.zeros(2,2,2,2,requires_grad=True)
        y=torch.zeros_like(x,dtype=torch.uint8)
        y[0,0,0,0]=1; y[1,1]=255
        mse=DistributedPixelLoss([1,2],overlap_weight=0,dice_scope='positive_only')
        dice=DistributedPixelLoss([1,2],mse_weight=0,dice_scope='positive_only')
        mse(x,y).backward(); gm=x.grad.clone(); x.grad.zero_()
        dice(x,y).backward(); gd=x.grad.clone()
        self.assertTrue((gm[0,1]>0).all()); self.assertTrue((gm[1,0]>0).all())
        self.assertEqual(float(gd[0,1].abs().sum()+gd[1,0].abs().sum()),0)
        self.assertEqual(float(gm[1,1].abs().sum()+gd[1,1].abs().sum()),0)
        self.assertLess(float(gd[0,0,0,0]),0)
        self.assertGreater(float(gd[0,0,1,1]),0)

    def test_value_and_evaluation_are_exact_for_mixed_and_empty_cases(self):
        x=torch.zeros(3,2,2,2)
        y=torch.zeros_like(x,dtype=torch.uint8);y[0,0,0,0]=1;y[2]=255
        criterion=DistributedPixelLoss([1,2],dice_scope='positive_only')
        expected=.25+(1-(1+1e-6)/(3+1e-6))/3
        self.assertAlmostEqual(float(criterion(x,y)),expected,places=6)
        accumulator=EvaluationLossAccumulator(criterion,'cpu')
        for i in range(3):accumulator.update(x[i:i+1],y[i:i+1])
        self.assertAlmostEqual(accumulator.result(),expected,places=6)
        self.assertEqual(accumulator.components['overlap_image_counts'],[1,0])
        self.assertAlmostEqual(float(criterion(x,torch.zeros_like(y))),.25,places=6)
        self.assertEqual(float(criterion(x,torch.full_like(y,255))),0)

    def test_crc33_sections_stay_in_one_patient_split(self):
        ids=[f'CRC{i:02}' for i in range(1,41) if i!=33]+['CRC33_01','CRC33_02']
        frame=patient_split(pd.DataFrame({'orion_slide_id':ids,'split':'train'}))
        self.assertEqual(frame.groupby('patient_id').split.nunique().max(),1)
        self.assertEqual(frame.groupby('split').patient_id.nunique().to_dict(),{'train':28,'val':6,'test':6})
        self.assertEqual(frame.loc[frame.patient_id=='CRC33','split'].tolist(),['test','test'])
        self.assertEqual(patient_id('CRC33_01'),patient_id('CRC33_02'))


if __name__=='__main__':unittest.main()
