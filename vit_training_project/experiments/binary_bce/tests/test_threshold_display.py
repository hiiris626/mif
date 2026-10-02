import unittest
import numpy as np
from vit_seg.display import argmax_display


class ThresholdDisplayTests(unittest.TestCase):
    def test_all_low_scores_stay_black(self):
        labels,rgb=argmax_display(np.full((16,2,2),.1),np.ones((2,2),bool))
        self.assertFalse(labels.any());self.assertFalse(rgb.any())

    def test_winner_must_pass_own_threshold(self):
        p=np.zeros((16,2,2));p[0]=.8;p[1]=.6
        thresholds=np.full(16,.5);thresholds[0]=.9
        tissue=np.array([[True,False],[True,True]])
        labels,_=argmax_display(p,tissue,thresholds=thresholds)
        np.testing.assert_array_equal(labels,[[2,0],[2,2]])
        available=np.ones(16,bool);available[1]=False
        labels,_=argmax_display(p,tissue,available,thresholds)
        self.assertFalse(labels.any())

    def test_threshold_equality_and_input_validation(self):
        p=np.zeros((16,1,1));p[3]=.5
        self.assertEqual(argmax_display(p,[[True]])[0].item(),4)
        for value in (0,1,float('nan')):
            with self.assertRaises(ValueError):argmax_display(p,[[True]],thresholds=value)
        with self.assertRaises(ValueError):argmax_display(p+2,[[True]])
