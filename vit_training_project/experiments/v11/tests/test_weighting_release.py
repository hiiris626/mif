"""Check capped weights and reproducible four-rank replacement sampling."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from vit_seg.bce_balance import load_pixel_weights
from vit_seg.data import CHANNELS
from vit_seg.pixel_sampling import DistributedWeightedPatchSampler

class ReleaseTests(unittest.TestCase):
    def weights(self,power=.6,declared=None):
        ratio=np.array([.5,1,4,256]*4)
        return dict(mode='capped_power',fitted_split='train',channels=CHANNELS,grid=256,
            positive_counts=[2]*16,valid_counts=(2+2*ratio).tolist(),power=power,maximum=16,minimum=1,
            positive_weights=(np.minimum(16,np.maximum(1,ratio)**.6) if declared is None else declared).tolist(),negative_weights=[1]*16)
    def load(self,value):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'weights.json';p.write_text(json.dumps(value))
            return load_pixel_weights(dict(bce_pixel_weights_file=str(p),tile_size=256))
    def test_floor_cap_and_negative_unit(self):
        w,n=self.load(self.weights());self.assertEqual(w[0],1);self.assertEqual(w[3],16)
        self.assertAlmostEqual(w[2],4**.6);self.assertEqual(n,[1]*16)
    def test_bad_power_rejected(self):
        with self.assertRaises(ValueError):self.load(self.weights(power=float('nan')))
    def test_tampered_weights_rejected(self):
        with self.assertRaises(ValueError):self.load(self.weights(declared=np.ones(16)))
    def test_ddp_draws_replay_and_change_by_epoch(self):
        q=np.arange(1,9,dtype=float);q/=q.sum()
        samplers=[DistributedWeightedPatchSampler(q.copy(),4,r,42) for r in range(4)]
        expected=samplers[0].global_indices()
        actual=np.array([list(s) for s in samplers]).T.ravel()
        np.testing.assert_array_equal(actual,expected)
        samplers[0].set_epoch(1);self.assertFalse(np.array_equal(expected,samplers[0].global_indices()))

if __name__=='__main__':unittest.main()
