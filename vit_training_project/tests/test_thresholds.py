import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from vit_seg.artifacts import file_hash
from vit_seg.data import CHANNELS
from vit_seg.metrics import MultilabelMetrics
from vit_seg.thresholds import select_thresholds, load_thresholds


class ThresholdTests(unittest.TestCase):
    def fixture(self):
        metric = MultilabelMetrics()
        probabilities = np.tile(np.array([.1,.2,.3,.4],np.float32),(1,16,1,1))
        truth = np.tile(np.array([0,0,1,1]),(1,16,1,1))
        metric.update(probabilities,truth)
        config=dict(min=.05,max=.95,min_positive_pixels=1,min_negative_pixels=1,min_patients_per_label=2)
        return metric,config

    def test_grid_agrees_with_direct_decisions(self):
        metric,config=self.fixture()
        result=select_thresholds(metric,[2]*16,[2]*16,config)
        self.assertTrue(.2 < result['thresholds'][0] <= .3)
        self.assertEqual(result['per_channel'][0]['val_f1_at_selected'],1)
        for threshold,f1 in zip(result['per_channel'][0]['grid'],result['per_channel'][0]['val_f1']):
            predictions=np.array([.1,.2,.3,.4],np.float32)>=threshold
            target=np.array([0,0,1,1],bool)
            tp=(predictions & target).sum();fp=(predictions & ~target).sum();fn=(~predictions & target).sum()
            self.assertAlmostEqual(f1,2*tp/(2*tp+fp+fn))

    def test_sparse_patient_support_falls_back(self):
        metric,config=self.fixture()
        result=select_thresholds(metric,[1]*16,[2]*16,config)
        self.assertEqual(result['thresholds'],[.5]*16)
        self.assertFalse(result['per_channel'][0]['tuned'])

    def test_threshold_model_and_data_identity(self):
        metric,config=self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); model=root/'best.pt';model.write_bytes(b'checkpoint')
            result=select_thresholds(metric,[2]*16,[2]*16,config)
            result.update(checkpoint_sha256=file_hash(model),data_signature={'x':'y'})
            path=root/'thresholds.json';path.write_text(json.dumps(result))
            self.assertEqual(load_thresholds(path,model,{'x':'y'}),result['thresholds'])
            with self.assertRaisesRegex(ValueError,'data mismatch'): load_thresholds(path,model,{'x':'changed'})
            result['fitted_split']='test';path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'validation'): load_thresholds(path,model)


if __name__=='__main__': unittest.main()
