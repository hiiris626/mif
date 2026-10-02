import unittest
import numpy as np
import torch
from vit_seg.metrics import MultilabelMetrics, DeviceMultilabelMetrics


def compare(device):
    rng=np.random.default_rng(73)
    p=rng.random((5,3,19,17),dtype=np.float32)
    y=rng.integers(0,2,p.shape,dtype=np.uint8)
    y[0]=255; y[:,2,:4]=255
    # Exact threshold and histogram boundary cases, including p=0 and p=1.
    boundaries=np.array([0,1,.5,1/256,127/256,128/256,255/256,1/15,14/15],np.float32)
    p[1,0,0,:len(boundaries)]=boundaries
    for threshold in (.5,[.35,.5,.7]):
        ref=MultilabelMetrics(3,threshold=threshold)
        actual=DeviceMultilabelMetrics(device,3,threshold=threshold)
        for a,b in ((0,2),(2,5)):
            ref.update(p[a:b],y[a:b])
            actual.update(torch.from_numpy(p[a:b]).to(device),torch.from_numpy(y[a:b]).to(device))
        actual=actual.as_numpy()
        for key in ('confusion','pos','neg','cal_n','cal_y'):
            np.testing.assert_array_equal(getattr(ref,key),getattr(actual,key),err_msg=key)
        for key in ('cal_p','brier'):
            np.testing.assert_allclose(getattr(ref,key),getattr(actual,key),rtol=1e-12,atol=1e-10)
        assert (ref.n_pixels,ref.exact)==(actual.n_pixels,actual.exact)
        assert ref.result(['a','b','c'])['macro']==actual.result(['a','b','c'])['macro']


class DeviceMetricsTests(unittest.TestCase):
    def test_reference_equivalence(self): compare(torch.device('cpu'))


if __name__=='__main__': unittest.main()
