"""Four CPU processes: production loss gradient matches a single global batch."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from vit_seg.distributed import DistributedPixelLoss, EvaluationLossAccumulator, reduce_metrics, EvaluationSampler
from vit_seg.metrics import MultilabelMetrics


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank=dist.get_rank()
    torch.manual_seed(42)
    x=torch.randn(8,3,5,5)
    y=torch.randint(0,2,(8,3,5,5))
    y[2:4,0]=255; y[4:6]=255; y[:,2,:,0]=255
    y[:2,0]=0; y[6:,1]=0
    for overlap, scope in (("dice","all_valid"),("dice","positive_only"),("lovasz_hinge","all_valid")):
        torch.manual_seed(6)
        local=torch.nn.Conv2d(3,3,1)
        reference=torch.nn.Conv2d(3,3,1); reference.load_state_dict(local.state_dict())
        ddp=DDP(local)
        objective=DistributedPixelLoss([1,2,3],overlap,dice_scope=scope,
                                       positive_weights=[2.,20.,.7],negative_weights=[.7,.51,1.75])
        loss=objective(ddp(x[rank*2:(rank+1)*2]), y[rank*2:(rank+1)*2])
        loss.backward()
        with patch("vit_seg.distributed.world_size",return_value=1), patch("vit_seg.distributed.summed",side_effect=lambda v:v.detach().clone()):
            expected=objective(reference(x),y); expected.backward()
        torch.testing.assert_close(loss,expected,rtol=2e-6,atol=2e-6)
        for a,b in zip(local.parameters(),reference.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=2e-6)
        for size in (3, 7):
            accumulator=EvaluationLossAccumulator(objective, 'cpu')
            for index in EvaluationSampler(range(size),rank,4):
                accumulator.update(reference(x[index:index+1]),y[index:index+1])
            actual=accumulator.result()
            with patch('vit_seg.distributed.world_size',return_value=1), patch('vit_seg.distributed.summed',side_effect=lambda v:v.detach().clone()):
                wanted=objective(reference(x[:size]),y[:size]).item()
            assert abs(actual-wanted)<2e-6, (actual,wanted)
    probability=torch.rand(7,3,5,5)
    truth=torch.randint(0,2,(7,3,5,5)); truth[0,0]=255
    metrics=MultilabelMetrics(3)
    indices=list(EvaluationSampler(range(7),rank,4))
    metrics.update(probability[indices],truth[indices]); reduce_metrics(metrics,"cpu")
    reference=MultilabelMetrics(3);reference.update(probability,truth)
    np.testing.assert_array_equal(metrics.confusion, reference.confusion)
    np.testing.assert_array_equal(metrics.pos, reference.pos)
    np.testing.assert_array_equal(metrics.neg, reference.neg)
    np.testing.assert_allclose(metrics.brier, reference.brier, rtol=1e-12)
    assert metrics.n_pixels == reference.n_pixels and metrics.exact == reference.exact
    control=[True if rank==0 else None];dist.broadcast_object_list(control,src=0)
    assert control[0] is True
    if rank == 0:
        Path("review/ddp_cpu_verification.json").write_text(json.dumps(dict(world_size=4,backend="gloo",device="cpu",
            dice_global_gradient=True,hinge_global_gradient=True,all_ignored_rank_safe=True,
            uneven_validation_exact=True,synchronized_stop_broadcast=True,
            gpu_training_executed=False),indent=2))
        print("PASS: four-process CPU DDP global gradients, uneven metrics, stop broadcast",flush=True)
    dist.destroy_process_group()


if __name__ == "__main__": main()
