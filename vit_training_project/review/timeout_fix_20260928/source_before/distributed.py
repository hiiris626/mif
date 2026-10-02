"""DDP reductions, non-padding evaluation, optimizer and schedule primitives."""
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from .data import IGNORE
from .losses import lovasz_hinge_class


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def summed(value):
    result = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(result)
    return result


def global_value_local_grad(value):
    return value + (summed(value) - value.detach())


class DistributedPixelLoss(torch.nn.Module):
    """Global channel denominators; DDP gradient averaging is compensated.

    Dice/Hinge average over eligible image/channel pairs. MSE averages over
    eligible pixels per channel. Accumulation averages microbatch objectives.
    """
    def __init__(self, weights, overlap="dice", mse_weight=1., overlap_weight=1.):
        super().__init__()
        if overlap not in ("dice", "lovasz_hinge"):
            raise ValueError("Expected Dice or independent binary Lovasz-Hinge")
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))
        self.overlap, self.mse_weight, self.overlap_weight = overlap, mse_weight, overlap_weight

    def forward(self, logits, labels):
        valid = labels != IGNORE
        y = labels.masked_fill(~valid, 0).float()
        p = logits.float().sigmoid()
        counts = summed(valid.sum((0, 2, 3)).float())
        numerator = global_value_local_grad(((p-y).square()*valid).sum((0, 2, 3)))
        mse = numerator/counts.clamp_min(1)
        available = valid.any((2, 3))
        if self.overlap == "dice":
            inter = (p*y*valid).sum((2, 3))
            denominator = ((p+y)*valid).sum((2, 3))
            overlap = 1-(2*inter+1e-6)/(denominator+1e-6)
        else:
            overlap = torch.stack([torch.stack([
                lovasz_hinge_class(logits[b,c][valid[b,c]].float(), y[b,c][valid[b,c]])
                if available[b,c] else logits[b,c].sum()*0
                for c in range(logits.shape[1])]) for b in range(logits.shape[0])])
        ov = global_value_local_grad((overlap*available).sum(0))/summed(available.sum(0).float()).clamp_min(1)
        weights = self.weights*(counts > 0)
        loss = ((self.mse_weight*mse+self.overlap_weight*ov)*weights).sum()/weights.sum().clamp_min(1e-8)
        # Each rank sees the global value but contributes its own derivative.
        return loss*world_size() + loss.detach()*(1-world_size())


class EvaluationLossAccumulator:
    """Reduce sufficient statistics once after uneven validation shards finish.

    MSE uses all valid pixels per channel; overlap uses all valid image/channel
    pairs. Empty ranks participate in the final reduction without dropping data.
    """
    def __init__(self, criterion, device):
        self.criterion = criterion
        self.values = torch.zeros((4, len(criterion.weights)), device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, labels):
        valid = labels != IGNORE
        y = labels.masked_fill(~valid, 0).float()
        p = logits.float().sigmoid()
        available = valid.any((2, 3))
        self.values[0] += ((p-y).square()*valid).sum((0, 2, 3), dtype=torch.float64)
        self.values[1] += valid.sum((0, 2, 3))
        if self.criterion.overlap == "dice":
            inter = (p*y*valid).sum((2, 3))
            denominator = ((p+y)*valid).sum((2, 3))
            overlap = 1-(2*inter+1e-6)/(denominator+1e-6)
        else:
            overlap = torch.stack([torch.stack([
                lovasz_hinge_class(logits[b,c][valid[b,c]].float(), y[b,c][valid[b,c]])
                if available[b,c] else logits.new_zeros(())
                for c in range(logits.shape[1])]) for b in range(logits.shape[0])])
        self.values[2] += (overlap*available).sum(0, dtype=torch.float64)
        self.values[3] += available.sum(0)

    def result(self):
        mse_sum, pixels, overlap_sum, images = summed(self.values)
        if not (pixels > 0).any():
            raise ValueError("Validation contains no supervised pixels")
        weights = self.criterion.weights*(pixels > 0)
        loss = ((self.criterion.mse_weight*mse_sum/pixels.clamp_min(1)
                 + self.criterion.overlap_weight*overlap_sum/images.clamp_min(1))*weights).sum()/weights.sum().clamp_min(1e-8)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        return float(loss.item())


class EvaluationSampler(Sampler):
    """Each sample occurs exactly once, including uneven split lengths."""
    def __init__(self, dataset, rank, world):
        self.n, self.rank, self.world = len(dataset), rank, world
    def __iter__(self):
        return iter(range(self.rank, self.n, self.world))
    def __len__(self):
        return len(range(self.rank, self.n, self.world))


def reduce_metrics(metrics, device):
    for name in ("confusion", "pos", "neg", "cal_n", "cal_y", "cal_p", "brier"):
        value = getattr(metrics, name)
        tensor = torch.as_tensor(value, device=device)
        dist.all_reduce(tensor)
        setattr(metrics, name, tensor.cpu().numpy())
    counts = torch.tensor([metrics.n_pixels, metrics.exact], dtype=torch.long, device=device)
    dist.all_reduce(counts)
    metrics.n_pixels, metrics.exact = counts.tolist()


def optimizer_for(model, cfg):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        branch = "lora" if name.startswith("encoder.") else "decoder"
        decay = parameter.ndim > 1 and not name.endswith("bias")
        groups.setdefault((branch, decay), []).append(parameter)
    params = [dict(params=p, lr=cfg[f"{branch}_lr"], initial_lr=cfg[f"{branch}_lr"],
                   weight_decay=cfg["weight_decay"] if decay else 0., name=f"{branch}_{decay}")
              for (branch, decay), p in groups.items()]
    return torch.optim.AdamW(params, betas=tuple(cfg["betas"]), eps=cfg["eps"])


def lr_factor(step, total, warmup, minimum):
    if step <= warmup:
        return step/max(warmup, 1)
    progress = min(1., (step-warmup)/max(total-warmup, 1))
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*progress))


def resolve_batch(cfg, batch):
    if batch < 1:
        raise ValueError("Batch must be positive")
    accumulation = max(1, math.ceil(cfg["target_global_batch"]/(cfg["world_size"]*batch)))
    return dict(batch_size=batch, grad_accum=accumulation,
                effective_global_batch=cfg["world_size"]*batch*accumulation)
