"""DDP reductions, non-padding evaluation, optimizer and schedule primitives."""
import math
import numpy as np
import torch
import torch.nn.functional as F
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

    Dice/Hinge average over eligible image/channel pairs. BCE averages over
    eligible pixels per channel. Accumulation averages microbatch objectives.
    """
    def __init__(self, weights, overlap="dice", bce_weight=1., overlap_weight=1., dice_scope="all_valid", positive_weights=None, negative_weights=None):
        super().__init__()
        if overlap not in ("dice", "lovasz_hinge"):
            raise ValueError("Expected Dice or independent binary Lovasz-Hinge")
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))
        for name,value in [('positive_weights',positive_weights),('negative_weights',negative_weights)]:
            tensor=torch.ones_like(self.weights) if value is None else torch.as_tensor(value,dtype=torch.float32)
            if tensor.shape!=self.weights.shape or not torch.isfinite(tensor).all() or (tensor<=0).any():
                raise ValueError('Pixel weights must be finite, positive and match channel count')
            self.register_buffer(name,tensor)
        self.overlap, self.bce_weight, self.overlap_weight = overlap, bce_weight, overlap_weight
        if dice_scope not in ("all_valid", "positive_only"):
            raise ValueError("Unknown Dice image/channel scope")
        self.dice_scope = dice_scope
        self.last_components = None

    def forward(self, logits, labels):
        valid = labels != IGNORE
        y = labels.masked_fill(~valid, 0).float()
        p = logits.float().sigmoid()
        counts = summed(valid.sum((0, 2, 3)).float())
        pixel_bce=F.binary_cross_entropy_with_logits(logits.float(),y,reduction='none')
        pixel_weight=y*self.positive_weights[None,:,None,None]+(1-y)*self.negative_weights[None,:,None,None]
        numerator = global_value_local_grad((pixel_bce*valid*pixel_weight).sum((0, 2, 3)))
        bce = numerator/counts.clamp_min(1)
        available = valid.any((2, 3))
        if self.overlap == "dice":
            if self.dice_scope == "positive_only":
                available = ((y > 0) & valid).any((2, 3))
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
        bce_component = (bce*weights).sum()/weights.sum().clamp_min(1e-8)
        overlap_component = (ov*weights).sum()/weights.sum().clamp_min(1e-8)
        self.last_components = torch.stack((bce_component, overlap_component)).detach()
        loss = self.bce_weight*bce_component+self.overlap_weight*overlap_component
        # Each rank sees the global value but contributes its own derivative.
        return loss*world_size() + loss.detach()*(1-world_size())


class EvaluationLossAccumulator:
    """Reduce sufficient statistics once after uneven validation shards finish.

    BCE uses all valid pixels per channel; overlap uses all valid image/channel
    pairs. Empty ranks participate in the final reduction without dropping data.
    """
    def __init__(self, criterion, device):
        self.criterion = criterion
        self.values = torch.zeros((8, len(criterion.weights)), device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, labels):
        valid = labels != IGNORE
        y = labels.masked_fill(~valid, 0).float()
        p = logits.float().sigmoid()
        available = valid.any((2, 3))
        pixel_bce=F.binary_cross_entropy_with_logits(logits.float(),y,reduction='none')
        pixel_weight=y*self.criterion.positive_weights[None,:,None,None]+(1-y)*self.criterion.negative_weights[None,:,None,None]
        self.values[0] += (pixel_bce*valid*pixel_weight).sum((0, 2, 3), dtype=torch.float64)
        self.values[1] += valid.sum((0, 2, 3))
        pos=valid&(y==1);neg=valid&(y==0)
        self.values[4] += (pixel_bce*pos).sum((0,2,3),dtype=torch.float64)
        self.values[5] += pos.sum((0,2,3))
        self.values[6] += (pixel_bce*neg).sum((0,2,3),dtype=torch.float64)
        self.values[7] += neg.sum((0,2,3))
        if self.criterion.overlap == "dice":
            if self.criterion.dice_scope == "positive_only":
                available = ((y > 0) & valid).any((2, 3))
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

    def result(self, group=None):
        if group is None:
            values = summed(self.values)
        else:
            # Validation reductions use Gloo/CPU after all ranks are ready.
            values = self.values.detach().cpu().clone()
            dist.all_reduce(values, group=group)
        bce_sum, pixels, overlap_sum, images, pos_sum, pos_n, neg_sum, neg_n = values
        if not (pixels > 0).any():
            raise ValueError("Validation contains no supervised pixels")
        weights = self.criterion.weights.to(pixels.device)*(pixels > 0)
        bce = (bce_sum/pixels.clamp_min(1)*weights).sum()/weights.sum().clamp_min(1e-8)
        overlap = (overlap_sum/images.clamp_min(1)*weights).sum()/weights.sum().clamp_min(1e-8)
        self.components = dict(bce=float(bce), overlap=float(overlap),
            per_channel_bce=(bce_sum/pixels.clamp_min(1)).tolist(),
            per_channel_overlap=(overlap_sum/images.clamp_min(1)).tolist(),
            overlap_image_counts=images.tolist(), supervised_pixel_counts=pixels.tolist(),
            positive_pixel_counts=pos_n.tolist(),negative_pixel_counts=neg_n.tolist(),
            per_channel_positive_bce=(pos_sum/pos_n.clamp_min(1)).tolist(),
            per_channel_negative_bce=(neg_sum/neg_n.clamp_min(1)).tolist(),
            positive_pixel_weights=self.criterion.positive_weights.tolist(),
            negative_pixel_weights=self.criterion.negative_weights.tolist())
        loss = self.criterion.bce_weight*bce+self.criterion.overlap_weight*overlap
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


def reduce_metrics(metrics, device, group=None):
    for name in ("confusion", "pos", "neg", "cal_n", "cal_y", "cal_p", "brier"):
        value = getattr(metrics, name)
        tensor = torch.as_tensor(value, device=device)
        dist.all_reduce(tensor, group=group)
        setattr(metrics, name, tensor.cpu().numpy())
    counts = torch.tensor([metrics.n_pixels, metrics.exact], dtype=torch.long, device=device)
    dist.all_reduce(counts, group=group)
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
