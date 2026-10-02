"""Train-only inverse-sigma weights and independent binary Lovasz-Hinge."""
import torch

def channel_weights(std, minimum=.25, maximum=4., eps=1e-3):
    s = torch.as_tensor(std, dtype=torch.float32)
    active = torch.isfinite(s) & (s >= 0)
    w = torch.where(active, 1 / s.clamp_min(eps), 0)
    if not active.any():
        raise ValueError("No valid training channel standard deviations")
    w = w / w[active].mean()
    w = torch.where(active, w.clamp(minimum, maximum), 0)
    return w / w[active].mean()

def lovasz_hinge_class(logits, truth):
    errors, order = (1-logits*(2*truth-1)).sort(descending=True)
    gt = truth[order]
    intersection = gt.sum()-gt.cumsum(0)
    union = gt.sum()+(1-gt).cumsum(0)
    grad = 1-intersection/union.clamp_min(1e-7)
    grad = torch.cat((grad[:1],grad[1:]-grad[:-1]))
    return torch.dot(torch.relu(errors),grad)
