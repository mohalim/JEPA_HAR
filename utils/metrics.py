# utils/metrics.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def representation_stats(z):
    std = z.std(dim=0).mean().item()
    norm = z.norm(dim=-1).mean().item()

    return std, norm
