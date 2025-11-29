# uncertainty/effective_distance.py
import torch

def compute_effective_distance(mu, sigma, lam=2.0):
    return mu - lam * sigma

def compute_speed_from_distance(d_eff, d_min=0.0, d_max=20.0):
    """
    Maps effective distance into [0,1] speed.
    Linear ramp + clipping.
    """
    S = (d_eff - d_min) / (d_max - d_min)
    return torch.clamp(S, 0.0, 1.0)
