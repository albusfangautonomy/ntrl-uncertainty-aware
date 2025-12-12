import torch

def symmetric_operator(a, b):
    """
    N(a, b) = concat(max(a, b), min(a, b))
    Each of a, b is (B, D)
    """
    max_ab = torch.max(a, b)
    min_ab = torch.min(a, b)
    return torch.cat([max_ab, min_ab], dim=-1)  # (B, 2D)
