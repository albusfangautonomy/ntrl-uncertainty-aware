# uncertainty/variance_models.py
import torch

def simple_edge_uncertainty(mu_sdf, scale=1.0):
    """
    Higher uncertainty near obstacle boundaries.
    sigma = exp(-|mu| / scale)
    """
    return torch.exp(-torch.abs(mu_sdf) / scale)

def uniform_uncertainty(mu_sdf, sigma_value=0.5):
    """
    Constant uncertainty everywhere.
    """
    return torch.full_like(mu_sdf, sigma_value)
