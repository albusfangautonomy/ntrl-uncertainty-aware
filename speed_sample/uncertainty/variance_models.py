# uncertainty/variance_models.py
import torch

def simple_edge_uncertainty(mu_sdf, scale=1.0, max_sigma= 0.5):
    """
    Higher uncertainty near obstacle boundaries.
    sigma = exp(-|mu| / scale)
    """
    return max_sigma * torch.exp(-torch.abs(mu_sdf) / scale)

def uniform_uncertainty(mu_sdf, sigma_value=0.5):
    """
    Constant uncertainty everywhere.
    """
    return torch.full_like(mu_sdf, sigma_value)



def lidar_range_variance(
    r,
    theta=None,
    sigma0=0.02,      # base noise (meters)
    alpha=0.003,      # range-growth factor
    beta=0.001        # grazing-angle factor
):
    """
    Returns per-ray LiDAR range variance σ(r,θ)^2.

    Parameters:
        r:      torch.Tensor [...], distances in meters
        theta:  torch.Tensor [...], incidence angle in radians (optional)
                0 = perpendicular hit
                π/2 = grazing hit
        sigma0: base noise (m)
        alpha:  multiplicative range noise growth
        beta:   incidence-angle noise growth factor

    Model:
        σ(r)^2 = σ0^2 + α r^2           (basic)
        if theta supplied:
            σ(r,θ)^2 += β * tan(theta)^2

    Returns:
        torch.Tensor of variances with same shape as r.
    """

    # Ensure tensor
    r = torch.as_tensor(r, dtype=torch.float32)

    # Basic range-dependent variance
    var = sigma0**2 + alpha * (r**2)

    # Optional incidence-angle-dependent term
    if theta is not None:
        theta = torch.as_tensor(theta, dtype=torch.float32)
        # Avoid numerical blow-up at grazing angles
        tan_term = torch.tan(theta).clamp(min=-10.0, max=10.0)
        var = var + beta * (tan_term**2)

    return var
