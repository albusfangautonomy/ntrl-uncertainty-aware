import torch

def gaussian_nll(mu, var, target):
    """
    Gaussian negative log-likelihood:
        0.5 * log(var) + 0.5 * (y - mu)^2 / var
    """
    return 0.5 * torch.log(var) + 0.5 * (target - mu)**2 / var
