import numpy as np
from .sdf_sampling import sample_sdf
from .speed_mapping import sdf_to_speed


def monte_carlo_speedfield(mean_sdf, std_sdf, num_samples, rng_seed=None):
    """
    Monte Carlo sampling of S*(x) = speed field derived from sampled SDFs.

    Returns:
        mean_speed : mean S* over samples
        var_speed  : variance of S* over samples
    """

    rng = np.random.default_rng(rng_seed)

    nx, ny = mean_sdf.shape
    sum_S = np.zeros((nx, ny))
    sum_S2 = np.zeros((nx, ny))

    for _ in range(num_samples):
        # Sample SDF realization
        sdf_k = sample_sdf(mean_sdf, std_sdf, rng)

        # Convert to speed
        S_k = sdf_to_speed(sdf_k)

        sum_S  += S_k
        sum_S2 += S_k**2

    mean_S = sum_S / num_samples
    var_S  = sum_S2 / num_samples - mean_S**2

    return mean_S, var_S
