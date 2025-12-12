import numpy as np
from .sdf_sampling import sample_sdf
from .speed_mapping import sdf_to_speed


def monte_carlo_speedfield(mean_sdf: np.ndarray,
                           std_sdf: np.ndarray,
                           num_samples: int,
                           rng_seed=None):
    """
    Monte Carlo estimation of E[S*(x)] and Var[S*(x)] from distributional SDF.
    Works for 3D (and also 2D) arrays.
    """
    if mean_sdf.shape != std_sdf.shape:
        raise ValueError("mean_sdf and std_sdf must have same shape.")

    rng = np.random.default_rng(rng_seed)

    shape = mean_sdf.shape
    sum_S = np.zeros(shape, dtype=np.float64)
    sum_S2 = np.zeros(shape, dtype=np.float64)

    for k in range(num_samples):
        sdf_k = sample_sdf(mean_sdf, std_sdf, rng)
        S_k = sdf_to_speed(sdf_k)

        sum_S += S_k
        sum_S2 += S_k ** 2

    mean_S = sum_S / num_samples
    var_S = sum_S2 / num_samples - mean_S ** 2

    return mean_S, var_S
