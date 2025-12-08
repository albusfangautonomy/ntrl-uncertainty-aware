import numpy as np
from .sdf_sampling import sample_sdf
from .speed_mapping import sdf_to_speed
from .solver import setup_solver_from_speed


def monte_carlo_traveltime(mean_sdf: np.ndarray,
                           std_sdf: np.ndarray,
                           num_samples: int,
                           src_idx=(0, 0, 0),
                           min_coords=(0.0, 0.0, 0.0),
                           node_intervals=(1.0, 1.0, 1.0),
                           rng_seed=None):
    """
    Monte Carlo evaluation of travel-time field.
    """

    if mean_sdf.ndim == 2:
        shape_3d = mean_sdf.shape + (1,)
    else:
        shape_3d = mean_sdf.shape

    rng = np.random.default_rng(rng_seed)

    sum_T = np.zeros(shape_3d)
    sum_T2 = np.zeros(shape_3d)

    for k in range(num_samples):
        sdf_k = sample_sdf(mean_sdf, std_sdf, rng)
        speed_k = sdf_to_speed(sdf_k)

        solver = setup_solver_from_speed(
            speed_k,
            min_coords=min_coords,
            node_intervals=node_intervals,
            src_idx=src_idx,
        )
        solver.solve()

        T = solver.traveltime.values.copy()
        sum_T += T
        sum_T2 += T**2

    mean_T = sum_T / num_samples
    var_T = sum_T2 / num_samples - mean_T**2

    return mean_T, var_T
