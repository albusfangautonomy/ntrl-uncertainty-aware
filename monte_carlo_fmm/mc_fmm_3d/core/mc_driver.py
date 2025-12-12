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
    Monte Carlo estimation of E[T(x)] and Var[T(x)] over a 3D grid.
    """

    if mean_sdf.shape != std_sdf.shape:
        raise ValueError("mean_sdf and std_sdf must have same shape.")

    if mean_sdf.ndim != 3:
        raise ValueError("mean_sdf must be 3D for 3D traveltime MC.")

    rng = np.random.default_rng(rng_seed)
    shape = mean_sdf.shape

    sum_T = np.zeros(shape, dtype=np.float64)
    sum_T2 = np.zeros(shape, dtype=np.float64)

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

        T_k = solver.traveltime.values.copy()
        sum_T += T_k
        sum_T2 += T_k ** 2

    mean_T = sum_T / num_samples
    var_T = sum_T2 / num_samples - mean_T ** 2

    return mean_T, var_T
