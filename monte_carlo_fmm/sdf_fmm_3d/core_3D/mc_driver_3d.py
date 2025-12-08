import numpy as np
from .solver_3d import setup_solver_from_speed_3d
from .speed_mapping_3d import sdf_to_speed_3d
from core_3D.sdf_sampling import sample_sdf   # identical in 3D


def monte_carlo_traveltime_3d(mean_sdf, std_sdf, num_samples,
                               src_idx=(0,0,0), rng_seed=None):

    nx, ny, nz = mean_sdf.shape
    
    rng = np.random.default_rng(rng_seed)
    sum_T = np.zeros((nx, ny, nz))
    sum_T2 = np.zeros((nx, ny, nz))

    for _ in range(num_samples):
        sdf_k = sample_sdf(mean_sdf, std_sdf, rng)
        speed_k = sdf_to_speed_3d(sdf_k)

        solver = setup_solver_from_speed_3d(speed_k, src_idx=src_idx)
        solver.solve()
        T = solver.traveltime.values.copy()

        sum_T += T
        sum_T2 += T**2

    mean_T = sum_T / num_samples
    var_T = sum_T2 / num_samples - mean_T**2

    return mean_T, var_T
