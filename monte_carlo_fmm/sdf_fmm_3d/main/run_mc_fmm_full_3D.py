import numpy as np
import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from sdf_generators_3D.world_3d import generate_world_with_uncertainty_3d
from core_3D.mc_driver_3d import monte_carlo_traveltime_3d
from core_3D.speed_mapping_3d import sdf_to_speed_3d
from viz_3D.plot_3d import plot_sdf_slice


if __name__ == "__main__":
    nx, ny, nz = 64, 64, 32

    mean_sdf, std_sdf = generate_world_with_uncertainty_3d(nx, ny, nz, n_obstacles=4)

    # Visualize a few mid-plane slices
    plot_sdf_slice(mean_sdf, nz//2, "Mean 3D SDF (mid z-slice)")

    mean_T, var_T = monte_carlo_traveltime_3d(
        mean_sdf,
        std_sdf,
        num_samples=20,
        src_idx=(0,0,0),
        rng_seed=42
    )

    plot_sdf_slice(mean_T, nz//2, "Mean Travel Time (mid z-slice)")
    plot_sdf_slice(var_T, nz//2, "Variance of T (mid z-slice)")
