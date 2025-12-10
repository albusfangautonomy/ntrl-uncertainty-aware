import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
from sdf_generators_3D.world_3d import generate_world_with_uncertainty_3d
from core_3D.mc_driver_3d import monte_carlo_traveltime_3d
from core_3D.speed_mapping_3d import sdf_to_speed_3d
from viz_3D.plot_3d import plot_sdf_slice
# Reuse your 2D visualization functions
from viz_3D.plot_fields import (
    plot_sdf_binary,
    plot_speed,
    plot_traveltime,
    plot_variance
)


if __name__ == "__main__":

    nx, ny, nz = 64, 64, 32

    # Generate multi-obstacle 3D world with uncertainty
    mean_sdf, std_sdf = generate_world_with_uncertainty_3d(
        nx, ny, nz, n_obstacles=4, scale=0.05
    )

    # --------------------------------------------------------
    # VISUALIZE FLOOR SLICE (z = 0)
    # --------------------------------------------------------
    floor = 0

    mean_sdf_2d = mean_sdf[:, :, floor]
    std_sdf_2d = std_sdf[:, :, floor]

    # SDF binary obstacle plot (same as 2D)
    plot_sdf_binary(mean_sdf_2d, title="3D World: Floor SDF Slice (z=0)")

    # Speed field at floor level
    speed_floor = sdf_to_speed_3d(mean_sdf)[:, :, floor]
    plot_speed(speed_floor, "Speed Field S* (Floor Slice)")

    # --------------------------------------------------------
    # Monte Carlo FMM in full 3D
    # --------------------------------------------------------
    mean_T, var_T = monte_carlo_traveltime_3d(
        mean_sdf,
        std_sdf,
        num_samples=20,
        src_idx=(0, 0, 0),
        rng_seed=42
    )

    # Extract floor slice for visualization
    mean_T_2d = mean_T[:, :, floor]
    var_T_2d = var_T[:, :, floor]

    plot_traveltime(mean_T_2d, "Mean Travel Time T (Floor Slice)")
    plot_variance(var_T_2d, "Variance of T (Floor Slice)")

    print("\n=== 3D Monte Carlo FMM Complete ===\n")
