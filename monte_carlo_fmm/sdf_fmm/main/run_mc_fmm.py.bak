import numpy as np
import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from sdf_gen.random_shapes import (
    random_circle_sdf,
    random_box_sdf,
    add_boundary_uncertainty
)
from core.mc_driver import monte_carlo_traveltime
from core.speed_mapping import sdf_to_speed
from viz.plot_fields import (
    plot_sdf,
    plot_speed,
    plot_traveltime,
    plot_variance,
)

if __name__ == "__main__":
    nx, ny = 64, 64

    # Choose obstacle type
    use_circle = False

    if use_circle:
        mean_sdf = random_circle_sdf(nx, ny)
    else:
        mean_sdf = random_box_sdf(nx, ny)

    std_sdf = add_boundary_uncertainty(mean_sdf)

    # Visualize mean SDF
    plot_sdf(mean_sdf, "Mean SDF")

    # Visualize speed field from SDF
    plot_speed(sdf_to_speed(mean_sdf), "Speed from Mean SDF")

    # Monte Carlo FMM
    mean_T, var_T = monte_carlo_traveltime(
        mean_sdf,
        std_sdf,
        num_samples=50,
        src_idx=(0, 0, 0),
        rng_seed=42
    )

    T2 = mean_T[:, :, 0]
    VarT2 = var_T[:, :, 0]

    plot_traveltime(T2, "Mean Travel Time T")
    plot_variance(VarT2, "Variance of T")
