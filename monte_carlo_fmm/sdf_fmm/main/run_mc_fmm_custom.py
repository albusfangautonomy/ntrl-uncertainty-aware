import numpy as np
import sys, os

# Fix sys.path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import SDF components
from sdf_gen.random_shapes import circle_sdf, box_sdf
from sdf_gen.utils import combine_sdfs

# Import speed + Monte Carlo modules
from core.speed_mapping import sdf_to_speed
from core.mc_driver import monte_carlo_traveltime
from core.mc_speedfield import monte_carlo_speedfield

# Import visualizations
from viz.plot_fields import (
    plot_sdf_binary,
    plot_speed,
    plot_traveltime,
    plot_variance,
    plot_traveltime_variance_masked,
    plot_speedfield_variance_masked,
    heatmap2d,
)

# ----------------------------------------------------------
# MAIN EXECUTION — CUSTOM WORLD VERSION
# ----------------------------------------------------------
if __name__ == "__main__":

    nx, ny = 64, 64
    map_size = 5.0

    # ------------------------------------------------------
    # CUSTOM MANUAL WORLD: CIRCLE + BOX
    # ------------------------------------------------------
    print("\n=== Using custom manually-defined world (Circle + Box) ===\n")

    # Circle obstacle
    circle = circle_sdf(
        nx, ny,
        cx=-1.0, cy=0.0,
        r=0.2
    )

    # Box obstacle
    box = box_sdf(
        nx, ny,
        cx=1.0, cy=-0.5,
        half_w=0.7, half_h=0.4
    )

    # Combine into world SDF
    mean_sdf = combine_sdfs(circle, box)

    # Assign uncertainty based on boundary proximity
    std_sdf = 0.05 * np.exp(-(np.abs(mean_sdf) / 0.2)**2)

    # ------------------------------------------------------
    # Visualize SDF & uncertainty
    # ------------------------------------------------------
    plot_sdf_binary(mean_sdf, "Custom World: Circle + Box (Mean SDF)")
    heatmap2d(std_sdf**2, "Custom World: Raw SDF Variance", cmap="magma")

    # Speed from mean SDF
    speed_mean = sdf_to_speed(mean_sdf)
    plot_speed(speed_mean, "Speed Field S* from Mean SDF")

    # ------------------------------------------------------
    # Monte Carlo SPEED FIELD S*
    # ------------------------------------------------------
    num_samples = 50

    print("\nRunning Monte Carlo for speed field S* ...\n")

    mean_S, var_S = monte_carlo_speedfield(
        mean_sdf,
        std_sdf,
        num_samples=num_samples,
        rng_seed=1
    )

    plot_speed(mean_S, "Monte Carlo Mean Speed Field E[S*]")
    plot_variance(var_S, "Monte Carlo Speed Variance Var[S*]")
    plot_speedfield_variance_masked(var_S, mean_sdf, "Masked Variance of S*")

    # ------------------------------------------------------
    # Monte Carlo FAST MARCHING METHOD
    # ------------------------------------------------------
    src_idx = (0, 0, 0)

    print("\nRunning Monte Carlo FMM ...\n")

    mean_T, var_T = monte_carlo_traveltime(
        mean_sdf,
        std_sdf,
        num_samples=num_samples,
        src_idx=src_idx,
        rng_seed=42
    )

    mean_T2 = mean_T[:, :, 0]
    var_T2  = var_T[:, :, 0]

    plot_traveltime(mean_T2, "Mean Travel Time T(q)")
    plot_variance(var_T2, "Variance of Travel Time Var[T(q)]")
    plot_traveltime_variance_masked(var_T2, mean_sdf, "Masked T Variance")

    print("\n=== Custom Monte Carlo FMM Complete ===\n")
