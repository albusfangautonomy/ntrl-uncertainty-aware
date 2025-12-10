import numpy as np
import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from sdf_gen.random_world import generate_world_with_uncertainty
from core.speed_mapping import sdf_to_speed
from core.mc_driver import monte_carlo_traveltime
from core.mc_speedfield import monte_carlo_speedfield
from viz.plot_fields import (
    plot_sdf_binary,
    plot_speed,
    plot_traveltime,
    plot_variance,
    plot_traveltime_variance_masked,
    heatmap2d,
    plot_speedfield_variance_masked
)


# ----------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------
if __name__ == "__main__":

    # Grid resolution
    nx, ny = 64, 64

    # Number of obstacles per world
    n_obstacles = 1

    print(f"\n=== Generating world with {n_obstacles} obstacles ===\n")

    # Generate full world SDF + uncertainty
    mean_sdf, std_sdf = generate_world_with_uncertainty(
        nx, ny,
        n_obstacles=n_obstacles,
        scale=0.05
    )

    # ------------------------------------------------------
    # Visualization of base SDF and speed from S*
    # ------------------------------------------------------
    plot_sdf_binary(mean_sdf, "Mean SDF (Multi-Obstacle World)")
    sdf_variance = std_sdf**2
    heatmap2d(sdf_variance, "Raw SDF Variance (std^2)", cmap="magma")
    speed_mean = sdf_to_speed(mean_sdf)
    plot_speed(speed_mean, "Speed Field S*(q) from Mean SDF")
    
    # ------------------------------------------------------
    # Monte Carlo SPEED FIELD computation
    # ------------------------------------------------------
    num_samples = 50

    print(f"\nRunning Monte Carlo for speed field S*(q) with {num_samples} samples...\n")

    mean_S, var_S = monte_carlo_speedfield(
        mean_sdf,
        std_sdf,
        num_samples=num_samples,
        rng_seed=1
    )

    # Visualize mean & variance of S*
    plot_speed(mean_S, "Monte Carlo Mean speed field E[S*]")
    plot_variance(var_S, "Monte Carlo Variance Var[S*]")
    plot_speedfield_variance_masked(var_S, mean_sdf, "Variance of S* (Masked Obstacles)")

    # ------------------------------------------------------
    # Monte Carlo FMM execution
    # ------------------------------------------------------
    num_samples = 50
    src_idx = (0, 0, 0)  # starting point

    print(f"Running Monte Carlo FMM with {num_samples} samples...")
    print("This may take a few seconds depending on grid size.")

    mean_T, var_T = monte_carlo_traveltime(
        mean_sdf,
        std_sdf,
        num_samples=num_samples,
        src_idx=src_idx,
        rng_seed=42
    )

    # Squeeze final dimension for 2D visualization
    mean_T2 = mean_T[:, :, 0]
    var_T2 = var_T[:, :, 0]

    # ------------------------------------------------------
    # Visualize final results
    # ------------------------------------------------------
    plot_traveltime(mean_T2, "Mean Travel Time T(q)")
    plot_variance(var_T2, "Variance of Travel Time Var[T(q)]")
    plot_traveltime_variance_masked(var_T2, mean_sdf, "Variance of T (Masked Obstacles)")

    print("\n=== Monte Carlo FMM Complete ===")
    print("Visualizations generated successfully.\n")