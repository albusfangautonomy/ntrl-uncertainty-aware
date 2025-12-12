import numpy as np
import sys, os

# Path fix
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from sdf_gen.random_world_3d import generate_world_with_uncertainty_3d
from core.speed_mapping import sdf_to_speed
from core.mc_speedfield import monte_carlo_speedfield
from core.mc_driver import monte_carlo_traveltime
from viz3d.plot_fields_3d import (
    plot_sdf_slice,
    plot_field_slice,
    plot_variance_masked_slice,
    plot_sdf_voxels,
    plot_floor_slice,
    plot_3d_slices,
    plot_3d_variance_masked
)


if __name__ == "__main__":

    # Grid resolution
    nx, ny, nz = 48, 48, 48
    map_size = 5.0  # meters, for your own semantic use

    n_obstacles = 3
    num_samples = 30

    print(f"\n=== 3D MC FMM: {n_obstacles} obstacles, {num_samples} samples ===\n")

    # Generate 3D world
    mean_sdf, std_sdf = generate_world_with_uncertainty_3d(
        nx, ny, nz,
        n_obstacles=n_obstacles,
        map_size=map_size,
        scale_range=(0.02, 0.08),
    )

    # Voxel spacing (for physical units in PyKonal)
    dx = map_size / (nx - 1)
    dy = map_size / (ny - 1)
    dz = map_size / (nz - 1)
    node_intervals = (dx, dy, dz)

    # ------------------------------------------------------
    # Plot mean SDF slice + SDF variance slice
    # ------------------------------------------------------
    plot_sdf_slice(mean_sdf, title="Mean SDF (3D)")
    print("Showing 3D voxel plot of obstacles...")

    plot_sdf_voxels(mean_sdf, title="3D Obstacle Voxel View")

    print("Showing 2D floor slice at z = 0...")
    plot_floor_slice(mean_sdf, z_world=0.0, map_min=-map_size/2, map_max=map_size/2)


    sdf_var = std_sdf ** 2
    plot_field_slice(sdf_var, title="SDF Variance (3D)", cmap="magma")

    # ------------------------------------------------------
    # Monte Carlo speed field S*
    # ------------------------------------------------------
    print("Running Monte Carlo for speed field S*(x)...")
    mean_S, var_S = monte_carlo_speedfield(
        mean_sdf,
        std_sdf,
        num_samples=num_samples,
        rng_seed=123,
    )

    plot_field_slice(mean_S, title="Mean Speed Field E[S*]")
    plot_variance_masked_slice(var_S, mean_sdf, title="Var[S*] masked (free space)")
    print("3D Slice Views of S*:")
    plot_3d_slices(mean_S, title="Mean S* Slices")
    plot_3d_slices(var_S, title="Var(S*) Slices", cmap="magma")

    print("3D Volume Rendering of Var(S*):")
    plot_3d_variance_masked(var_S, mean_sdf, title="Masked Var(S*) Volume")

    # ------------------------------------------------------
    # Monte Carlo traveltime T(x)
    # ------------------------------------------------------
    print("Running Monte Carlo FMM (traveltime)...")
    src_idx = (0, 0, 0)  # lower corner

    mean_T, var_T = monte_carlo_traveltime(
        mean_sdf,
        std_sdf,
        num_samples=num_samples,
        src_idx=src_idx,
        min_coords=(-map_size/2, -map_size/2, -map_size/2),
        node_intervals=node_intervals,
        rng_seed=42,
    )

    plot_field_slice(mean_T, title="Mean Travel Time E[T]")
    plot_variance_masked_slice(var_T, mean_sdf, title="Var[T] masked (free space)")

    print("3D Slice Views of T:")
    plot_3d_slices(mean_T, title="Mean Travel Time T Slices", cmap="plasma")
    plot_3d_slices(var_T, title="Var(T) Slices (Masked)", cmap="inferno")

    print("3D Volume Rendering of T variance:")
    plot_3d_variance_masked(var_T, mean_sdf, title="3D Var(T) (Free Space Only)")


    print("\n=== 3D Monte Carlo FMM Complete ===\n")
