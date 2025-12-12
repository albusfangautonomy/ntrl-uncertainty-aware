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
    plot_sdf_voxels,
    plot_floor_slice,
    plot_3d_field,
    plot_3d_field_masked,
    plot_3d_isosurface,
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
    # Plot SDF — voxel view + floor slice
    # ------------------------------------------------------
    print("Showing 3D voxel plot of obstacles...")
    plot_sdf_voxels(mean_sdf, title="3D Obstacle Voxel View")

    print("Showing 2D floor slice at z = 0...")
    plot_floor_slice(mean_sdf, z_world=0.0, map_min=-map_size/2, map_max=map_size/2)


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

    # ----------------------------
    # 3D BODY PLOTS — SPEED FIELD
    # ----------------------------
    print("Plotting 3D Mean Speed Field...")
    plot_3d_field(mean_S, title="3D Body: Mean Speed Field E[S*]", cmap="viridis")

    print("Plotting 3D Var(S*)...")
    plot_3d_field_masked(var_S, mean_sdf, title="3D Body: Var(S*) (Free Space Only)", cmap="magma")

    print("Plotting S* Isosurface...")
    plot_3d_isosurface(mean_S, title="S* Isosurface", color="cyan")


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

    # ----------------------------
    # 3D BODY PLOTS — TRAVEL TIME
    # ----------------------------
    print("Plotting 3D Mean Travel Time Field...")
    plot_3d_field(mean_T, title="3D Body: Mean Travel Time E[T]", cmap="plasma")

    print("Plotting 3D Var(T)...")
    plot_3d_field_masked(var_T, mean_sdf, title="3D Body: Var(T) (Free Space Only)", cmap="inferno")

    print("\n=== 3D Monte Carlo FMM Complete ===\n")
