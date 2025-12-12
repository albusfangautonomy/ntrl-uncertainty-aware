import numpy as np
from .utils import combine_sdfs
from .shapes3d import sphere_sdf, box_sdf_3d, add_boundary_uncertainty


def generate_multi_obstacle_sdf_with_uncertainty_3d(
    nx, ny, nz,
    n_obstacles=3,
    map_size=5.0,
    scale_range=(0.02, 0.10),
):
    """
    3D version:
    - Multiple obstacles (spheres and boxes) in a cube domain of side `map_size`.
    - Each obstacle has its own uncertainty scale.
    - Returns mean_sdf and std_sdf over the whole world.
    """

    # We'll parametrize centers and sizes in world coords within [-L, L]
    L = map_size / 2.0

    mean_sdfs = []
    std_sdfs = []

    for _ in range(n_obstacles):
        obstacle_type = np.random.choice(["sphere", "box"])

        # center inside ±40% of domain
        center_range = 0.4 * L
        cx = np.random.uniform(-center_range, center_range)
        cy = np.random.uniform(-center_range, center_range)
        cz = np.random.uniform(-center_range, center_range)

        if obstacle_type == "sphere":
            r = np.random.uniform(0.1 * L, 0.3 * L)
            sdf_i = sphere_sdf(nx, ny, nz, cx, cy, cz, r)
        else:
            hx = np.random.uniform(0.05 * L, 0.25 * L)
            hy = np.random.uniform(0.05 * L, 0.25 * L)
            hz = np.random.uniform(0.05 * L, 0.25 * L)
            sdf_i = box_sdf_3d(nx, ny, nz, cx, cy, cz, hx, hy, hz)

        scale_i = np.random.uniform(*scale_range)
        std_i = add_boundary_uncertainty(sdf_i, scale=scale_i, band_width=0.1 * L)

        mean_sdfs.append(sdf_i)
        std_sdfs.append(std_i)

    # Stack along obstacle axis
    all_sdfs = np.stack(mean_sdfs, axis=-1)  # (nx, ny, nz, n_obst)
    all_stds = np.stack(std_sdfs, axis=-1)

    # Min over obstacles = union SDF
    mean_sdf = np.min(all_sdfs, axis=-1)

    # Argmin to know which obstacle dominates each voxel
    argmin = np.argmin(all_sdfs, axis=-1)

    nx_, ny_, nz_ = mean_sdf.shape
    std_sdf = np.zeros_like(mean_sdf)

    # Fancy indexing to pick std from winning obstacle
    idx_x = np.arange(nx_)[:, None, None]
    idx_y = np.arange(ny_)[None, :, None]
    idx_z = np.arange(nz_)[None, None, :]

    std_sdf = all_stds[idx_x, idx_y, idx_z, argmin]

    return mean_sdf, std_sdf


def generate_world_with_uncertainty_3d(
    nx, ny, nz,
    n_obstacles=3,
    map_size=5.0,
    scale_range=(0.02, 0.10),
):
    return generate_multi_obstacle_sdf_with_uncertainty_3d(
        nx, ny, nz,
        n_obstacles=n_obstacles,
        map_size=map_size,
        scale_range=scale_range,
    )
