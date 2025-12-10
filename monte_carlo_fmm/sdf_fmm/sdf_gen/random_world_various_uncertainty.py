import numpy as np
from .utils import combine_sdfs
from .random_shapes import circle_sdf, box_sdf


def generate_multi_obstacle_sdf_with_uncertainty(
    nx, ny, n_obstacles=3, map_size=5.0,
    scale_range=(0.02, 0.10)   # each obstacle gets a random uncertainty scale
):
    """
    Generate mean_sdf and std_sdf for multiple obstacles.
    Each obstacle has its OWN uncertainty scale.
    
    RETURNS:
        mean_sdf : Nx × Ny world SDF
        std_sdf  : Nx × Ny world uncertainty
    """

    L = map_size / 2
    mean_sdfs = []    # list of per-obstacle SDFs
    std_sdfs  = []    # list of per-obstacle STD fields (per obstacle)

    for _ in range(n_obstacles):

        obstacle_type = np.random.choice(["circle", "box"])

        # Random placement within ±40% of map
        center_range = 0.6 * map_size
        cx = np.random.uniform(-center_range, center_range)
        cy = np.random.uniform(-center_range, center_range)

        # Obstacle geometry
        if obstacle_type == "circle":
            r = np.random.uniform(0.1 * map_size, 0.25 * map_size)
            sdf_i = circle_sdf(nx, ny, cx, cy, r)

        else:
            hw = np.random.uniform(0.05 * map_size, 0.25 * map_size)
            hh = np.random.uniform(0.05 * map_size, 0.25 * map_size)
            sdf_i = box_sdf(nx, ny, cx, cy, hw, hh)

        # Obstacle-specific uncertainty scale
        scale_i = np.random.uniform(*scale_range)

        # Boundary-local uncertainty per obstacle
        std_i = scale_i * np.exp(-(np.abs(sdf_i) / 0.1)**2)

        mean_sdfs.append(sdf_i)
        std_sdfs.append(std_i)

    # -------------------------------
    # Combine obstacles: mean SDF = min over all obstacles
    # -------------------------------
    mean_sdf = mean_sdfs[0].copy()
    for i in range(1, n_obstacles):
        mean_sdf = np.minimum(mean_sdf, mean_sdfs[i])


    # -------------------------------
    # Combine uncertainties:
    # STD(x) = uncertainty of the nearest obstacle (min SDF)
    # -------------------------------
    std_sdf = np.zeros_like(mean_sdf)

    # Compute argmin over SDFs to find closest obstacle at each voxel
    all_sdfs = np.stack(mean_sdfs, axis=-1)   # shape (nx, ny, n_obst)
    argmin = np.argmin(all_sdfs, axis=-1)     # which obstacle is closest

    # Assign uncertainty based on winning obstacle
    all_stds = np.stack(std_sdfs, axis=-1)
    std_sdf = all_stds[np.arange(nx)[:,None], np.arange(ny)[None,:], argmin]

    return mean_sdf, std_sdf


# ------------------------------------------------------------
# Legacy API you previously used—now updated to call new version
# ------------------------------------------------------------
def generate_world_with_uncertainty(
    nx, ny, n_obstacles=3,
    map_size=5.0, scale_range=(0.02, 0.10)
):
    return generate_multi_obstacle_sdf_with_uncertainty(
        nx, ny, n_obstacles, map_size, scale_range
    )
