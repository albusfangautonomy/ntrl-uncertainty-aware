import numpy as np
from .utils import combine_sdfs
from .random_shapes import circle_sdf, box_sdf, add_boundary_uncertainty


def generate_multi_obstacle_sdf(nx, ny, n_obstacles=3, map_size=5.0):
    """
    Generate a world with multiple random obstacles inside a map of size map_size (meters).
    Domain = [-map_size/2, map_size/2].
    """

    L = map_size / 2      # half-size of map

    sdfs = []

    for _ in range(n_obstacles):
        obstacle_type = np.random.choice(["circle", "box"])

        # Allow obstacles anywhere within ±40% of map
        center_range = 0.4 * map_size

        cx = np.random.uniform(-center_range, center_range)
        cy = np.random.uniform(-center_range, center_range)

        if obstacle_type == "circle":
            # Circle radius scaled to map size
            r = np.random.uniform(0.1, 0.35)
            sdf_i = circle_sdf(nx, ny, cx, cy, r)

        else:
            # Box half-dimensions scaled to map size
            hw = np.random.uniform(0.1, 0.25)
            hh = np.random.uniform(0.1, 0.25)
            sdf_i = box_sdf(nx, ny, cx, cy, hw, hh)

        sdfs.append(sdf_i)

    return combine_sdfs(*sdfs)


def generate_world_with_uncertainty(nx, ny, n_obstacles=3, map_size=5.0, scale=0.05):
    mean_sdf = generate_multi_obstacle_sdf(nx, ny, n_obstacles, map_size)
    std_sdf = add_boundary_uncertainty(mean_sdf, scale)
    return mean_sdf, std_sdf
