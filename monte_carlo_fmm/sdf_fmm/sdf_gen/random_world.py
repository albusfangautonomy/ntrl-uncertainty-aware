import numpy as np
from .utils import combine_sdfs
from .random_shapes import circle_sdf, box_sdf, add_boundary_uncertainty


def generate_multi_obstacle_sdf(nx, ny, n_obstacles=3):
    """
    Generate a world with multiple random obstacles.
    Supports circles + boxes in random combinations.
    """

    sdfs = []

    for i in range(n_obstacles):
        obstacle_type = np.random.choice(["circle", "box"])

        # random center
        cx = np.random.uniform(-0.6, 0.6)
        cy = np.random.uniform(-0.6, 0.6)

        if obstacle_type == "circle":
            r = np.random.uniform(0.15, 0.35)
            sdf_i = circle_sdf(nx, ny, cx, cy, r)
        else:
            half_w = np.random.uniform(0.1, 0.25)
            half_h = np.random.uniform(0.1, 0.25)
            sdf_i = box_sdf(nx, ny, cx, cy, half_w, half_h)

        sdfs.append(sdf_i)

    # CSG union of all obstacles
    sdf_world = combine_sdfs(*sdfs)
    return sdf_world


def generate_world_with_uncertainty(nx, ny, n_obstacles=3, scale=0.05):
    """
    Generate both mean_sdf and std_sdf for a multi-obstacle world.
    """
    mean_sdf = generate_multi_obstacle_sdf(nx, ny, n_obstacles)
    std_sdf = add_boundary_uncertainty(mean_sdf, scale)
    return mean_sdf, std_sdf
