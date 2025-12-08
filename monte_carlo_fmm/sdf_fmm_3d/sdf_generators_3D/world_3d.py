import numpy as np
from .shapes_3d import sphere_sdf, box_sdf_3d, add_boundary_uncertainty_3d
from sdf_generators_3D.utils import combine_sdfs  # same min(...) rule


def generate_world_3d(nx, ny, nz, n_obstacles=3):
    """
    Generate a 3D world made of multiple random spheres or boxes.
    """
    sdfs = []

    for i in range(n_obstacles):
        obstacle_type = np.random.choice(["sphere", "box"])

        cx = np.random.uniform(-0.6, 0.6)
        cy = np.random.uniform(-0.6, 0.6)
        cz = np.random.uniform(-0.6, 0.6)

        if obstacle_type == "sphere":
            r = np.random.uniform(0.15, 0.35)
            sdf_i = sphere_sdf(nx, ny, nz, cx, cy, cz, r)

        else:  # 3D rectangular box
            half_wx = np.random.uniform(0.1, 0.25)
            half_wy = np.random.uniform(0.1, 0.25)
            half_wz = np.random.uniform(0.1, 0.25)
            sdf_i = box_sdf_3d(nx, ny, nz, cx, cy, cz, half_wx, half_wy, half_wz)

        sdfs.append(sdf_i)

    return combine_sdfs(*sdfs)


def generate_world_with_uncertainty_3d(nx, ny, nz, n_obstacles=3, scale=0.05):
    mean_sdf = generate_world_3d(nx, ny, nz, n_obstacles)
    std_sdf = add_boundary_uncertainty_3d(mean_sdf, scale)
    return mean_sdf, std_sdf
