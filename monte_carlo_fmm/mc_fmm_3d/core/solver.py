import numpy as np
import pykonal


def setup_solver_from_speed(speed: np.ndarray,
                            min_coords=(0.0, 0.0, 0.0),
                            node_intervals=(1.0, 1.0, 1.0),
                            src_idx=(0, 0, 0)) -> pykonal.EikonalSolver:
    """
    Create and initialize a 3D PyKonal EikonalSolver from a speed field.

    speed: 3D array (nx, ny, nz)
    min_coords: (x_min, y_min, z_min)
    node_intervals: (dx, dy, dz)
    src_idx: index tuple (ix, iy, iz)
    """
    if speed.ndim != 3:
        raise ValueError("speed must be a 3D array for 3D FMM.")

    nx, ny, nz = speed.shape

    solver = pykonal.EikonalSolver(coord_sys="cartesian")

    solver.velocity.min_coords = tuple(min_coords)
    solver.velocity.node_intervals = tuple(node_intervals)
    solver.velocity.npts = (nx, ny, nz)
    solver.velocity.values = speed

    solver.traveltime.min_coords = tuple(min_coords)
    solver.traveltime.node_intervals = tuple(node_intervals)
    solver.traveltime.npts = (nx, ny, nz)

    # Initialize source
    solver.traveltime.values[:] = np.inf
    solver.unknown[:] = True

    solver.traveltime.values[src_idx] = 0.0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)

    return solver
