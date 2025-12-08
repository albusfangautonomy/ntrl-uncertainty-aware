import numpy as np
import pykonal


def setup_solver_from_speed(speed: np.ndarray,
                            min_coords=(0.0, 0.0, 0.0),
                            node_intervals=(1.0, 1.0, 1.0),
                            src_idx=(0, 0, 0)) -> pykonal.EikonalSolver:
    """
    Creates and initializes a PyKonal solver.
    """
    # Expand 2D → 3D for PyKonal
    speed_3d = speed[:, :, None] if speed.ndim == 2 else speed

    nx, ny, nz = speed_3d.shape

    solver = pykonal.EikonalSolver(coord_sys="cartesian")

    solver.velocity.min_coords = min_coords
    solver.velocity.node_intervals = node_intervals
    solver.velocity.npts = (nx, ny, nz)
    solver.velocity.values = speed_3d

    solver.traveltime.min_coords = min_coords
    solver.traveltime.node_intervals = node_intervals
    solver.traveltime.npts = (nx, ny, nz)

    solver.traveltime.values[src_idx] = 0.0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)

    return solver
