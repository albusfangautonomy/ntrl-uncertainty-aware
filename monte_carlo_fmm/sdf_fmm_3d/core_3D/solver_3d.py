import numpy as np
import pykonal


def setup_solver_from_speed_3d(speed, min_coords=(0,0,0),
                               node_intervals=(1,1,1),
                               src_idx=(0,0,0)):
    nx, ny, nz = speed.shape

    solver = pykonal.EikonalSolver(coord_sys="cartesian")

    solver.velocity.min_coords = min_coords
    solver.velocity.node_intervals = node_intervals
    solver.velocity.npts = (nx, ny, nz)
    solver.velocity.values = speed

    solver.traveltime.min_coords = min_coords
    solver.traveltime.node_intervals = node_intervals
    solver.traveltime.npts = (nx, ny, nz)

    solver.traveltime.values[src_idx] = 0.0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)

    return solver
