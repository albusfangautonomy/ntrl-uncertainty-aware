import numpy as np


def sphere_sdf(nx, ny, nz, cx, cy, cz, r):
    """
    3D sphere SDF centered at (cx, cy, cz) with radius r.
    Domain assumed in the same units as (cx, cy, cz).
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    return np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) - r


def box_sdf_3d(nx, ny, nz, cx, cy, cz, hx, hy, hz):
    """
    Axis-aligned 3D box SDF centered at (cx, cy, cz) with half-sizes (hx, hy, hz).
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    dx = np.abs(X - cx) - hx
    dy = np.abs(Y - cy) - hy
    dz = np.abs(Z - cz) - hz

    dx_out = np.maximum(dx, 0.0)
    dy_out = np.maximum(dy, 0.0)
    dz_out = np.maximum(dz, 0.0)
    outside = np.sqrt(dx_out**2 + dy_out**2 + dz_out**2)

    inside = np.minimum(np.maximum.reduce([dx, dy, dz]), 0.0)

    return outside + inside


def add_boundary_uncertainty(sdf: np.ndarray, scale=0.05, band_width=0.1) -> np.ndarray:
    """
    Uncertainty concentrated near SDF=0 boundary.
    """
    return scale * np.exp(-(np.abs(sdf) / band_width) ** 2)
