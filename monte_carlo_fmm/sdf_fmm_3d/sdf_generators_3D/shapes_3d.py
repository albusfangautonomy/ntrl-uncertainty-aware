import numpy as np


def sphere_sdf(nx, ny, nz, cx, cy, cz, r):
    """
    3D sphere signed distance field.
    Center (cx,cy,cz), radius r.
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) - r


def box_sdf_3d(nx, ny, nz, cx, cy, cz, half_wx, half_wy, half_wz):
    """
    3D axis-aligned rectangular prism SDF.
    Center (cx, cy, cz) with half-sizes.
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    dx = np.abs(X - cx) - half_wx
    dy = np.abs(Y - cy) - half_wy
    dz = np.abs(Z - cz) - half_wz

    # Outside distance
    dx_out = np.maximum(dx, 0)
    dy_out = np.maximum(dy, 0)
    dz_out = np.maximum(dz, 0)
    outside = np.sqrt(dx_out**2 + dy_out**2 + dz_out**2)

    # Inside penetration (negative)
    inside = np.minimum(np.maximum.reduce([dx, dy, dz]), 0.0)

    return outside + inside


def add_boundary_uncertainty_3d(sdf, scale=0.05):
    return scale * np.exp(-(np.abs(sdf) / 0.1)**2)
