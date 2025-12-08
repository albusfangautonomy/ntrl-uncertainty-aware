import numpy as np


def random_circle_sdf(nx, ny, radius_range=(0.2, 0.5)):
    """
    Generate a random circle SDF in [-1,1]^2.
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    r = np.random.uniform(*radius_range)
    sdf = np.sqrt(X**2 + Y**2) - r
    return sdf


def random_box_sdf(nx, ny, box_range=((0.3, 0.7), (0.3, 0.7))):
    """
    Generate an axis-aligned box SDF in [-1,1]^2.

    box_range:
        ((xmin_ratio, xmax_ratio), (ymin_ratio, ymax_ratio))
        defines the fractional extent of the box within [-1,1].

    Example:
        (0.3, 0.7) means box spans x in [-0.4, 0.4].
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Convert fractional to coordinates
    (rx0, rx1), (ry0, ry1) = box_range
    xmin = -1 + 2 * rx0
    xmax = -1 + 2 * rx1
    ymin = -1 + 2 * ry0
    ymax = -1 + 2 * ry1

    # Signed distance to axis-aligned box
    dx = np.maximum.reduce([xmin - X, 0.0, X - xmax])
    dy = np.maximum.reduce([ymin - Y, 0.0, Y - ymax])
    outside_dist = np.sqrt(dx**2 + dy**2)

    # Interior distance (negative)
    inside_dx = np.minimum(X - xmin, xmax - X)
    inside_dy = np.minimum(Y - ymin, ymax - Y)
    inside = np.minimum(inside_dx, inside_dy)

    sdf = outside_dist.copy()
    sdf[inside > 0] = -inside[inside > 0]
    return sdf


def circle_sdf(nx, ny, cx, cy, r):
    """
    Analytic circle SDF: distance to circle centered at (cx,cy).
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.sqrt((X - cx)**2 + (Y - cy)**2) - r


def box_sdf(nx, ny, cx, cy, half_w, half_h):
    """
    Axis-aligned box SDF centered at (cx, cy) with half widths.
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Signed distance to AABB
    dx = np.abs(X - cx) - half_w
    dy = np.abs(Y - cy) - half_h

    # Outside distance
    dx_out = np.maximum(dx, 0.0)
    dy_out = np.maximum(dy, 0.0)
    outside = np.sqrt(dx_out**2 + dy_out**2)

    # Inside distance (negative)
    inside = np.minimum(np.maximum(dx, dy), 0.0)

    return outside + inside


def add_boundary_uncertainty(sdf, scale=0.05):
    """
    Gaussian uncertainty concentrated around SDF=0 boundary.
    """
    return scale * np.exp(-(np.abs(sdf) / 0.1)**2)
