import numpy as np
import matplotlib.pyplot as plt


def _check_3d(arr: np.ndarray):
    if arr.ndim != 3:
        raise ValueError("Expected a 3D array.")

def plot_3d_field(field, title="3D Field", cmap="viridis", alpha_scale=0.7, threshold_ratio=0.05):
    """
    Render a full 3D volumetric heatmap of a scalar field.

    - field: 3D ndarray
    - cmap: colormap
    - alpha_scale: max transparency level
    - threshold_ratio: minimum normalized value to show (filters noise)

    This produces a true 3D volume of colored voxels.
    """
    assert field.ndim == 3, "Field must be 3D"

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # Normalize field to [0, 1]
    fmin, fmax = np.min(field), np.max(field)
    norm = (field - fmin) / (fmax - fmin + 1e-12)

    # Mask very small values (cleans up plot)
    mask = norm > threshold_ratio

    # Color + transparency
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm)
    colors[..., -1] = norm * alpha_scale  # alpha = normalized magnitude

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(mask, facecolors=colors, edgecolor='k', linewidth=0.1)

    ax.set_title(f"{title}\n(min={fmin:.3f}, max={fmax:.3f})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()

def plot_3d_isosurface(field, level=None, title="3D Isosurface", color="cyan"):
    """
    Plot 3D isosurface using marching cubes.
    If level is None → automatically uses median value.
    """
    from skimage import measure
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    assert field.ndim == 3, "Field must be 3D"

    if level is None:
        level = np.median(field)

    verts, faces, normals, values = measure.marching_cubes(field, level)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.5)
    mesh.set_facecolor(color)
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlim(0, field.shape[0])
    ax.set_ylim(0, field.shape[1])
    ax.set_zlim(0, field.shape[2])
    ax.set_title(f"{title}\n(level={level:.3f})")

    plt.tight_layout()
    plt.show()

def plot_3d_field_masked(field, mean_sdf, title="3D Field (Free Space Only)", cmap="magma"):
    """
    Shows a 3D volume but masks obstacle interiors.
    """
    mask = mean_sdf >= 0
    masked_field = field * mask  # zero-out obstacles

    plot_3d_field(masked_field, title=title, cmap=cmap)


def plot_sdf_slice(sdf: np.ndarray,
                   z_index: int | None = None,
                   title="SDF Slice",
                   cmap="coolwarm"):
    """
    Plot a 2D slice of a 3D SDF at given z_index (or mid-plane if None).
    """
    _check_3d(sdf)
    if z_index is None:
        z_index = sdf.shape[2] // 2

    slice_ = sdf[:, :, z_index]

    plt.figure(figsize=(6, 6))
    im = plt.imshow(slice_.T, origin="lower", cmap=cmap)
    plt.colorbar(im)
    plt.title(f"{title} (z={z_index})")
    plt.tight_layout()
    plt.show()


def plot_field_slice(field: np.ndarray,
                     z_index: int | None = None,
                     title="Field Slice",
                     cmap="viridis"):
    _check_3d(field)
    if z_index is None:
        z_index = field.shape[2] // 2

    slice_ = field[:, :, z_index]

    plt.figure(figsize=(6, 6))
    im = plt.imshow(slice_.T, origin="lower", cmap=cmap)
    plt.colorbar(im)
    plt.title(f"{title} (z={z_index})")
    plt.tight_layout()
    plt.show()


def plot_variance_masked_slice(var_field: np.ndarray,
                               sdf_mean: np.ndarray,
                               z_index: int | None = None,
                               title="Variance (masked free space)",
                               cmap="magma"):
    """
    Mask obstacle interior (mean SDF < 0) before plotting a slice.
    """
    _check_3d(var_field)
    _check_3d(sdf_mean)

    if z_index is None:
        z_index = var_field.shape[2] // 2

    var_slice = var_field[:, :, z_index].copy()
    sdf_slice = sdf_mean[:, :, z_index]

    var_slice[sdf_slice < 0] = np.nan

    plt.figure(figsize=(6, 6))
    im = plt.imshow(var_slice.T, origin="lower", cmap=cmap)
    plt.colorbar(im)
    plt.title(f"{title} (z={z_index})")
    plt.tight_layout()
    plt.show()

def plot_sdf_voxels(mean_sdf, threshold=0.0, title="3D Obstacle Voxel Plot"):
    """
    Render a 3D voxel view of obstacles (mean_sdf < 0).
    Useful for visualizing full 3D worlds.
    """
    _check_3d(mean_sdf)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    mask = mean_sdf < threshold  # obstacles where SDF < 0

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.voxels(mask, facecolors='red', edgecolor='k', alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

def plot_floor_slice(mean_sdf, z_world=0.0, map_min=-1.0, map_max=1.0, title="Floor Slice (z=0)"):
    """
    Extract and plot the SDF slice at world-coordinate z = z_world.
    If z_world=0, this gives the literal floor-level slice.
    """
    _check_3d(mean_sdf)

    nx, ny, nz = mean_sdf.shape
    
    # Compute corresponding index
    z_lin = np.linspace(map_min, map_max, nz)
    z_idx = np.argmin(np.abs(z_lin - z_world))

    slice_ = mean_sdf[:, :, z_idx]

    plt.figure(figsize=(6,6))
    im = plt.imshow(slice_.T, origin='lower', cmap='coolwarm')
    plt.colorbar(im)
    plt.title(f"{title} (z={z_world}, idx={z_idx})")
    plt.tight_layout()
    plt.show()

def plot_3d_slices(field, title="3D Field Slices", cmap="viridis",
                   x_slice=None, y_slice=None, z_slice=None):
    """
    Show 3 orthogonal slices (XY, XZ, YZ).
    Automatically picks mid-slices if none are provided.
    """
    assert field.ndim == 3, "field must be 3D"
    nx, ny, nz = field.shape

    if x_slice is None: x_slice = nx // 2
    if y_slice is None: y_slice = ny // 2
    if z_slice is None: z_slice = nz // 2

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(field[:, :, z_slice].T, origin="lower", cmap=cmap)
    axs[0].set_title(f"XY plane (z={z_slice})")

    axs[1].imshow(field[:, y_slice, :].T, origin="lower", cmap=cmap)
    axs[1].set_title(f"XZ plane (y={y_slice})")

    axs[2].imshow(field[x_slice, :, :].T, origin="lower", cmap=cmap)
    axs[2].set_title(f"YZ plane (x={x_slice})")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_3d_volume(field, title="3D Volume Heatmap", cmap="magma", alpha_scale=0.8):
    """
    Render a 3D volume by mapping values to RGBA and using voxels().
    Suitable for variance fields, speed fields, etc.
    """
    assert field.ndim == 3

    field_norm = (field - field.min()) / (field.max() - field.min() + 1e-12)
    alpha = field_norm * alpha_scale

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Voxel mask = True everywhere except extremely small values
    mask = alpha > 0.05

    # Get colormap
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(field_norm)
    colors[..., -1] = alpha  # set transparency

    ax.voxels(mask, facecolors=colors, edgecolor='k', linewidth=0.1)

    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_3d_variance_masked(var_field, mean_sdf, title="3D Variance (Free Space Only)"):
    """
    3D volume rendering of variance while masking out obstacle interiors.
    """
    mask = mean_sdf >= 0  # free space only
    var_masked = var_field.copy()
    var_masked[~mask] = 0.0  # remove obstacle noise

    plot_3d_volume(var_masked, title=title, cmap="magma")

def plot_isosurface(field, level, title="3D Isosurface"):
    from skimage import measure
    verts, faces, normals, values = measure.marching_cubes(field, level=level)

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor('cyan')
    ax.add_collection3d(mesh)

    ax.set_xlim(0, field.shape[0])
    ax.set_ylim(0, field.shape[1])
    ax.set_zlim(0, field.shape[2])
    ax.set_title(title)

    plt.tight_layout()
    plt.show()
