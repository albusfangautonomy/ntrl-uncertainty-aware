# visualize/world_viz.py
import torch
import matplotlib.pyplot as plt

def show_occupancy_slice(occ, z=0, title="Occupancy slice"):
    """
    2D slice of occupancy grid at height z.
    1 = obstacle, 0 = free.
    """
    occ2d = occ[:,:,z].cpu().float().numpy()

    plt.figure(figsize=(6,6))
    plt.imshow(occ2d.T, origin='lower', cmap='gray_r')
    plt.title(title + f" (z={z})")
    plt.tight_layout()
    plt.show()


def show_3d_points(occ, title="3D Occupancy (scatter)"):
    """
    Scatter plot of all occupied voxels.
    Good for understanding obstacle geometry.
    """
    occ_idx = occ.nonzero().cpu().numpy()
    xs, ys, zs = occ_idx[:,0], occ_idx[:,1], occ_idx[:,2]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, s=1)
    ax.set_title(title)

    plt.show()
