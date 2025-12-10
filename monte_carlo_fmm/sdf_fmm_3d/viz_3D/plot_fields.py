import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

def plot_sdf_binary(sdf, title="Obstacle Map (Binary SDF)"):
    """
    Plot the SDF as a binary occupancy image.
    Black = obstacle (SDF <= 0)
    White = free (SDF > 0)
    """
    occ = sdf <= 0.0   # boolean obstacle mask

    plt.figure(figsize=(5, 5))
    plt.imshow(occ.T, origin="lower", cmap="gray_r")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def heatmap2d(arr, title="", cmap="viridis"):
    """
    Clean 2D heatmap with axes removed.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(arr.T, origin="lower", cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_sdf_heatmap(sdf, title="Signed Distance Field"):
    heatmap2d(sdf, title, cmap="coolwarm")


def plot_speed(speed, title="Speed Field S*"):
    heatmap2d(speed, title, cmap="viridis")


def plot_traveltime(T, title="Travel Time T"):
    heatmap2d(T, title, cmap="plasma")


def plot_variance(VarT, title="Variance of T"):
    heatmap2d(VarT, title, cmap="magma")
