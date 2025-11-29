# main.py
import torch

from world.world_gen import generate_random_world
from world.sdf import compute_sdf
from uncertainty.variance_models import simple_edge_uncertainty
from uncertainty.effective_distance import compute_effective_distance, compute_speed_from_distance
from visualize.heatmap import heatmap_2d
from visualize.world_viz import show_occupancy_slice, show_3d_points
DEBUG = True
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. World generation
    occ = generate_random_world(device=device)
    if DEBUG:
        # 1. World generation
        occ = generate_random_world(device=device)

        # Visualize world
        show_occupancy_slice(occ, z=0, title="Ground occupancy")
        show_occupancy_slice(occ, z=5, title="Height slice")
        show_3d_points(occ, title="3D obstacle scatter")

    # 2. Mean SDF (μ)
    mu = compute_sdf(occ)

    # 3. Uncertainty σ
    sigma = simple_edge_uncertainty(mu, scale=5.0)

    # 4. Effective distance
    d_eff = compute_effective_distance(mu, sigma, lam=2.0)

    # 5. Speed S*
    S = compute_speed_from_distance(d_eff, d_min=0.0, d_max=30.0)

    # 6. Visualize ground slice (z=1)
    z = 1
    heatmap_2d(mu[:,:,z],     "Mean SDF slice")
    heatmap_2d(sigma[:,:,z],  "Uncertainty σ slice")
    heatmap_2d(d_eff[:,:,z],  "Effective distance slice")
    heatmap_2d(S[:,:,z],      "Speed S*(q) slice")

if __name__ == "__main__":
    main()
