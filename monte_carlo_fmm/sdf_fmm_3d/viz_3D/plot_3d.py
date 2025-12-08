import matplotlib.pyplot as plt

def plot_sdf_slice(sdf, z_idx, title="SDF Slice"):
    plt.figure(figsize=(5,5))
    plt.imshow((sdf[:,:,z_idx] <= 0).T, origin='lower', cmap="gray_r")
    plt.title(f"{title} (z={z_idx})")
    plt.tight_layout()
    plt.show()
