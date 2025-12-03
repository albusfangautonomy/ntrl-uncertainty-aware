# visualize/heatmap.py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch

def heatmap_2d(field_2d, title="", cmap="viridis"):
    f = field_2d.cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(f.T, origin='lower', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()
