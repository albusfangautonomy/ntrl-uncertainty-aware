import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------
# ----------- Load the dataset ----------
# ---------------------------------------

path = "datasets/test/box"
path = "."

pts   = np.load(f"{path}/sampled_points.npy")
s_mu  = np.load(f"{path}/speed_mean.npy")
s_var = np.load(f"{path}/speed_var.npy")
n_mu  = np.load(f"{path}/normal_mean.npy")
n_var = np.load(f"{path}/normal_var.npy")

print("points:", pts.shape)
print("speed μ:", s_mu.shape)
print("speed σ²:", s_var.shape)
print("normal μ:", n_mu.shape)
print("normal σ²:", n_var.shape)

x = pts[:,0]
y = pts[:,1]

nx = n_mu[:,0]
ny = n_mu[:,1]

# ---------------------------------------
# -----   Visualize Speed Mean      -----
# ---------------------------------------

plt.figure(figsize=(6,5))
sc = plt.scatter(x, y, c=s_mu[:,0], s=3, cmap="viridis")
plt.colorbar(sc, label="speed mean (start)")
plt.title("Speed Mean Field (start point)")
plt.axis("equal")
plt.show()

# ---------------------------------------
# ----- Visualize Speed Uncertainty -----
# ---------------------------------------

plt.figure(figsize=(6,5))
sc = plt.scatter(x, y, c=s_var[:,0], s=3, cmap="magma")
plt.colorbar(sc, label="speed variance (start)")
plt.title("Speed Uncertainty")
plt.axis("equal")
plt.show()

# ---------------------------------------
# -------- Visualize Normal Mean --------
# ---------------------------------------

plt.figure(figsize=(6,6))
plt.quiver(x, y, nx, ny, angles="xy", scale_units="xy", scale=10)
plt.title("Normal Mean Vectors")
plt.axis("equal")
plt.show()

# ---------------------------------------
# ------ Visualize Normal Uncertainty ---
# ---------------------------------------

plt.figure(figsize=(6,5))
sc = plt.scatter(x, y, c=n_var[:,0], s=3, cmap="plasma")
plt.colorbar(sc, label="normal variance (start)")
plt.title("Normal Uncertainty")
plt.axis("equal")
plt.show()

# ---------------------------------------
# ---- Overlay: Speed + Normal Arrows ---
# ---------------------------------------

plt.figure(figsize=(6,6))
sc = plt.scatter(x, y, c=s_mu[:,0], s=3, cmap="viridis")
plt.quiver(x, y, nx, ny, color="red", scale=20)
plt.colorbar(sc, label="speed mean (start)")
plt.title("Speed + Normals")
plt.axis("equal")
plt.show()
