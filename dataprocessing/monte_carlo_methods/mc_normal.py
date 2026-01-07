import torch
from scipy.spatial import cKDTree
import torch.nn.functional as F
from dataprocessing.speed_sampling_gpu_kdtree_normal import point_obstacle_distance

def mc_normal_stats(query_points,
                    v_obs,
                    normal_obs,
                    base_kdtree,
                    K=8,
                    sigma_geom=0.01):
    """
    Monte Carlo normal estimation.

    Args:
        query_points : (N, dim) CUDA tensor
        v_obs        : (M, dim) CUDA tensor of obstacle surface samples
        normal_obs   : (M, dim) CUDA tensor of surface normals
        base_kdtree  : cKDTree built from the *mean* geometry (optional baseline)
        K            : number of MC samples
        sigma_geom   : std dev of geometry perturbation (in normalized coordinates)

    Returns:
        normal_mean : (N, dim) CUDA tensor (unit vectors)
        normal_var  : (N,)     CUDA tensor (angular dispersion)
    """

    normals_mc = []

    for k in range(K):
        # ---- sample noisy obstacle geometry ----
        v_noisy = v_obs + sigma_geom * torch.randn_like(v_obs)

        # KDTree needs CPU numpy
        kdt = cKDTree(v_noisy.detach().cpu().numpy())

        # reuse distance+normal routine
        _, _, n_k = point_obstacle_distance(query_points, kdt, v_noisy, normal_obs)

        # ensure unit length
        n_k = F.normalize(n_k, dim=-1)

        normals_mc.append(n_k)

    # (K, N, dim)
    normals_mc = torch.stack(normals_mc, dim=0)

    # ---- mean direction ----
    normal_mean = F.normalize(normals_mc.mean(dim=0), dim=-1)

    # ---- angular variance (1 - cosine agreement) ----
    cos_sim = (normals_mc * normal_mean.unsqueeze(0)).sum(dim=-1)   # (K, N)
    normal_var = (1 - cos_sim).mean(dim=0)                          # (N,)

    return normal_mean, normal_var
