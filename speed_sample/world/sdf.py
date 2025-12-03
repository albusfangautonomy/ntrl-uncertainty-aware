# world/sdf.py
import torch
import torch.nn.functional as F

def compute_sdf_bfs(occ):
    """
    occ: boolean occupancy grid (1=obstacle)
    returns: signed distance field (float), same size
    """

    device = occ.device

    def edt_approx(binary):
        """
        binary: bool tensor indicating seed locations (1 = seed)
        Returns float distance map (Manhattan-ish approx).
        """
        # Float tensor for distances
        dist = torch.full(binary.shape, 1e6, dtype=torch.float32, device=device)

        # Boolean frontier mask
        frontier = binary.clone()

        # Initialize distances of seeds
        dist[frontier] = 0.0

        # 3x3x3 kernel for BFS propagation
        kernel = torch.ones((1,1,3,3,3), device=device)

        for step in range(200):  # enough for typical world sizes
            # Expand frontier by 1 voxel
            expanded = F.conv3d(
                frontier.float().unsqueeze(0).unsqueeze(0),
                kernel,
                padding=1
            )[0,0] > 0

            # Update new frontier: voxels not assigned a shorter distance yet
            update_mask = expanded & (dist > step + 1)

            if not update_mask.any():
                break

            dist[update_mask] = float(step + 1)
            frontier = update_mask

        return dist

    # Distance outside obstacles
    dist_out = edt_approx(~occ)

    # Distance inside obstacles
    dist_in  = edt_approx(occ)

    # Signed SDF = outside distance minus inside distance
    sdf = dist_out - dist_in

    return sdf

import numpy as np
from scipy import ndimage

def compute_sdf(occ):
    """
    occ: boolean torch tensor where True = obstacle
    Returns signed Euclidean SDF as a torch float32 tensor.
    +dist outside obstacles
    -dist inside obstacles
    """

    # Convert to numpy
    occ_np = occ.cpu().numpy()

    # Distance OUTSIDE obstacles (distance to nearest obstacle)
    dist_out = ndimage.distance_transform_edt(~occ_np)

    # Distance INSIDE obstacles (distance to nearest free voxel)
    dist_in  = ndimage.distance_transform_edt(occ_np)

    # Signed distance: positive outside, negative inside
    sdf = dist_out - dist_in

    # Back to torch
    return torch.from_numpy(sdf).float().to(occ.device)
