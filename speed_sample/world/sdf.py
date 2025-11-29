# world/sdf.py
import torch
import torch.nn.functional as F

def compute_sdf(occ):
    """
    occ: boolean occupancy grid (1=obstacle)
    returns: signed distance field (float), same size
    """

    device = occ.device
    occ_f = occ.float()

    # Distance to obstacle = distance transform on ~occ (free space)
    # Distance inside obstacle = negative distance transform on occ
    #
    # We approximate EDT using repeated 3D pooling (fast and differentiable).

    def edt_approx(binary):
        # binary = 1 where "seed", 0 elsewhere
        # convolutional growth: at each step, wavefront expands by 1 voxel
        dist = torch.full_like(binary, 1e6)
        frontier = (binary > 0)
        dist[frontier] = 0

        # BFS-like multi-iteration convolution
        kernel = torch.ones((1,1,3,3,3), device=device)
        for step in range(100):   # enough to fill typical grid
            expanded = F.conv3d(frontier.float().unsqueeze(0).unsqueeze(0), 
                                kernel, padding=1)[0,0] > 0
            update_mask = (expanded & (dist > step+1))
            if not update_mask.any():
                break
            dist[update_mask] = step+1
            frontier = update_mask

        return dist

    # Distance to nearest obstacle from outside
    dist_out = edt_approx(~occ)

    # Distance to nearest free region from inside
    dist_in  = edt_approx(occ)

    # Signed: outside positive, inside negative
    sdf = dist_out - dist_in

    return sdf.float()
