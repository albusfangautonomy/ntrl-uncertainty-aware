import numpy as np


def sample_sdf(mean_sdf: np.ndarray,
               std_sdf: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """
    Sample one SDF realization from a Gaussian distribution per voxel.
    Works for 2D or 3D arrays.
    """
    if mean_sdf.shape != std_sdf.shape:
        raise ValueError("mean_sdf and std_sdf must have same shape.")
    return rng.normal(loc=mean_sdf, scale=std_sdf)
