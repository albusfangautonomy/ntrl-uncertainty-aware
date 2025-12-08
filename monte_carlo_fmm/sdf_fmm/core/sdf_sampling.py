import numpy as np


def sample_sdf(mean_sdf: np.ndarray,
               std_sdf: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """
    Sample one SDF realization from a Gaussian distribution per voxel.
    """
    return rng.normal(loc=mean_sdf, scale=std_sdf)
