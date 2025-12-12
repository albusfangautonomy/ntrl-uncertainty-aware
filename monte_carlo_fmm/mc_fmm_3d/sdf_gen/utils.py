import numpy as np


def combine_sdfs(*sdfs: np.ndarray) -> np.ndarray:
    """
    Union of multiple SDFs: d_union(x) = min_i d_i(x)
    """
    if not sdfs:
        raise ValueError("At least one SDF required.")
    out = sdfs[0].copy()
    for sdf in sdfs[1:]:
        out = np.minimum(out, sdf)
    return out
