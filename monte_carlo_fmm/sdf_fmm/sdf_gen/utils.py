import numpy as np


def combine_sdfs(*sdfs):
    """
    Base SDF combination rule: union = min(d1, d2, ...)
    """
    out = sdfs[0].copy()
    for sdf in sdfs[1:]:
        out = np.minimum(out, sdf)
    return out
