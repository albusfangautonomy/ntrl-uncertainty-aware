import numpy as np


def sdf_to_speed(sdf: np.ndarray,
                 base_speed: float = 1.0,
                 d_safe: float = 1.0,
                 v_min: float = 1e-3) -> np.ndarray:
    """
    Example mapping from SDF d(x) -> speed S*(x), dimension-agnostic.

    - Inside / near obstacle (d <= 0): very small speed.
    - 0 < d < d_safe : smooth increase from v_min to base_speed.
    - d >= d_safe : base_speed.

    Replace with your own S*(d) when ready.
    """
    speed = np.empty_like(sdf, dtype=np.float64)

    inside = sdf <= 0.0
    band = (sdf > 0.0) & (sdf < d_safe)
    outside = sdf >= d_safe

    speed[inside] = v_min

    if np.any(band):
        t = np.clip(sdf[band] / d_safe, 0.0, 1.0)
        speed[band] = v_min + (base_speed - v_min) * t**2

    speed[outside] = base_speed

    return speed
