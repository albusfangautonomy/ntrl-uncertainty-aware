import numpy as np


def sdf_to_speed(sdf: np.ndarray,
                 base_speed: float = 1.0,
                 d_safe: float = 1.0,
                 v_min: float = 1e-3) -> np.ndarray:
    """
    Placeholder mapping SDF d(q) -> S*(q).
    Replace with your real S* model.

    Speed is low inside obstacles and increases smoothly with distance.
    """
    speed = np.empty_like(sdf, dtype=np.float64)

    # Inside or touching obstacle
    inside = sdf <= 0.0
    speed[inside] = v_min

    # Transition band
    band = (sdf > 0.0) & (sdf < d_safe)
    t = np.clip(sdf[band] / d_safe, 0.0, 1.0)
    speed[band] = v_min + (base_speed - v_min) * t**2

    # Far from obstacles
    speed[sdf >= d_safe] = base_speed

    return speed
