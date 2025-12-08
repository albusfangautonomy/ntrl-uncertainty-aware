import numpy as np

def sdf_to_speed_3d(sdf, base_speed=1.0, d_safe=1.0, v_min=1e-3):
    speed = np.empty_like(sdf)

    inside = sdf <= 0
    speed[inside] = v_min

    band = (sdf > 0) & (sdf < d_safe)
    t = np.clip(sdf[band] / d_safe, 0.0, 1.0)
    speed[band] = v_min + (base_speed - v_min) * t**2

    speed[sdf >= d_safe] = base_speed
    return speed
