import numpy as np


def sdf_to_speed_smooth(sdf: np.ndarray,
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

def sdf_to_speed(
    sdf: np.ndarray,
    s_const: float = 1.0,
    d_min: float = 0.005,
    d_max: float = 0.8,
) -> np.ndarray:
    """
    Compute S*(q) from SDF using the original linear clipped formula:

        S*(q) = (s_const / d_max) * clip(d(q, X_obs), d_min, d_max)

    where:
      - d(q, X_obs) is the distance to the obstacle set.
      - We assume SDF is negative inside obstacles and positive outside.
        Then we take:
            d(q, X_obs) = max(SDF(q), 0).

    Args:
        sdf: Signed distance field, negative inside obstacles.
        s_const: Base speed scaling constant (s_const > 0).
        d_min: Minimum distance for clipping.
        d_max: Maximum distance for clipping.

    Returns:
        speed: S*(q) with same shape as sdf.
    """

    # Distance to obstacles: 0 inside, positive outside
    dist = np.maximum(sdf, 0.0)

    # Clip distance into [d_min, d_max]
    dist_clipped = np.clip(dist, d_min, d_max)

    # Apply S*(q) formula
    speed = (s_const / d_max) * dist_clipped

    return speed

import numpy as np


def sdf_to_speed_uncertainty_aware(
    sdf_mean: np.ndarray,
    sdf_std: np.ndarray,
    s_const: float = 1.0,
    d_min: float = 0.1,
    d_max: float = 1.0,
    lam: float = 2.0,
    gamma: float = 2.0
) -> np.ndarray:
    """
    Uncertainty-aware S*(q) mapping.

    Implements:
        d_u(q) = max(sdf_mean - lam * sdf_std, 0)
        d_c(q) = clip(d_u, d_min, d_max)
        r(q)   = (d_c - d_min) / (d_max - d_min)
        r_g    = r^gamma
        S*(q)  = s_const * r_g

    This:
        - Pushes obstacles outward based on uncertainty.
        - Treats variance as safety inflation.
        - Applies nonlinear smoothing for safe speed modulation.
    """

    # 1. Conservative distance (std inflation)
    d_u = np.maximum(sdf_mean - lam * sdf_std, 0.0)

    # 2. Clip to [d_min, d_max]
    d_c = np.clip(d_u, d_min, d_max)

    # 3. Normalize to [0,1]
    r = (d_c - d_min) / (d_max - d_min)
    r = np.clip(r, 0.0, 1.0)

    # 4. Apply smoothing exponent gamma ≥ 1
    r_g = r ** gamma

    # 5. Scale final speed
    speed = s_const * r_g

    return speed

def sdf_to_speed_long_tail(sdf,
                           base_speed=1.0,
                           d_safe=0.5,
                           v_min=1e-3,
                           L_transition=0.5,
                           L_tail=1.0,
                           k_tail=0.1):
    """
    Long-tail speed model that preserves obstacle slowdown but introduces
    small global variation so Monte Carlo uncertainties propagate across
    the entire domain.

    Args:
        sdf : signed distance field
        base_speed : max speed
        d_safe : transition region for obstacle margin
        v_min : minimum speed inside obstacles
        L_transition : length scale for near-obstacle smoothing
        L_tail : global long-tail decay length scale
        k_tail : amplitude of long-tail modulation

    Returns:
        speed field S*(q) with global sensitivity to sdf
    """
    speed = np.empty_like(sdf, dtype=np.float64)

    # ------------------------------
    # 1. Inside obstacle
    # ------------------------------
    inside = sdf <= 0.0
    speed[inside] = v_min

    # ------------------------------
    # 2. Near-obstacle transition region
    #    Smooth ramp to base speed
    # ------------------------------
    band = (sdf > 0.0) & (sdf < d_safe)
    t = sdf[band] / d_safe
    speed[band] = v_min + (base_speed - v_min) * (1 - np.exp(-(t / L_transition)**2))

    # ------------------------------
    # 3. Far region with long-tail modulation
    # ------------------------------
    far = sdf >= d_safe

    # Long-tail Gaussian dip:
    # base_speed - k * exp(-(d/L_tail)^2)
    # gives global sensitivity without violating physical structure
    tail = k_tail * np.exp(-(sdf[far] / L_tail)**2)
    speed[far] = base_speed - tail

    return speed