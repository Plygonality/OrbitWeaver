"""Scalar elliptical Kepler map for the Blender preview.

This is a display copy of the v0.1 state map. Scientific products come from
``orbitweaver.kepler``. Tests compare the two on representative ellipses.
No Blender import lives in this module.
"""

from __future__ import annotations

import math

_FAIL = 1e-10
_TOL = 1e-14
_MAX_ITER = 80


def eccentric_anomaly(mean_anomaly: float, eccentricity: float) -> tuple[float, float, int]:
    """Return folded eccentric anomaly, residual, and iteration count."""
    if eccentricity < 0.0 or eccentricity >= 1.0:
        raise ValueError("preview solver requires 0 <= e < 1")
    two_pi = 2.0 * math.pi
    revolutions = round(mean_anomaly / two_pi)
    folded = mean_anomaly - revolutions * two_pi
    if folded > math.pi:
        folded -= two_pi
    elif folded <= -math.pi:
        folded += two_pi
    sign = -1.0 if folded < 0.0 else 1.0
    magnitude = abs(folded)
    if eccentricity == 0.0 or magnitude == 0.0:
        anomaly = magnitude
        iterations = 0
    else:
        starter = magnitude + eccentricity * math.sin(magnitude) * (1.0 + eccentricity * math.cos(magnitude))
        if eccentricity > 0.8:
            starter = max(starter, (6.0 * magnitude) ** (1.0 / 3.0))
        anomaly = min(math.pi - 1e-18, max(1e-18, starter))
        lower = 0.0
        upper = math.pi
        iterations = 0
        for _ in range(_MAX_ITER):
            sine = math.sin(anomaly)
            cosine = math.cos(anomaly)
            residual_now = anomaly - eccentricity * sine - magnitude
            if abs(residual_now) <= _TOL:
                break
            first = 1.0 - eccentricity * cosine
            second = eccentricity * sine
            denominator = first * first - 0.5 * residual_now * second
            step = -(residual_now * first) / denominator if abs(denominator) > 1e-30 else math.nan
            trial = anomaly + step
            width = upper - lower
            inside = (
                math.isfinite(trial)
                and lower < trial < upper
                and abs(step) <= 0.75 * max(width, 0.0)
            )
            if inside:
                trial_residual = trial - eccentricity * math.sin(trial) - magnitude
                improved = math.isfinite(trial_residual) and abs(trial_residual) <= 0.5 * max(abs(residual_now), _TOL)
            else:
                improved = False
            anomaly = trial if inside and improved else 0.5 * (lower + upper)
            updated = anomaly - eccentricity * math.sin(anomaly) - magnitude
            if updated < 0.0:
                lower = anomaly
            else:
                upper = anomaly
            iterations += 1
            if abs(updated) <= _TOL:
                break
    signed = sign * anomaly
    residual = signed - eccentricity * math.sin(signed) - folded
    if abs(residual) > _FAIL:
        raise ValueError(f"preview Kepler residual {residual}")
    return signed, residual, iterations


def inertial_state(
    a_m: float,
    eccentricity: float,
    inclination_rad: float,
    raan_rad: float,
    argp_rad: float,
    mean_anomaly_rad: float,
    mu_m3_s2: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Inertial position and velocity in metres and metres per second."""
    anomaly, _residual, _iterations = eccentric_anomaly(mean_anomaly_rad, eccentricity)
    cosine = math.cos(anomaly)
    sine = math.sin(anomaly)
    radius = a_m * (1.0 - eccentricity * cosine)
    factor = math.sqrt(max(0.0, 1.0 - eccentricity * eccentricity))
    x_p = a_m * (cosine - eccentricity)
    y_p = a_m * factor * sine
    speed_factor = math.sqrt(mu_m3_s2 * a_m) / radius
    vx_p = -speed_factor * sine
    vy_p = speed_factor * factor * cosine
    cos_o, sin_o = math.cos(raan_rad), math.sin(raan_rad)
    cos_w, sin_w = math.cos(argp_rad), math.sin(argp_rad)
    cos_i, sin_i = math.cos(inclination_rad), math.sin(inclination_rad)
    px = cos_o * cos_w - sin_o * sin_w * cos_i
    py = sin_o * cos_w + cos_o * sin_w * cos_i
    pz = sin_w * sin_i
    qx = -cos_o * sin_w - sin_o * cos_w * cos_i
    qy = -sin_o * sin_w + cos_o * cos_w * cos_i
    qz = cos_w * sin_i
    position = (x_p * px + y_p * qx, x_p * py + y_p * qy, x_p * pz + y_p * qz)
    velocity = (vx_p * px + vy_p * qx, vx_p * py + vy_p * qy, vx_p * pz + vy_p * qz)
    return position, velocity
