"""Classical element conversions and scalar orbital integrals.

The forward map is ``kepler.state_from_eccentric_anomaly``. This module inverts
that map and evaluates energy, angular momentum, and apsides. Singular angles
follow the conventions in ``docs/physics.md``.
"""

from __future__ import annotations

import numpy as np

from orbitweaver.constants import INVERSE_CIRCULAR_ECCENTRICITY, INVERSE_NODE_RATIO
from orbitweaver.errors import OrbitWeaverError
from orbitweaver.kepler import perifocal_axes


def periapsis_radius(a_m: np.ndarray, eccentricity: np.ndarray) -> np.ndarray:
    return np.asarray(a_m, dtype=np.float64) * (1.0 - np.asarray(eccentricity, dtype=np.float64))


def apoapsis_radius(a_m: np.ndarray, eccentricity: np.ndarray) -> np.ndarray:
    return np.asarray(a_m, dtype=np.float64) * (1.0 + np.asarray(eccentricity, dtype=np.float64))


def analytic_specific_energy(a_m: np.ndarray, mu_m3_s2: float) -> np.ndarray:
    """``-mu / (2a)``."""
    return -mu_m3_s2 / (2.0 * np.asarray(a_m, dtype=np.float64))


def analytic_h_magnitude(a_m: np.ndarray, eccentricity: np.ndarray, mu_m3_s2: float) -> np.ndarray:
    """``sqrt(mu * a * (1 - e**2))``."""
    a = np.asarray(a_m, dtype=np.float64)
    ecc = np.asarray(eccentricity, dtype=np.float64)
    return np.sqrt(mu_m3_s2 * a * (1.0 - ecc * ecc))


def periapsis_speed(a_m: np.ndarray, eccentricity: np.ndarray, mu_m3_s2: float) -> np.ndarray:
    """Vis-viva speed at pericentre, an upper bound on orbital speed."""
    a = np.asarray(a_m, dtype=np.float64)
    ecc = np.asarray(eccentricity, dtype=np.float64)
    return np.sqrt(mu_m3_s2 * (1.0 + ecc) / (a * (1.0 - ecc)))


def specific_energy(position_m: np.ndarray, velocity_m_s: np.ndarray, mu_m3_s2: float) -> np.ndarray:
    radius = np.linalg.norm(position_m, axis=-1)
    speed2 = np.sum(np.asarray(velocity_m_s, dtype=np.float64) ** 2, axis=-1)
    return 0.5 * speed2 - mu_m3_s2 / radius


def specific_angular_momentum(position_m: np.ndarray, velocity_m_s: np.ndarray) -> np.ndarray:
    return np.cross(position_m, velocity_m_s)


def cartesian_to_elements(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    mu_m3_s2: float,
) -> dict[str, np.ndarray]:
    """Invert a bound Cartesian state to classical elements.

    Accepts shape ``(3,)`` or ``(N, 3)``. Angles are wrapped to ``[0, 2 pi)``.
    On circular or equatorial singularities the documented convention replaces
    the undefined angles; the returned ``convention`` array names that choice.
    """
    position = np.asarray(position_m, dtype=np.float64)
    velocity = np.asarray(velocity_m_s, dtype=np.float64)
    single = position.ndim == 1
    if single:
        position = position[None, :]
        velocity = velocity[None, :]
    if position.ndim != 2 or position.shape[-1] != 3 or velocity.shape != position.shape:
        raise OrbitWeaverError("position and velocity must have shape (3,) or (N, 3).")
    if not np.isfinite(mu_m3_s2) or mu_m3_s2 <= 0.0:
        raise OrbitWeaverError("Gravitational parameter mu must be positive and finite.")

    radius = np.linalg.norm(position, axis=-1)
    speed = np.linalg.norm(velocity, axis=-1)
    if np.any(~np.isfinite(position)) or np.any(~np.isfinite(velocity)):
        raise OrbitWeaverError("Cartesian state contains a non-finite value.")
    if np.any(radius == 0.0):
        raise OrbitWeaverError("A state at the central point mass has no Keplerian ellipse.")
    angular = np.cross(position, velocity)
    angular_norm = np.linalg.norm(angular, axis=-1)
    if np.any(angular_norm == 0.0):
        raise OrbitWeaverError("Zero angular momentum is outside the v0.1 ellipse converter.")

    radial_velocity = np.sum(position * velocity, axis=-1)
    ecc_vector = (
        (speed**2 - mu_m3_s2 / radius)[..., None] * position - radial_velocity[..., None] * velocity
    ) / mu_m3_s2
    eccentricity = np.linalg.norm(ecc_vector, axis=-1)
    energy = 0.5 * speed**2 - mu_m3_s2 / radius
    if np.any(energy >= 0.0):
        raise OrbitWeaverError(
            "Specific energy is non-negative, so the state is not a bound ellipse. "
            "Hyperbolic trajectories are outside v0.1."
        )
    semi_major = -mu_m3_s2 / (2.0 * energy)
    inclination = np.arccos(np.clip(angular[..., 2] / angular_norm, -1.0, 1.0))

    reference_z = np.zeros_like(position)
    reference_z[..., 2] = 1.0
    node = np.cross(reference_z, angular)
    node_norm = np.linalg.norm(node, axis=-1)
    angular_hat = angular / angular_norm[..., None]

    equatorial = node_norm <= (INVERSE_NODE_RATIO * angular_norm)
    circular = eccentricity <= INVERSE_CIRCULAR_ECCENTRICITY
    prograde = angular[..., 2] >= 0.0

    raan = np.where(
        equatorial,
        0.0,
        np.arctan2(node[..., 1], node[..., 0]),
    )
    node_hat = np.zeros_like(node)
    has_node = node_norm > 0.0
    node_hat[has_node] = node[has_node] / node_norm[has_node, None]
    ecc_hat = np.zeros_like(ecc_vector)
    has_ecc = eccentricity > 0.0
    ecc_hat[has_ecc] = ecc_vector[has_ecc] / eccentricity[has_ecc, None]
    radius_hat = position / radius[..., None]

    argp = np.zeros_like(eccentricity)
    true_anomaly = np.zeros_like(eccentricity)

    classical = ~equatorial & ~circular
    argp = np.where(
        classical,
        np.arctan2(
            np.sum(angular_hat * np.cross(node_hat, ecc_hat), axis=-1),
            np.sum(node_hat * ecc_hat, axis=-1),
        ),
        argp,
    )
    elliptical_equatorial = equatorial & ~circular
    argp = np.where(
        elliptical_equatorial & prograde,
        np.arctan2(ecc_vector[..., 1], ecc_vector[..., 0]),
        argp,
    )
    argp = np.where(
        elliptical_equatorial & ~prograde,
        np.arctan2(-ecc_vector[..., 1], ecc_vector[..., 0]),
        argp,
    )

    true_anomaly = np.where(
        ~circular,
        np.arctan2(
            np.sum(angular_hat * np.cross(ecc_hat, radius_hat), axis=-1),
            np.sum(ecc_hat * radius_hat, axis=-1),
        ),
        true_anomaly,
    )
    circular_inclined = circular & ~equatorial
    true_anomaly = np.where(
        circular_inclined,
        np.arctan2(
            np.sum(angular_hat * np.cross(node_hat, radius_hat), axis=-1),
            np.sum(node_hat * radius_hat, axis=-1),
        ),
        true_anomaly,
    )
    circular_equatorial = circular & equatorial
    true_anomaly = np.where(
        circular_equatorial & prograde,
        np.arctan2(position[..., 1], position[..., 0]),
        true_anomaly,
    )
    true_anomaly = np.where(
        circular_equatorial & ~prograde,
        np.arctan2(-position[..., 1], position[..., 0]),
        true_anomaly,
    )

    cosine_nu = np.cos(true_anomaly)
    sine_nu = np.sin(true_anomaly)
    denominator = 1.0 + eccentricity * cosine_nu
    cosine_e = (eccentricity + cosine_nu) / denominator
    sine_e = np.sqrt(np.maximum(0.0, 1.0 - eccentricity**2)) * sine_nu / denominator
    eccentric_anomaly = np.where(circular, true_anomaly, np.arctan2(sine_e, cosine_e))
    mean_anomaly = np.where(
        circular,
        true_anomaly,
        eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly),
    )

    convention = np.full(eccentricity.shape, "classical", dtype=object)
    convention = np.where(circular_equatorial, "circular_equatorial", convention)
    convention = np.where(circular_inclined, "circular_inclined", convention)
    convention = np.where(elliptical_equatorial, "elliptical_equatorial", convention)

    result = {
        "a_m": semi_major,
        "e": np.where(circular, 0.0, eccentricity),
        "inclination_rad": inclination,
        "raan_rad": _wrap(raan),
        "argp_rad": _wrap(np.where(circular, 0.0, argp)),
        "mean_anomaly_rad": _wrap(mean_anomaly),
        "convention": convention,
    }
    if single:
        return {key: value[0] for key, value in result.items()}
    return result


def angular_momentum_direction(
    inclination_rad: np.ndarray,
    raan_rad: np.ndarray,
) -> np.ndarray:
    """Unit angular-momentum direction W for the stored elements. Shape ``(N, 3)``."""
    inc = np.asarray(inclination_rad, dtype=np.float64)
    raan = np.asarray(raan_rad, dtype=np.float64)
    return np.stack(
        (np.sin(raan) * np.sin(inc), -np.cos(raan) * np.sin(inc), np.cos(inc)),
        axis=-1,
    )


def perifocal_basis_matrix(
    inclination_rad: np.ndarray,
    raan_rad: np.ndarray,
    argp_rad: np.ndarray,
) -> np.ndarray:
    """Rows are not used; returns ``(N, 3, 3)`` with columns P, Q, W."""
    px, py, pz, qx, qy, qz = perifocal_axes(inclination_rad, raan_rad, argp_rad)
    direction = angular_momentum_direction(inclination_rad, raan_rad)
    basis = np.stack(
        (
            np.stack((px, qx, direction[..., 0]), axis=-1),
            np.stack((py, qy, direction[..., 1]), axis=-1),
            np.stack((pz, qz, direction[..., 2]), axis=-1),
        ),
        axis=-2,
    )
    return basis


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.mod(angle, 2.0 * np.pi)
