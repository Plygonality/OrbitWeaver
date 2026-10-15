"""Safeguarded elliptical Kepler solver and vectorised two-body states.

Kepler's equation is ``M = E - e sin E``. On each folded anomaly the root is
bracketed in ``[0, pi]`` and advanced with Halley's method. A trial step that
leaves the bracket, or that fails to reduce the residual, is replaced by
bisection, so the iteration remains convergent for eccentricities approaching 1.

There is no per-satellite Python loop in the state map. Satellite and time
axes are NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbitweaver.constants import KEPLER_ABS_TOL, KEPLER_FAIL_RESIDUAL, KEPLER_MAX_ITER
from orbitweaver.errors import OrbitWeaverError


@dataclass(frozen=True)
class KeplerSolution:
    """Eccentric anomaly on the branch ``(-pi, pi]`` matching the folded mean anomaly.

    ``residual`` is ``E - e sin(E) - M_folded`` in radians, where ``M_folded`` is
    the input mean anomaly reduced into ``(-pi, pi]``. That residual stays
    meaningful after many revolutions. ``sin`` and ``cos`` of this branch are
    the values used for Cartesian states.
    """

    eccentric_anomaly: np.ndarray
    residual: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray

    @property
    def max_abs_residual(self) -> float:
        if self.residual.size == 0:
            return 0.0
        return float(np.max(np.abs(self.residual)))

    @property
    def max_iterations(self) -> int:
        if self.iterations.size == 0:
            return 0
        return int(np.max(self.iterations))


@dataclass(frozen=True)
class CartesianState:
    """Inertial states in the OrbitWeaver frame, SI units, float64.

    ``position_m`` and ``velocity_m_s`` have shape ``(n_times, n_orbits, 3)``.
    """

    time_s: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    solution: KeplerSolution


def mean_motion(a_m: np.ndarray, mu_m3_s2: float) -> np.ndarray:
    """Mean motion ``sqrt(mu / a**3)`` in radians per second."""
    a = np.asarray(a_m, dtype=np.float64)
    _require_mu(mu_m3_s2)
    if np.any(~np.isfinite(a)) or np.any(a <= 0.0):
        raise OrbitWeaverError("Semi-major axis must be positive and finite.")
    return np.sqrt(mu_m3_s2 / a**3)


def orbital_period(a_m: np.ndarray, mu_m3_s2: float) -> np.ndarray:
    """Keplerian period ``2 pi sqrt(a**3 / mu)`` in seconds."""
    return 2.0 * np.pi / mean_motion(a_m, mu_m3_s2)


def solve_kepler(
    mean_anomaly: np.ndarray,
    eccentricity: np.ndarray,
    *,
    tol: float = KEPLER_ABS_TOL,
    max_iter: int = KEPLER_MAX_ITER,
) -> KeplerSolution:
    """Solve ``M = E - e sin E`` for array inputs.

    ``mean_anomaly`` and ``eccentricity`` broadcast together. Eccentricity must
    satisfy ``0 <= e < 1``.
    """
    mean = np.asarray(mean_anomaly, dtype=np.float64)
    ecc = np.asarray(eccentricity, dtype=np.float64)
    try:
        mean_b, ecc_b = np.broadcast_arrays(mean, ecc)
    except ValueError as exc:
        raise OrbitWeaverError(
            f"Mean anomaly and eccentricity cannot be broadcast together: {exc}"
        ) from exc
    if mean_b.size == 0:
        empty = np.zeros(mean_b.shape, dtype=np.float64)
        return KeplerSolution(
            empty,
            empty,
            np.zeros(mean_b.shape, dtype=np.int32),
            np.ones(mean_b.shape, dtype=bool),
        )
    if not np.all(np.isfinite(mean_b)):
        raise OrbitWeaverError("Mean anomaly contains a non-finite value.")
    if not np.all(np.isfinite(ecc_b)) or np.any(ecc_b < 0.0) or np.any(ecc_b >= 1.0):
        raise OrbitWeaverError(
            "Eccentricity must be finite and satisfy 0 <= e < 1. "
            "Hyperbolic trajectories are outside the v0.1 Kepler model."
        )
    shape = mean_b.shape
    anomaly, residual, iterations, converged = _solve_flat(
        np.ascontiguousarray(mean_b, dtype=np.float64).ravel(),
        np.ascontiguousarray(ecc_b, dtype=np.float64).ravel(),
        tol=tol,
        max_iter=max_iter,
    )
    return KeplerSolution(
        anomaly.reshape(shape),
        residual.reshape(shape),
        iterations.reshape(shape),
        converged.reshape(shape),
    )


def propagate_elements(
    a_m: np.ndarray,
    eccentricity: np.ndarray,
    inclination_rad: np.ndarray,
    raan_rad: np.ndarray,
    argp_rad: np.ndarray,
    mean_anomaly0_rad: np.ndarray,
    mu_m3_s2: float,
    times_s: np.ndarray,
    *,
    orbit_ids: list[str] | tuple[str, ...] | None = None,
    raise_on_failure: bool = True,
) -> CartesianState:
    """Inertial position and velocity of many ellipses at many times.

    Element arrays share one satellite axis. ``times_s`` is one time axis of
    SI seconds past the epoch. Output shape is ``(n_times, n_orbits, 3)``.
    """
    a = np.asarray(a_m, dtype=np.float64).reshape(-1)
    ecc = np.asarray(eccentricity, dtype=np.float64).reshape(-1)
    inc = np.asarray(inclination_rad, dtype=np.float64).reshape(-1)
    raan = np.asarray(raan_rad, dtype=np.float64).reshape(-1)
    argp = np.asarray(argp_rad, dtype=np.float64).reshape(-1)
    mean0 = np.asarray(mean_anomaly0_rad, dtype=np.float64).reshape(-1)
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    n_orbits = a.shape[0]
    for name, array in (
        ("e", ecc),
        ("inclination", inc),
        ("raan", raan),
        ("argp", argp),
        ("mean_anomaly", mean0),
    ):
        if array.shape != (n_orbits,):
            raise OrbitWeaverError(f"{name} must have the same length as semi-major axis.")
    if np.any(~np.isfinite(times)):
        raise OrbitWeaverError("A requested propagation time is not finite.")
    motion = mean_motion(a, mu_m3_s2)
    mean = mean0[None, :] + times[:, None] * motion[None, :]
    solution = solve_kepler(mean, ecc[None, :])
    if raise_on_failure and not np.all(solution.converged):
        failed = np.where(np.any(~solution.converged, axis=0))[0]
        ids = _ids_for(failed, orbit_ids)
        raise OrbitWeaverError(
            "Kepler solver residual stayed above "
            f"{KEPLER_FAIL_RESIDUAL:.1e} rad "
            f"(max |residual| = {solution.max_abs_residual:.3e} rad).",
            ids,
        )
    position, velocity = state_from_eccentric_anomaly(
        solution.eccentric_anomaly, a, ecc, inc, raan, argp, mu_m3_s2
    )
    return CartesianState(times, position, velocity, solution)


def state_from_eccentric_anomaly(
    eccentric_anomaly: np.ndarray,
    a_m: np.ndarray,
    eccentricity: np.ndarray,
    inclination_rad: np.ndarray,
    raan_rad: np.ndarray,
    argp_rad: np.ndarray,
    mu_m3_s2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map eccentric anomaly ``(..., N)`` and elements ``(N,)`` to Cartesian states."""
    anomaly = np.asarray(eccentric_anomaly, dtype=np.float64)
    a = np.asarray(a_m, dtype=np.float64)
    ecc = np.asarray(eccentricity, dtype=np.float64)
    cos_e = np.cos(anomaly)
    sin_e = np.sin(anomaly)
    a_b = a[None, :] if anomaly.ndim == 2 else a
    e_b = ecc[None, :] if anomaly.ndim == 2 else ecc
    radius = a_b * (1.0 - e_b * cos_e)
    one_minus_e2 = np.sqrt(np.maximum(0.0, 1.0 - ecc * ecc))
    one_minus_e2_b = one_minus_e2[None, :] if anomaly.ndim == 2 else one_minus_e2
    x_p = a_b * (cos_e - e_b)
    y_p = a_b * one_minus_e2_b * sin_e
    factor = np.sqrt(mu_m3_s2 * a_b) / radius
    vx_p = -factor * sin_e
    vy_p = factor * one_minus_e2_b * cos_e
    px, py, pz, qx, qy, qz = perifocal_axes(inclination_rad, raan_rad, argp_rad)
    if anomaly.ndim == 2:
        px, py, pz, qx, qy, qz = (component[None, :] for component in (px, py, pz, qx, qy, qz))
    position = np.stack(
        (x_p * px + y_p * qx, x_p * py + y_p * qy, x_p * pz + y_p * qz),
        axis=-1,
    )
    velocity = np.stack(
        (vx_p * px + vy_p * qx, vx_p * py + vy_p * qy, vx_p * pz + vy_p * qz),
        axis=-1,
    )
    return position, velocity


def perifocal_axes(
    inclination_rad: np.ndarray,
    raan_rad: np.ndarray,
    argp_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perifocal P and Q axes in the inertial frame.

    P points at periapsis. Q is the in-plane axis 90 degrees ahead in the
    direction of motion. W = P × Q is the angular-momentum direction and is
    not needed for the position map.
    """
    inc = np.asarray(inclination_rad, dtype=np.float64)
    raan = np.asarray(raan_rad, dtype=np.float64)
    argp = np.asarray(argp_rad, dtype=np.float64)
    cos_o = np.cos(raan)
    sin_o = np.sin(raan)
    cos_w = np.cos(argp)
    sin_w = np.sin(argp)
    cos_i = np.cos(inc)
    sin_i = np.sin(inc)
    px = cos_o * cos_w - sin_o * sin_w * cos_i
    py = sin_o * cos_w + cos_o * sin_w * cos_i
    pz = sin_w * sin_i
    qx = -cos_o * sin_w - sin_o * cos_w * cos_i
    qy = -sin_o * sin_w + cos_o * cos_w * cos_i
    qz = cos_w * sin_i
    return px, py, pz, qx, qy, qz


def _solve_flat(
    mean_anomaly: np.ndarray,
    eccentricity: np.ndarray,
    *,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    two_pi = 2.0 * np.pi
    revolutions = np.round(mean_anomaly / two_pi)
    folded = mean_anomaly - revolutions * two_pi
    folded = np.where(folded > np.pi, folded - two_pi, folded)
    folded = np.where(folded <= -np.pi, folded + two_pi, folded)

    sign = np.where(folded < 0.0, -1.0, 1.0)
    magnitude = np.abs(folded)
    anomaly = magnitude.copy()
    iterations = np.zeros(magnitude.shape, dtype=np.int32)
    active = (eccentricity > 0.0) & (magnitude > 0.0)

    starter = magnitude + eccentricity * np.sin(magnitude) * (1.0 + eccentricity * np.cos(magnitude))
    # A cubic estimate covers the flat root near e -> 1, M -> 0.
    cubic = np.cbrt(np.maximum(6.0 * magnitude, 0.0))
    high = eccentricity > 0.8
    starter = np.where(high, np.maximum(starter, cubic), starter)
    anomaly = np.where(active, np.clip(starter, 1e-18, np.pi - 1e-18), anomaly)

    lower = np.zeros_like(magnitude)
    upper = np.full_like(magnitude, np.pi)

    for _ in range(max_iter):
        if not np.any(active):
            break
        sine = np.sin(anomaly)
        cosine = np.cos(anomaly)
        residual_now = anomaly - eccentricity * sine - magnitude
        first = 1.0 - eccentricity * cosine
        second = eccentricity * sine
        denominator = first * first - 0.5 * residual_now * second
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(np.abs(denominator) > 1e-30, -(residual_now * first) / denominator, np.nan)
        trial = anomaly + step
        width = upper - lower
        inside = (
            np.isfinite(trial)
            & (trial > lower)
            & (trial < upper)
            & (np.abs(step) <= 0.75 * np.maximum(width, 0.0))
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            trial_residual = trial - eccentricity * np.sin(trial) - magnitude
        improved = np.isfinite(trial_residual) & (
            np.abs(trial_residual) <= 0.5 * np.maximum(np.abs(residual_now), tol)
        )
        use_halley = active & inside & improved
        bisect = 0.5 * (lower + upper)
        updated = np.where(use_halley, trial, np.where(active, bisect, anomaly))
        updated_residual = updated - eccentricity * np.sin(updated) - magnitude
        lower = np.where(active & (updated_residual < 0.0), updated, lower)
        upper = np.where(active & (updated_residual >= 0.0), updated, upper)
        anomaly = updated
        iterations = np.where(active, iterations + 1, iterations)
        finished = active & (np.abs(updated_residual) <= tol)
        active = active & ~finished

    signed = sign * anomaly
    # Residual against the folded mean anomaly, which is the equation we solved.
    residual = signed - eccentricity * np.sin(signed) - folded
    converged = np.abs(residual) <= KEPLER_FAIL_RESIDUAL
    return signed, residual, iterations, converged


def propagate_array(array, times_s: np.ndarray, *, batch_orbits: int = 8192) -> CartesianState:
    """Propagate an :class:`~orbitweaver.models.OrbitArray` in bounded orbit batches."""
    orbits = list(array.orbits)
    if not orbits:
        raise OrbitWeaverError("There are no orbits to propagate.")
    if batch_orbits < 1:
        raise OrbitWeaverError("batch_orbits must be >= 1.")
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise OrbitWeaverError("At least one propagation time is required.")
    pieces: list[CartesianState] = []
    for start in range(0, len(orbits), batch_orbits):
        chunk = orbits[start : start + batch_orbits]
        pieces.append(
            propagate_elements(
                [orbit.a_m for orbit in chunk],
                [orbit.e for orbit in chunk],
                [orbit.inclination_rad for orbit in chunk],
                [orbit.raan_rad for orbit in chunk],
                [orbit.argp_rad for orbit in chunk],
                [orbit.mean_anomaly_rad for orbit in chunk],
                array.body.mu_m3_s2,
                times,
                orbit_ids=[orbit.satellite_id for orbit in chunk],
            )
        )
    if len(pieces) == 1:
        return pieces[0]
    return CartesianState(
        time_s=times,
        position_m=np.concatenate([piece.position_m for piece in pieces], axis=1),
        velocity_m_s=np.concatenate([piece.velocity_m_s for piece in pieces], axis=1),
        solution=KeplerSolution(
            eccentric_anomaly=np.concatenate(
                [piece.solution.eccentric_anomaly for piece in pieces], axis=1
            ),
            residual=np.concatenate([piece.solution.residual for piece in pieces], axis=1),
            iterations=np.concatenate([piece.solution.iterations for piece in pieces], axis=1),
            converged=np.concatenate([piece.solution.converged for piece in pieces], axis=1),
        ),
    )


def _require_mu(mu_m3_s2: float) -> None:
    if not np.isfinite(mu_m3_s2) or mu_m3_s2 <= 0.0:
        raise OrbitWeaverError(
            f"Gravitational parameter mu must be positive and finite, got {mu_m3_s2}."
        )


def _ids_for(indexes: np.ndarray, orbit_ids: list[str] | tuple[str, ...] | None) -> list[str]:
    if orbit_ids is None:
        return [str(int(index)) for index in indexes]
    return [str(orbit_ids[int(index)]) for index in indexes]
