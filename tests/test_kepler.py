"""Kepler residual, closure, energy, and angular-momentum checks."""

from __future__ import annotations

import numpy as np

from orbitweaver.kepler import orbital_period, propagate_elements, solve_kepler
from orbitweaver.orbital_elements import (
    analytic_h_magnitude,
    analytic_specific_energy,
    cartesian_to_elements,
    specific_angular_momentum,
    specific_energy,
)


def test_kepler_residual_grid_including_high_eccentricity():
    eccentricity = np.array(
        [0.0, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0 - 1.0e-6, 1.0 - 1.0e-8]
    )
    mean_anomaly = np.linspace(-np.pi, np.pi, 361)
    solution = solve_kepler(mean_anomaly[:, None], eccentricity[None, :])
    residual = np.abs(solution.residual)
    moderate = eccentricity <= (1.0 - 1.0e-6)
    assert np.max(residual[:, moderate]) < 1e-12
    assert np.max(residual[:, ~moderate]) < 1e-8
    assert solution.max_abs_residual == np.max(residual)
    assert np.all(solution.converged)
    assert solution.iterations.shape == residual.shape


def test_circular_orbit_matches_analytic_and_closes():
    mu = 3.986004e14
    a = 6_778_137.0
    period = float(orbital_period(a, mu))
    expected = 2.0 * np.pi * np.sqrt(a**3 / mu)
    assert period == expected
    times = np.array([0.0, period / 4.0, period / 2.0, period])
    state = propagate_elements(
        [a],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        mu,
        times,
        orbit_ids=["REF-CIRC-1"],
    )
    speed = np.sqrt(mu / a)
    np.testing.assert_allclose(state.position_m[0, 0], [a, 0.0, 0.0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(state.velocity_m_s[0, 0], [0.0, speed, 0.0], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(state.position_m[1, 0], [0.0, a, 0.0], rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(state.velocity_m_s[1, 0], [-speed, 0.0, 0.0], rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(state.position_m[-1, 0], state.position_m[0, 0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(state.velocity_m_s[-1, 0], state.velocity_m_s[0, 0], rtol=0.0, atol=1e-9)
    assert state.position_m.dtype == np.float64
    assert state.position_m.shape == (4, 1, 3)


def test_energy_and_angular_momentum_are_conserved():
    mu = 1.3271244e20
    a = np.array([1.496e11, 1.2e11, 2.0e11])
    ecc = np.array([0.0, 0.3, 0.8])
    inc = np.array([0.2, 0.7, 1.2])
    raan = np.array([0.1, 1.0, 2.0])
    argp = np.array([0.3, 0.4, 0.5])
    mean0 = np.array([0.0, 1.0, 2.0])
    periods = orbital_period(a, mu)
    times = np.linspace(0.0, float(np.max(periods)), 9)
    state = propagate_elements(a, ecc, inc, raan, argp, mean0, mu, times)
    energy = specific_energy(state.position_m, state.velocity_m_s, mu)
    expected_energy = analytic_specific_energy(a, mu)
    np.testing.assert_allclose(energy, np.broadcast_to(expected_energy, energy.shape), rtol=1e-10, atol=0.0)
    angular = specific_angular_momentum(state.position_m, state.velocity_m_s)
    magnitude = np.linalg.norm(angular, axis=-1)
    expected_h = analytic_h_magnitude(a, ecc, mu)
    np.testing.assert_allclose(magnitude, np.broadcast_to(expected_h, magnitude.shape), rtol=1e-10, atol=0.0)
    # Direction is constant along each trajectory.
    reference = angular[0]
    cosine = np.sum(angular * reference[None, :, :], axis=-1) / (
        magnitude * np.linalg.norm(reference, axis=-1)[None, :]
    )
    np.testing.assert_allclose(cosine, 1.0, rtol=0.0, atol=1e-10)


def test_high_eccentricity_state_roundtrip_and_period_closure():
    mu = 3.986004e14
    a = 2.0e7
    ecc = 0.99
    elements = dict(
        a_m=np.array([a]),
        eccentricity=np.array([ecc]),
        inclination_rad=np.array([0.4]),
        raan_rad=np.array([0.7]),
        argp_rad=np.array([1.1]),
        mean_anomaly0_rad=np.array([0.2]),
    )
    period = float(orbital_period(a, mu))
    state = propagate_elements(
        elements["a_m"],
        elements["eccentricity"],
        elements["inclination_rad"],
        elements["raan_rad"],
        elements["argp_rad"],
        elements["mean_anomaly0_rad"],
        mu,
        np.array([0.0, period]),
    )
    np.testing.assert_allclose(state.position_m[1, 0], state.position_m[0, 0], rtol=0.0, atol=1e-3)
    recovered = cartesian_to_elements(state.position_m[0, 0], state.velocity_m_s[0, 0], mu)
    assert recovered["convention"] == "classical"
    assert recovered["a_m"] == np.array([a])[0] or abs(recovered["a_m"] - a) / a < 1e-10
    assert abs(recovered["e"] - ecc) < 1e-10


def test_many_revolutions_do_not_inflate_folded_residual():
    mu = 3.986004e14
    a = 7.0e6
    period = float(orbital_period(a, mu))
    times = np.array([0.0, 10_000.0 * period])
    state = propagate_elements([a], [0.2], [0.5], [0.3], [0.4], [0.1], mu, times)
    assert state.solution.max_abs_residual < 1e-12
    radius = np.linalg.norm(state.position_m, axis=-1)
    assert np.all(radius > a * (1 - 0.2) * 0.999)
    assert np.all(radius < a * (1 + 0.2) * 1.001)
