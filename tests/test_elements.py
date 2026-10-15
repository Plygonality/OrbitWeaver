"""Element singularities and Cartesian round trips."""

from __future__ import annotations

import numpy as np
import pytest

from orbitweaver.errors import OrbitWeaverError
from orbitweaver.kepler import propagate_elements, solve_kepler
from orbitweaver.orbital_elements import cartesian_to_elements


def _roundtrip(mu, a, e, inc, raan, argp, mean):
    state = propagate_elements([a], [e], [inc], [raan], [argp], [mean], mu, [0.0])
    recovered = cartesian_to_elements(state.position_m[0, 0], state.velocity_m_s[0, 0], mu)
    again = propagate_elements(
        [recovered["a_m"]],
        [recovered["e"]],
        [recovered["inclination_rad"]],
        [recovered["raan_rad"]],
        [recovered["argp_rad"]],
        [recovered["mean_anomaly_rad"]],
        mu,
        [0.0],
    )
    np.testing.assert_allclose(again.position_m[0, 0], state.position_m[0, 0], rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(again.velocity_m_s[0, 0], state.velocity_m_s[0, 0], rtol=1e-10, atol=1e-8)
    return recovered, state


def test_classical_angles_roundtrip():
    mu = 1.0
    recovered, _state = _roundtrip(mu, 2.0, 0.3, 0.8, 1.2, 0.4, 1.7)
    assert recovered["convention"] == "classical"
    assert abs(recovered["a_m"] - 2.0) / 2.0 < 1e-12
    assert abs(recovered["e"] - 0.3) < 1e-12
    assert abs(recovered["inclination_rad"] - 0.8) < 1e-12
    assert abs(_angle_diff(recovered["raan_rad"], 1.2)) < 1e-10
    assert abs(_angle_diff(recovered["argp_rad"], 0.4)) < 1e-10
    assert abs(_angle_diff(recovered["mean_anomaly_rad"], 1.7)) < 1e-10


def test_circular_inclined_convention_preserves_state():
    recovered, _state = _roundtrip(1.0, 1.5, 0.0, 0.6, 0.9, 0.4, 1.1)
    assert recovered["convention"] == "circular_inclined"
    assert recovered["e"] == 0.0
    assert recovered["argp_rad"] == 0.0 or abs(recovered["argp_rad"]) < 1e-12


def test_equatorial_elliptical_convention_preserves_state():
    recovered, _state = _roundtrip(1.0, 3.0, 0.25, 0.0, 0.7, 0.5, 0.2)
    assert recovered["convention"] == "elliptical_equatorial"
    assert recovered["raan_rad"] == 0.0 or abs(recovered["raan_rad"]) < 1e-12


def test_circular_equatorial_depends_on_true_longitude_only():
    mu = 1.0
    a = 4.0
    # Two angle sets with the same true longitude λ = Ω + ω + M.
    first = propagate_elements([a], [0.0], [0.0], [0.4], [0.5], [0.2], mu, [0.0])
    second = propagate_elements([a], [0.0], [0.0], [0.0], [0.0], [1.1], mu, [0.0])
    np.testing.assert_allclose(first.position_m, second.position_m, rtol=0.0, atol=1e-12)
    recovered, _state = _roundtrip(mu, a, 0.0, 0.0, 0.4, 0.5, 0.2)
    assert recovered["convention"] == "circular_equatorial"
    assert abs(_angle_diff(recovered["mean_anomaly_rad"], 1.1)) < 1e-10


def test_retrograde_equatorial_roundtrip():
    recovered, _state = _roundtrip(1.0, 2.0, 0.0, np.pi, 0.2, 0.3, 0.4)
    assert recovered["convention"] == "circular_equatorial"
    recovered_e, _state = _roundtrip(1.0, 2.0, 0.4, np.pi, 0.2, 0.3, 0.8)
    assert recovered_e["convention"] == "elliptical_equatorial"


def test_hyperbolic_state_is_rejected():
    with pytest.raises(OrbitWeaverError, match="bound ellipse"):
        cartesian_to_elements(np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0]), 1.0)


def test_solver_rejects_hyperbolic_eccentricity():
    with pytest.raises(OrbitWeaverError, match="0 <= e < 1"):
        solve_kepler(0.2, 1.2)


def _angle_diff(left: float, right: float) -> float:
    return (left - right + np.pi) % (2.0 * np.pi) - np.pi
