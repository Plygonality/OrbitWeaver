"""Design diagnostics derived from elements and a short state sample.

These figures describe the Keplerian layout. They do not certify stability.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from orbitweaver.kepler import orbital_period, propagate_array
from orbitweaver.models import OrbitArray
from orbitweaver.orbital_elements import (
    analytic_h_magnitude,
    analytic_specific_energy,
    apoapsis_radius,
    periapsis_radius,
    periapsis_speed,
    specific_angular_momentum,
    specific_energy,
)


def design_diagnostics(array: OrbitArray) -> dict[str, object]:
    """Summarise one array at epoch and over one median orbital period."""
    orbits = array.orbits
    count = len(orbits)
    if count == 0:
        return {"n_orbits": 0, "warnings": [note.to_dict() for note in array.warnings]}
    semi_major = array.column("a_m")
    eccentricity = array.column("e")
    inclination = array.column("inclination_rad")
    radii = array.column("body_radius_m")
    mu = array.body.mu_m3_s2
    period = orbital_period(semi_major, mu)
    peri = periapsis_radius(semi_major, eccentricity)
    apo = apoapsis_radius(semi_major, eccentricity)
    median_period = float(np.median(period))
    sample_times = np.array(
        [0.0, 0.25 * median_period, 0.5 * median_period, 0.75 * median_period, median_period],
        dtype=np.float64,
    )
    state = propagate_array(array, sample_times)
    energy = specific_energy(state.position_m, state.velocity_m_s, mu)
    expected_energy = analytic_specific_energy(semi_major, mu)
    energy_error = np.max(np.abs(energy - expected_energy[None, :]) / np.abs(expected_energy))
    angular = specific_angular_momentum(state.position_m, state.velocity_m_s)
    angular_mag = np.linalg.norm(angular, axis=-1)
    expected_h = analytic_h_magnitude(semi_major, eccentricity, mu)
    h_error = np.max(np.abs(angular_mag - expected_h[None, :]) / expected_h)
    epoch_speed = np.linalg.norm(state.velocity_m_s[0], axis=-1)
    epoch_position = state.position_m[0]
    neighbors = _nearest_neighbors(epoch_position)
    density = _peak_density(epoch_position, neighbors["distances_m"])
    return {
        "n_orbits": count,
        "groups": sorted({orbit.group for orbit in orbits}),
        "shells": sorted({orbit.shell for orbit in orbits}),
        "semi_major_m": _stats(semi_major),
        "eccentricity": _stats(eccentricity),
        "inclination_rad": _stats(inclination),
        "period_s": _stats(period),
        "periapsis_m": _stats(peri),
        "apoapsis_m": _stats(apo),
        "epoch_speed_m_s": _stats(epoch_speed),
        "periapsis_speed_m_s": _stats(periapsis_speed(semi_major, eccentricity, mu)),
        "body_radius_m": _stats(radii),
        "conservation": {
            "sample_times_s": sample_times.tolist(),
            "sample_basis": "five samples spanning the median Keplerian period",
            "max_relative_energy_error": float(energy_error),
            "max_relative_angular_momentum_error": float(h_error),
            "kepler_residual_max_rad": state.solution.max_abs_residual,
            "meaning": (
                "Specific energy and angular momentum are integrals of the "
                "two-body problem. Agreement here checks the state map. "
                "It is not a multi-body stability test."
            ),
        },
        "radial_envelope": envelope_report(peri, apo),
        "nearest_neighbor": neighbors,
        "peak_density": density,
        "warnings": [note.to_dict() for note in array.warnings],
    }


def envelope_report(periapsis_m: np.ndarray, apoapsis_m: np.ndarray) -> dict[str, object]:
    """Union of radial intervals ``[periapsis, apoapsis]`` and pairwise overlap count.

    Overlap of two radial intervals is a necessary condition for a shared
    geocentric distance. It is not a conjunction and not a collision.
    """
    peri = np.asarray(periapsis_m, dtype=np.float64).reshape(-1)
    apo = np.asarray(apoapsis_m, dtype=np.float64).reshape(-1)
    count = int(peri.size)
    if count == 0:
        return {"n_orbits": 0}
    order = np.argsort(peri, kind="mergesort")
    left = peri[order]
    right = apo[order]
    overlaps = _count_interval_overlaps(left, right)
    possible = count * (count - 1) // 2
    union, components, gaps = _union_intervals(left, right)
    span = float(np.max(apo) - np.min(peri))
    return {
        "n_orbits": count,
        "overlapping_pairs": overlaps,
        "possible_pairs": possible,
        "overlap_fraction": (overlaps / possible) if possible else 0.0,
        "union_length_m": union,
        "span_m": span,
        "occupancy": (union / span) if span > 0.0 else 1.0,
        "components": components,
        "min_gap_m": None if not gaps else float(min(gaps)),
        "meaning": (
            "Radial-envelope overlap means the osculating periapsis/apoapsis "
            "intervals share a radius. Orbits in different planes can overlap "
            "in radius and still miss in space."
        ),
    }


def _count_interval_overlaps(left: np.ndarray, right: np.ndarray) -> int:
    """Count pairs with ``left`` sorted ascending.

    For ``j > i``, the intervals overlap exactly when ``left[j] <= right[i]``.
    """
    if left.size < 2:
        return 0
    end_index = np.searchsorted(left, right, side="right")
    counts = end_index - np.arange(left.size) - 1
    return int(np.sum(np.maximum(counts, 0)))


def _union_intervals(left: np.ndarray, right: np.ndarray) -> tuple[float, int, list[float]]:
    total = 0.0
    components = 0
    gaps: list[float] = []
    cursor_left = float(left[0])
    cursor_right = float(right[0])
    for start, stop in zip(left[1:], right[1:], strict=True):
        if start <= cursor_right:
            cursor_right = max(cursor_right, float(stop))
        else:
            total += cursor_right - cursor_left
            components += 1
            gaps.append(float(start) - cursor_right)
            cursor_left = float(start)
            cursor_right = float(stop)
    total += cursor_right - cursor_left
    components += 1
    return total, components, gaps


def _nearest_neighbors(position_m: np.ndarray) -> dict[str, object]:
    count = int(position_m.shape[0])
    if count < 2:
        return {
            "n_orbits": count,
            "min_m": None,
            "median_m": None,
            "distances_m": [],
            "closest_pair_indexes": None,
        }
    distances, indexes = cKDTree(position_m).query(position_m, k=2)
    nearest = distances[:, 1]
    closest = int(np.argmin(nearest))
    return {
        "n_orbits": count,
        "min_m": float(np.min(nearest)),
        "percentile_5_m": float(np.percentile(nearest, 5)),
        "median_m": float(np.median(nearest)),
        "max_m": float(np.max(nearest)),
        "distances_m": nearest.tolist() if count <= 4000 else [],
        "closest_pair_indexes": [closest, int(indexes[closest, 1])],
    }


def _peak_density(position_m: np.ndarray, nearest_m: list[float] | np.ndarray) -> dict[str, object]:
    count = int(position_m.shape[0])
    if count < 2:
        return {
            "defined": False,
            "reason": "Peak spatial density needs at least two satellites.",
        }
    nearest = np.asarray(nearest_m, dtype=np.float64)
    if nearest.size == 0:
        _tree_distances, _indexes = cKDTree(position_m).query(position_m, k=2)
        nearest = _tree_distances[:, 1]
    radius = float(np.median(nearest))
    if radius <= 0.0:
        return {
            "defined": False,
            "reason": "Median nearest-neighbour distance is zero, so the density ball is undefined.",
        }
    counts = cKDTree(position_m).query_ball_point(position_m, r=radius, return_length=True)
    peak = int(np.max(counts))
    volume = 4.0 / 3.0 * np.pi * radius**3
    return {
        "defined": True,
        "ball_radius_m": radius,
        "ball_rule": "median nearest-neighbour centre distance at the sampled positions",
        "peak_count_including_self": peak,
        "peak_number_density_per_m3": peak / volume,
        "meaning": (
            "This is an epoch snapshot inside one ball per satellite. "
            "It is not a time-averaged collision probability."
        ),
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }
