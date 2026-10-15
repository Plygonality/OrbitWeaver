"""Interval-limited encounter screening.

The minimum separation reported here is the minimum among the evaluated
samples and any locally refined candidate windows. Finite sampling can miss a
closer approach between samples. The speed bound below is the documented
ceiling on that miss; it is not a proof that an array is collision-free for
all time.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from orbitweaver.diagnostics import envelope_report
from orbitweaver.errors import OrbitWeaverError
from orbitweaver.kepler import propagate_array, propagate_elements
from orbitweaver.models import OrbitArray
from orbitweaver.orbital_elements import apoapsis_radius, periapsis_radius, periapsis_speed

_TIME_BATCH = 48
_PAIR_STORE_LIMIT = 4000


def screen_array(
    array: OrbitArray,
    times_s: np.ndarray,
    *,
    search_radius_m: float | None = None,
    refine: bool = True,
    max_refined_pairs: int = 16,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Screen ``array`` on an explicit time grid.

    ``search_radius_m`` is the additional spatial-index radius used to collect
    close-approach candidates. Volume intersections are always queried out to
    the sum of the largest body radii.
    """
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if times.size == 0 or np.any(~np.isfinite(times)):
        raise OrbitWeaverError("Screening times must be a non-empty finite grid.")
    if np.any(np.diff(times) < 0.0):
        raise OrbitWeaverError("Screening times must be sorted in ascending order.")
    orbits = list(array.orbits)
    count = len(orbits)
    ids = [orbit.satellite_id for orbit in orbits]
    radii = np.array([orbit.body_radius_m for orbit in orbits], dtype=np.float64)
    mu = array.body.mu_m3_s2
    semi_major = array.column("a_m") if count else np.zeros(0)
    eccentricity = array.column("e") if count else np.zeros(0)
    central = _central_body_report(array)
    envelope = (
        envelope_report(periapsis_radius(semi_major, eccentricity), apoapsis_radius(semi_major, eccentricity))
        if count
        else {"n_orbits": 0}
    )
    speed = periapsis_speed(semi_major, eccentricity, mu) if count else np.zeros(0)
    relative_speed_bound = float(2.0 * np.max(speed)) if count else 0.0
    gaps = np.diff(times)
    max_gap = float(np.max(gaps)) if gaps.size else 0.0
    missed_approach_m = relative_speed_bound * max_gap / 2.0

    pair_radius = _pair_radius(search_radius_m, radii)
    tracked: dict[tuple[int, int], dict[str, float]] = {}
    nearest_series: list[dict[str, object]] = []
    truncated = False
    collisions: list[dict[str, object]] = []
    sampled_min = None
    sampled_pair: tuple[int, int] | None = None
    sampled_time = None

    if count >= 2:
        for start in range(0, times.size, _TIME_BATCH):
            batch = times[start : start + _TIME_BATCH]
            state = propagate_array(array, batch)
            for local, instant in enumerate(batch):
                positions = state.position_m[local]
                tree = cKDTree(positions)
                neighbor_distance, neighbor_index = tree.query(positions, k=2)
                nearest_row = int(np.argmin(neighbor_distance[:, 1]))
                right = int(neighbor_index[nearest_row, 1])
                distance = float(neighbor_distance[nearest_row, 1])
                key = (nearest_row, right) if nearest_row < right else (right, nearest_row)
                nearest_series.append(
                    {"time_s": float(instant), "distance_m": distance, "key": key}
                )
                if sampled_min is None or distance < sampled_min:
                    sampled_min = distance
                    sampled_pair = key
                    sampled_time = float(instant)
                if pair_radius > 0.0:
                    for left, right_index in tree.query_pairs(r=pair_radius):
                        pair = (int(left), int(right_index))
                        if len(tracked) >= _PAIR_STORE_LIMIT and pair not in tracked:
                            truncated = True
                            continue
                        pair_distance = float(np.linalg.norm(positions[left] - positions[right_index]))
                        _remember(tracked, pair, pair_distance, float(instant))
                        if pair_distance < radii[left] + radii[right_index]:
                            collisions.append(_hit(ids, pair, pair_distance, float(instant), "sample"))
            if progress is not None:
                progress(min(start + batch.size, times.size), times.size)
        if sampled_pair is not None and sampled_min is not None and sampled_time is not None:
            _remember(tracked, sampled_pair, float(sampled_min), float(sampled_time))

    epoch = propagate_array(array, np.array([0.0])) if count else None
    initial = []
    if epoch is not None and count >= 2 and float(np.max(radii)) > 0.0:
        initial = _volume_hits(epoch.position_m[0], radii, ids, "epoch")

    refinements = []
    if refine and nearest_series and max_gap > 0.0:
        for sample in _local_minima(nearest_series)[:max_refined_pairs]:
            key = sample["key"]
            refined = _refine_pair(
                array,
                key,
                float(sample["time_s"]),
                max_gap,
                float(times[0]),
                float(times[-1]),
            )
            refinements.append(refined)
            separation = refined.get("separation_m")
            if separation is not None and separation < radii[key[0]] + radii[key[1]]:
                collisions.append(_hit(ids, key, float(separation), float(refined["time_s"]), "refined"))

    reported = _reported_minimum(sampled_min, sampled_pair, sampled_time, refinements, ids)
    certificate = _certificate(
        count=count,
        radii=radii,
        collisions=collisions or initial,
        sampled_min=sampled_min,
        missed_approach_m=missed_approach_m,
    )
    return {
        "format": "orbitweaver.screening",
        "format_version": "1.0.0",
        "guarantee": {
            "label": "sampled_interval_only",
            "statement": (
                "The minimum reported separation was screened on the stated time "
                "interval at the stated resolution, including local refinement of "
                "candidate windows when that option was on. It is not a guarantee "
                "of collision-free motion over all time, and it is not a dynamical "
                "stability result."
            ),
            "interval_start_s": float(times[0]),
            "interval_stop_s": float(times[-1]),
            "n_samples": int(times.size),
            "max_sample_gap_s": max_gap,
            "search_radius_m": pair_radius,
            "candidate_store_truncated": truncated,
            "relative_speed_bound_m_s": relative_speed_bound,
            "max_missed_approach_m": missed_approach_m,
            "missed_encounter_caveat": (
                "Let δ(t) be the centre distance of a pair and let V bound the "
                "relative speed. Then |δ(t) - δ(s)| <= V |t - s|. Between samples "
                "separated by Δt, the true minimum can sit below the lower sample "
                "by at most V Δt / 2. OrbitWeaver uses V = 2 max periapsis speed "
                "and Δt = the largest gap in the requested grid. A real encounter "
                "closer than the sampled minimum by up to that amount can be absent "
                "from the sample list. Refinement reduces this only inside the "
                "windows of the tracked candidate pairs."
            ),
            "certificate": certificate["name"],
            "certificate_meaning": certificate["meaning"],
        },
        "minimum_separation": reported,
        "initial_volume_intersections": initial,
        "sampled_or_refined_volume_intersections": _dedupe_hits(collisions),
        "central_body": central,
        "radial_envelope": envelope,
        "refined_pairs": refinements,
        "closest_candidates": _closest_payload(tracked, ids, limit=32),
        "warnings": _screen_warnings(certificate["name"], missed_approach_m, initial, central),
    }


def _pair_radius(requested: float | None, radii: np.ndarray) -> float:
    collision_span = float(2.0 * np.max(radii)) if radii.size else 0.0
    if requested is None:
        return max(1000.0, collision_span)
    if requested < 0.0:
        raise OrbitWeaverError("Search radius must be >= 0.")
    return max(float(requested), collision_span)


def _local_minima(series: list[dict[str, object]]) -> list[dict[str, object]]:
    """Deepest sampled nearest-neighbour dips, including endpoints.

    A fast crossing can sit in a gap whose samples are not the global minimum.
    Refining every local dip, not only the single closest sample, is what makes
    that crossing visible when the step still reaches its neighbourhood.
    """
    dips: list[dict[str, object]] = []
    for index, sample in enumerate(series):
        previous = float("inf") if index == 0 else float(series[index - 1]["distance_m"])
        following = float("inf") if index + 1 == len(series) else float(series[index + 1]["distance_m"])
        if float(sample["distance_m"]) <= previous and float(sample["distance_m"]) <= following:
            dips.append(sample)
    dips.sort(key=lambda item: float(item["distance_m"]))
    return dips


def _remember(tracked: dict[tuple[int, int], dict[str, float]], key, distance: float, time_s: float) -> None:
    current = tracked.get(key)
    if current is None or distance < current["distance_m"]:
        tracked[key] = {"distance_m": distance, "time_s": time_s}


def _hit(ids: list[str], key: tuple[int, int], distance: float, time_s: float, source: str) -> dict[str, object]:
    return {
        "id_a": ids[key[0]],
        "id_b": ids[key[1]],
        "separation_m": distance,
        "time_offset_s": time_s,
        "source": source,
    }


def _volume_hits(positions: np.ndarray, radii: np.ndarray, ids: list[str], source: str) -> list[dict[str, object]]:
    span = float(2.0 * np.max(radii))
    hits = []
    tree = cKDTree(positions)
    for left, right in tree.query_pairs(r=span):
        distance = float(np.linalg.norm(positions[left] - positions[right]))
        if distance < radii[left] + radii[right]:
            hits.append(_hit(ids, (int(left), int(right)), distance, 0.0, source))
    return hits


def _refine_pair(array: OrbitArray, key: tuple[int, int], centre: float, half_window: float, start: float, stop: float):
    left, right = array.orbits[key[0]], array.orbits[key[1]]
    lo = max(start, centre - half_window)
    hi = min(stop, centre + half_window)
    if hi <= lo:
        return {
            "id_a": left.satellite_id,
            "id_b": right.satellite_id,
            "separation_m": None,
            "time_s": centre,
            "status": "skipped_empty_window",
        }

    def separation(instant: float) -> float:
        state = propagate_elements(
            [left.a_m, right.a_m],
            [left.e, right.e],
            [left.inclination_rad, right.inclination_rad],
            [left.raan_rad, right.raan_rad],
            [left.argp_rad, right.argp_rad],
            [left.mean_anomaly_rad, right.mean_anomaly_rad],
            array.body.mu_m3_s2,
            [instant],
            orbit_ids=[left.satellite_id, right.satellite_id],
        )
        return float(np.linalg.norm(state.position_m[0, 0] - state.position_m[0, 1]))

    result = minimize_scalar(separation, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3, "maxiter": 80})
    return {
        "id_a": left.satellite_id,
        "id_b": right.satellite_id,
        "separation_m": float(result.fun),
        "time_s": float(result.x),
        "window_start_s": lo,
        "window_stop_s": hi,
        "status": "refined",
    }


def _reported_minimum(sampled_min, sampled_pair, sampled_time, refinements, ids):
    best_value = sampled_min
    source = "sampled"
    pair = None if sampled_pair is None else [ids[sampled_pair[0]], ids[sampled_pair[1]]]
    time_s = sampled_time
    for item in refinements:
        value = item.get("separation_m")
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_value = float(value)
            source = "refined"
            pair = [item["id_a"], item["id_b"]]
            time_s = item["time_s"]
    return {
        "metres": None if best_value is None else float(best_value),
        "source": source,
        "pair": pair,
        "time_offset_s": time_s,
        "sampled_min_metres": None if sampled_min is None else float(sampled_min),
        "note": (
            "This is the smallest centre distance evaluated on the time grid "
            "and inside refined candidate windows. It is not the minimum over all time."
        ),
    }


def _certificate(*, count, radii, collisions, sampled_min, missed_approach_m):
    if collisions:
        return {
            "name": "encounter_detected",
            "meaning": (
                "At least one evaluated sample or refined window brings two body "
                "volumes into intersection, or the epoch state already does. "
                "See the intersection lists for orbit ids."
            ),
        }
    if count < 2 or sampled_min is None:
        return {
            "name": "no_pair",
            "meaning": "Fewer than two satellites were screened, so there is no pair separation.",
        }
    collision_distance = float(np.max(radii) + np.partition(radii, -2)[-2]) if count >= 2 else 0.0
    if sampled_min >= collision_distance + missed_approach_m:
        return {
            "name": "no_collision_consistent_with_samples",
            "meaning": (
                "Under the two-body model and the stated speed bound, a "
                "centre-to-centre approach inside the largest pair of body radii "
                "on this grid would contradict the sampled separations. "
                "The certificate covers this interval only. It is not a stability proof "
                "and it says nothing about times outside the grid."
            ),
        }
    return {
        "name": "not_certified",
        "meaning": (
            "No evaluated sample showed a body-volume intersection, and the "
            "sampling bound is too wide to rule one out between samples. "
            "A closer approach may have been missed. Tighten the step or inspect "
            "the refined candidates. This state is not a clean bill of health."
        ),
    }


def _closest_payload(tracked, ids, limit: int) -> list[dict[str, object]]:
    ranked = sorted(tracked.items(), key=lambda item: item[1]["distance_m"])[:limit]
    payload = []
    for key, record in ranked:
        payload.append(
            {
                "id_a": ids[key[0]],
                "id_b": ids[key[1]],
                "sampled_min_separation_m": record["distance_m"],
                "time_offset_s": record["time_s"],
            }
        )
    return payload


def _dedupe_hits(hits: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for hit in hits:
        marker = (str(hit["id_a"]), str(hit["id_b"]), str(hit["source"]))
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(hit)
    return unique


def _central_body_report(array: OrbitArray) -> dict[str, object]:
    violations = []
    for orbit in array.orbits:
        peri = orbit.a_m * (1.0 - orbit.e)
        physical_limit = array.body.radius_m + orbit.body_radius_m
        exclusion_hit = peri <= array.body.exclusion_radius_m
        physical_hit = peri <= physical_limit
        if exclusion_hit or physical_hit:
            violations.append(
                {
                    "satellite_id": orbit.satellite_id,
                    "periapsis_m": peri,
                    "exclusion_radius_m": array.body.exclusion_radius_m,
                    "physical_contact_radius_m": physical_limit,
                    "deliberately_invalid": orbit.deliberately_invalid,
                    "analytic": True,
                    "note": (
                        "Pericentre is an exact two-body result. This central-body "
                        "check does not depend on the sample step."
                    ),
                }
            )
    return {
        "violations": violations,
        "n_checked": len(array.orbits),
        "clear": not violations,
    }


def _screen_warnings(certificate: str, missed_approach_m: float, initial, central) -> list[dict[str, object]]:
    notes = [
        {
            "code": "SAMPLED_SCREEN_NOT_ALL_TIME",
            "severity": "info",
            "message": (
                "Minimum separation is reported for the sampled interval and the "
                f"documented miss bound of {missed_approach_m:.6e} m. "
                "It does not certify the array for all time."
            ),
            "orbit_ids": [],
        }
    ]
    if certificate == "not_certified":
        notes.append(
            {
                "code": "SCREEN_NOT_CERTIFIED",
                "severity": "warning",
                "message": (
                    "The sample step is coarse relative to orbital speed. "
                    "Absence of a sampled intersection does not rule out an encounter between samples."
                ),
                "orbit_ids": [],
            }
        )
    if initial:
        notes.append(
            {
                "code": "INITIAL_VOLUME_INTERSECTION",
                "severity": "warning",
                "message": "Body volumes intersect at the epoch.",
                "orbit_ids": sorted({item["id_a"] for item in initial} | {item["id_b"] for item in initial}),
            }
        )
    central_ids = [item["satellite_id"] for item in central["violations"]]
    if central_ids:
        notes.append(
            {
                "code": "CENTRAL_BODY_PERICENTRE",
                "severity": "warning",
                "message": "Analytic pericentre reaches the central body or its exclusion radius.",
                "orbit_ids": central_ids,
            }
        )
    return notes
