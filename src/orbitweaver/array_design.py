"""Deterministic, configuration-driven Keplerian layouts.

A shell, ring family, or inclination family is a set of independent ellipses.
Equal spacing is a draw of orbital elements. It is not a rigid body and it is
not evidence that the array remains collision-free once the model assumptions
are left behind.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from orbitweaver import STABILITY_STATEMENT
from orbitweaver.config import AngleSpec, DesignConfig, RangeSpec, ShellSpec
from orbitweaver.errors import OrbitWeaverError
from orbitweaver.kepler import propagate_elements
from orbitweaver.models import (
    OrbitArray,
    OrbitRecord,
    WarningNote,
    element_convention,
    singularity_warnings,
    validate_orbit,
)

_TWO_PI = 2.0 * np.pi


def generate_array(config: DesignConfig) -> OrbitArray:
    """Draw a reproducible array. Equal seeds and equal configs match exactly."""
    rng = np.random.Generator(np.random.PCG64(config.seed))
    warnings: list[WarningNote] = [
        WarningNote("NOT_A_STABILITY_PROOF", "info", STABILITY_STATEMENT),
    ]
    if config.intent == "visual_only":
        warnings.append(
            WarningNote(
                code="VISUAL_ONLY_NEWTONIAN_POINT_MASS",
                severity="warning",
                message=(
                    "This configuration is marked visual_only. The integrator is the "
                    "Newtonian point-mass ellipse. It does not contain relativistic "
                    "orbits, an event horizon, frame dragging, or a photon sphere. "
                    "Rendering scale is not part of the physical state."
                ),
            )
        )
    if config.physics_note:
        warnings.append(WarningNote("PHYSICS_NOTE", "info", config.physics_note))

    explicit = [_explicit_record(item) for item in config.explicit]
    for orbit in explicit:
        validate_orbit(orbit, config.body)
    occupied = _epoch_occupation(explicit, config.body.mu_m3_s2)
    _enforce_separation(
        occupied,
        config.min_initial_separation_m,
        allow_overlap=config.allow_initial_overlap,
        context="explicit orbits",
    )

    generated: list[OrbitRecord] = []
    for shell in config.shells:
        records, shell_warnings, occupied = _generate_shell(shell, config, rng, occupied)
        generated.extend(records)
        warnings.extend(shell_warnings)

    orbits = tuple(explicit + generated)
    _unique_ids(orbits)
    warnings.extend(singularity_warnings(orbits))
    invalid_ids = tuple(orbit.satellite_id for orbit in orbits if orbit.deliberately_invalid)
    if invalid_ids:
        warnings.append(
            WarningNote(
                code="DELIBERATELY_INVALID",
                severity="warning",
                message=(
                    "These orbits failed the pericentre keep-out and were kept because "
                    "deliberately_invalid is set. They are negative-test records, not a nominal design."
                ),
                orbit_ids=invalid_ids,
            )
        )
    return OrbitArray(
        body=config.body,
        epoch=config.epoch,
        orbits=orbits,
        warnings=tuple(warnings),
        seed=config.seed,
        config_sha256=config.config_sha256,
        intent=config.intent,
        source_name=config.source_name,
    )


def _generate_shell(
    shell: ShellSpec,
    config: DesignConfig,
    rng: np.random.Generator,
    occupied: list[tuple[np.ndarray, float]],
):
    warnings: list[WarningNote] = []
    if shell.kind == "ring_family":
        warnings.append(
            WarningNote(
                code="RING_FAMILY_IS_NOT_A_RIGID_BODY",
                severity="info",
                message=(
                    f"Shell {shell.name!r} is a ring family: independent Keplerian ellipses "
                    "that share a semi-major axis, eccentricity, and inclination. "
                    "Phased mean anomalies are a layout. They are not a rigid ring and "
                    "they do not demonstrate dynamical stability."
                ),
            )
        )
    if shell.inclination.mode in {"uniform", "range"} and (
        shell.inclination.high - shell.inclination.low
    ) > 2.0:
        warnings.append(
            WarningNote(
                code="INCLINATION_UNIFORM_NOT_ISOTROPIC",
                severity="warning",
                message=(
                    f"Shell {shell.name!r} draws inclination uniformly in angle over a wide "
                    "span. That oversamples the poles relative to an isotropic shell. "
                    "Use inclination mode 'isotropic' when the intended distribution is "
                    "uniform on the sphere."
                ),
            )
        )

    plane_of, local_index, local_count = _plane_maps(shell)
    candidates: list[dict[str, float | int]] = []
    for _attempt in range(config.max_placement_attempts):
        for slot in range(shell.count):
            phase_slot = local_index[slot] if shell.phase_within_plane else slot
            phase_count = local_count[slot] if shell.phase_within_plane else shell.count
            drawn = _draw_slot(shell, slot, phase_slot, phase_count, plane_of[slot], rng)
            drawn["radius"] = shell.body_radius_m
            candidates.append(drawn)

    positions = _positions_for(candidates, config.body.mu_m3_s2)
    chosen, retries = _select_separated(
        candidates,
        positions,
        shell.count,
        occupied,
        config.min_initial_separation_m,
    )
    if len(chosen) < shell.count and not config.allow_shortfall:
        missing = sorted(set(range(shell.count)) - {int(candidates[index]["slot"]) for index in chosen})
        raise OrbitWeaverError(
            f"Shell {shell.name!r} placed {len(chosen)} of {shell.count} orbits "
            f"under the initial-separation constraint "
            f"({config.min_initial_separation_m:.6e} m) after "
            f"{config.max_placement_attempts} attempts. Unfilled slots: {missing[:12]}.",
        )
    if len(chosen) < shell.count:
        warnings.append(
            WarningNote(
                code="CAPACITY_PRESSURE",
                severity="warning",
                message=(
                    f"Shell {shell.name!r} placed {len(chosen)} of {shell.count} requested "
                    "orbits. allow_shortfall is set, so the short design was kept. "
                    "This is a packing limit of the initial-separation rule, not a stability result."
                ),
            )
        )
    elif retries:
        warnings.append(
            WarningNote(
                code="PLACEMENT_RETRIES",
                severity="info",
                message=(
                    f"Shell {shell.name!r} needed extra placement attempts for "
                    f"{len(retries)} slots before the initial-separation constraint held."
                ),
                orbit_ids=tuple(f"{shell.id_prefix}-{slot:06d}" for slot in retries),
            )
        )

    records: list[OrbitRecord] = []
    for index in chosen:
        item = candidates[index]
        slot = int(item["slot"])
        satellite_id = f"{shell.id_prefix}-{slot:06d}"
        record = OrbitRecord(
            satellite_id=satellite_id,
            a_m=float(item["a"]),
            e=float(item["e"]),
            inclination_rad=float(item["inc"]),
            raan_rad=float(item["raan"]),
            argp_rad=float(item["argp"]),
            mean_anomaly_rad=float(item["mean"]),
            body_radius_m=shell.body_radius_m,
            group=shell.group,
            shell=shell.name,
            element_convention=element_convention(float(item["e"]), float(item["inc"])),
            deliberately_invalid=shell.deliberately_invalid,
            rendering=dict(shell.rendering),
        )
        validate_orbit(record, config.body)
        records.append(record)
        occupied.append((positions[index], shell.body_radius_m))
    records.sort(key=lambda orbit: orbit.satellite_id)
    if shell.kind == "ring_family" and records:
        warnings[0] = WarningNote(
            code=warnings[0].code,
            severity=warnings[0].severity,
            message=warnings[0].message,
            orbit_ids=tuple(orbit.satellite_id for orbit in records),
        )
    return records, warnings, occupied


def _draw_slot(
    shell: ShellSpec,
    slot: int,
    phase_slot: int,
    phase_count: int,
    plane: int | None,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    # Draw order is part of the seed contract: a, e, inclination, node, argument, anomaly.
    return {
        "slot": slot,
        "a": _sample_range(shell.a, rng),
        "e": _sample_range(shell.eccentricity, rng),
        "inc": _sample_inclination(shell.inclination, plane, rng),
        "raan": _sample_angle(shell.raan, phase_slot, phase_count, rng),
        "argp": _sample_angle(shell.argp, phase_slot, phase_count, rng),
        "mean": _sample_angle(shell.mean_anomaly, phase_slot, phase_count, rng),
    }


def _sample_range(spec: RangeSpec, rng: np.random.Generator) -> float:
    if spec.low == spec.high:
        return float(spec.low)
    return float(rng.uniform(spec.low, spec.high))


def _sample_inclination(spec: AngleSpec, plane: int | None, rng: np.random.Generator) -> float:
    if spec.mode == "list":
        assert plane is not None
        return float(spec.values[plane])
    return _sample_angle(spec, 0, 1, rng)


def _sample_angle(spec: AngleSpec, slot: int, count: int, rng: np.random.Generator) -> float:
    if spec.mode == "fixed":
        return float(np.mod(spec.value, _TWO_PI))
    if spec.mode in {"uniform", "range"}:
        return float(rng.uniform(spec.low, spec.high))
    if spec.mode == "isotropic":
        return float(np.arccos(rng.uniform(-1.0, 1.0)))
    if spec.mode == "phased":
        base = spec.offset + _TWO_PI * (slot / count)
        if spec.jitter > 0.0:
            base += float(rng.uniform(-spec.jitter, spec.jitter))
        return float(np.mod(base, _TWO_PI))
    if spec.mode == "list":
        return float(np.mod(spec.values[slot % len(spec.values)], _TWO_PI))
    raise OrbitWeaverError(f"Unsupported angle mode {spec.mode!r}.")


def _plane_maps(shell: ShellSpec) -> tuple[list[int | None], list[int], list[int]]:
    if shell.inclination.mode != "list":
        return [None] * shell.count, list(range(shell.count)), [shell.count] * shell.count
    n_planes = len(shell.inclination.values)
    sizes = [shell.count // n_planes + (1 if plane < shell.count % n_planes else 0) for plane in range(n_planes)]
    plane_of: list[int | None] = []
    local_index: list[int] = []
    local_count: list[int] = []
    for plane, size in enumerate(sizes):
        for local in range(size):
            plane_of.append(plane)
            local_index.append(local)
            local_count.append(size)
    return plane_of, local_index, local_count


def _select_separated(candidates, positions, count, occupied, min_sep):
    taken: set[int] = set()
    chosen: list[int] = []
    retried: list[int] = []
    acc_pos = [np.asarray(pos, dtype=np.float64) for pos, _radius in occupied]
    acc_rad = [float(radius) for _pos, radius in occupied]
    tree = cKDTree(np.vstack(acc_pos)) if acc_pos else None
    seen_attempt_fail: set[int] = set()
    for index, item in enumerate(candidates):
        slot = int(item["slot"])
        if slot in taken:
            continue
        radius = float(item.get("radius", 0.0))
        if not _far_enough(positions[index], radius, acc_pos, acc_rad, tree, min_sep):
            if slot in seen_attempt_fail:
                pass
            seen_attempt_fail.add(slot)
            continue
        if slot in seen_attempt_fail:
            retried.append(slot)
        taken.add(slot)
        chosen.append(index)
        acc_pos.append(positions[index])
        acc_rad.append(radius)
        tree = cKDTree(np.vstack(acc_pos))
        if len(taken) == count:
            break
    return chosen, retried


def _far_enough(pos, radius, acc_pos, acc_rad, tree, min_sep) -> bool:
    if tree is None:
        return True
    limit = max(float(min_sep), float(radius) + (max(acc_rad) if acc_rad else 0.0))
    if limit <= 0.0:
        return True
    neighbors = tree.query_ball_point(pos, r=limit)
    for neighbor in neighbors:
        need = max(float(min_sep), float(radius) + acc_rad[neighbor])
        if float(np.linalg.norm(pos - acc_pos[neighbor])) < need:
            return False
    return True


def _explicit_record(item) -> OrbitRecord:
    convention = element_convention(item.e, item.inclination_rad)
    return OrbitRecord(
        satellite_id=item.satellite_id,
        a_m=item.a_m,
        e=item.e,
        inclination_rad=item.inclination_rad,
        raan_rad=float(np.mod(item.raan_rad, _TWO_PI)),
        argp_rad=float(np.mod(item.argp_rad, _TWO_PI)),
        mean_anomaly_rad=float(np.mod(item.mean_anomaly_rad, _TWO_PI)),
        body_radius_m=item.body_radius_m,
        group=item.group,
        shell=item.shell,
        element_convention=convention,
        deliberately_invalid=item.deliberately_invalid,
        rendering=dict(item.rendering),
    )


def _epoch_occupation(orbits: list[OrbitRecord], mu: float) -> list[tuple[np.ndarray, float]]:
    if not orbits:
        return []
    state = propagate_elements(
        [orbit.a_m for orbit in orbits],
        [orbit.e for orbit in orbits],
        [orbit.inclination_rad for orbit in orbits],
        [orbit.raan_rad for orbit in orbits],
        [orbit.argp_rad for orbit in orbits],
        [orbit.mean_anomaly_rad for orbit in orbits],
        mu,
        [0.0],
        orbit_ids=[orbit.satellite_id for orbit in orbits],
    )
    return [
        (state.position_m[0, index], orbit.body_radius_m) for index, orbit in enumerate(orbits)
    ]


def _positions_for(candidates: list[dict[str, float | int]], mu: float) -> np.ndarray:
    state = propagate_elements(
        [float(item["a"]) for item in candidates],
        [float(item["e"]) for item in candidates],
        [float(item["inc"]) for item in candidates],
        [float(item["raan"]) for item in candidates],
        [float(item["argp"]) for item in candidates],
        [float(item["mean"]) for item in candidates],
        mu,
        [0.0],
    )
    return state.position_m[0]


def _enforce_separation(occupied, min_sep: float, *, allow_overlap: bool, context: str) -> None:
    if allow_overlap or len(occupied) < 2:
        return
    positions = np.vstack([pos for pos, _radius in occupied])
    radii = [radius for _pos, radius in occupied]
    limit = max(min_sep, (max(radii) * 2.0 if radii else 0.0))
    if limit <= 0.0:
        return
    pairs = cKDTree(positions).query_pairs(r=limit)
    bad: list[str] = []
    for left, right in pairs:
        need = max(min_sep, radii[left] + radii[right])
        if float(np.linalg.norm(positions[left] - positions[right])) < need:
            bad.append(f"{left}:{right}")
    if bad:
        raise OrbitWeaverError(
            f"Initial separation constraint failed for {context}. "
            f"Pair indexes {bad[:8]} are closer than the required centre distance "
            f"or their body volumes intersect. Set allow_initial_overlap only for a negative test."
        )


def _unique_ids(orbits: tuple[OrbitRecord, ...]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for orbit in orbits:
        if orbit.satellite_id in seen:
            duplicates.append(orbit.satellite_id)
        seen.add(orbit.satellite_id)
    if duplicates:
        raise OrbitWeaverError("Satellite ids must be unique.", duplicates)
