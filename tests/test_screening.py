"""Controlled encounter, non-encounter, and sampling-bound tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orbitweaver.array_design import generate_array
from orbitweaver.config import load_config
from orbitweaver.kepler import mean_motion, propagate_array
from orbitweaver.screening import screen_array

ROOT = Path(__file__).resolve().parents[1]


def test_known_crossing_is_detected_when_refined():
    array = generate_array(load_config(ROOT / "examples/close_approach.toml"))
    mu = array.body.mu_m3_s2
    period = float(2.0 * np.pi / mean_motion(array.column("a_m"), mu)[0])
    state = propagate_array(array, np.array([period / 4.0]))
    distance = float(np.linalg.norm(state.position_m[0, 0] - state.position_m[0, 1]))
    assert distance < 1e-4

    coarse = screen_array(array, np.array([0.0, period / 2.0]), refine=False)
    assert coarse["guarantee"]["certificate"] == "not_certified"
    assert coarse["sampled_or_refined_volume_intersections"] == []
    assert coarse["guarantee"]["max_missed_approach_m"] > coarse["minimum_separation"]["metres"]

    refined = screen_array(array, np.array([0.0, period / 2.0]), refine=True)
    assert refined["guarantee"]["certificate"] == "encounter_detected"
    assert refined["minimum_separation"]["metres"] < 50.0
    pair = set(refined["minimum_separation"]["pair"])
    assert pair == {"ENC-A", "ENC-B"}


def test_separated_pair_is_certified_on_a_fine_grid():
    array = generate_array(load_config(ROOT / "examples/reference_circular.toml"))
    # Rebuild a two-orbit case by screening the reference together with a far twin
    # constructed through the close-approach loader is unnecessary: use two
    # explicit radii from a tiny inline propagation via the elliptical file's body.
    from orbitweaver.models import Epoch, OrbitArray, OrbitRecord, element_convention

    body = array.body
    epoch = array.epoch
    orbits = []
    for index, semi_major in enumerate((7_000_000.0, 21_000_000.0)):
        orbits.append(
            OrbitRecord(
                satellite_id=f"SEP-{index}",
                a_m=semi_major,
                e=0.0,
                inclination_rad=0.1,
                raan_rad=0.2,
                argp_rad=0.0,
                mean_anomaly_rad=0.3,
                body_radius_m=1.0,
                group="separated",
                shell="test",
                element_convention=element_convention(0.0, 0.1),
            )
        )
    separated = OrbitArray(
        body=body,
        epoch=Epoch(epoch.iso, epoch.scale, epoch.jd_tt),
        orbits=tuple(orbits),
        warnings=(),
        seed=None,
        config_sha256=None,
        intent="science",
    )
    period = float(2.0 * np.pi / mean_motion(np.array([7_000_000.0]), body.mu_m3_s2)[0])
    result = screen_array(separated, np.linspace(0.0, period, 40), refine=False)
    assert result["guarantee"]["certificate"] == "no_collision_consistent_with_samples"
    assert result["minimum_separation"]["metres"] > 1.0e6


def test_invalid_pericentre_is_analytic():
    array = generate_array(load_config(ROOT / "examples/deliberately_invalid.toml"))
    result = screen_array(array, np.array([0.0]), refine=False)
    assert result["central_body"]["clear"] is False
    assert result["central_body"]["violations"][0]["satellite_id"] == "INV-PERI"
    assert any(note["code"] == "CENTRAL_BODY_PERICENTRE" for note in result["warnings"])
