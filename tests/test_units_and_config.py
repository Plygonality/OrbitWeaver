"""Time parsing, config rejection, and reproducible design."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orbitweaver.array_design import generate_array
from orbitweaver.config import load_config
from orbitweaver.errors import ConfigError, OrbitWeaverError
from orbitweaver.units import parse_duration_seconds

ROOT = Path(__file__).resolve().parents[1]


def test_duration_requires_a_unit():
    assert parse_duration_seconds("365d") == 365 * 86400
    assert parse_duration_seconds("1h") == 3600
    assert parse_duration_seconds("1.5 day") == 1.5 * 86400
    assert parse_duration_seconds("1 yr") == 365.25 * 86400
    with pytest.raises(ConfigError, match="unitless"):
        parse_duration_seconds("365")
    with pytest.raises(ConfigError, match="unitless"):
        parse_duration_seconds("365.0")
    with pytest.raises(ConfigError, match="Unrecognised"):
        parse_duration_seconds("1 fortnight")


def test_epoch_scale_is_required(tmp_path: Path):
    text = """
format = "orbitweaver.config"
format_version = "1.0.0"
[physics]
model = "newtonian_two_body_kepler"
[epoch]
iso = "2000-01-01T12:00:00"
[central_body]
preset = "earth"
[[orbits]]
id = "A"
a_m = 7000000.0
e = 0.0
"""
    path = tmp_path / "missing_scale.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="scale"):
        load_config(path)


def test_equal_seeds_match_and_elements_stay_in_range():
    first = generate_array(load_config(ROOT / "examples/elliptical_shell.toml"))
    second = generate_array(load_config(ROOT / "examples/elliptical_shell.toml"))
    assert [orbit.satellite_id for orbit in first.orbits] == [orbit.satellite_id for orbit in second.orbits]
    for left, right in zip(first.orbits, second.orbits, strict=True):
        assert left.to_dict() == right.to_dict()
        assert 8_000_000.0 <= left.a_m <= 12_000_000.0
        assert 0.05 <= left.e <= 0.35
        assert np.deg2rad(15) <= left.inclination_rad <= np.deg2rad(75)
    assert first.seed == 42
    assert any(note.code == "NOT_A_STABILITY_PROOF" for note in first.warnings)


def test_ring_family_is_labelled_and_phased():
    array = generate_array(load_config(ROOT / "examples/ring_family.toml"))
    assert len(array.orbits) == 24
    assert {orbit.a_m for orbit in array.orbits} == {42_164_000.0}
    assert {orbit.e for orbit in array.orbits} == {0.0}
    anomalies = np.array([orbit.mean_anomaly_rad for orbit in array.orbits])
    assert np.allclose(np.sort(anomalies), np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False))
    assert any(note.code == "RING_FAMILY_IS_NOT_A_RIGID_BODY" for note in array.warnings)


def test_invalid_eccentricity_names_the_orbit():
    with pytest.raises(OrbitWeaverError, match="BAD-HYPERBOLA") as caught:
        generate_array(load_config(ROOT / "examples/invalid_elements.toml"))
    assert "BAD-HYPERBOLA" in caught.value.orbit_ids


def test_reference_manifest_roundtrip_preserves_elements(tmp_path: Path):
    from orbitweaver.export import load_manifest, write_manifest

    array = generate_array(load_config(ROOT / "examples/reference_circular.toml"))
    path = write_manifest(array, tmp_path / "manifest.json")
    loaded = load_manifest(path)
    assert loaded.orbits[0].satellite_id == "REF-CIRC-1"
    assert loaded.orbits[0].a_m == array.orbits[0].a_m
    assert loaded.body.mu_m3_s2 == array.body.mu_m3_s2
    assert loaded.epoch.scale == "tt"
