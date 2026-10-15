"""Manifest, chunked ephemeris, and CLI contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbitweaver.cli import main
from orbitweaver.export import read_state_npz
from orbitweaver.kepler import propagate_elements
from orbitweaver.models import central_body_preset

ROOT = Path(__file__).resolve().parents[1]


def test_cli_generate_propagate_report_and_reject_unitless_time(tmp_path: Path):
    manifest_dir = tmp_path / "shell"
    assert main(["generate", str(ROOT / "examples/reference_circular.toml"), "--out", str(manifest_dir)]) == 0
    manifest = manifest_dir / "manifest.json"
    ephemeris = tmp_path / "ephemeris"
    assert (
        main(
            [
                "propagate",
                str(manifest),
                "--duration",
                "2h",
                "--step",
                "10min",
                "--out",
                str(ephemeris),
                "--preview-metres-per-unit",
                "1e5",
            ]
        )
        == 0
    )
    archive = read_state_npz(ephemeris / "states.npz")
    body = central_body_preset("earth")
    period_state = propagate_elements(
        [6_778_137.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        body.mu_m3_s2,
        archive["time_s"],
    )
    np.testing.assert_allclose(archive["position_m"], period_state.position_m, rtol=0.0, atol=1e-4)
    assert (ephemeris / "preview.owbin").is_file()
    report = tmp_path / "design.html"
    assert main(["report", str(manifest), "--out", str(report)]) == 0
    text = report.read_text(encoding="utf-8")
    assert "REF-CIRC-1" in text
    assert "not a proof" in text.lower() or "Not a proof" in text or "not a" in text.lower()
    assert main(["propagate", str(manifest), "--duration", "365", "--step", "1d", "--out", str(tmp_path / "nope")]) == 2


def test_chunked_ephemeris_matches_direct_propagation(tmp_path: Path):
    from orbitweaver.array_design import generate_array
    from orbitweaver.config import load_config
    from orbitweaver.export import write_ephemeris
    from orbitweaver.kepler import propagate_array

    array = generate_array(load_config(ROOT / "examples/elliptical_shell.toml"))
    times = np.array([0.0, 600.0, 1200.0])
    index = write_ephemeris(array, times, tmp_path / "chunks", batch_orbits=5, single_file_limit=1)
    assert index["single_file"] is False
    pieces = []
    for chunk in index["chunks"]:
        pieces.append(read_state_npz(tmp_path / "chunks" / chunk["file"])["position_m"])
    stacked = np.concatenate(pieces, axis=1)
    direct = propagate_array(array, times)
    np.testing.assert_allclose(stacked, direct.position_m, rtol=0.0, atol=1e-5)


def test_screen_cli_on_close_approach(tmp_path: Path):
    manifest_dir = tmp_path / "encounter"
    assert main(["generate", str(ROOT / "examples/close_approach.toml"), "--out", str(manifest_dir)]) == 0
    out = tmp_path / "encounters"
    assert (
        main(
            [
                "screen",
                str(manifest_dir / "manifest.json"),
                "--duration",
                "2h",
                "--step",
                "15min",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads((out / "encounters.json").read_text(encoding="utf-8"))
    assert payload["guarantee"]["certificate"] == "encounter_detected"
    assert "all time" in payload["guarantee"]["statement"]


def test_doctor_runs():
    assert main(["doctor"]) == 0
