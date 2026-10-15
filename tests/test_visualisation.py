"""Render-space scale, floating origin, and preview-cache round trip."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from orbitweaver.visualisation import (
    float32_quantisation_m,
    floating_origin_positions,
    read_preview_cache,
    to_render_space,
    write_preview_cache,
)

ROOT = Path(__file__).resolve().parents[1]


def test_scale_and_floating_origin_keep_science_separate():
    au = 149_597_870_700.0
    scientific = np.array([au, 0.0, 0.0])
    rendered = to_render_space(scientific, metres_per_unit=1.0e9, origin_m=np.zeros(3))
    np.testing.assert_allclose(rendered, [au / 1.0e9, 0.0, 0.0])
    centred = to_render_space(scientific, metres_per_unit=1.0e9, origin_m=scientific)
    np.testing.assert_allclose(centred, 0.0, atol=0.0)
    step = float32_quantisation_m(au, 1.0)
    assert step > 1000.0
    positions = np.zeros((2, 1, 3))
    positions[:, 0, 0] = [au, au * 1.01]
    tracked = floating_origin_positions(positions, metres_per_unit=1.0e9, origin_index=0)
    np.testing.assert_allclose(tracked[:, 0, :], 0.0, atol=0.0)


def test_preview_cache_roundtrip(tmp_path: Path):
    times = np.array([0.0, 10.0])
    render = np.zeros((2, 3, 3), dtype=np.float64)
    render[:, :, 0] = np.arange(3)
    write_preview_cache(
        tmp_path / "preview.owbin",
        render,
        times,
        metres_per_unit=1.0e9,
        origin_m=np.array([1.0, 2.0, 3.0]),
        satellite_ids=["A", "B", "C"],
        manifest_name="manifest.json",
    )
    loaded = read_preview_cache(tmp_path / "preview.owbin")
    np.testing.assert_allclose(loaded["times_s"], times)
    np.testing.assert_allclose(loaded["render_positions"], render, atol=1e-5)
    assert loaded["metres_per_unit"] == 1.0e9


def test_preview_solver_matches_vectorised_kepler():
    module_path = ROOT / "blender_addon" / "orbitweaver_preview" / "kepler_preview.py"
    spec = importlib.util.spec_from_file_location("kepler_preview", module_path)
    assert spec and spec.loader
    preview = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preview)
    from orbitweaver.kepler import propagate_elements

    mu = 3.986004e14
    samples = [
        (7_000_000.0, 0.0, 0.2, 0.4, 0.1, 1.2),
        (1.2e7, 0.7, 1.1, 0.3, 2.0, -0.4),
        (2.0e7, 0.99, 0.4, 1.5, 0.2, 0.8),
    ]
    for a, e, inc, raan, argp, mean in samples:
        state = propagate_elements([a], [e], [inc], [raan], [argp], [mean], mu, [0.0])
        position, _velocity = preview.inertial_state(a, e, inc, raan, argp, mean, mu)
        np.testing.assert_allclose(position, state.position_m[0, 0], rtol=1e-8, atol=1e-2)

