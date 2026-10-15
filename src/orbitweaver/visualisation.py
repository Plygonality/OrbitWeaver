"""Render-space transforms. This module does not import Blender.

Scientific coordinates stay in SI metres. Render coordinates are a presentation
choice: a scale plus a floating origin. A uniform scale does not recover
precision that float32 loses far from the origin; moving the origin does.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from orbitweaver.errors import OrbitWeaverError

PREVIEW_MAGIC = b"OWPV"
PREVIEW_VERSION = 1


def to_render_space(
    position_m: np.ndarray,
    *,
    metres_per_unit: float,
    origin_m: np.ndarray | None = None,
) -> np.ndarray:
    """Map scientific metres to render units.

    ``metres_per_unit`` is the number of SI metres represented by one render
    unit (one Blender unit, when the add-on is used). The origin is subtracted
    before scaling.
    """
    if not np.isfinite(metres_per_unit) or metres_per_unit <= 0.0:
        raise OrbitWeaverError("metres_per_unit must be a positive finite length.")
    positions = np.asarray(position_m, dtype=np.float64)
    origin = np.zeros(3, dtype=np.float64) if origin_m is None else np.asarray(origin_m, dtype=np.float64)
    if origin.shape != (3,):
        raise OrbitWeaverError("Floating origin must have shape (3,).")
    return (positions - origin) / metres_per_unit


def floating_origin_positions(
    position_m: np.ndarray,
    *,
    metres_per_unit: float,
    origin_index: int | None = None,
) -> np.ndarray:
    """Scale about the central body, or about one satellite at every sample.

    ``position_m`` has shape ``(n_times, n_orbits, 3)``. When ``origin_index``
    is set, that satellite is subtracted at each time so it stays at the render
    origin while the scientific state is left unchanged.
    """
    positions = np.asarray(position_m, dtype=np.float64)
    if origin_index is None:
        return to_render_space(positions, metres_per_unit=metres_per_unit)
    if positions.ndim != 3:
        raise OrbitWeaverError("Satellite-tracking origin expects shape (time, orbit, 3).")
    origin = positions[:, origin_index, :]
    return (positions - origin[:, None, :]) / metres_per_unit


def float32_quantisation_m(distance_from_origin_m: float, metres_per_unit: float) -> float:
    """SI metres represented by one float32 step at this render coordinate.

    Mesh coordinates in Blender are float32. The step depends on the distance
    from the floating origin after scaling, then converted back to metres.
    """
    render = np.float32(distance_from_origin_m / metres_per_unit)
    nxt = np.nextafter(render, np.float32(np.inf))
    return float(abs(np.float64(nxt) - np.float64(render)) * metres_per_unit)


def precision_note(distance_from_origin_m: float, metres_per_unit: float, body_radius_m: float) -> dict[str, object]:
    step = float32_quantisation_m(distance_from_origin_m, metres_per_unit)
    return {
        "distance_from_origin_m": distance_from_origin_m,
        "metres_per_unit": metres_per_unit,
        "float32_step_m": step,
        "body_radius_m": body_radius_m,
        "step_exceeds_body_radius": bool(body_radius_m > 0.0 and step > body_radius_m),
        "meaning": (
            "Float32 render coordinates quantise position in steps of about "
            "this size at the stated distance from the floating origin. "
            "Choose an origin near the camera subject when the step is larger "
            "than the detail you need to see. The scientific ephemeris remains float64 SI."
        ),
    }


def write_preview_cache(
    path: Path | str,
    render_positions: np.ndarray,
    times_s: np.ndarray,
    *,
    metres_per_unit: float,
    origin_m: np.ndarray,
    satellite_ids: list[str],
    manifest_name: str,
) -> dict[str, object]:
    """Write a compact float32 render cache and a JSON sidecar.

    The binary file contains render units, not scientific metres. The sidecar
    records the scale, the origin, and the separation from the scientific manifest.
    """
    positions = np.asarray(render_positions, dtype=np.float32)
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape[0] != times.size:
        raise OrbitWeaverError("Preview positions must have shape (time, orbit, 3).")
    if len(satellite_ids) != positions.shape[1]:
        raise OrbitWeaverError("Preview satellite ids do not match the position array.")
    origin = np.asarray(origin_m, dtype=np.float64).reshape(3)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<4sIIIdddd",
        PREVIEW_MAGIC,
        PREVIEW_VERSION,
        positions.shape[0],
        positions.shape[1],
        float(metres_per_unit),
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
    )
    time_bytes = times.astype("<f8", copy=False).tobytes()
    point_bytes = np.ascontiguousarray(positions, dtype="<f4").tobytes()
    destination.write_bytes(header + time_bytes + point_bytes)
    sidecar = {
        "format": "orbitweaver.preview",
        "format_version": "1.0.0",
        "cache_file": destination.name,
        "manifest": manifest_name,
        "coordinates": "render_units",
        "metres_per_unit": float(metres_per_unit),
        "origin_m": origin.tolist(),
        "n_times": int(positions.shape[0]),
        "n_orbits": int(positions.shape[1]),
        "satellite_ids": list(satellite_ids),
        "scientific_coordinates": "kept in the manifest and ephemeris; not in this cache",
    }
    destination.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return sidecar


def read_preview_cache(path: Path | str) -> dict[str, object]:
    """Read a preview cache written by :func:`write_preview_cache`."""
    blob = Path(path).read_bytes()
    header = struct.unpack_from("<4sIIIdddd", blob, 0)
    magic, version, n_times, n_orbits, metres_per_unit, ox, oy, oz = header
    if magic != PREVIEW_MAGIC or version != PREVIEW_VERSION:
        raise OrbitWeaverError(f"Unrecognised preview cache header in {path}.")
    offset = struct.calcsize("<4sIIIdddd")
    time_count = n_times * 8
    times = np.frombuffer(blob, dtype="<f8", count=n_times, offset=offset).copy()
    point_offset = offset + time_count
    points = np.frombuffer(blob, dtype="<f4", count=n_times * n_orbits * 3, offset=point_offset).copy()
    return {
        "times_s": times,
        "render_positions": points.reshape(n_times, n_orbits, 3),
        "metres_per_unit": float(metres_per_unit),
        "origin_m": np.array([ox, oy, oz], dtype=np.float64),
    }
