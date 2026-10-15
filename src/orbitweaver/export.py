"""Versioned manifests and chunked ephemerides.

JSON manifests carry the orbit records. Position archives are written only for
the requested times, in one NPZ file when the product is moderate and in
orbit-chunks otherwise. A full all-times cube is not the default above that
threshold because the caller chooses the time grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbitweaver import __version__
from orbitweaver.constants import FORMAT_VERSION, FRAME_ID, MODEL_ID
from orbitweaver.errors import OrbitWeaverError
from orbitweaver.kepler import propagate_array
from orbitweaver.models import (
    CentralBody,
    Epoch,
    OrbitArray,
    OrbitRecord,
    WarningNote,
    validate_body,
    validate_orbit,
)

_SINGLE_FILE_LIMIT = 2_000_000


def manifest_dict(array: OrbitArray) -> dict[str, object]:
    import astropy

    return {
        "format": "orbitweaver.manifest",
        "format_version": FORMAT_VERSION,
        "software": {
            "name": "orbitweaver",
            "version": __version__,
            "astropy": astropy.__version__,
            "numpy": np.__version__,
        },
        "physics": {
            "model": MODEL_ID,
            "mutual_gravity": False,
            "central_body_fixed": True,
            "test_particles": True,
            "intent": array.intent,
            "stability_statement": _stability_text(array),
        },
        "frame": {
            "id": FRAME_ID,
            "origin": "central body centre of mass",
            "axes": "right-handed, non-rotating, parallel to ICRS X/Y/Z",
            "units": {"length": "m", "time": "s", "mass": "kg", "angle": "rad"},
        },
        "epoch": array.epoch.to_dict(),
        "central_body": array.body.to_dict(),
        "seed": array.seed,
        "config_sha256": array.config_sha256,
        "source_name": array.source_name,
        "warnings": [note.to_dict() for note in array.warnings],
        "orbits": [orbit.to_dict() for orbit in array.orbits],
    }


def write_manifest(array: OrbitArray, path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _dump_json(destination, manifest_dict(array))
    return destination


def load_manifest(path: Path | str) -> OrbitArray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "orbitweaver.manifest":
        raise OrbitWeaverError(f"{path} is not an OrbitWeaver manifest.")
    version = str(payload.get("format_version", ""))
    if not version.startswith("1."):
        raise OrbitWeaverError(f"Unsupported manifest format_version {version!r}.")
    physics = payload.get("physics") or {}
    if physics.get("model") != MODEL_ID:
        raise OrbitWeaverError(
            f"Manifest model is {physics.get('model')!r}. This build propagates {MODEL_ID} only."
        )
    body = CentralBody.from_dict(payload["central_body"])
    validate_body(body)
    if body.frame != FRAME_ID:
        raise OrbitWeaverError(f"Manifest frame {body.frame!r} is not {FRAME_ID}.")
    epoch_payload = payload["epoch"]
    epoch = Epoch.from_dict(epoch_payload)
    orbits = tuple(OrbitRecord.from_dict(item) for item in payload.get("orbits", []))
    for orbit in orbits:
        validate_orbit(orbit, body)
    warnings = tuple(
        WarningNote(
            code=str(item["code"]),
            severity=str(item["severity"]),
            message=str(item["message"]),
            orbit_ids=tuple(item.get("orbit_ids") or ()),
        )
        for item in payload.get("warnings", [])
    )
    return OrbitArray(
        body=body,
        epoch=epoch,
        orbits=orbits,
        warnings=warnings,
        seed=None if payload.get("seed") is None else int(payload["seed"]),
        config_sha256=payload.get("config_sha256"),
        intent=str(physics.get("intent", "science")),
        source_name=payload.get("source_name"),
    )


def write_ephemeris(
    array: OrbitArray,
    times_s: np.ndarray,
    out_dir: Path | str,
    *,
    batch_orbits: int = 8192,
    single_file_limit: int = _SINGLE_FILE_LIMIT,
) -> dict[str, object]:
    """Write requested states. Large products are split on the orbit axis."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    orbits = list(array.orbits)
    product = len(orbits) * int(times.size)
    write_manifest(array, destination / "manifest.json")
    chunks: list[dict[str, object]] = []
    residual_max = 0.0
    use_single = product <= single_file_limit
    if use_single:
        state = propagate_array(array, times, batch_orbits=batch_orbits)
        residual_max = state.solution.max_abs_residual
        _write_state_npz(
            destination / "states.npz",
            times,
            state.position_m,
            state.velocity_m_s,
            state.solution.residual,
        )
        chunks.append(
            {
                "file": "states.npz",
                "orbit_start": 0,
                "orbit_count": len(orbits),
                "time_count": int(times.size),
            }
        )
    else:
        chunk_dir = destination / "chunks"
        chunk_dir.mkdir(exist_ok=True)
        for start in range(0, len(orbits), batch_orbits):
            subset = OrbitArray(
                body=array.body,
                epoch=array.epoch,
                orbits=tuple(orbits[start : start + batch_orbits]),
                warnings=(),
                seed=array.seed,
                config_sha256=array.config_sha256,
                intent=array.intent,
                source_name=array.source_name,
            )
            state = propagate_array(subset, times, batch_orbits=batch_orbits)
            residual_max = max(residual_max, state.solution.max_abs_residual)
            name = f"chunks/{start:06d}.npz"
            _write_state_npz(
                destination / name,
                times,
                state.position_m,
                state.velocity_m_s,
                state.solution.residual,
            )
            chunks.append(
                {
                    "file": name,
                    "orbit_start": start,
                    "orbit_count": len(subset.orbits),
                    "time_count": int(times.size),
                }
            )
    index = {
        "format": "orbitweaver.ephemeris",
        "format_version": FORMAT_VERSION,
        "manifest": "manifest.json",
        "units": {"time": "s", "position": "m", "velocity": "m/s", "angle": "rad"},
        "frame": FRAME_ID,
        "physics_model": MODEL_ID,
        "times_s": times.tolist() if times.size <= 100_000 else None,
        "time_count": int(times.size),
        "time_start_s": float(times[0]) if times.size else None,
        "time_stop_s": float(times[-1]) if times.size else None,
        "satellite_ids": [orbit.satellite_id for orbit in orbits],
        "chunks": chunks,
        "kepler_residual_max_rad": residual_max,
        "single_file": use_single,
        "requested_times_only": True,
        "stability_statement": _stability_text(array),
    }
    _dump_json(destination / "ephemeris_index.json", index)
    return index


def read_state_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {
            "time_s": np.array(archive["time_s"], dtype=np.float64),
            "position_m": np.array(archive["position_m"], dtype=np.float64),
            "velocity_m_s": np.array(archive["velocity_m_s"], dtype=np.float64),
            "kepler_residual": np.array(archive["kepler_residual"], dtype=np.float64),
        }


def _write_state_npz(path: Path, times, position, velocity, residual) -> None:
    np.savez_compressed(
        path,
        time_s=np.asarray(times, dtype=np.float64),
        position_m=np.asarray(position, dtype=np.float64),
        velocity_m_s=np.asarray(velocity, dtype=np.float64),
        kepler_residual=np.asarray(residual, dtype=np.float64),
    )


def _dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _stability_text(array: OrbitArray) -> str:
    for note in array.warnings:
        if note.code == "NOT_A_STABILITY_PROOF":
            return note.message
    from orbitweaver import STABILITY_STATEMENT

    return STABILITY_STATEMENT
