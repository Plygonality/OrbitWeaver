"""Command line for designing, propagating, screening, and reporting arrays."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from orbitweaver import __version__
from orbitweaver.array_design import generate_array
from orbitweaver.benchmark import run_benchmark, write_benchmark
from orbitweaver.config import load_config
from orbitweaver.errors import OrbitWeaverError
from orbitweaver.export import load_manifest, read_state_npz, write_ephemeris, write_manifest
from orbitweaver.kepler import orbital_period, propagate_elements
from orbitweaver.rebound_check import rebound_status
from orbitweaver.report import write_design_report, write_screening_report
from orbitweaver.screening import screen_array
from orbitweaver.units import parse_duration_seconds, parse_length_m, time_grid_seconds
from orbitweaver.visualisation import (
    floating_origin_positions,
    precision_note,
    write_preview_cache,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orbitweaver", description=__doc__)
    parser.add_argument("--version", action="version", version=f"orbitweaver {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Design an array from TOML or JSON.")
    generate.add_argument("config")
    generate.add_argument("--out", required=True)
    generate.set_defaults(func=_generate)

    propagate = commands.add_parser("propagate", help="Write states at the requested times.")
    propagate.add_argument("manifest")
    propagate.add_argument("--duration", required=True)
    propagate.add_argument("--step", required=True)
    propagate.add_argument("--out", required=True)
    propagate.add_argument("--preview-metres-per-unit", type=float, default=None)
    propagate.add_argument("--floating-origin-id", default=None)
    propagate.set_defaults(func=_propagate)

    screen = commands.add_parser("screen", help="Screen a manifest over a finite time grid.")
    screen.add_argument("manifest")
    screen.add_argument("--duration", required=True)
    screen.add_argument("--step", required=True)
    screen.add_argument("--out", required=True)
    screen.add_argument("--distance", default=None, help="Candidate radius, metres or a unit string.")
    screen.add_argument("--no-refine", action="store_true")
    screen.set_defaults(func=_screen)

    report = commands.add_parser("report", help="Write an HTML design report.")
    report.add_argument("manifest")
    report.add_argument("--out", required=True)
    report.set_defaults(func=_report)

    doctor = commands.add_parser("doctor", help="Check imports and a numeric self-test.")
    doctor.set_defaults(func=_doctor)

    benchmark = commands.add_parser("benchmark", help="Record observed propagation timings.")
    benchmark.add_argument("--out", required=True)
    benchmark.set_defaults(func=_benchmark)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except OrbitWeaverError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _generate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    array = generate_array(config)
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    path = write_manifest(array, destination / "manifest.json")
    print(f"wrote {path} ({len(array.orbits)} orbits)")


def _propagate(args: argparse.Namespace) -> None:
    array = load_manifest(args.manifest)
    times = time_grid_seconds(
        parse_duration_seconds(args.duration),
        parse_duration_seconds(args.step),
    )
    index = write_ephemeris(array, times, args.out)
    print(
        f"wrote {args.out} ({index['time_count']} times, "
        f"{len(index['satellite_ids'])} orbits, "
        f"kepler residual {index['kepler_residual_max_rad']:.3e} rad)"
    )
    if args.preview_metres_per_unit is None:
        return
    if not index["single_file"]:
        raise OrbitWeaverError(
            "Preview cache is written for a single-file ephemeris. "
            "Shorten the time grid or raise the chunk threshold."
        )
    archive = read_state_npz(Path(args.out) / "states.npz")
    origin_index = None
    if args.floating_origin_id:
        try:
            origin_index = index["satellite_ids"].index(args.floating_origin_id)
        except ValueError as exc:
            raise OrbitWeaverError(
                f"Floating origin id {args.floating_origin_id!r} is not in the manifest.",
                [args.floating_origin_id],
            ) from exc
    render = floating_origin_positions(
        archive["position_m"],
        metres_per_unit=args.preview_metres_per_unit,
        origin_index=origin_index,
    )
    characteristic = float(np.max(np.linalg.norm(archive["position_m"], axis=-1)))
    note = precision_note(
        0.0 if origin_index is not None else characteristic,
        args.preview_metres_per_unit,
        float(max(orbit.body_radius_m for orbit in array.orbits)),
    )
    sidecar = write_preview_cache(
        Path(args.out) / "preview.owbin",
        render,
        archive["time_s"],
        metres_per_unit=args.preview_metres_per_unit,
        origin_m=np.zeros(3),
        satellite_ids=index["satellite_ids"],
        manifest_name="manifest.json",
    )
    sidecar["precision"] = note
    sidecar["floating_origin_id"] = args.floating_origin_id
    (Path(args.out) / "preview.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote preview cache; float32 step at the reference distance is "
        f"{note['float32_step_m']:.6e} m"
    )


def _screen(args: argparse.Namespace) -> None:
    array = load_manifest(args.manifest)
    times = time_grid_seconds(
        parse_duration_seconds(args.duration),
        parse_duration_seconds(args.step),
    )
    distance = None if args.distance is None else parse_length_m(args.distance, "--distance")

    def progress(done: int, total: int) -> None:
        if done == total or done % max(total // 10, 1) == 0:
            print(f"screen: sample {done}/{total}", file=sys.stderr)

    result = screen_array(
        array,
        times,
        search_radius_m=distance,
        refine=not args.no_refine,
        progress=progress,
    )
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "encounters.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_screening_report(result, destination / "encounters.html")
    certificate = result["guarantee"]["certificate"]
    minimum = result["minimum_separation"]["metres"]
    print(f"wrote {destination} certificate={certificate} reported_min_m={minimum}")


def _report(args: argparse.Namespace) -> None:
    array = load_manifest(args.manifest)
    path = write_design_report(array, args.out)
    print(f"wrote {path}")


def _doctor(_args: argparse.Namespace) -> None:
    import astropy
    import scipy

    check = _numeric_self_check()
    rebound = rebound_status()
    blender = shutil.which("blender")
    payload = {
        "orbitweaver": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "astropy": astropy.__version__,
        "self_check": check,
        "rebound": rebound,
        "blender": {
            "available": blender is not None,
            "path": blender,
            "preview_status": "available" if blender else "unverified",
        },
        "physics_model": "newtonian_two_body_kepler",
        "frame": "OW_INERTIAL_V1",
    }
    print(json.dumps(payload, indent=2))
    if not check["passed"]:
        raise OrbitWeaverError("Numeric self-check failed.")


def _benchmark(args: argparse.Namespace) -> None:
    payload = run_benchmark()
    path = write_benchmark(args.out, payload)
    print(f"wrote {path}")
    for run in payload["runs"]:
        print(f"n={run['n_orbits']} min_s={run['min_s']:.4f} repeats_s={run['repeats_s']}")


def _numeric_self_check() -> dict[str, object]:
    mu = 1.0
    semi_major = 1.0
    period = float(orbital_period(np.array([semi_major]), mu)[0])
    state = propagate_elements([semi_major], [0.0], [0.0], [0.0], [0.0], [0.0], mu, [0.0, period])
    closure = float(np.max(np.abs(state.position_m[1, 0] - state.position_m[0, 0])))
    residual = state.solution.max_abs_residual
    return {
        "circular_closure_m": closure,
        "kepler_residual_max_rad": residual,
        "passed": closure < 1e-12 and residual < 1e-12,
    }
