"""Observed propagation timings.

Numbers written by this module are measurements from the process that ran it.
They are not a performance commitment for other machines or array sizes.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from orbitweaver.kepler import propagate_elements


def machine_specs() -> dict[str, object]:
    processor = platform.processor() or _cpu_model()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": processor,
        "cpu_model": _cpu_model(),
        "logical_cpus": _logical_cpus(),
        "memory_bytes": _memory_bytes(),
        "numpy": np.__version__,
    }


def run_benchmark(
    *,
    sizes: tuple[int, ...] = (1_000, 100_000),
    n_times: int = 4,
    repeats: int = 3,
    batch_orbits: int = 8_192,
) -> dict[str, object]:
    """Time batched elliptical propagation. Includes one warmup per size."""
    mu = 3.986004e14
    runs = []
    for size in sizes:
        elements = _elements(size)
        times = np.linspace(0.0, 3600.0, n_times)
        _propagate_batched(elements, mu, times[:1], batch_orbits)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            state = _propagate_batched(elements, mu, times, batch_orbits)
            elapsed = time.perf_counter() - started
            samples.append(
                {
                    "seconds": elapsed,
                    "kepler_residual_max_rad": state["residual"],
                    "position_shape": state["shape"],
                }
            )
        runs.append(
            {
                "n_orbits": size,
                "n_times": n_times,
                "batch_orbits": batch_orbits,
                "repeats_s": [sample["seconds"] for sample in samples],
                "min_s": min(sample["seconds"] for sample in samples),
                "kepler_residual_max_rad": max(sample["kepler_residual_max_rad"] for sample in samples),
            }
        )
    return {
        "observed_only": True,
        "statement": (
            "These are elapsed times measured on the machine described here. "
            "They are not a guarantee of throughput."
        ),
        "machine": machine_specs(),
        "runs": runs,
    }


def write_benchmark(path: Path | str, payload: dict[str, object] | None = None) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = payload if payload is not None else run_benchmark()
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return destination


def _elements(size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "a": rng.uniform(7.0e6, 4.2e7, size),
        "e": rng.uniform(0.0, 0.2, size),
        "inc": np.arccos(rng.uniform(-1.0, 1.0, size)),
        "raan": rng.uniform(0.0, 2.0 * np.pi, size),
        "argp": rng.uniform(0.0, 2.0 * np.pi, size),
        "mean": rng.uniform(0.0, 2.0 * np.pi, size),
    }


def _propagate_batched(elements: dict[str, np.ndarray], mu: float, times: np.ndarray, batch: int) -> dict:
    residual = 0.0
    shape = None
    count = elements["a"].size
    for start in range(0, count, batch):
        stop = min(count, start + batch)
        state = propagate_elements(
            elements["a"][start:stop],
            elements["e"][start:stop],
            elements["inc"][start:stop],
            elements["raan"][start:stop],
            elements["argp"][start:stop],
            elements["mean"][start:stop],
            mu,
            times,
        )
        residual = max(residual, state.solution.max_abs_residual)
        shape = list(state.position_m.shape)
    return {"residual": residual, "shape": shape}


def _cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return platform.processor()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def _logical_cpus() -> int | None:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if line.startswith("processor"))


def _memory_bytes() -> int | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemTotal:"):
            kib = int(line.split()[1])
            return kib * 1024
    return None
