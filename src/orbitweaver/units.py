"""SI internals and strict parsing at the public boundary.

Computations use float64 SI quantities. ``astropy.units`` and ``astropy.time``
are used where a user string or an absolute epoch enters the program.
"""

from __future__ import annotations

import re

import astropy.units as u
import numpy as np
from astropy.time import Time

from orbitweaver.errors import ConfigError

_DURATION = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]+)\s*$"
)

# Julian year matches astropy ``u.yr`` (365.25 days of exactly 86400 SI seconds).
_DURATION_UNITS: dict[str, u.Unit] = {
    "s": u.s,
    "sec": u.s,
    "second": u.s,
    "seconds": u.s,
    "min": u.min,
    "mins": u.min,
    "minute": u.min,
    "minutes": u.min,
    "h": u.h,
    "hr": u.h,
    "hrs": u.h,
    "hour": u.h,
    "hours": u.h,
    "d": u.d,
    "day": u.d,
    "days": u.d,
    "yr": u.yr,
    "yrs": u.yr,
    "year": u.yr,
    "years": u.yr,
    "julianyear": u.yr,
}

_SCALES = ("tai", "tcb", "tcg", "tdb", "tt", "ut1", "utc")


def parse_duration_seconds(text: str) -> float:
    """Parse a duration into SI seconds.

    A unit is mandatory. ``365`` and ``365.0`` are rejected so a bare count
    cannot be mistaken for days or seconds.
    """
    if not isinstance(text, str):
        raise ConfigError(
            "Duration must be a string with a unit, for example '365d' "
            f"(got {type(text).__name__})."
        )
    match = _DURATION.match(text.strip())
    if match is None:
        raise ConfigError(
            f"Ambiguous or unitless time expression {text!r}. "
            "Use a number and a unit: s, min, h, d, or yr (Julian year, 365.25 d)."
        )
    magnitude = float(match.group(1))
    token = match.group(2).lower()
    unit = _DURATION_UNITS.get(token)
    if unit is None:
        raise ConfigError(
            f"Unrecognised time unit {match.group(2)!r} in {text!r}. "
            "Accepted units: s, min, h, d, yr."
        )
    if magnitude < 0.0:
        raise ConfigError(f"Duration must be >= 0, got {text!r}.")
    seconds = float((magnitude * unit).to(u.s).value)
    if not _finite(seconds):
        raise ConfigError(f"Duration {text!r} is not finite.")
    return seconds


def parse_length_m(value: object, field: str) -> float:
    """Parse a length to metres.

    A bare number is metres. A string must carry a length unit
    (``'1.0 AU'``, ``'7000 km'``). Dimensionless strings are rejected.
    """
    if isinstance(value, bool) or isinstance(value, str) and not value.strip():
        raise ConfigError(f"{field} must be a length in metres or a string with a unit.")
    if isinstance(value, str):
        try:
            quantity = u.Quantity(value.strip())
        except Exception as exc:
            raise ConfigError(f"{field}: could not parse length {value!r}: {exc}") from exc
        if quantity.unit.is_equivalent(u.one) and not quantity.unit.is_equivalent(u.m):
            raise ConfigError(f"{field}: unitless length {value!r} is ambiguous.")
        if not quantity.unit.is_equivalent(u.m):
            raise ConfigError(f"{field}: {value!r} is not a length.")
        metres = float(quantity.to(u.m).value)
    elif isinstance(value, (int, float)):
        metres = float(value)
    else:
        raise ConfigError(
            f"{field} must be metres (a number) or a string with a unit, "
            f"got {type(value).__name__}."
        )
    if not _finite(metres):
        raise ConfigError(f"{field} is not finite.")
    return metres


def parse_epoch(iso: str, scale: str) -> Time:
    """Parse an absolute epoch. The time scale is required."""
    if not isinstance(scale, str) or scale.lower() not in _SCALES:
        raise ConfigError(
            f"Epoch scale must be one of {list(_SCALES)}, got {scale!r}. "
            "A civil date without a scale mixes UTC, TT, and TDB."
        )
    if not isinstance(iso, str) or not iso.strip():
        raise ConfigError("Epoch iso must be a non-empty ISO-8601 string.")
    try:
        epoch = Time(iso.strip(), format="isot", scale=scale.lower())
    except Exception as exc:
        raise ConfigError(f"Could not parse epoch {iso!r} on scale {scale!r}: {exc}") from exc
    return epoch


def epoch_to_dict(epoch: Time) -> dict[str, object]:
    """JSON record of an absolute epoch.

    Propagator offsets are SI seconds after this instant. They are not UTC
    clock strings and they do not insert leap seconds.
    """
    tt = epoch.tt
    return {
        "iso": epoch.isot,
        "scale": str(epoch.scale),
        "jd_tt": float(tt.jd),
        "offset_definition": (
            "SI seconds after this instant. Leap seconds are not inserted "
            "into propagation offsets."
        ),
    }


def time_grid_seconds(duration_s: float, step_s: float) -> np.ndarray:
    """Inclusive grid from 0 through ``duration_s``.

    The last sample is the requested duration when the duration is an integer
    number of steps within a small tolerance. Otherwise the endpoint is added
    and the final gap is shorter than ``step_s``.
    """
    if not _finite(duration_s) or not _finite(step_s):
        raise ConfigError("Duration and step must be finite.")
    if duration_s < 0.0:
        raise ConfigError("Duration must be >= 0.")
    if step_s <= 0.0:
        raise ConfigError("Step must be > 0.")
    if duration_s == 0.0:
        return np.array([0.0], dtype=np.float64)
    n_steps = int(np.round(duration_s / step_s))
    if n_steps < 1:
        return np.array([0.0, duration_s], dtype=np.float64)
    times = np.arange(n_steps + 1, dtype=np.float64) * step_s
    if abs(times[-1] - duration_s) <= 1e-9 * max(duration_s, 1.0):
        times[-1] = duration_s
    elif times[-1] < duration_s:
        times = np.append(times, duration_s)
    return times


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
