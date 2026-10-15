"""Body, epoch, and orbit records.

Internal numeric fields are float64 SI: metres, seconds, kilograms, radians.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy import constants as astropy_constants

from orbitweaver.constants import (
    CIRCULAR_ECCENTRICITY,
    EQUATORIAL_INCLINATION_RAD,
    FRAME_ID,
)
from orbitweaver.errors import OrbitWeaverError

WARNING_SEVERITIES = ("info", "warning", "error")


@dataclass(frozen=True)
class WarningNote:
    """A named, id-bearing diagnostic attached to a design or a screen."""

    code: str
    severity: str
    message: str
    orbit_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "orbit_ids": list(self.orbit_ids),
        }


@dataclass(frozen=True)
class CentralBody:
    """Point-mass central body and its keep-out radius.

    ``mu_m3_s2`` is the gravitational parameter used by the propagator.
    ``mass_kg`` is recorded for metadata. When both are supplied independently
    they need not satisfy ``mu = G * mass`` to machine precision; physics uses
    ``mu``.
    """

    name: str
    mass_kg: float
    radius_m: float
    mu_m3_s2: float
    exclusion_radius_m: float
    frame: str = FRAME_ID
    mu_source: str = "user"
    mass_source: str = "user"
    radius_source: str = "user"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mass_kg": self.mass_kg,
            "radius_m": self.radius_m,
            "mu_m3_s2": self.mu_m3_s2,
            "exclusion_radius_m": self.exclusion_radius_m,
            "frame": self.frame,
            "mu_source": self.mu_source,
            "mass_source": self.mass_source,
            "radius_source": self.radius_source,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> CentralBody:
        return cls(
            name=str(payload["name"]),
            mass_kg=float(payload["mass_kg"]),
            radius_m=float(payload["radius_m"]),
            mu_m3_s2=float(payload["mu_m3_s2"]),
            exclusion_radius_m=float(payload["exclusion_radius_m"]),
            frame=str(payload.get("frame", FRAME_ID)),
            mu_source=str(payload.get("mu_source", "manifest")),
            mass_source=str(payload.get("mass_source", "manifest")),
            radius_source=str(payload.get("radius_source", "manifest")),
        )


@dataclass(frozen=True)
class Epoch:
    """Absolute epoch. Propagation never reads the calendar, only SI offsets."""

    iso: str
    scale: str
    jd_tt: float

    def to_dict(self) -> dict[str, object]:
        return {
            "iso": self.iso,
            "scale": self.scale,
            "jd_tt": self.jd_tt,
            "offset_definition": (
                "SI seconds after this instant. Leap seconds are not inserted "
                "into propagation offsets."
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Epoch:
        return cls(iso=str(payload["iso"]), scale=str(payload["scale"]), jd_tt=float(payload["jd_tt"]))


@dataclass(frozen=True)
class OrbitRecord:
    """One independently propagated ellipse.

    ``satellite_id`` is immutable once written to a manifest. Angular elements
    are radians. ``element_convention`` records whether those angles sit on a
    classical singularity; the stored numbers are still what the propagator uses.
    """

    satellite_id: str
    a_m: float
    e: float
    inclination_rad: float
    raan_rad: float
    argp_rad: float
    mean_anomaly_rad: float
    body_radius_m: float
    group: str
    shell: str
    element_convention: str
    deliberately_invalid: bool = False
    rendering: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "satellite_id": self.satellite_id,
            "a_m": self.a_m,
            "e": self.e,
            "inclination_rad": self.inclination_rad,
            "raan_rad": self.raan_rad,
            "argp_rad": self.argp_rad,
            "mean_anomaly_rad": self.mean_anomaly_rad,
            "body_radius_m": self.body_radius_m,
            "group": self.group,
            "shell": self.shell,
            "element_convention": self.element_convention,
            "deliberately_invalid": self.deliberately_invalid,
            "rendering": dict(self.rendering),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> OrbitRecord:
        rendering = payload.get("rendering") or {}
        if not isinstance(rendering, dict):
            raise OrbitWeaverError(
                f"Orbit {payload.get('satellite_id')!r} has a non-object rendering record."
            )
        return cls(
            satellite_id=str(payload["satellite_id"]),
            a_m=float(payload["a_m"]),
            e=float(payload["e"]),
            inclination_rad=float(payload["inclination_rad"]),
            raan_rad=float(payload["raan_rad"]),
            argp_rad=float(payload["argp_rad"]),
            mean_anomaly_rad=float(payload["mean_anomaly_rad"]),
            body_radius_m=float(payload["body_radius_m"]),
            group=str(payload.get("group", "")),
            shell=str(payload.get("shell", "")),
            element_convention=str(payload.get("element_convention", "classical")),
            deliberately_invalid=bool(payload.get("deliberately_invalid", False)),
            rendering=dict(rendering),
        )


@dataclass(frozen=True)
class OrbitArray:
    """A designed collection bound to one central body and one epoch."""

    body: CentralBody
    epoch: Epoch
    orbits: tuple[OrbitRecord, ...]
    warnings: tuple[WarningNote, ...]
    seed: int | None
    config_sha256: str | None
    intent: str
    source_name: str | None = None

    def ids(self) -> list[str]:
        return [orbit.satellite_id for orbit in self.orbits]

    def column(self, name: str) -> np.ndarray:
        return np.array([getattr(orbit, name) for orbit in self.orbits], dtype=np.float64)


def element_convention(eccentricity: float, inclination_rad: float) -> str:
    """Classify whether the classical angles are unique.

    The thresholds are documented in ``docs/physics.md``. Classification does
    not rewrite the stored angles.
    """
    circular = eccentricity < CIRCULAR_ECCENTRICITY
    equatorial = min(inclination_rad, np.pi - inclination_rad) < EQUATORIAL_INCLINATION_RAD
    if circular and equatorial:
        return "circular_equatorial"
    if circular:
        return "circular_inclined"
    if equatorial:
        return "elliptical_equatorial"
    return "classical"


def central_body_preset(name: str) -> CentralBody:
    """Build a body from Astropy nominal constants.

    The gravitational parameter is the IAU mass parameter, not ``G * M``
    recomputed by the caller. Astropy's solar and terrestrial masses are
    consistent with those mass parameters and the CODATA gravitational constant.
    """
    key = name.strip().lower()
    constants = astropy_constants
    if key == "sun":
        return CentralBody(
            name="Sun",
            mass_kg=float(constants.M_sun.si.value),
            radius_m=float(constants.R_sun.si.value),
            mu_m3_s2=float(constants.GM_sun.si.value),
            exclusion_radius_m=float(constants.R_sun.si.value),
            frame=FRAME_ID,
            mu_source=f"astropy.constants.GM_sun ({constants.GM_sun.reference})",
            mass_source=f"astropy.constants.M_sun ({constants.M_sun.reference})",
            radius_source=f"astropy.constants.R_sun ({constants.R_sun.reference})",
        )
    if key == "earth":
        return CentralBody(
            name="Earth",
            mass_kg=float(constants.M_earth.si.value),
            radius_m=float(constants.R_earth.si.value),
            mu_m3_s2=float(constants.GM_earth.si.value),
            exclusion_radius_m=float(constants.R_earth.si.value),
            frame=FRAME_ID,
            mu_source=f"astropy.constants.GM_earth ({constants.GM_earth.reference})",
            mass_source=f"astropy.constants.M_earth ({constants.M_earth.reference})",
            radius_source=f"astropy.constants.R_earth ({constants.R_earth.reference})",
        )
    raise OrbitWeaverError(
        f"Unknown central-body preset {name!r}. v0.1 provides 'sun' and 'earth', "
        "or a custom body with an explicit gravitational parameter."
    )


def validate_body(body: CentralBody) -> None:
    if body.frame != FRAME_ID:
        raise OrbitWeaverError(
            f"Central body {body.name!r} uses frame {body.frame!r}. "
            f"v0.1 propagates in {FRAME_ID} only."
        )
    _require_positive("mass_kg", body.mass_kg, body.name)
    _require_finite("radius_m", body.radius_m, body.name)
    if body.radius_m < 0.0:
        raise OrbitWeaverError(f"Central body {body.name!r} has a negative radius.")
    _require_positive("mu_m3_s2", body.mu_m3_s2, body.name)
    _require_finite("exclusion_radius_m", body.exclusion_radius_m, body.name)
    if body.exclusion_radius_m < 0.0:
        raise OrbitWeaverError(f"Central body {body.name!r} has a negative exclusion radius.")


def validate_orbit(orbit: OrbitRecord, body: CentralBody) -> None:
    """Raise ``OrbitWeaverError`` naming ``orbit.satellite_id`` when the record is unusable.

    A pericentre inside the central body or the exclusion radius is rejected
    unless ``deliberately_invalid`` is set for a negative test.
    """
    oid = orbit.satellite_id
    _require_finite("a_m", orbit.a_m, oid)
    _require_finite("e", orbit.e, oid)
    _require_finite("inclination_rad", orbit.inclination_rad, oid)
    _require_finite("raan_rad", orbit.raan_rad, oid)
    _require_finite("argp_rad", orbit.argp_rad, oid)
    _require_finite("mean_anomaly_rad", orbit.mean_anomaly_rad, oid)
    _require_finite("body_radius_m", orbit.body_radius_m, oid)
    if orbit.a_m <= 0.0:
        raise OrbitWeaverError(
            f"Semi-major axis must be > 0 (got {orbit.a_m}).",
            [oid],
        )
    if orbit.e < 0.0 or orbit.e >= 1.0:
        raise OrbitWeaverError(
            f"Eccentricity must satisfy 0 <= e < 1 (got {orbit.e}). "
            "v0.1 propagates bound ellipses only; hyperbolic trajectories are out of scope.",
            [oid],
        )
    if orbit.inclination_rad < 0.0 or orbit.inclination_rad > np.pi:
        raise OrbitWeaverError(
            f"Inclination must lie in [0, pi] radians (got {orbit.inclination_rad}).",
            [oid],
        )
    if orbit.body_radius_m < 0.0:
        raise OrbitWeaverError(f"Body radius must be >= 0 (got {orbit.body_radius_m}).", [oid])
    peri = orbit.a_m * (1.0 - orbit.e)
    keep_out = max(body.exclusion_radius_m, body.radius_m + orbit.body_radius_m)
    if peri <= keep_out and not orbit.deliberately_invalid:
        raise OrbitWeaverError(
            "Pericentre "
            f"{peri:.6e} m is inside the central/exclusion radius "
            f"(central radius {body.radius_m:.6e} m, exclusion {body.exclusion_radius_m:.6e} m, "
            f"satellite radius {orbit.body_radius_m:.6e} m). "
            "Mark the orbit deliberately_invalid only for a negative test.",
            [oid],
        )


def singularity_warnings(orbits: tuple[OrbitRecord, ...] | list[OrbitRecord]) -> list[WarningNote]:
    """One warning per singularity class, listing every affected id."""
    buckets: dict[str, list[str]] = {
        "circular_equatorial": [],
        "circular_inclined": [],
        "elliptical_equatorial": [],
    }
    messages = {
        "circular_equatorial": (
            "Eccentricity and inclination are below the documented singularity "
            "thresholds. The node, argument of periapsis, and mean anomaly are "
            "not unique. The inertial state of a circular equatorial orbit depends "
            "on the true longitude λ = Ω + ω + ν. v0.1 keeps the stored angles. "
            "The Cartesian inverse sets Ω = 0, ω = 0 and stores λ as the mean anomaly."
        ),
        "circular_inclined": (
            "Eccentricity is below the documented circular threshold, so the "
            "argument of periapsis is not unique. The inertial state depends on "
            "the argument of latitude u = ω + ν. v0.1 keeps the stored angles. "
            "The Cartesian inverse sets ω = 0 and stores u as the mean anomaly."
        ),
        "elliptical_equatorial": (
            "Inclination is below the documented equatorial threshold, so the "
            "node and argument of periapsis are not unique separately. The "
            "inertial state depends on the longitude of periapsis ϖ = Ω + ω "
            "(with a reversed in-plane sense when the orbit is retrograde). "
            "v0.1 keeps the stored angles. The Cartesian inverse sets Ω = 0 and "
            "stores ϖ as the argument of periapsis."
        ),
    }
    for orbit in orbits:
        kind = orbit.element_convention
        if kind in buckets:
            buckets[kind].append(orbit.satellite_id)
    notes: list[WarningNote] = []
    for kind, ids in buckets.items():
        if not ids:
            continue
        notes.append(
            WarningNote(
                code=f"ELEMENT_SINGULARITY_{kind.upper()}",
                severity="warning",
                message=messages[kind],
                orbit_ids=tuple(ids),
            )
        )
    return notes


def _require_finite(name: str, value: float, owner: str) -> None:
    if not np.isfinite(value):
        raise OrbitWeaverError(f"{name} is not finite (got {value}).", [owner] if owner else None)


def _require_positive(name: str, value: float, owner: str) -> None:
    _require_finite(name, value, owner)
    if value <= 0.0:
        raise OrbitWeaverError(f"{name} must be > 0 (got {value}).", [owner] if owner else None)
