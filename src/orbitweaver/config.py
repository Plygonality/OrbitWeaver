"""TOML and JSON design files.

Angle fields are degrees. Lengths are metres when numeric, or unit-bearing
strings such as ``'1.0 AU'``. Time scales and duration units are mandatory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orbitweaver.constants import FORMAT_VERSION, FRAME_ID, MODEL_ID
from orbitweaver.errors import ConfigError
from orbitweaver.models import CentralBody, Epoch, central_body_preset, validate_body
from orbitweaver.units import parse_epoch, parse_length_m

_TOP_KEYS = {
    "format",
    "format_version",
    "physics",
    "epoch",
    "central_body",
    "array",
    "shells",
    "orbits",
}
_SHELL_KEYS = {
    "name",
    "kind",
    "count",
    "group",
    "body_radius_m",
    "a_m",
    "e",
    "inclination",
    "raan",
    "argp",
    "mean_anomaly",
    "rendering",
    "deliberately_invalid",
    "id_prefix",
}
_ORBIT_KEYS = {
    "id",
    "satellite_id",
    "a_m",
    "e",
    "inclination_deg",
    "raan_deg",
    "argp_deg",
    "mean_anomaly_deg",
    "body_radius_m",
    "group",
    "deliberately_invalid",
    "rendering",
    "shell",
}
_KINDS = {"shell", "ring_family", "inclination_family"}
_ANGLE_MODES = {"fixed", "uniform", "range", "phased", "list", "isotropic"}


@dataclass(frozen=True)
class RangeSpec:
    low: float
    high: float


@dataclass(frozen=True)
class AngleSpec:
    mode: str
    low: float = 0.0
    high: float = 2.0 * np.pi
    value: float = 0.0
    values: tuple[float, ...] = ()
    jitter: float = 0.0
    offset: float = 0.0


@dataclass(frozen=True)
class ExplicitOrbitSpec:
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
    deliberately_invalid: bool
    rendering: dict[str, object]


@dataclass(frozen=True)
class ShellSpec:
    name: str
    kind: str
    count: int
    group: str
    body_radius_m: float
    id_prefix: str
    a: RangeSpec
    eccentricity: RangeSpec
    inclination: AngleSpec
    raan: AngleSpec
    argp: AngleSpec
    mean_anomaly: AngleSpec
    phase_within_plane: bool
    rendering: dict[str, object]
    deliberately_invalid: bool


@dataclass(frozen=True)
class DesignConfig:
    body: CentralBody
    epoch: Epoch
    seed: int
    id_prefix: str
    min_initial_separation_m: float
    max_placement_attempts: int
    allow_shortfall: bool
    allow_initial_overlap: bool
    acknowledge_large_array: bool
    shells: tuple[ShellSpec, ...]
    explicit: tuple[ExplicitOrbitSpec, ...]
    intent: str
    physics_note: str
    config_sha256: str
    source_name: str | None


def load_config(path: Path | str) -> DesignConfig:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"Config file not found: {file_path}")
    raw = file_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {file_path}: {exc}") from exc
    elif suffix == ".toml":
        import tomllib

        try:
            payload = tomllib.loads(text)
        except Exception as exc:
            raise ConfigError(f"Invalid TOML in {file_path}: {exc}") from exc
    else:
        raise ConfigError(f"Config {file_path} must be .toml or .json.")
    if not isinstance(payload, dict):
        raise ConfigError(f"Config {file_path} must be a table at the top level.")
    return parse_config(payload, source_name=str(file_path), config_sha256=digest)


def parse_config(payload: dict, *, source_name: str | None, config_sha256: str) -> DesignConfig:
    unknown = set(payload) - _TOP_KEYS
    if unknown:
        raise ConfigError(f"Unknown config keys: {sorted(unknown)}.")
    _require_format(payload)
    physics = payload.get("physics") or {}
    if not isinstance(physics, dict):
        raise ConfigError("physics must be a table.")
    model = str(physics.get("model", MODEL_ID))
    if model != MODEL_ID:
        raise ConfigError(
            f"physics.model must be {MODEL_ID!r} in v0.1 (got {model!r}). "
            "N-body, relativistic, and rigidly connected rings are not this model."
        )
    intent = str(physics.get("intent", "science"))
    if intent not in {"science", "visual_only"}:
        raise ConfigError("physics.intent must be 'science' or 'visual_only'.")
    note = str(physics.get("note", ""))
    epoch = _parse_epoch_table(payload.get("epoch"))
    body = _parse_body(payload.get("central_body"))
    array = payload.get("array") or {}
    if not isinstance(array, dict):
        raise ConfigError("array must be a table.")
    seed = _as_int(array.get("seed", 0), "array.seed")
    prefix = str(array.get("id_prefix", "SAT"))
    if not prefix:
        raise ConfigError("array.id_prefix must be a non-empty string.")
    min_sep = parse_length_m(array.get("min_initial_separation_m", 0.0), "array.min_initial_separation_m")
    if min_sep < 0.0:
        raise ConfigError("array.min_initial_separation_m must be >= 0.")
    attempts = _as_int(array.get("max_placement_attempts", 32), "array.max_placement_attempts")
    if attempts < 1:
        raise ConfigError("array.max_placement_attempts must be >= 1.")
    shells_raw = payload.get("shells") or []
    orbits_raw = payload.get("orbits") or []
    if not isinstance(shells_raw, list) or not isinstance(orbits_raw, list):
        raise ConfigError("shells and orbits must be arrays.")
    if not shells_raw and not orbits_raw:
        raise ConfigError("Config must contain at least one shell or one explicit orbit.")
    shells = tuple(
        _parse_shell(item, index, prefix) for index, item in enumerate(shells_raw)
    )
    explicit = tuple(_parse_explicit(item, index) for index, item in enumerate(orbits_raw))
    total = sum(shell.count for shell in shells) + len(explicit)
    acknowledge = bool(array.get("acknowledge_large_array", False))
    if total > 2_000_000:
        raise ConfigError(f"Array count {total} exceeds the v0.1 designer limit of 2000000.")
    if total > 100_000 and not acknowledge:
        raise ConfigError(
            f"Array count {total} is above 100000. Set array.acknowledge_large_array = true "
            "to confirm the runtime and memory cost."
        )
    return DesignConfig(
        body=body,
        epoch=epoch,
        seed=seed,
        id_prefix=prefix,
        min_initial_separation_m=min_sep,
        max_placement_attempts=attempts,
        allow_shortfall=bool(array.get("allow_shortfall", False)),
        allow_initial_overlap=bool(array.get("allow_initial_overlap", False)),
        acknowledge_large_array=acknowledge,
        shells=shells,
        explicit=explicit,
        intent=intent,
        physics_note=note,
        config_sha256=config_sha256,
        source_name=source_name,
    )


def _require_format(payload: dict) -> None:
    marker = payload.get("format")
    version = str(payload.get("format_version", ""))
    if marker != "orbitweaver.config":
        raise ConfigError("format must be 'orbitweaver.config'.")
    if not version.startswith("1."):
        raise ConfigError(
            f"format_version {version!r} is not supported. This build reads {FORMAT_VERSION}."
        )


def _parse_epoch_table(payload: object) -> Epoch:
    if not isinstance(payload, dict):
        raise ConfigError("epoch must be a table with iso and scale.")
    if "scale" not in payload:
        raise ConfigError("epoch.scale is required (tt, tdb, utc, tai, ut1, tcg, or tcb).")
    stamp = parse_epoch(str(payload.get("iso", "")), str(payload["scale"]))
    return Epoch(iso=stamp.isot, scale=str(stamp.scale), jd_tt=float(stamp.tt.jd))


def _parse_body(payload: object) -> CentralBody:
    if not isinstance(payload, dict):
        raise ConfigError("central_body must be a table.")
    preset = payload.get("preset")
    if preset and str(preset).lower() != "custom":
        body = central_body_preset(str(preset))
        if "exclusion_radius_m" in payload:
            body = CentralBody(
                name=body.name,
                mass_kg=body.mass_kg,
                radius_m=body.radius_m,
                mu_m3_s2=body.mu_m3_s2,
                exclusion_radius_m=parse_length_m(payload["exclusion_radius_m"], "exclusion_radius_m"),
                frame=body.frame,
                mu_source=body.mu_source,
                mass_source=body.mass_source,
                radius_source=body.radius_source,
            )
        if "frame" in payload and str(payload["frame"]) != FRAME_ID:
            raise ConfigError(
                f"Unsupported frame {payload['frame']!r}. v0.1 uses {FRAME_ID} only."
            )
        validate_body(body)
        return body
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ConfigError("A custom central_body needs a name.")
    if "mass_kg" not in payload or "radius_m" not in payload:
        raise ConfigError("A custom central_body needs mass_kg and radius_m.")
    mass = float(payload["mass_kg"])
    radius = parse_length_m(payload["radius_m"], "central_body.radius_m")
    if "mu_m3_s2" in payload:
        mu = float(payload["mu_m3_s2"])
        mu_source = "config.mu_m3_s2"
    else:
        from astropy import constants as astropy_constants

        mu = float(astropy_constants.G.si.value) * mass
        mu_source = f"astropy.constants.G ({astropy_constants.G.reference}) * mass_kg"
    exclusion = parse_length_m(
        payload.get("exclusion_radius_m", radius),
        "central_body.exclusion_radius_m",
    )
    frame = str(payload.get("frame", FRAME_ID))
    if frame != FRAME_ID:
        raise ConfigError(f"Unsupported frame {frame!r}. v0.1 uses {FRAME_ID} only.")
    body = CentralBody(
        name=name,
        mass_kg=mass,
        radius_m=radius,
        mu_m3_s2=mu,
        exclusion_radius_m=exclusion,
        frame=frame,
        mu_source=mu_source,
        mass_source="config.mass_kg",
        radius_source="config.radius_m",
    )
    validate_body(body)
    return body


def _parse_shell(payload: object, index: int, default_prefix: str) -> ShellSpec:
    field = f"shells[{index}]"
    if not isinstance(payload, dict):
        raise ConfigError(f"{field} must be a table.")
    unknown = set(payload) - _SHELL_KEYS
    if unknown:
        raise ConfigError(f"{field} has unknown keys: {sorted(unknown)}.")
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ConfigError(f"{field}.name is required.")
    kind = str(payload.get("kind", "shell"))
    if kind not in _KINDS:
        raise ConfigError(f"{field}.kind must be one of {sorted(_KINDS)}.")
    count = _as_int(payload.get("count"), f"{field}.count")
    if count < 1:
        raise ConfigError(f"{field}.count must be >= 1.")
    group = str(payload.get("group", name))
    radius = parse_length_m(payload.get("body_radius_m", 0.0), f"{field}.body_radius_m")
    if radius < 0.0:
        raise ConfigError(f"{field}.body_radius_m must be >= 0.")
    if "a_m" not in payload:
        raise ConfigError(f"{field}.a_m is required.")
    semi_major = _parse_positive_range(payload["a_m"], f"{field}.a_m")
    eccentricity = _parse_unit_range(payload.get("e", 0.0), f"{field}.e", low_bound=0.0, high_bound=1.0)
    if eccentricity.high >= 1.0:
        raise ConfigError(f"{field}.e must satisfy 0 <= e < 1.")
    inclination = _parse_angle(
        payload.get("inclination", {"mode": "fixed", "deg": 0.0}),
        f"{field}.inclination",
        allow_isotropic=True,
    )
    _check_inclination(inclination, f"{field}.inclination")
    raan = _parse_angle(payload.get("raan", {"mode": "uniform"}), f"{field}.raan", allow_isotropic=False)
    argp = _parse_angle(payload.get("argp", {"mode": "uniform"}), f"{field}.argp", allow_isotropic=False)
    if "mean_anomaly" in payload:
        mean_anomaly = _parse_angle(payload["mean_anomaly"], f"{field}.mean_anomaly", allow_isotropic=False)
    elif kind == "ring_family":
        mean_anomaly = AngleSpec(mode="phased")
    else:
        mean_anomaly = _parse_angle({"mode": "uniform"}, f"{field}.mean_anomaly", allow_isotropic=False)
    if kind == "ring_family":
        if semi_major.low != semi_major.high or eccentricity.low != eccentricity.high:
            raise ConfigError(
                f"{field} is a ring_family, so a_m and e must be single values. "
                "A ring family is a set of independent ellipses, not a radial band."
            )
        if inclination.mode != "fixed":
            raise ConfigError(f"{field} is a ring_family, so inclination must be fixed.")
    if kind == "inclination_family" and inclination.mode != "list":
        raise ConfigError(f"{field} is an inclination_family, so inclination mode must be 'list'.")
    phase_within_plane = inclination.mode == "list"
    rendering = _rendering(payload.get("rendering", {}), f"{field}.rendering")
    return ShellSpec(
        name=name,
        kind=kind,
        count=count,
        group=group,
        body_radius_m=radius,
        id_prefix=str(payload.get("id_prefix", default_prefix)),
        a=semi_major,
        eccentricity=eccentricity,
        inclination=inclination,
        raan=raan,
        argp=argp,
        mean_anomaly=mean_anomaly,
        phase_within_plane=phase_within_plane,
        rendering=rendering,
        deliberately_invalid=bool(payload.get("deliberately_invalid", False)),
    )


def _parse_explicit(payload: object, index: int) -> ExplicitOrbitSpec:
    field = f"orbits[{index}]"
    if not isinstance(payload, dict):
        raise ConfigError(f"{field} must be a table.")
    unknown = set(payload) - _ORBIT_KEYS
    if unknown:
        raise ConfigError(f"{field} has unknown keys: {sorted(unknown)}.")
    satellite_id = str(payload.get("satellite_id", payload.get("id", ""))).strip()
    if not satellite_id:
        raise ConfigError(f"{field} needs id.")
    if "a_m" not in payload or "e" not in payload:
        raise ConfigError(f"{field} needs a_m and e.")
    return ExplicitOrbitSpec(
        satellite_id=satellite_id,
        a_m=parse_length_m(payload["a_m"], f"{field}.a_m"),
        e=float(payload["e"]),
        inclination_rad=_degrees(payload.get("inclination_deg", 0.0), f"{field}.inclination_deg"),
        raan_rad=_degrees(payload.get("raan_deg", 0.0), f"{field}.raan_deg"),
        argp_rad=_degrees(payload.get("argp_deg", 0.0), f"{field}.argp_deg"),
        mean_anomaly_rad=_degrees(payload.get("mean_anomaly_deg", 0.0), f"{field}.mean_anomaly_deg"),
        body_radius_m=parse_length_m(payload.get("body_radius_m", 0.0), f"{field}.body_radius_m"),
        group=str(payload.get("group", "explicit")),
        shell=str(payload.get("shell", "explicit")),
        deliberately_invalid=bool(payload.get("deliberately_invalid", False)),
        rendering=_rendering(payload.get("rendering", {}), f"{field}.rendering"),
    )


def _parse_positive_range(value: object, field: str) -> RangeSpec:
    low, high = _parse_range_values(value, field, parse_length_m)
    if low <= 0.0 or high <= 0.0:
        raise ConfigError(f"{field} must be > 0.")
    if high < low:
        raise ConfigError(f"{field} max is below min.")
    return RangeSpec(low, high)


def _parse_unit_range(value: object, field: str, *, low_bound: float, high_bound: float) -> RangeSpec:
    def _plain(item: object, item_field: str) -> float:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError(f"{item_field} must be a number.")
        return float(item)

    low, high = _parse_range_values(value, field, _plain)
    if low < low_bound or high > high_bound:
        raise ConfigError(f"{field} must lie in [{low_bound}, {high_bound}].")
    if high < low:
        raise ConfigError(f"{field} max is below min.")
    return RangeSpec(low, high)


def _parse_range_values(value: object, field: str, convert) -> tuple[float, float]:
    if isinstance(value, (int, float, str)) and not isinstance(value, bool):
        number = convert(value, field)
        return number, number
    if isinstance(value, list):
        if len(value) != 2:
            raise ConfigError(f"{field} range must have two values.")
        return convert(value[0], field), convert(value[1], field)
    if isinstance(value, dict):
        if "min" in value or "max" in value:
            if "min" not in value or "max" not in value:
                raise ConfigError(f"{field} needs both min and max.")
            return convert(value["min"], f"{field}.min"), convert(value["max"], f"{field}.max")
        if "value" in value:
            number = convert(value["value"], field)
            return number, number
    raise ConfigError(f"{field} must be a number, a unit string, or a min/max table.")


def _parse_angle(value: object, field: str, *, allow_isotropic: bool) -> AngleSpec:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be a table with a mode.")
    mode = str(value.get("mode", ""))
    if mode not in _ANGLE_MODES:
        raise ConfigError(f"{field}.mode must be one of {sorted(_ANGLE_MODES)}.")
    if mode == "isotropic":
        if not allow_isotropic:
            raise ConfigError(f"{field} does not accept mode 'isotropic'.")
        return AngleSpec(mode="isotropic")
    if mode == "fixed":
        return AngleSpec(mode="fixed", value=_degrees(value.get("deg", 0.0), f"{field}.deg"))
    if mode == "uniform":
        if "min_deg" in value or "max_deg" in value:
            low = _degrees(value.get("min_deg", 0.0), f"{field}.min_deg")
            high = _degrees(value.get("max_deg", 360.0), f"{field}.max_deg")
        else:
            low, high = 0.0, 2.0 * np.pi
        if high < low:
            raise ConfigError(f"{field} max_deg is below min_deg.")
        return AngleSpec(mode="uniform", low=low, high=high)
    if mode == "range":
        low = _degrees(value["min_deg"], f"{field}.min_deg") if "min_deg" in value else None
        if low is None:
            raise ConfigError(f"{field}.range needs min_deg and max_deg.")
        high = _degrees(value["max_deg"], f"{field}.max_deg")
        if high < low:
            raise ConfigError(f"{field} max_deg is below min_deg.")
        return AngleSpec(mode="range", low=low, high=high)
    if mode == "phased":
        return AngleSpec(
            mode="phased",
            jitter=_degrees(value.get("jitter_deg", 0.0), f"{field}.jitter_deg"),
            offset=_degrees(value.get("offset_deg", 0.0), f"{field}.offset_deg"),
        )
    values = value.get("deg")
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{field}.list needs a non-empty deg array.")
    return AngleSpec(mode="list", values=tuple(_degrees(item, f"{field}.deg") for item in values))


def _check_inclination(spec: AngleSpec, field: str) -> None:
    if spec.mode == "isotropic":
        return
    if spec.mode == "phased":
        raise ConfigError(f"{field} cannot be phased. Use fixed, range, list, or isotropic.")
    values: list[float] = []
    if spec.mode == "fixed":
        values = [spec.value]
    elif spec.mode in {"range", "uniform"}:
        values = [spec.low, spec.high]
    elif spec.mode == "list":
        values = list(spec.values)
    if any(item < -1e-12 or item > np.pi + 1e-12 for item in values):
        raise ConfigError(f"{field} must stay inside [0, 180] degrees.")


def _rendering(value: object, field: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be a table of strings and numbers.")
    cleaned: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(item, bool) or isinstance(item, (int, float, str)):
            cleaned[str(key)] = item
        else:
            raise ConfigError(f"{field}.{key} must be a string, number, or boolean.")
    return cleaned


def _degrees(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number of degrees.")
    return float(value) * np.pi / 180.0


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer.")
    return int(value)
