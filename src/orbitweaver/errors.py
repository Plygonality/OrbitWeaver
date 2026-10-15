"""User-facing errors that name the orbits involved."""

from __future__ import annotations


class OrbitWeaverError(Exception):
    """A failed check, with the satellite ids that triggered it when known."""

    def __init__(self, message: str, orbit_ids: list[str] | tuple[str, ...] | None = None) -> None:
        self.orbit_ids = tuple(orbit_ids or ())
        super().__init__(message)

    def __str__(self) -> str:
        text = super().__str__()
        if self.orbit_ids:
            listed = ", ".join(self.orbit_ids[:12])
            extra = len(self.orbit_ids) - 12
            suffix = f" (+{extra} more)" if extra > 0 else ""
            return f"{text} Offending orbit ids: {listed}{suffix}."
        return text


class ConfigError(OrbitWeaverError):
    """Invalid configuration, unit, or time expression."""
