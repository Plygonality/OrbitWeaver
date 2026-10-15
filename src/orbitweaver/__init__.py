"""OrbitWeaver: Newtonian Keplerian layouts for large orbital arrays.

Scientific propagation lives in this package. The Blender add-on is separate
and is not imported here.
"""

from __future__ import annotations

__version__ = "0.1.0"

STABILITY_STATEMENT = (
    "OrbitWeaver v0.1 propagates independent test particles on bound Keplerian "
    "ellipses about a fixed point mass. A manifest, a sampled ephemeris, or a "
    "symmetric layout is a design and a screen over stated times. It is not a "
    "proof of long-term dynamical stability, collision avoidance for all time, "
    "or of behaviour under mutual gravity, oblateness, drag, radiation pressure, "
    "station-keeping, or relativity."
)
