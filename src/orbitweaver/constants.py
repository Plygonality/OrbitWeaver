"""Identifiers, singularity thresholds, and solver limits for v0.1."""

from __future__ import annotations

FORMAT_VERSION = "1.0.0"
MODEL_ID = "newtonian_two_body_kepler"
FRAME_ID = "OW_INERTIAL_V1"

# Documented classification thresholds. Below these, the classical angular
# elements are not unique. See docs/physics.md.
CIRCULAR_ECCENTRICITY = 1e-8
EQUATORIAL_INCLINATION_RAD = 1e-8

# Numerical branch used by the Cartesian inverse. Tighter than the warning
# thresholds so a merely small eccentricity is not rewritten.
INVERSE_CIRCULAR_ECCENTRICITY = 1e-12
INVERSE_NODE_RATIO = 1e-12

# Internal Halley step aims at this residual on the folded mean anomaly.
KEPLER_ABS_TOL = 1e-14
KEPLER_MAX_ITER = 80
# Propagation fails if the folded residual stays above this bound.
KEPLER_FAIL_RESIDUAL = 1e-10
