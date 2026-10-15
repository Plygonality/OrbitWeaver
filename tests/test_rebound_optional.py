"""Optional IAS15 comparison.

A missing REBOUND install is unverified, which is not a pass. When the import
succeeds, one circular and one high-eccentricity ellipse must match.
"""

from __future__ import annotations

import numpy as np

from orbitweaver.rebound_check import compare_ias15, rebound_status


def test_rebound_is_unverified_or_matches_ias15():
    status = rebound_status()
    if not status["available"]:
        result = compare_ias15(
            mass_kg=5.972168e24,
            a_m=6_778_137.0,
            eccentricity=0.0,
            inclination_rad=0.0,
            raan_rad=0.0,
            argp_rad=0.0,
            mean_anomaly_rad=0.0,
            times_s=np.array([0.0, 1000.0]),
        )
        assert result["status"] == "unverified"
        assert result["passed"] is False
        assert result["max_position_error_m"] is None
        assert result["integrator_used"] is None
        return

    circular = compare_ias15(
        mass_kg=5.972168e24,
        a_m=6_778_137.0,
        eccentricity=0.0,
        inclination_rad=0.0,
        raan_rad=0.0,
        argp_rad=0.0,
        mean_anomaly_rad=0.0,
        times_s=np.linspace(0.0, 5553.6245624479825, 9),
    )
    high = compare_ias15(
        mass_kg=1.988409870698051e30,
        a_m=1.496e11,
        eccentricity=0.99,
        inclination_rad=1.1,
        raan_rad=0.3,
        argp_rad=2.2,
        mean_anomaly_rad=0.2,
        times_s=np.linspace(0.0, 3.15576e7, 5),
    )
    for result in (circular, high):
        assert result["status"] == "compared"
        assert result["passed"] is True
        assert result["integrator_used"] == "ias15"
