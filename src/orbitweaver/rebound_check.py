"""Optional comparison against REBOUND's IAS15 integrator.

REBOUND is not required to design or propagate a Keplerian scene. WHFast is a
symplectic Wisdom–Holman map for weakly perturbed systems and is not used
here: a close-encounter or a pure high-accuracy check is the role of IAS15
(Rein & Spiegel 2015). When REBOUND is absent the result is ``unverified``,
which is not a pass.
"""

from __future__ import annotations

import numpy as np

from orbitweaver.kepler import propagate_elements


def rebound_status() -> dict[str, object]:
    try:
        import rebound
    except Exception as exc:
        return {
            "status": "unverified",
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "integrator_used": None,
            "whfast_role": (
                "WHFast targets weakly perturbed dominant-central-mass systems "
                "and is not a general close-encounter solver. It is not invoked "
                "by this check."
            ),
        }
    return {
        "status": "available",
        "available": True,
        "version": getattr(rebound, "__version__", "unknown"),
        "integrator_used": None,
        "reason": "",
        "whfast_role": (
            "WHFast targets weakly perturbed dominant-central-mass systems "
            "and is not a general close-encounter solver. The optional check uses IAS15."
        ),
    }


def compare_ias15(
    *,
    mass_kg: float,
    a_m: float,
    eccentricity: float,
    inclination_rad: float,
    raan_rad: float,
    argp_rad: float,
    mean_anomaly_rad: float,
    times_s: np.ndarray,
) -> dict[str, object]:
    """Compare one massless ellipse with REBOUND IAS15 in SI units.

    The central body is fixed by giving the test particle zero mass. This is
    the same two-body assumption as OrbitWeaver, not an N-body validation.
    """
    status = rebound_status()
    if not status["available"]:
        return {**status, "passed": False, "max_position_error_m": None}
    import rebound

    gravitational_parameter = None
    # REBOUND's unit helper sets G. Place a central mass and a massless satellite.
    simulation = rebound.Simulation()
    simulation.units = ("m", "s", "kg")
    simulation.integrator = "ias15"
    simulation.add(m=mass_kg)
    simulation.add(
        m=0.0,
        a=a_m,
        e=eccentricity,
        inc=inclination_rad,
        Omega=raan_rad,
        omega=argp_rad,
        M=mean_anomaly_rad,
        primary=simulation.particles[0],
    )
    gravitational_parameter = float(simulation.G) * mass_kg
    ours = propagate_elements(
        [a_m],
        [eccentricity],
        [inclination_rad],
        [raan_rad],
        [argp_rad],
        [mean_anomaly_rad],
        gravitational_parameter,
        times_s,
        orbit_ids=["rebound-check"],
    )
    errors = []
    for index, instant in enumerate(np.asarray(times_s, dtype=np.float64)):
        simulation.integrate(float(instant))
        particle = simulation.particles[1]
        reference = np.array([particle.x, particle.y, particle.z], dtype=np.float64)
        errors.append(float(np.linalg.norm(reference - ours.position_m[index, 0])))
    max_error = float(np.max(errors)) if errors else 0.0
    scale = max(a_m, 1.0)
    passed = max_error / scale < 1e-6
    return {
        "status": "compared" if passed else "failed",
        "available": True,
        "passed": passed,
        "version": status["version"],
        "integrator_used": "ias15",
        "max_position_error_m": max_error,
        "relative_tolerance": 1e-6,
        "whfast_role": status["whfast_role"],
        "assumption": "massless test particle, SI units, IAS15, no extra forces",
    }
