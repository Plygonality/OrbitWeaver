# Limitations

v0.1 is a Newtonian, central-mass, test-particle ellipse tool. Read this beside any manifest or HTML report.

## Dynamical claims that this version does not make

- Sampled positions are not a proof of dynamical stability.
- Even spacing, a symmetric shell, or a phased ring family is not a proof of collision avoidance once mutual gravity, oblateness, drag, or radiation pressure exists.
- A ring family is not a rigid structure. The satellites are separate initial-value problems that happen to share some elements.
- The screen certificate `no_collision_consistent_with_samples` covers one time interval, one step, and the two-body speed bound \(V \Delta t / 2\). It says nothing about times outside that interval.
- `not_certified` means a closer approach may have been missed. It is not a quiet success.
- `intent = "visual_only"` does not add relativity. There is no horizon, no frame dragging, and no photon orbit in the integrator.

## Sampling can miss an encounter

Between samples separated by \(\Delta t\), centre distance can fall by as much as \(V \Delta t / 2\) below the lower sample, with \(V\) twice the fastest pericentre speed in the array. Refinement only inspects local dips of the nearest-neighbour series. A pair that is never among those dips, and never inside the spatial query radius at a sample, can cross unseen. The JSON report always carries `max_missed_approach_m` and the certificate name.

The central-body pericentre check is analytic and does not depend on the step. Satellite–satellite encounters do.

## Numerical scope

- Eccentricity must satisfy \(0 \le e < 1\). Hyperbolas raise an error that names the orbit id.
- The Kepler residual that is tested is the folded residual. It is reported on every ephemeris.
- Element angles on circular or equatorial orbits are not unique. Warnings name the ids. The stored numbers are what gets propagated.
- Very long propagations remain accurate in the folded anomaly. The Cartesian map does not feed a multi-revolution angle into `sin` and `cos`.

## Rendering

- Scientific coordinates and render coordinates are different files.
- A presentation scale does not restore float32 precision at astronomical distances. A nearby floating origin does.
- The Blender add-on instances one prototype on a point cloud. It does not create one object per satellite by default.
- If Blender is not installed, `orbitweaver doctor` marks the preview **unverified**. That is not a claim that the add-on was executed.

## Optional REBOUND

REBOUND is GPL-3.0-or-later (see [NOTICE](../NOTICE)) and is not a required dependency. The core package is BSD-3-Clause. The optional comparison uses IAS15 on one massless particle. WHFast is not called: it is for weakly perturbed systems, not for close encounters and not for this check. A missing install is **unverified**, not a pass. A match does not turn the array into an N-body integration.

## Ecosystem contracts

The worldbuilding projects named as later integration points — Unit-canon, Habitat-kit, Probe-kit, gn-as-code, and Epektasis Cosmica — were not readable as repositories from this environment. No fiction from those projects is a physical default here, and no sibling repository was modified. The accessible Plygonality repositories at the time of writing (WCSScene, RomanRecover, NodeBridge, ParityForge, Plygon-mcp) are separate tools and were left untouched.

## What v0.1 does not yet do

These are the highest-impact gaps, in order:

1. A continuous Keplerian closest-approach certificate (minimum orbit intersection distance, or an interval bound tighter than the global speed ceiling) so coarse samples cannot hide a node crossing.
2. A CI job that installs the GPL REBOUND extra. Locally, `tests/test_rebound_optional.py` checks two ellipses against IAS15 when the import succeeds, and records **unverified** when it does not. CI does not install REBOUND, so the default pipeline result stays unverified. A machine-local comparison is in `docs/benchmarks/rebound_ias15.json`.
3. A headless Blender execution of the instancing add-on. Until that exists, the add-on is source plus a scalar solver checked against the NumPy map, not a viewed animation.
4. Perturbations (\(J_2\), drag, radiation pressure) and mutual gravity, kept outside the Kepler core so a layout can be stress-tested without pretending the ellipse map includes them.
5. Out-of-core design and screening for well beyond \(10^5\) objects, with a pair stream that does not retain every candidate.

Station-keeping, hyperbolic escape, and relativistic orbits are further milestones, not bugs in the ellipse map.
