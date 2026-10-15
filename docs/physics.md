# Physics model, v0.1

OrbitWeaver v0.1 propagates **independent test particles on bound Keplerian ellipses** about one fixed point mass. The particles do not attract each other. The central body does not move. This is the Newtonian two-body problem, repeated once per satellite.

That model is useful for laying out shells, ring-like families, and for screening trajectories. A manifest, a picture, or a list of sampled positions is a design plus a screen over stated times. It is not a proof of long-term stability, and it is not evidence that equal spacing will survive once the assumptions stop holding.

## What the integrator integrates

For each satellite the specific acceleration is

\[
\ddot{\mathbf{r}} = -\mu \frac{\mathbf{r}}{|\mathbf{r}|^3}
\]

with \(\mu\) the gravitational parameter of the central body. The closed solutions used here are the elliptical ones: semi-major axis \(a > 0\) and eccentricity \(0 \le e < 1\).

Out of scope for v0.1, and rejected or simply not modelled:

- hyperbolic and parabolic flybys (\(e \ge 1\))
- mutual gravity, including the stability of a Dyson swarm
- \(J_2\), drag, solar radiation pressure, third bodies
- active station-keeping
- relativistic precession, frame dragging, horizons, and photon orbits

A file marked `intent = "visual_only"` still uses this Newtonian point mass. The label is there so a picture is not mistaken for a relativistic solution.

## Elements

Each orbit record stores, in SI units and radians:

| Symbol | Field | Meaning |
| --- | --- | --- |
| \(a\) | `a_m` | semi-major axis |
| \(e\) | `e` | eccentricity |
| \(i\) | `inclination_rad` | inclination, \(0\) to \(\pi\) |
| \(\Omega\) | `raan_rad` | longitude of the ascending node |
| \(\omega\) | `argp_rad` | argument of periapsis |
| \(M_0\) | `mean_anomaly_rad` | mean anomaly at the epoch |
| — | `satellite_id` | immutable id |
| — | `body_radius_m` | spherical body radius used only for volume checks |
| — | `group` | design group |

The mean motion and the period are the elementary formulae

\[
n = \sqrt{\mu / a^3}, \qquad T = 2\pi \sqrt{a^3 / \mu}, \qquad M(t) = M_0 + n t.
\]

\(t\) is SI seconds after the epoch. Specific energy and specific angular momentum of a Keplerian ellipse are

\[
\varepsilon = -\frac{\mu}{2a}, \qquad |\mathbf{h}| = \sqrt{\mu a (1-e^2)}.
\]

Pericentre and apocentre radii are \(a(1-e)\) and \(a(1+e)\). Speed at pericentre, used as a per-orbit speed ceiling, is

\[
v_p = \sqrt{\mu \frac{1+e}{a(1-e)}}.
\]

A generated orbit is rejected when the pericentre of the satellite centre is less than or equal to the exclusion radius or the sum of the central radius and the satellite radius, unless `deliberately_invalid` is set for a negative test.

## Kepler’s equation

The eccentric anomaly \(E\) solves

\[
M = E - e \sin E.
\]

The implementation folds \(M\) into \((-\pi, \pi]\), solves the odd equation on \([0, \pi]\), and reports the residual of that folded equation. Folding keeps the residual meaningful after many revolutions: the absolute spacing of a huge unwrapped mean anomaly is coarser than \(10^{-14}\) rad, so a residual against the raw \(M\) would measure floating-point magnitude rather than the solver.

The root on \([0, \pi]\) is unique for \(e \in [0, 1)\). The iteration is Halley’s method,

\[
E_{n+1} = E_n - \frac{f f'}{f'^2 - \tfrac12 f f''},
\]

with \(f(E) = E - e\sin E - |M|\), \(f' = 1 - e\cos E\), and \(f'' = e\sin E\). A trial step is kept only when it stays inside the bracket and reduces the residual. Otherwise the step is a bisection. The bracket starts at \([0, \pi]\) and always contains the root, so the iteration cannot walk off to a spurious attractor when \(e\) is close to 1 and \(M\) is close to 0. The internal aim is a residual of \(10^{-14}\) rad. Propagation raises, and names the orbit ids, if a folded residual remains above \(10^{-10}\) rad.

Trigonometric functions in the state map use this folded eccentric anomaly, not a multi-revolution unwrapped angle.

## Inertial state

Perifocal coordinates, with \(P\) toward periapsis and \(Q\) 90° ahead in the direction of motion:

\[
\begin{aligned}
x_P &= a(\cos E - e), &
y_P &= a\sqrt{1-e^2}\sin E, \\
v_{xP} &= -\frac{\sqrt{\mu a}}{r}\sin E, &
v_{yP} &= \frac{\sqrt{\mu a}}{r}\sqrt{1-e^2}\cos E,
\end{aligned}
\]

where \(r = a(1 - e\cos E)\). The inertial components are \(x_P \mathbf{P} + y_P \mathbf{Q}\), with

\[
\begin{aligned}
\mathbf{P} &= \begin{pmatrix}
\cos\Omega\cos\omega - \sin\Omega\sin\omega\cos i \\
\sin\Omega\cos\omega + \cos\Omega\sin\omega\cos i \\
\sin\omega\sin i
\end{pmatrix}, &
\mathbf{Q} &= \begin{pmatrix}
-\cos\Omega\sin\omega - \sin\Omega\cos\omega\cos i \\
-\sin\Omega\sin\omega + \cos\Omega\cos\omega\cos i \\
\cos\omega\sin i
\end{pmatrix}.
\end{aligned}
\]

The angular-momentum direction is \(\mathbf{W} = \mathbf{P}\times\mathbf{Q} = (\sin\Omega\sin i,\ -\cos\Omega\sin i,\ \cos i)\).

There is no per-satellite Python loop in this map. Times and satellites are NumPy axes, and large propagations are split into orbit batches so the whole cube is not required at once.

## Singular angles

Below the documented thresholds \(e < 10^{-8}\) or \(\min(i, \pi-i) < 10^{-8}\) rad, the classical angles are not unique. OrbitWeaver **keeps the stored angles** and attaches a warning that names the orbit ids. The inertial state is still well defined:

| Case | What is actually fixed | Inverse-converter convention |
| --- | --- | --- |
| Circular and equatorial | true longitude \(\lambda = \Omega + \omega + \nu\) (retrograde: the in-plane sense reverses) | \(\Omega = 0\), \(\omega = 0\), mean anomaly holds \(\lambda\) |
| Circular and inclined | argument of latitude \(u = \omega + \nu\) | \(\omega = 0\), mean anomaly holds \(u\) |
| Elliptical and equatorial | longitude of periapsis \(\varpi = \Omega + \omega\) (retrograde sense reversed) | \(\Omega = 0\), argument of periapsis holds \(\varpi\) |

The Cartesian inverse uses a tighter numerical branch, \(10^{-12}\), so a merely small eccentricity is not rewritten. Round-trip tests check the Cartesian state, not the individual undefined angles.

## Layouts are draws of elements

`shell`, `ring_family`, and `inclination_family` all emit ordinary ellipses.

- A **shell** draws \(a\), \(e\), and angles from the configured distributions. `inclination.mode = "isotropic"` draws \(\cos i\) uniformly on \([-1, 1]\), which is uniform on the sphere. A wide uniform draw in \(i\) itself oversamples the poles and is warned about.
- A **ring family** requires a single \(a\), a single \(e\), and a fixed inclination. Mean anomaly defaults to an even phase. The satellites remain separate initial-value problems. On a shared circular orbit the mean-anomaly spacing is constant only while every particle has that exact \(a\) and nothing else acts. That is a property of this model, not a rigid body and not a stability result.
- An **inclination family** places a list of planes and phases mean anomaly inside each plane.

The designer uses NumPy’s PCG64 generator seeded by `array.seed`. Stochastic fields consume the stream in the order semi-major axis, eccentricity, inclination, node, argument of periapsis, mean anomaly. Fixed fields do not consume a draw. Equal configs and equal seeds reproduce the same records in any UI.

Initial centre separation is enforced at the epoch with a KD-tree. Failure to place the requested count raises an error that names the shell. `allow_shortfall` keeps a partial layout and records capacity pressure. Meeting the epoch separation does not constrain later encounters; that is what screening is for.

## Screening and the missed-encounter bound

Let \(\delta(t) = |\mathbf{r}_i(t) - \mathbf{r}_j(t)|\). Wherever the derivative exists,

\[
|\delta'(t)| \le |\mathbf{v}_i - \mathbf{v}_j| \le v_{p,i} + v_{p,j}.
\]

OrbitWeaver uses the uniform ceiling \(V = 2 \max v_p\). On a sample gap of length \(\Delta t\),

\[
\min \delta \ge \frac{\delta(t_k) + \delta(t_{k+1}) - V\Delta t}{2},
\]

so the true minimum on that gap can lie at most \(V\Delta t / 2\) below the lower neighbouring sample. The reported `max_missed_approach_m` is that quantity on the widest gap of the requested grid.

Consequences, all of which are printed with the screen:

- **encounter_detected** — a sample or a refined candidate window brought two body volumes into intersection, or the epoch state already does, or analytic pericentre meets the central body.
- **no_collision_consistent_with_samples** — the smallest sampled nearest-neighbour distance is at least the largest pair of body radii plus \(V\Delta t / 2\). Under the two-body model and that speed ceiling, a closer approach on this grid would contradict the samples. The certificate stops at the ends of the grid.
- **not_certified** — nothing evaluated intersected, and the bound is too wide to rule an intersection out between samples. This is not a clean result. Tighten the step.

Refinement minimises centre distance with a bounded scalar solver on a window of one sample gap about each local minimum of the nearest-neighbour series. It can reveal a crossing that the samples themselves step over, and it can still miss a crossing whose samples were never among those dips. The miss bound remains the statement of that risk.

Radial-envelope overlap counts pairs whose \([a(1-e), a(1+e)]\) intervals share a radius. That is necessary for a shared geocentric distance. It is not a conjunction.

## Optional REBOUND check

REBOUND is not imported unless the optional check runs, and it is not required to build a scene. Official guidance used here:

- **WHFast** is a symplectic Wisdom–Holman integrator for systems with a dominant central mass and small perturbations to the Keplerian orbits. It is not an adaptive close-encounter solver. OrbitWeaver does not call it.
- **IAS15** is REBOUND’s default high-order adaptive integrator and is appropriate for a short, high-accuracy numerical comparison (Rein & Spiegel 2015; Rein & Tamayo 2015 for the WHFast scope).

The optional check places one massless particle on the same elements and compares Cartesian position. Agreement supports the two-body state map. It does not promote the array to an N-body integration. If REBOUND is not installed the check is **unverified**, which is not a pass.

## References

- Kepler’s equation and the classical rotation: standard two-body relations as in Bate, Mueller & White, *Fundamentals of Astrodynamics*, and Vallado, *Fundamentals of Astrodynamics and Applications* (singularity conventions for true longitude, argument of latitude, and longitude of periapsis).
- Halley’s iteration with a bisection safeguard: the classical Halley step on a bracket that is known to contain the single elliptical root.
- Rein & Tamayo, MNRAS 452, 376 (2015), WHFast.
- Rein & Spiegel, MNRAS 446, 1424 (2015), IAS15.
- IAU 2015 Resolution B3 nominal solar and terrestrial mass parameters, via Astropy constants. The astronomical unit is the SI exact value 149597870700 m.
