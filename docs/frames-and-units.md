# Frames and units

## Internal arithmetic

Computations use IEEE-754 float64 and SI units: metres, seconds, kilograms, and radians. Angles in configuration files are degrees. The manifest stores radians so a downstream tool does not have to guess.

Public inputs that carry a human unit go through Astropy:

- durations such as `365d`, `1h`, `1.5 day`, and `1 yr`
- length strings such as `0.98 AU` and `7000 km`
- absolute epochs, which require an explicit time scale

A bare number is rejected for durations. `365` is not interpreted as days or as seconds. `yr` is the Julian year, 365.25 days of exactly 86400 SI seconds, matching `astropy.units.yr`.

## Epoch and time

An epoch is an ISO-8601 instant plus a scale: `tt`, `tdb`, `utc`, `tai`, `ut1`, `tcg`, or `tcb`. The scale is not optional. A civil date alone mixes UTC and the dynamical scales.

The manifest records the instant and its Julian date on TT. Propagation then ignores the calendar. Every later instant is an SI-second offset from that epoch. Leap seconds are not inserted into those offsets. A duration of `1d` is 86400 SI seconds, including when the epoch scale is UTC.

## The inertial frame

v0.1 has one frame, `OW_INERTIAL_V1`.

- Origin at the central body’s centre of mass, which is the point mass.
- Right-handed Cartesian axes.
- For the Sun and Earth presets, the axes are parallel to ICRS: +Z toward the ICRS pole, +X toward the ICRS origin of right ascension.
- The axes do not rotate. This is not ITRS, not a body-fixed geographic frame, and not the ecliptic.
- Earth preset orbits are geocentric and inertial. There is no sidereal rotation, polar motion, precession, or nutation. A ground track is not available.
- It is not a barycentric multi-body ephemeris. The central mass is alone and fixed.

Other frame names in a config are rejected rather than silently relabelled.

## Central body

A body carries mass, physical radius, gravitational parameter \(\mu\), an exclusion radius, and the frame id. Presets read Astropy’s IAU 2015 nominal values (`GM_sun`, `GM_earth`, `R_sun`, `R_earth`, and the corresponding masses) at the moment the manifest is written. The numbers are then stored in the manifest. Later propagation trusts the manifest, so a later Astropy release does not silently change an old array.

Physics uses \(\mu\). Mass is metadata. A custom body may set `mu_m3_s2` directly; if it does not, \(\mu = G M\) with Astropy’s CODATA gravitational constant, and the source string is stored.

The exclusion radius defaults to the physical radius. Pericentre of the satellite centre must lie outside both the exclusion radius and the sum of the two physical radii, unless the orbit is marked `deliberately_invalid`.

## Render space is a different coordinate system

Blender mesh coordinates are float32. At one astronomical unit from the origin, one float32 step is on the order of ten kilometres. Dividing all coordinates by a constant scale does **not** remove that error: the relative precision of float32 is unchanged, so the metre error stays about \(10^{-7}\) times the distance from the origin.

What does help is a floating origin. Render coordinates are

\[
\mathbf{x}_{\text{render}} = \frac{\mathbf{r}_{\text{SI}} - \mathbf{r}_{\text{origin}}}{s}
\]

where \(s\) is `metres_per_unit` (SI metres per render unit). The scientific ephemeris is left in float64 metres. The preview cache stores float32 render units plus a sidecar that records \(s\) and the origin. A scale chosen for a picture is not a change of the physical semi-major axis.

`orbitweaver propagate --preview-metres-per-unit 1e9` writes that cache beside the scientific NPZ. `--floating-origin-id COL-000000` subtracts that satellite at every sample so it sits at the render origin.

## Exports

| Product | Contents | Units |
| --- | --- | --- |
| `manifest.json` | body, epoch, frame, elements, warnings | SI, radians |
| `states.npz` or `chunks/*.npz` | requested times only, position and velocity | s, m, m/s |
| `encounters.json` | screen, bound, certificate | s, m |
| `preview.owbin` | render-space float32 positions | render units |

JSON numbers use Python’s round-trip float formatting. A regression test checks that a manifest reloads to the same elements.
