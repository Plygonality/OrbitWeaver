"""HTML design and screening reports.

The pages restate the two-body scope beside the numbers so a layout plot cannot
be read as a stability certificate.
"""

from __future__ import annotations

import html
from pathlib import Path

from orbitweaver.diagnostics import design_diagnostics
from orbitweaver.models import OrbitArray


def write_design_report(array: OrbitArray, path: Path | str) -> Path:
    diagnostics = design_diagnostics(array)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_design_html(array, diagnostics), encoding="utf-8")
    return destination


def write_screening_report(screening: dict, path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_screening_html(screening), encoding="utf-8")
    return destination


def _design_html(array: OrbitArray, diagnostics: dict) -> str:
    warning_rows = "".join(
        "<tr>"
        f"<td>{_esc(note.severity)}</td>"
        f"<td>{_esc(note.code)}</td>"
        f"<td>{_esc(note.message)}</td>"
        f"<td>{_esc(', '.join(note.orbit_ids[:8]))}</td>"
        "</tr>"
        for note in array.warnings
    )
    conservation = diagnostics["conservation"]
    envelope = diagnostics["radial_envelope"]
    neighbors = diagnostics["nearest_neighbor"]
    density = diagnostics["peak_density"]
    preview_rows = "".join(_orbit_row(orbit) for orbit in array.orbits[:24])
    extra = ""
    if len(array.orbits) > 24:
        extra = f"<p>{len(array.orbits) - 24} further orbits are in the manifest.</p>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>OrbitWeaver design report</title>
<style>
  body {{ margin: 0; background: #f4f1ea; color: #1c1915; font: 16px/1.5 "Iowan Old Style", Palatino, Georgia, serif; }}
  main {{ max-width: 920px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }}
  h1 {{ font-weight: 500; letter-spacing: -0.03em; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
  .banner {{ background: #1c1915; color: #f4f1ea; padding: 0.9rem 1rem; border-radius: 6px; }}
  .meta {{ color: #5c564c; }}
  table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; background: white; }}
  th, td {{ text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid #e4dfd4; vertical-align: top; }}
  th {{ font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; color: #5c564c; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em; }}
</style>
</head>
<body>
<main>
  <p class="banner">{_esc(_statement(array))}</p>
  <h1>Design report</h1>
  <p class="meta">{_esc(array.body.name)} · frame {_esc(array.body.frame)} ·
  epoch {_esc(array.epoch.iso)} {_esc(array.epoch.scale)} · intent {_esc(array.intent)}</p>
  <p>{diagnostics["n_orbits"]} orbits. Seed {_esc(array.seed)}.
  Gravitational parameter {_fmt(array.body.mu_m3_s2)} m³/s².</p>
  <h2>Keplerian summary</h2>
  {_stat_table(diagnostics)}
  <h2>Conservation sample</h2>
  <p>{_esc(conservation["meaning"])}</p>
  <p>Max relative specific-energy error {_fmt(conservation["max_relative_energy_error"])}.
     Max relative angular-momentum error {_fmt(conservation["max_relative_angular_momentum_error"])}.
     Max folded Kepler residual {_fmt(conservation["kepler_residual_max_rad"])} rad.</p>
  <h2>Radial envelopes</h2>
  <p>{_esc(envelope.get("meaning", ""))}</p>
  <p>Overlapping pairs {envelope.get("overlapping_pairs")} of {envelope.get("possible_pairs")}
     (fraction {_fmt(envelope.get("overlap_fraction"))}).
     Occupancy of the radial span {_fmt(envelope.get("occupancy"))}.
     Components {envelope.get("components")}.</p>
  <h2>Epoch nearest neighbours</h2>
  <p>Minimum {_fmt(neighbors.get("min_m"))} m.
     5th percentile {_fmt(neighbors.get("percentile_5_m"))} m.
     Median {_fmt(neighbors.get("median_m"))} m.</p>
  <h2>Peak spatial density</h2>
  <p>{_density_text(density)}</p>
  <h2>Orbit sample</h2>
  <table>
    <thead><tr><th>Id</th><th>a (m)</th><th>e</th><th>i (rad)</th><th>convention</th><th>group</th></tr></thead>
    <tbody>{preview_rows}</tbody>
  </table>
  {extra}
  <h2>Warnings</h2>
  <table>
    <thead><tr><th>Severity</th><th>Code</th><th>Message</th><th>Orbits</th></tr></thead>
    <tbody>{warning_rows}</tbody>
  </table>
</main>
</body>
</html>
"""


def _screening_html(screening: dict) -> str:
    guarantee = screening["guarantee"]
    minimum = screening["minimum_separation"]
    rows = "".join(
        "<tr>"
        f"<td>{_esc(item['id_a'])}</td><td>{_esc(item['id_b'])}</td>"
        f"<td>{_fmt(item['sampled_min_separation_m'])}</td>"
        f"<td>{_fmt(item['time_offset_s'])}</td></tr>"
        for item in screening.get("closest_candidates", [])[:20]
    )
    hits = screening.get("sampled_or_refined_volume_intersections", [])
    hit_rows = "".join(
        "<tr>"
        f"<td>{_esc(item['id_a'])}</td><td>{_esc(item['id_b'])}</td>"
        f"<td>{_fmt(item['separation_m'])}</td><td>{_esc(item['source'])}</td></tr>"
        for item in hits[:20]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>OrbitWeaver screening report</title>
<style>
  body {{ margin: 0; background: #f4f1ea; color: #1c1915; font: 16px/1.5 "Iowan Old Style", Palatino, Georgia, serif; }}
  main {{ max-width: 920px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }}
  .banner {{ background: #1c1915; color: #f4f1ea; padding: 0.9rem 1rem; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; background: white; }}
  th, td {{ text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid #e4dfd4; }}
</style>
</head>
<body>
<main>
  <p class="banner">{_esc(guarantee["statement"])}</p>
  <h1>Screening report</h1>
  <p>Certificate <strong>{_esc(guarantee["certificate"])}</strong>. {_esc(guarantee["certificate_meaning"])}</p>
  <p>Samples {guarantee["n_samples"]} from {_fmt(guarantee["interval_start_s"])} s
     to {_fmt(guarantee["interval_stop_s"])} s.
     Largest gap {_fmt(guarantee["max_sample_gap_s"])} s.
     Speed bound {_fmt(guarantee["relative_speed_bound_m_s"])} m/s.
     Maximum missed approach {_fmt(guarantee["max_missed_approach_m"])} m.</p>
  <p>{_esc(guarantee["missed_encounter_caveat"])}</p>
  <h2>Reported minimum separation</h2>
  <p>{_fmt(minimum["metres"])} m via {_esc(minimum["source"])}
     for pair {_esc(minimum["pair"])} at offset {_fmt(minimum["time_offset_s"])} s.
     {_esc(minimum["note"])}</p>
  <h2>Volume intersections</h2>
  <table><thead><tr><th>A</th><th>B</th><th>Separation (m)</th><th>Source</th></tr></thead>
  <tbody>{hit_rows or "<tr><td colspan='4'>None evaluated.</td></tr>"}</tbody></table>
  <h2>Closest sampled candidates</h2>
  <table><thead><tr><th>A</th><th>B</th><th>Separation (m)</th><th>Time (s)</th></tr></thead>
  <tbody>{rows or "<tr><td colspan='4'>None.</td></tr>"}</tbody></table>
</main>
</body>
</html>
"""


def _statement(array: OrbitArray) -> str:
    for note in array.warnings:
        if note.code == "NOT_A_STABILITY_PROOF":
            return note.message
    return "Newtonian two-body screen. Not a stability proof."


def _stat_table(diagnostics: dict) -> str:
    rows = []
    labels = (
        ("semi_major_m", "Semi-major axis (m)"),
        ("eccentricity", "Eccentricity"),
        ("inclination_rad", "Inclination (rad)"),
        ("period_s", "Period (s)"),
        ("periapsis_m", "Periapsis (m)"),
        ("apoapsis_m", "Apoapsis (m)"),
        ("epoch_speed_m_s", "Epoch speed (m/s)"),
    )
    for key, label in labels:
        stats = diagnostics[key]
        rows.append(
            "<tr>"
            f"<td>{label}</td><td>{_fmt(stats['min'])}</td>"
            f"<td>{_fmt(stats['median'])}</td><td>{_fmt(stats['max'])}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Quantity</th><th>Min</th><th>Median</th><th>Max</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _orbit_row(orbit) -> str:
    return (
        "<tr>"
        f"<td>{_esc(orbit.satellite_id)}</td>"
        f"<td>{_fmt(orbit.a_m)}</td>"
        f"<td>{_fmt(orbit.e)}</td>"
        f"<td>{_fmt(orbit.inclination_rad)}</td>"
        f"<td>{_esc(orbit.element_convention)}</td>"
        f"<td>{_esc(orbit.group)}</td></tr>"
    )


def _density_text(density: dict) -> str:
    if not density.get("defined"):
        return _esc(density.get("reason", "Undefined."))
    return (
        f"Ball radius {_fmt(density['ball_radius_m'])} m "
        f"({_esc(density['ball_rule'])}). "
        f"Peak count {density['peak_count_including_self']}. "
        f"Number density {_fmt(density['peak_number_density_per_m3'])} per m³. "
        f"{_esc(density['meaning'])}"
    )


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6e}"
    return _esc(value)


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))
