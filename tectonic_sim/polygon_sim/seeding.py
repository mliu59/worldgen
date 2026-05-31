"""Initial state: per-plate cell-grid construction.

Self-contained: places plate seeds, assigns types, builds the per-plate
cell paint grids via domain-warped power-weighted Voronoi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from tectonic_sim.noise import PerlinNoise2D, fbm_grid
from tectonic_sim.rng import RngStream
from tectonic_sim.types import CRUST_CONTINENTAL, WorldRect, crust_type_code

from tectonic_sim.polygon_sim.topology import _cell_centres, _grid_dims
from tectonic_sim.polygon_sim.types import (
    PolygonPlate,
    _CONTINENTAL_RELIEF_RNG_TAG,
    _ROT_RNG_TAG,
    _VELOCITY_RNG_TAG,
    _VORONOI_RNG_TAG)


@dataclass(frozen=True)
class _PlateSeed:
    """Plate seed metadata: id, crust type, anchor (km).

    Used internally by ``_initial_state`` between the place-and-type
    sampling pass and the cell-grid Voronoi assignment. Velocity and
    angular velocity are NOT carried here — the cell-grid pass assigns
    them per-plate independently.
    """
    id: int
    type: str
    anchor_position_km: tuple[float, float]


def _plate_seed_min_separation_km(
    domain: WorldRect, plate_count: int, hint_spacing_km: float) -> float:
    """Derived minimum separation between plate seeds. Heuristic: scale
    with ``min(w, h) / (sqrt(N) + 1)`` so plates spread out as N grows,
    with a floor of ``2 * hint_spacing_km`` so adjacent plates don't
    seed inside each other's Voronoi cells. ``hint_spacing_km`` is the
    cell resolution (sim_config.particle_spacing_km — kept around as
    a knob even though polygon_sim doesn't use particles)."""
    base = min(domain.width_km, domain.height_km) / (math.sqrt(plate_count) + 1.0)
    floor = 2.0 * hint_spacing_km
    return max(base, floor)


def _place_plate_seeds(
    domain: WorldRect, sim_config, rng: np.random.Generator) -> np.ndarray:
    """Place ``plate_count`` seed positions in the domain.

    Returns (N, 2) km. Rejection-sampled with derived min-separation +
    optional radial bias. Toroidal-aware separation check.
    """
    n = sim_config.plate_count
    if n <= 0:
        raise ValueError(f"plate_count must be > 0, got {n}")

    hw = domain.half_width_km
    hh = domain.half_height_km
    min_sep = _plate_seed_min_separation_km(
        domain, n, sim_config.particle_spacing_km)
    bias = sim_config.seed_radial_bias

    pool_size = max(8 * n, 128)
    xs = rng.uniform(-hw, hw, pool_size)
    ys = rng.uniform(-hh, hh, pool_size)

    if bias != 0.0:
        d_to_edge = np.minimum(hw - np.abs(xs), hh - np.abs(ys))
        d_max = min(hw, hh)
        centre_score = np.clip(d_to_edge / d_max, 0.0, 1.0)
        jitter = rng.uniform(0.0, 0.15, pool_size)
        key = bias * centre_score + jitter
        order = np.argsort(-key)
    else:
        order = rng.permutation(pool_size)

    accepted: list[tuple[float, float]] = []
    separation = min_sep
    for _ in range(6):
        accepted.clear()
        sep2 = separation * separation
        for i in order:
            x, y = float(xs[i]), float(ys[i])
            ok = True
            for ax, ay in accepted:
                dx, dy = ax - x, ay - y
                dx, dy = domain.wrapped_delta_xy(dx, dy)
                if dx * dx + dy * dy < sep2:
                    ok = False
                    break
            if ok:
                accepted.append((x, y))
                if len(accepted) == n:
                    return np.asarray(accepted, dtype=np.float64)
        separation *= 0.7

    # Fallback: take the first N in order.
    chosen = order[:n]
    return np.stack([xs[chosen], ys[chosen]], axis=1)


def _seed_plates(
    domain: WorldRect, sim_config, seed: int) -> tuple[_PlateSeed, ...]:
    """Sample plate seed positions + types. Replaces the metadata path
    of the legacy ``build_initial_state`` — particle data is discarded.
    """
    rng = RngStream(seed)
    seed_xy = _place_plate_seeds(
        domain, sim_config, rng.child("seeding", "plate_seeds"))
    plates: list[_PlateSeed] = []
    for i, (ax, ay) in enumerate(seed_xy):
        type_rng = rng.child("seeding", "plate", i, "type")
        ptype = (
            "continental"
            if float(type_rng.uniform(0.0, 1.0)) < sim_config.continental_fraction
            else "oceanic"
        )
        plates.append(_PlateSeed(
            id=i, type=ptype, anchor_position_km=(float(ax), float(ay))))
    return tuple(plates)


# ---------------------------------------------------------------------------


def _initial_state(
    domain: WorldRect, sim_config, seed: int) -> tuple[list[PolygonPlate], float]:
    """Build the per-plate paint grids using **domain-warped weighted
    Voronoi** against plate seed anchors.

    Two naturalisation techniques layered on baseline Voronoi:

      * **Domain warp (Method 1):** Each cell's position is displaced by
        a noise-driven vector field before the distance-to-seed lookup.
        Adjacent cells get coherent displacements, so boundaries become
        wavy curves instead of straight perpendicular bisectors.
      * **Power weights (Method 2):** Each plate has an additive weight
        drawn log-normally. Cells go to ``argmin(distance − weight)`` so
        higher-weight plates win more territory — gives size variety
        (some plates Pacific-sized, some Juan-de-Fuca tiny).

    Crust type is taken straight from each plate's seeded type
    (``plates_seed[i].type``), so every cell in a plate has that
    plate's crust. Thickness defaults to the continental/oceanic
    constant. Initial polygons are NOT built here — the per-tick loop's
    ``_re_extract_polygons`` rebuilds them from each plate's cell mask.
    """
    plates_seed = _seed_plates(domain, sim_config, seed)
    gy, gx, cell_km = _grid_dims(domain, sim_config)
    cell_xy = _cell_centres(gy, gx, cell_km)
    n_cells = cell_xy.shape[0]

    # Plate anchors and crust types come from the seeding pass.
    n_plates = len(plates_seed)
    seed_xy = np.array(
        [p.anchor_position_km for p in plates_seed],
        dtype=np.float64)
    seed_pid = np.array([p.id for p in plates_seed], dtype=np.int64)
    seed_crust = np.array(
        [crust_type_code(p.type) for p in plates_seed],
        dtype=np.int8)

    vor_rng = np.random.Generator(np.random.PCG64(seed ^ _VORONOI_RNG_TAG))

    # ----- Method 2: per-plate weights, log-normal × km scale.
    log_weights = vor_rng.normal(0.0, sim_config.voronoi_weight_sigma, n_plates)
    weights = (np.exp(log_weights) - 1.0) * sim_config.voronoi_weight_scale_km

    # ----- Method 1: domain-warp displacement fields.
    # Smooth component: large-scale wavy wobble.
    raw_x = vor_rng.standard_normal((gy, gx))
    raw_y = vor_rng.standard_normal((gy, gx))
    warp_x = gaussian_filter(
        raw_x, sigma=sim_config.voronoi_warp_sigma_cells, mode="wrap")
    warp_y = gaussian_filter(
        raw_y, sigma=sim_config.voronoi_warp_sigma_cells, mode="wrap")
    if warp_x.std() > 1e-9:
        warp_x = warp_x * (sim_config.voronoi_warp_amplitude_km / warp_x.std())
    if warp_y.std() > 1e-9:
        warp_y = warp_y * (sim_config.voronoi_warp_amplitude_km / warp_y.std())
    # Optional jaggedness overlay: high-frequency component added on top
    # of the smooth warp. Cell-scale wobble that breaks up the perfectly
    # smooth curves into something more like a coastline.
    if sim_config.voronoi_warp_jaggedness > 0.0:
        raw_xj = vor_rng.standard_normal((gy, gx))
        raw_yj = vor_rng.standard_normal((gy, gx))
        jag_x = gaussian_filter(
            raw_xj, sigma=sim_config.voronoi_warp_jagged_sigma_cells, mode="wrap")
        jag_y = gaussian_filter(
            raw_yj, sigma=sim_config.voronoi_warp_jagged_sigma_cells, mode="wrap")
        jag_amp = sim_config.voronoi_warp_amplitude_km * sim_config.voronoi_warp_jaggedness
        if jag_x.std() > 1e-9:
            jag_x = jag_x * (jag_amp / jag_x.std())
        if jag_y.std() > 1e-9:
            jag_y = jag_y * (jag_amp / jag_y.std())
        warp_x = warp_x + jag_x
        warp_y = warp_y + jag_y
    warped_x = cell_xy[:, 0] + warp_x.ravel()
    warped_y = cell_xy[:, 1] + warp_y.ravel()

    # ----- Voronoi assignment: argmin over (warped distance − weight).
    # Memory bound: with 232k cells × 80 plates × float64 = 150 MB per
    # term; we chunk along cells to stay under 50 MB peak.
    chunk = 8192
    g_owner_idx = np.empty(n_cells, dtype=np.int64)
    for start in range(0, n_cells, chunk):
        end = min(start + chunk, n_cells)
        dx_chunk = warped_x[start:end, None] - seed_xy[None, :, 0]
        dy_chunk = warped_y[start:end, None] - seed_xy[None, :, 1]
        dx_chunk, dy_chunk = domain.wrapped_delta_xy(dx_chunk, dy_chunk)
        d_chunk = np.sqrt(dx_chunk * dx_chunk + dy_chunk * dy_chunk)
        d_chunk -= weights[None, :]
        g_owner_idx[start:end] = np.argmin(d_chunk, axis=1)

    g_owner = seed_pid[g_owner_idx].reshape(gy, gx)
    g_crust = seed_crust[g_owner_idx].reshape(gy, gx)
    g_thick = np.where(
        g_crust == CRUST_CONTINENTAL,
        sim_config.continental_thickness_km,
        sim_config.oceanic_thickness_km).astype(np.float64)

    # ----- Initial-thickness variation.
    # Per-plate scalar multiplier: log-normal so it stays positive and is
    # symmetric in log-space (1.2× and 1/1.2× equally likely). Knob for
    # plate-to-plate "this continent is on average thicker than that one"
    # variation.
    if sim_config.init_thickness_per_plate_sigma > 0.0:
        plate_thick_mult = np.exp(vor_rng.normal(
            0.0, sim_config.init_thickness_per_plate_sigma, n_plates))
        g_thick = g_thick * plate_thick_mult[g_owner_idx].reshape(gy, gx)
    # Continental relief: Perlin fBm perturbation in physical km,
    # applied only to continental cells, zero-mean per plate so total
    # continental mass is preserved exactly. After sea-level sampling,
    # the thin spots become shelves / inland seas / straits, the thick
    # spots stand proud — producing the "ancient basement topography"
    # that turns featureless plate interiors into varied continents.
    #
    # The noise lives on the same toroidal frame as the cell grid (cell
    # centres in [-half_w, +half_w] km). Frequency is set in physical km
    # — wavelength_km gives the lowest-octave wavelength.
    if sim_config.continental_relief_amplitude_km > 0.0:
        relief_rng = np.random.Generator(
            np.random.PCG64(seed ^ _CONTINENTAL_RELIEF_RNG_TAG))
        relief_noise = PerlinNoise2D.from_rng(relief_rng)
        x_km_grid = cell_xy[:, 0].reshape(gy, gx)
        y_km_grid = cell_xy[:, 1].reshape(gy, gx)
        relief = fbm_grid(
            relief_noise, x_km_grid, y_km_grid,
            octaves=sim_config.continental_relief_octaves,
            persistence=sim_config.continental_relief_persistence,
            base_frequency=1.0 / sim_config.continental_relief_wavelength_km)
        # Normalize to unit std so amplitude_km is a meaningful "typical
        # perturbation" rather than a peak. fBm output is approximately
        # in [-1, 1] but its std is octave-dependent (~0.3-0.5).
        std = float(relief.std())
        if std > 1e-9:
            relief = relief / std
        relief_km = sim_config.continental_relief_amplitude_km * relief
        # Apply only to continental cells; zero elsewhere.
        relief_km = np.where(
            g_crust == CRUST_CONTINENTAL, relief_km, 0.0)
        # Zero-mean per plate so total continental mass on each plate is
        # unchanged (preserves isostasy invariants and keeps plate-by-plate
        # average thickness anchored at continental_thickness_km).
        for k in range(n_plates):
            cont_mask = (g_owner == seed_pid[k]) & (g_crust == CRUST_CONTINENTAL)
            if cont_mask.any():
                relief_km[cont_mask] -= relief_km[cont_mask].mean()
        g_thick = g_thick + relief_km
    # Clamp to a sane floor — no zero/negative thickness from extreme
    # combined perturbations. Continental cells with deeply negative
    # relief stay above this floor; their crust type is unchanged, so
    # they just sit very thin and dip below sea level after isostasy.
    g_thick = np.maximum(g_thick, 1.0)

    # ----- Build per-plate state.
    vel_rng = np.random.Generator(np.random.PCG64(seed ^ _VELOCITY_RNG_TAG))
    plates: list[PolygonPlate] = []
    for ps in plates_seed:
        mask = g_owner == ps.id
        if not mask.any():
            continue
        speed = float(vel_rng.uniform(
            sim_config.init_speed_min_ratio * sim_config.motion_speed_kmpy,
            sim_config.motion_speed_kmpy))
        angle = float(vel_rng.uniform(0.0, 2.0 * np.pi))
        vx = speed * float(np.cos(angle))
        vy = speed * float(np.sin(angle))
        plate = PolygonPlate(
            pid=int(ps.id),
            velocity_kmpy=np.array([vx, vy], dtype=np.float64),
            accum=np.zeros(2, dtype=np.float64),
            cell_mask=mask.copy(),
            crust=np.where(mask, g_crust, 0).astype(np.int8),
            age=np.zeros((gy, gx), dtype=np.float64),
            thickness=np.where(mask, g_thick, 0.0).astype(np.float64),
            polygon=None,   # filled by _re_extract_polygons in the sim loop
            alive=True)
        plates.append(plate)

    # Per-plate random angular velocity (deterministic via seed XOR).
    rot_rng = np.random.Generator(np.random.PCG64(seed ^ _ROT_RNG_TAG))
    for p in plates:
        p.angular_velocity_rad_per_myr = float(rot_rng.uniform(
            -sim_config.init_angular_velocity_max_rad_per_myr, sim_config.init_angular_velocity_max_rad_per_myr))
    return plates, cell_km

