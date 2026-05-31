"""Climate layer: temperature (latitude + lapse rate) and precipitation
(prevailing-wind moisture sweep with orographic uplift).

The precipitation model uses a per-hex *cartesian* wind direction. The base
direction is set by latitude band (the three-cell zonal pattern), then
perturbed by an annual-mean **sea-breeze** onshore component near coasts
and a spatially-coherent **Perlin jitter**. Per target hex, the moisture
sweep is averaged over **multiple sample paths** spread in a cone around
the base direction — together these break the pure zonal "stripes" pattern
without simulating seasonal physics.
"""

from __future__ import annotations

import math
from collections import deque

from worldgen._log import progress
from worldgen.rng import RngHierarchy
from worldgen.hex import Hex
from tectonic_sim.noise import PerlinNoise2D
from worldgen.ocean import OceanLayer
from worldgen.types import (
    ClimateLayer,
    ElevationLayer,
    SeaLayer,
    WorldgenConfig,
)
from worldgen.world import map_half_extents_km

_SQRT3 = math.sqrt(3.0)


def hex_latitude_deg(
    h: Hex, half_height_km: float, config: WorldgenConfig,
) -> float:
    """Geographic latitude (degrees) the hex occupies on the planet.

    The map's pixel-y axis is a slice of the planet's latitude range:
    ``y = -half_height_km`` sits at ``map_lat_max`` (north edge),
    ``y = +half_height_km`` at ``map_lat_min`` (south edge). The slice's
    km extent is independent of how many degrees of latitude it spans —
    the user is free to simulate a 1000-km-tall map covering 1° or 60° of
    latitude alike.

    ``half_height_km`` is derived by the caller from the hex set's
    pixel-y bounding box (via ``world.map_half_extents_km``) so this
    function doesn't need the world's configured shape.
    """
    if half_height_km <= 0:
        return 0.5 * (config.map_lat_min + config.map_lat_max)
    pixel_y_km = _SQRT3 * (h.r + h.q / 2.0) * config.hex_size_km
    fraction_south = (pixel_y_km + half_height_km) / (2.0 * half_height_km)
    fraction_south = max(0.0, min(1.0, fraction_south))
    return (
        config.map_lat_max
        - fraction_south * (config.map_lat_max - config.map_lat_min)
    )


def compute_temperature(
    elevation: ElevationLayer,
    sea: SeaLayer,
    ocean: OceanLayer,
    half_height_km: float,
    config: WorldgenConfig,
    rng: RngHierarchy,
) -> dict[Hex, float]:
    """Per-hex annual mean temperature in °C.

    Ocean hex T = latitudinal baseline + ``ocean.current_temp_anomaly``
    Land hex T = latitudinal baseline + ``ocean.coastal_temp_anomaly``
                 − lapse(elev above sea) + small Perlin noise

    The current and coastal anomalies are pre-computed by the ocean layer:
    warm western-boundary currents push the adjacent coastline above the
    pure-latitude baseline, cold eastern-boundary currents push it below.
    """
    temp_noise = PerlinNoise2D.from_rng(rng.child("worldgen", "climate", "temp_noise"))
    eq = config.equator_temp_c
    pole = config.polar_temp_c

    temperatures: dict[Hex, float] = {}
    for h, elev in elevation.elevation.items():
        lat_deg = hex_latitude_deg(h, half_height_km, config)
        polar_fraction = min(1.0, abs(lat_deg) / 90.0)
        latitudinal = eq + (pole - eq) * (polar_fraction**1.3)

        if sea.is_ocean.get(h, False):
            ocean_anomaly = ocean.current_temp_anomaly.get(h, 0.0)
            n = temp_noise.sample(h.q * 0.04, h.r * 0.04)
            temperatures[h] = (
                latitudinal + ocean_anomaly + n * config.temp_noise_amplitude
            )
            continue

        elev_above_sea = max(0.0, elev - elevation.sea_level)
        km = elev_above_sea * config.max_elevation_km
        lapse = km * config.lapse_rate_c_per_km
        coastal_anomaly = ocean.coastal_temp_anomaly.get(h, 0.0)
        n = temp_noise.sample(h.q * 0.04, h.r * 0.04)
        temperatures[h] = (
            latitudinal + coastal_anomaly - lapse + n * config.temp_noise_amplitude
        )
    return temperatures


# Wind-band boundaries in geographic latitude (degrees). Earth-like:
#   |lat| ∈ [ 0°, 30°] → trade easterlies (wind blows east → west)
#   |lat| ∈ [30°, 60°] → westerlies (west → east)
#   |lat| ∈ [60°, 90°] → polar easterlies (east → west)
_WIND_BAND_LO_DEG = 30.0
_WIND_BAND_HI_DEG = 60.0
# Half-width of the smoothstep transition zone at each band boundary, in
# degrees. Wider than Earth's real polar-front zone but enough to suppress
# the "sharp line of opposite winds" artifact the hard step produced.
_WIND_TRANSITION_HALF_WIDTH_DEG = 3.6


def _zonal_wind_cartesian(lat_deg: float) -> tuple[float, float]:
    """Base prevailing wind unit vector by latitude band, in *cartesian*
    screen space (``+x`` = right, ``+y`` = down — matches the renderer).
    """
    sign = _zonal_wind_sign(abs(lat_deg))
    return (sign, 0.0)


def _zonal_wind_sign(lat_abs_deg: float) -> float:
    """E-W wind sign as a continuous function of |latitude| in degrees.

    Returns −1 (easterly) below 30° and above 60°, +1 (westerly) in
    between, with smoothstep transitions at the two band boundaries.
    """
    def smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    w = _WIND_TRANSITION_HALF_WIDTH_DEG
    lo = _WIND_BAND_LO_DEG
    hi = _WIND_BAND_HI_DEG
    if lat_abs_deg <= lo - w:
        return -1.0
    if lat_abs_deg < lo + w:
        t = (lat_abs_deg - (lo - w)) / (2.0 * w)
        return -1.0 + 2.0 * smoothstep(t)
    if lat_abs_deg <= hi - w:
        return 1.0
    if lat_abs_deg < hi + w:
        t = (lat_abs_deg - (hi - w)) / (2.0 * w)
        return 1.0 - 2.0 * smoothstep(t)
    return -1.0


def _hex_to_xy(h: Hex) -> tuple[float, float]:
    """Same flat-top projection the renderer uses (1 unit = one hex-step)."""
    return 1.5 * h.q, _SQRT3 * (h.r + h.q / 2.0)


def _hex_round(q_frac: float, r_frac: float) -> Hex:
    """Round fractional axial coords to the nearest hex (Patel's cube round)."""
    s_frac = -q_frac - r_frac
    qi = round(q_frac)
    ri = round(r_frac)
    si = round(s_frac)
    dq = abs(qi - q_frac)
    dr = abs(ri - r_frac)
    ds = abs(si - s_frac)
    if dq > dr and dq > ds:
        qi = -ri - si
    elif dr > ds:
        ri = -qi - si
    # else implicit: si = -qi - ri
    return Hex(qi, ri)


def _axial_step_for_theta(theta: float) -> tuple[float, float]:
    """Axial offset that moves one hex of *physical* distance in cart direction θ.

    One adjacent-hex spacing is √3 in our cartesian projection, so a unit
    cartesian step ``(cos θ, sin θ)`` scaled by √3 corresponds to one hex of
    travel. The cart→axial transform then gives the fractional ``(dq, dr)``
    to add per upwind step.
    """
    cx = math.cos(theta) * _SQRT3
    cy = math.sin(theta) * _SQRT3
    dq = (2.0 / 3.0) * cx
    dr = -cx / 3.0 + cy / _SQRT3
    return dq, dr


def _compute_sea_breeze_field(
    elevation: ElevationLayer,
    sea: SeaLayer,
    config: WorldgenConfig,
) -> dict[Hex, tuple[float, float, float]]:
    """For each land hex, return (onshore_x, onshore_y, strength).

    The vector points FROM the nearest sea hex TO this hex in cartesian
    space — that is the *inland* direction, which is also the direction the
    onshore wind blows (sea → land). Strength decays linearly from 1.0 at
    the coast to 0.0 at ``sea_breeze_reach_km``.

    Implemented as a multi-source BFS from every sea hex; cheap (single
    pass) and gives both distance and nearest-source in one go.
    """
    hexes = list(elevation.elevation.keys())
    hex_set = set(hexes)
    nearest_sea: dict[Hex, Hex] = {}
    distance_km: dict[Hex, float] = {h: math.inf for h in hexes}
    queue: deque[Hex] = deque()
    step_km = config.hex_size_km * _SQRT3
    for h in hexes:
        if sea.is_ocean[h]:
            nearest_sea[h] = h
            distance_km[h] = 0.0
            queue.append(h)
    while queue:
        h = queue.popleft()
        d = distance_km[h]
        src = nearest_sea[h]
        for nb in h.neighbors():
            if nb not in hex_set:
                continue
            new_d = d + step_km
            if new_d < distance_km[nb]:
                distance_km[nb] = new_d
                nearest_sea[nb] = src
                queue.append(nb)

    reach = config.sea_breeze_reach_km
    field: dict[Hex, tuple[float, float, float]] = {}
    for h in hexes:
        if sea.is_ocean[h] or h not in nearest_sea:
            field[h] = (0.0, 0.0, 0.0)
            continue
        d = distance_km[h]
        if d >= reach:
            field[h] = (0.0, 0.0, 0.0)
            continue
        sea_h = nearest_sea[h]
        hx, hy = _hex_to_xy(h)
        sx, sy = _hex_to_xy(sea_h)
        dx, dy = hx - sx, hy - sy
        norm = math.hypot(dx, dy)
        if norm == 0:  # shouldn't happen for non-ocean hex, but be safe
            field[h] = (0.0, 0.0, 0.0)
            continue
        # Smoothstep decay: strongest at the coast, zero at reach, with
        # zero derivative at both endpoints so the field tapers gracefully
        # into the inland baseline instead of cutting off at a sharp ring.
        t = 1.0 - d / reach
        strength = t * t * (3.0 - 2.0 * t)
        field[h] = (dx / norm, dy / norm, strength)
    return field


def _per_hex_wind_theta(
    h: Hex,
    half_height_km: float,
    sea_breeze_field: dict[Hex, tuple[float, float, float]],
    jitter_noise: PerlinNoise2D,
    config: WorldgenConfig,
) -> float:
    """Cartesian wind direction (radians, 0 = +x) for one hex.

    = normalize(zonal + sea_breeze_strength · onshore_unit · sea_breeze_falloff)
      rotated by Perlin jitter ∈ ±wind_jitter_amplitude_deg.
    """
    lat_deg = hex_latitude_deg(h, half_height_km, config)
    base_x, base_y = _zonal_wind_cartesian(lat_deg)
    sb_x, sb_y, sb_strength = sea_breeze_field.get(h, (0.0, 0.0, 0.0))
    wx = base_x + sb_x * sb_strength * config.sea_breeze_strength
    wy = base_y + sb_y * sb_strength * config.sea_breeze_strength
    cx, cy = _hex_to_xy(h)
    freq = config.hex_size_km / config.wind_jitter_wavelength_km
    jitter_sample = jitter_noise.sample(cx * freq, cy * freq)
    jitter_rad = jitter_sample * math.radians(config.wind_jitter_amplitude_deg)
    cos_j, sin_j = math.cos(jitter_rad), math.sin(jitter_rad)
    rx = cos_j * wx - sin_j * wy
    ry = sin_j * wx + cos_j * wy
    return math.atan2(ry, rx)


def _moisture_along_path(
    target: Hex,
    theta: float,
    elevation: ElevationLayer,
    sea: SeaLayer,
    temperatures: dict[Hex, float],
    max_steps: int,
    config: WorldgenConfig,
) -> float:
    """Walk upwind from ``target`` along cartesian direction ``theta`` and
    return the moisture deposited at ``target``."""
    dq_step, dr_step = _axial_step_for_theta(theta)
    moisture = 0.0
    deposited_here = 0.0
    prev_elev = elevation.sea_level
    prev_hex: Hex | None = None
    for step in range(max_steps, -1, -1):
        qf = target.q - dq_step * step
        rf = target.r - dr_step * step
        cur_hex = _hex_round(qf, rf)
        if cur_hex not in elevation.elevation:
            continue
        if sea.is_ocean[cur_hex]:
            t = temperatures[cur_hex]
            warmth = max(0.05, min(1.0, (t + 5.0) / 35.0))
            moisture += warmth * config.precip_max_ocean_pickup
            prev_elev = elevation.sea_level
        else:
            deposit = moisture * config.precip_loss_per_land
            cur_elev = elevation.elevation[cur_hex]
            dh = cur_elev - prev_elev
            if dh > 0:
                deposit += dh * config.precip_orographic_coef
            deposit = min(deposit, moisture)
            moisture -= deposit
            prev_elev = cur_elev
            if cur_hex == target:
                deposited_here = deposit
                break
        prev_hex = cur_hex
    _ = prev_hex  # quiet linters: kept for future debugging
    return deposited_here


def compute_wind_directions(
    elevation: ElevationLayer,
    sea: SeaLayer,
    half_height_km: float,
    config: WorldgenConfig,
    rng: RngHierarchy,
) -> dict[Hex, tuple[float, float]]:
    """Per-hex wind direction as a unit vector in cartesian screen space.

    Combines zonal latitude band + sea-breeze onshore component + Perlin
    jitter (the same combination ``compute_precipitation`` consumes). Run
    once and reused so both the moisture sweep and the wind preview see the
    same field.
    """
    jitter_noise = PerlinNoise2D.from_rng(rng.child("worldgen", "climate", "wind_jitter"))
    sea_breeze_field = _compute_sea_breeze_field(elevation, sea, config)
    out: dict[Hex, tuple[float, float]] = {}
    for h in elevation.elevation:
        theta = _per_hex_wind_theta(
            h, half_height_km, sea_breeze_field, jitter_noise, config,
        )
        out[h] = (math.cos(theta), math.sin(theta))
    return out


def compute_precipitation(
    elevation: ElevationLayer,
    sea: SeaLayer,
    ocean: OceanLayer,
    wind_direction: dict[Hex, tuple[float, float]],
    temperatures: dict[Hex, float],
    config: WorldgenConfig,
    rng: RngHierarchy,
) -> dict[Hex, float]:
    """Per-hex annual precipitation in mm.

    For each target hex, build a per-hex wind direction (zonal + sea-breeze
    + Perlin jitter), then average the moisture-sweep deposit over
    ``wind_path_samples`` paths spread evenly across ±``wind_path_spread_deg``
    around that base direction. Each individual sweep walks upwind in
    fractional axial space (one hex of physical distance per step),
    accumulates moisture over ocean and deposits with orographic uplift
    over land.
    """
    precip_noise = PerlinNoise2D.from_rng(rng.child("worldgen", "climate", "precip_noise"))

    # Cap the upwind walk at the world's longest dimension (derived from
    # the input hex set's bounding box), so we don't loop forever on a
    # small test world but cover the full map on a large one.
    half_w, half_h = map_half_extents_km(
        elevation.elevation.keys(), config.hex_size_km,
    )
    world_span_km = 2.0 * max(half_w, half_h)
    span_hexes = max(40, int(world_span_km / config.hex_size_km))
    max_steps = min(config.wind_reach_hexes, span_hexes)

    n_paths = max(1, config.wind_path_samples)
    spread_rad = math.radians(config.wind_path_spread_deg)

    precipitation: dict[Hex, float] = {}
    for h in progress(
        elevation.elevation,
        desc="precipitation",
        total=len(elevation.elevation),
    ):
        if sea.is_ocean[h]:
            precipitation[h] = 0.0
            continue
        dx, dy = wind_direction[h]
        base_theta = math.atan2(dy, dx)
        # Sample N angles evenly spread across the cone, average their deposits.
        total = 0.0
        for i in range(n_paths):
            if n_paths == 1:
                offset = 0.0
            else:
                offset = (i / (n_paths - 1) - 0.5) * 2.0 * spread_rad
            theta = base_theta + offset
            total += _moisture_along_path(
                h, theta, elevation, sea, temperatures, max_steps, config,
            )
        deposited = total / n_paths

        n = precip_noise.sample(h.q * 0.03, h.r * 0.03)
        # Continentality: drier baseline far from any ocean. The upwind
        # moisture sweep already drops precip in continental interiors, but
        # this damps the *floor* too so deep interiors don't get a uniform
        # "minimum 240 mm" carpet they shouldn't have.
        d_to_ocean = ocean.distance_to_ocean_km.get(h, 0.0)
        continentality = math.exp(
            -d_to_ocean / config.ocean.continentality_dry_scale_km
        )
        base = config.precip_base * 0.15 * continentality
        value = base + deposited + n * config.precip_noise_amplitude
        precipitation[h] = max(0.0, value)

    # Final spatial smoothing: each pass averages land hexes with their
    # land neighbors (oceans untouched). Cleans up hex-scale roughness
    # introduced by the upwind hex-rounding and the discrete band model
    # without erasing the rain-shadow / coastal-gradient signal.
    for _ in range(max(0, config.precip_smoothing_passes)):
        precipitation = _smooth_precipitation(precipitation, sea)

    return precipitation


def _smooth_precipitation(
    precipitation: dict[Hex, float],
    sea: SeaLayer,
) -> dict[Hex, float]:
    """One uniform smoothing pass: each land hex → mean(self, land neighbors).

    Oceans are passed through unchanged. Edge hexes with fewer in-bounds
    land neighbors still produce a valid mean over whatever neighbors they
    do have, so the boundary isn't distorted.
    """
    out: dict[Hex, float] = {}
    for h, p in precipitation.items():
        if sea.is_ocean[h]:
            out[h] = p
            continue
        total = p
        count = 1
        for nb in h.neighbors():
            if nb in precipitation and not sea.is_ocean[nb]:
                total += precipitation[nb]
                count += 1
        out[h] = total / count
    return out


def compute(
    elevation: ElevationLayer,
    sea: SeaLayer,
    ocean: OceanLayer,
    config: WorldgenConfig,
    rng: RngHierarchy,
) -> ClimateLayer:
    """Compute temperature, wind directions, then precipitation.

    Derives the map's vertical half-extent (for latitude mapping) from
    the elevation layer's hex set once, then threads it through the
    sub-steps. No reference to ``config.world`` — the input hex set
    *is* the map.
    """
    _half_w, half_h = map_half_extents_km(
        elevation.elevation.keys(), config.hex_size_km,
    )
    temperatures = compute_temperature(
        elevation, sea, ocean, half_h, config, rng,
    )
    wind_direction = compute_wind_directions(
        elevation, sea, half_h, config, rng,
    )
    precipitation = compute_precipitation(
        elevation, sea, ocean, wind_direction, temperatures, config, rng,
    )
    return ClimateLayer(
        temperature_c=temperatures,
        precipitation_mm=precipitation,
        wind_direction=wind_direction,
    )
