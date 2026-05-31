"""Bridge from the rigid-polygon ``tectonic_sim.polygon_sim`` package to
worldgen's hex-grid ``LithosphereState``.

Worldgen's downstream layers (elevation, climate, hydrology, biome,
preview, snapshot export) expect their tectonics input as the
dataclasses defined in ``worldgen/tectonics.py``: per-hex
``LithosphereColumn`` dicts, plate ownership keyed by ``Hex``. This
module:

  1. Optionally randomises the loaded ``SimConfig`` via
     ``randomize_sim_config(param_temperature)``.
  2. Derives ``plate_count`` and ``motion_speed_kmpy`` from the sim
     domain area + duration so plate population scales with world size.
  3. Runs ``simulate_rigid_polygon`` on a domain larger than the
     worldgen world (``_SIM_DOMAIN_MULTIPLIER``) so plates have room to
     drift without self-wrapping.
  4. Samples the final cell grid at every world-hex centre to build the
     ``LithosphereColumn`` / ``plate_id`` / ``elevation_km`` dicts.
  5. Packs the raw polygon-sim output onto
     ``LithosphereState.raw_snapshot`` so export-time renderers can
     produce the ``tectonic_sim_views/`` subdirectory without re-running
     the sim.
"""

from __future__ import annotations

import numpy as np

from worldgen._log import get_logger
from worldgen.hex import Hex
from worldgen.tectonics import (
    LithosphereColumn,
    LithosphereState,
    TectonicPlate,
)
from worldgen.types import TectonicsConfig, WorldShape
from worldgen.world import hex_to_xy_km

from tectonic_sim import (
    SimConfig,
    WorldRect,
    randomize_sim_config,
)
from tectonic_sim.polygon_sim import simulate_rigid_polygon
from tectonic_sim.polygon_sim.isostasy import particle_elevation_km
from tectonic_sim.types import CRUST_CONTINENTAL, CRUST_OCEANIC

_log = get_logger("tectonics_cast")


# Polygon-sim runs on a LARGER domain than the worldgen world. The
# central worldgen-sized region is what gets sampled for per-hex output
# and rendered as the cropped views in tectonic_sim_views/. Benefits:
#   - more plates (count scales with area),
#   - more velocity variety per plate,
#   - plates can drift in/out of the cropped view without wrapping
#     onto themselves over the sim duration.
# 2.0 matches the prototype default; setting to 1.0 disables the
# larger-sim pattern (sim == world).
_SIM_DOMAIN_MULTIPLIER: float = 2.0


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def simulate_tectonics_via_continuous_sim(
    world_hexes: list[Hex],
    config: TectonicsConfig,
    world_shape: WorldShape,
    hex_size_km: float,
    seed: int,
    *,
    param_temperature: float = 0.0,
) -> LithosphereState:
    """Run the polygon-sim and cast its output to a ``LithosphereState``.

    Pure function of all inputs. The same ``(config, seed,
    param_temperature)`` triple always produces the same
    ``LithosphereState``. ``param_temperature > 0`` perturbs every
    randomizable field of the loaded ``SimConfig`` via
    ``randomize_sim_config`` before the run.
    """
    # ``config`` IS already a SimConfig — the worldgen config loader
    # loads ``config/tectonic_sim.toml`` directly and assigns the
    # result to ``WorldgenConfig.tectonics``.
    sim_cfg = config
    if param_temperature > 0:
        # Tag the random draw's seed with a magic word so changing
        # ``param_temperature`` doesn't reshuffle the *base* seed of
        # the simulation's own RNG hierarchy.
        sim_cfg = randomize_sim_config(
            sim_cfg, param_temperature, seed=seed ^ 0xC07E,
        )
        _log.info(
            "applied param_temperature=%.2f → plate_count=%d, "
            "sea_level=%+.2f km, motion_speed=%.1f km/Myr",
            param_temperature, sim_cfg.plate_count, sim_cfg.sea_level_km,
            sim_cfg.motion_speed_kmpy,
        )

    # The worldgen "world" is the visible region. The polygon sim runs
    # on a LARGER domain so plates have room to drift without self-
    # wrapping and so the plate population is denser. ``world_domain``
    # is what gets sampled at hex centres (matches the user-visible
    # world). ``sim_domain`` is what the simulator sees.
    world_domain = WorldRect(
        width_km=world_shape.width_km, height_km=world_shape.height_km,
    )
    sim_domain = WorldRect(
        width_km=world_shape.width_km * _SIM_DOMAIN_MULTIPLIER,
        height_km=world_shape.height_km * _SIM_DOMAIN_MULTIPLIER,
    )

    # Override plate_count and motion_speed_kmpy with prototype-style
    # domain-derived values. The default SimConfig.plate_count and
    # motion_speed are tuned for a tiny test world, not for the larger
    # sim_domain that worldgen uses. Without these overrides we get ~5
    # plates on a 4M km² sim (prototype gets ~80) and motion speeds
    # uncalibrated to the sim duration.
    from dataclasses import replace as _dc_replace
    elapsed_total = sim_cfg.n_ticks * sim_cfg.dt_myr
    motion_cap = (
        sim_cfg.translation_speed_ratio
        * min(sim_domain.width_km, sim_domain.height_km)
        / max(elapsed_total, 1e-6)
    )
    if sim_cfg.plate_area_per_plate_km2 > 0:
        plate_count = max(
            5,
            int(round(sim_domain.width_km * sim_domain.height_km
                      / sim_cfg.plate_area_per_plate_km2)),
        )
    else:
        plate_count = sim_cfg.plate_count
    sim_cfg = _dc_replace(
        sim_cfg, plate_count=plate_count, motion_speed_kmpy=motion_cap,
    )
    _log.info(
        "polygon_sim domain overrides: sim=%.0fx%.0f km, "
        "plate_count=%d, motion_speed=%.2f km/Myr (cap from "
        "%.2f x min(sim_w, sim_h) / elapsed=%g Myr)",
        sim_domain.width_km, sim_domain.height_km,
        plate_count, motion_cap, sim_cfg.translation_speed_ratio, elapsed_total,
    )

    # Polygon-sim runs on the larger sim_domain and returns FULL-SIM
    # arrays + FULL-SIM captured frames. Worldgen reads them as-is:
    #   - per-hex sampling reads the full sim grid directly (the hex's
    #     world-frame km coordinate maps through wrap into the sim's
    #     central region — no cropping needed);
    #   - tectonic_sim_views/* PNGs and GIFs render the full sim, so
    #     plate drift outside the world is visible (matching the
    #     polygons.png convention).
    # Worldgen's own per-hex outputs (layers/elevation.png etc.) stay
    # world-sized because they're built from world_hexes.
    out = simulate_rigid_polygon(
        sim_domain, sim_cfg, seed=seed,
        capture_every=sim_cfg.snapshot_period_ticks,
        frame_upscale=4,
    )
    (
        polygon_plates, owner, crust, age, thick, cell_km, timeline,
        frames, frames_thickness, frames_topography, hotspots,
    ) = out
    _log.info(
        "polygon_sim done: sim grid %dx%d (cell=%.1fkm, sim=%gx%g km, "
        "world=%gx%g km), %d alive plates, %d hotspots, %d frames",
        owner.shape[1], owner.shape[0], cell_km,
        sim_domain.width_km, sim_domain.height_km,
        world_domain.width_km, world_domain.height_km,
        sum(1 for p in polygon_plates if p.alive),
        len(hotspots), len(frames),
    )

    # Sample at hex centres using the FULL sim grid — hex coords are
    # in world coords (centred at 0,0), which sit inside the larger
    # sim grid's centre. The sampler handles the index math.
    columns, plate_id_per_hex, elevation_per_hex = (
        _sample_polygon_at_hex_centres(
            owner, crust, age, thick, cell_km, sim_domain,
            sim_cfg, world_hexes, hex_size_km,
        )
    )

    tectonic_plates = _build_polygon_tectonic_plates(polygon_plates)

    return LithosphereState(
        columns=columns,
        plate_id=plate_id_per_hex,
        elevation_km=elevation_per_hex,
        sea_level_km=sim_cfg.sea_level_km,
        n_ticks_simulated=sim_cfg.n_ticks,
        plates=tectonic_plates,
        # Pack the raw polygon-sim output so export-time renderers can
        # produce the ``tectonic_sim_views/`` subdirectory directly via
        # ``tectonic_sim.polygon_sim`` renderers.
        raw_snapshot={
            "kind": "polygon_sim",
            "plates": polygon_plates,
            "owner": owner,
            "crust": crust,
            "age": age,
            "thickness": thick,
            "cell_km": cell_km,
            "sim_domain": sim_domain,
            "world_domain": world_domain,
            "sim_config": sim_cfg,
            "hotspots": hotspots,
            "timeline": timeline,
            "frames": frames,
            "frames_thickness": frames_thickness,
            "frames_topography": frames_topography,
        },
    )


# ---------------------------------------------------------------------------
# Polygon-sim: cell-grid → per-hex sampling
# ---------------------------------------------------------------------------


def _sample_polygon_at_hex_centres(
    owner: np.ndarray,
    crust: np.ndarray,
    age: np.ndarray,
    thick: np.ndarray,
    cell_km: float,
    domain: WorldRect,
    sim_config: SimConfig,
    world_hexes: list[Hex],
    hex_size_km: float,
) -> tuple[dict[Hex, LithosphereColumn], dict[Hex, int], dict[Hex, float]]:
    """Sample the polygon-sim cell grid at every world-hex centre.

    For each hex, project its centre to (x, y) km in the centred sim
    frame, convert to (cy, cx) cell index with toroidal wrap, and read
    the owner / crust / age / thickness arrays. Elevation comes from
    ``particle_elevation_km`` — the isostasy mapping that the polygon
    sim itself uses.
    """
    gy, gx = owner.shape
    half_w = 0.5 * gx * cell_km
    half_h = 0.5 * gy * cell_km

    columns: dict[Hex, LithosphereColumn] = {}
    plate_id_per_hex: dict[Hex, int] = {}
    elevation_per_hex: dict[Hex, float] = {}

    for h in world_hexes:
        x_km, y_km = hex_to_xy_km(h, hex_size_km)
        # Wrap to torus, then convert to cell index.
        wx = ((x_km + half_w) % (gx * cell_km)) / cell_km
        wy = ((y_km + half_h) % (gy * cell_km)) / cell_km
        cx = min(gx - 1, max(0, int(wx)))
        cy = min(gy - 1, max(0, int(wy)))

        pid = int(owner[cy, cx])
        if pid < 0:
            # Unowned (rare — transient gap between rifted plates).
            # Treat as oceanic ridge crust.
            ctype = CRUST_OCEANIC
            t_km = sim_config.oceanic_thickness_km
            a_myr = 0.0
        else:
            ctype = int(crust[cy, cx])
            t_km = float(thick[cy, cx])
            a_myr = float(age[cy, cx])

        plate_id_per_hex[h] = pid

        # Elevation via isostasy.
        ct_arr = np.array([ctype], dtype=np.int8)
        th_arr = np.array([t_km], dtype=np.float64)
        age_arr = np.array([a_myr], dtype=np.float64)
        elev = float(
            particle_elevation_km(ct_arr, th_arr, age_arr, sim_config)[0]
        )
        elevation_per_hex[h] = elev

        ctype_str = "continental" if ctype == CRUST_CONTINENTAL else "oceanic"
        columns[h] = LithosphereColumn(
            crust_type=ctype_str,
            thickness_km=t_km,
            age_myr=a_myr,
        )

    return columns, plate_id_per_hex, elevation_per_hex


def _build_polygon_tectonic_plates(polygon_plates) -> tuple[TectonicPlate, ...]:
    """Build the ``TectonicPlate`` tuple worldgen exposes downstream.

    Type is whichever crust dominates the plate's owned cells
    (continental if any continental cell exists, otherwise oceanic).
    The ``center_km`` anchor is left at (0, 0) — precise km centroids
    aren't consumed by any current renderer; the polygon-sim renderer
    works off ``raw_snapshot`` directly.
    """
    plates_out: list[TectonicPlate] = []
    for p in polygon_plates:
        if not p.alive or not p.cell_mask.any():
            continue
        cont_cells = int((p.cell_mask & (p.crust == CRUST_CONTINENTAL)).sum())
        crust_str = "continental" if cont_cells > 0 else "oceanic"
        plates_out.append(TectonicPlate(
            id=p.pid,
            initial_type=crust_str,
            center_km=(0.0, 0.0),
            velocity_kmpy=(float(p.velocity_kmpy[0]), float(p.velocity_kmpy[1])),
        ))
    return tuple(plates_out)
