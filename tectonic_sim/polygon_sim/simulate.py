"""Main per-tick orchestrator: simulate_rigid_polygon."""

from __future__ import annotations

import hashlib

import numpy as np
from tqdm.auto import tqdm

from tectonic_sim.types import CRUST_CONTINENTAL, WorldRect

from tectonic_sim.polygon_sim.accretion import _apply_co_accretion
from tectonic_sim.polygon_sim.aging import (
    _apply_aging,
    _apply_buoyancy_to_thickness,
    _apply_erosion,
    _apply_thinned_continental_absorption)
from tectonic_sim.polygon_sim.contention import _global_owner, _resolve_contention
from tectonic_sim.polygon_sim.culling import _cull_disconnected
from tectonic_sim.polygon_sim.damping import _apply_velocity_damping
from tectonic_sim.polygon_sim.divergent import _trailing_edge_fill
from tectonic_sim.polygon_sim.fusion import _apply_fusion
from tectonic_sim.polygon_sim.hotspots import (
    _apply_hotspot_eruptions,
    _apply_hotspot_prehistory,
    _initialize_hotspots)
from tectonic_sim.polygon_sim.kinematics import _rotate_plates, _stamp_paint
from tectonic_sim.polygon_sim.momentum import _apply_momentum_exchange
from tectonic_sim.polygon_sim.polygons import (
    _build_polygons_for_render,
    _mark_dead_small_plates)
from tectonic_sim.polygon_sim.rifting import _rift_plate
from tectonic_sim.polygon_sim.seeding import _initial_state
from tectonic_sim.polygon_sim.topology import _cell_centres, _grid_dims
from tectonic_sim.polygon_sim.types import (
    PolygonPlate,
    _ACCRETION_RNG_TAG,
    _HOTSPOT_RNG_TAG,
    _RIFT_RNG_TAG,
    _SPAWN_RNG_TAG)
from tectonic_sim.polygon_sim.viz import (
    _build_crust_image,
    _build_partition_image,
    _build_thickness_image,
    _build_topography_image,
    _combine_frame,
    _overlay_hotspots)


# Top-level simulation.
# ---------------------------------------------------------------------------


def simulate_rigid_polygon(
    domain: WorldRect, sim_config, seed: int,
    capture_every: int = 0, frame_upscale: int = 4):
    """Run the simulation.

    Returns ``(plates, owner, crust, age, thick, cell_km, timeline,
    frames)``. ``frames`` is a list of side-by-side partition+crust PIL
    Images, one per capture interval, or an empty list when
    ``capture_every == 0``. All output (returned ndarrays and captured
    frames) is the full sim domain — callers that want a sub-region
    apply their own slicing.
    """
    plates, cell_km = _initial_state(domain, sim_config, seed)
    gy, gx, _ = _grid_dims(domain, sim_config)
    cell_xy = _cell_centres(gy, gx, cell_km)
    rift_rng = np.random.Generator(np.random.PCG64(seed ^ _RIFT_RNG_TAG))
    spawn_rng = np.random.Generator(np.random.PCG64(seed ^ _SPAWN_RNG_TAG))
    hotspot_rng = np.random.Generator(np.random.PCG64(seed ^ _HOTSPOT_RNG_TAG))
    dt = sim_config.dt_myr
    divergence_kmpy = sim_config.rift_divergence_ratio * sim_config.motion_speed_kmpy
    elapsed_total = sim_config.n_ticks * dt

    hotspots = _initialize_hotspots(domain, sim_config, seed)
    # Pre-stamp trails for hotspots that have been active since before
    # t=0, so the initial frame already shows their footprint.
    n_prehistory = _apply_hotspot_prehistory(
        plates, hotspots,
        sim_config=sim_config, cell_km=cell_km, gy=gy, gx=gx,
        rng=hotspot_rng)
    for i, h in enumerate(hotspots):
        death = h.birth_tick + h.lifespan_ticks
        print(
            f"  hotspot H{i}: "
            f"pos=({h.position_xy_km[0]:+7.0f}, {h.position_xy_km[1]:+7.0f}) km   "
            f"born={h.birth_tick:+4d}   "
            f"lifespan={h.lifespan_ticks}   "
            f"dies@{death:+4d}"
        )

    timeline: list[tuple[int, int]] = [(0, sum(1 for p in plates if p.alive))]
    n_rifts = 0
    total_released = 0
    total_redistributed = 0
    total_spawned = 0
    total_hotspot_eruptions = n_prehistory
    # Three parallel frame lists:
    #   drift       — partition (left) + crust (right)
    #   thickness   — thickness heatmap + crust
    #   topography  — elevation map + crust
    # Same headers, same FPS — just different left-panel view.
    frames: list = []
    frames_thickness: list = []
    frames_topography: list = []

    def capture(tick: int) -> None:
        if capture_every <= 0:
            return
        owner, crust, age, thick = _flatten_state(plates, gy, gx)
        max_age = max(1.0, float(age.max()))
        elapsed = tick * dt
        n_alive = sum(1 for p in plates if p.alive)
        part = _build_partition_image(
            owner, "partition (cell-mask + edge boundaries)",
            cell_km=cell_km, upscale=frame_upscale)
        cr = _build_crust_image(
            owner, crust, age, thick, max_age, "crust (cont. tan / ocean age)",
            cell_km=cell_km, upscale=frame_upscale)
        thk = _build_thickness_image(
            owner, thick, "thickness (km: navy=0, cyan=12, gold=40, red=60+)",
            cell_km=cell_km, upscale=frame_upscale)
        topo = _build_topography_image(
            owner, crust, age, thick, sim_config,
            "topography (km: deep navy=-6, sandy=0, brown=3, white=5+)",
            cell_km=cell_km, upscale=frame_upscale)
        # Overlay hotspot markers on all per-tick frame panels.
        # Active = red filled disk; extinct = hollow grey ring (so the
        # provenance of older trails stays visible). No crop offset:
        # frames render the full sim, so the hotspot's mantle-frame
        # position maps directly onto the panel's cell grid.
        for _panel in (part, cr, thk, topo):
            _overlay_hotspots(
                _panel, hotspots, tick,
                cell_km=cell_km, gy=gy, gx=gx, upscale=frame_upscale,
                x0_cells=0, y0_cells=0,
                only_active=False)
        header = (
            f"tick={tick:3d}/{sim_config.n_ticks}   "
            f"t={elapsed:6.1f} Myr / {elapsed_total:g}   "
            f"alive plates = {n_alive}   "
            f"rifts = {n_rifts}   released = {total_released}   "
            f"spawned = {total_spawned}   "
            f"hotspots = {sum(1 for h in hotspots if h.is_active(tick))}"
            f"/{len(hotspots)}"
        )
        frames.append(_combine_frame(part, cr, header))
        frames_thickness.append(_combine_frame(thk, cr, header))
        frames_topography.append(_combine_frame(topo, cr, header))

    # Cull right away so any pre-existing nearest-particle Voronoi
    # fragments are dropped. Polygon construction is deferred to the
    # end of the sim — see _build_polygons_for_render below; the
    # alpha-complex is only consumed by polygons.png at render time,
    # so rebuilding it every tick was 36% of program self-time on
    # the v0.1 profile.
    rel, redist, spawn = _cull_disconnected(plates, spawn_rng, sim_config)
    total_released += rel
    total_redistributed += redist
    total_spawned += spawn
    _mark_dead_small_plates(plates)
    capture(0)

    total_collisions = 0
    total_fusions = 0
    total_accreted = 0
    total_absorbed = 0
    # Net continental cell change per tick (sum over the run). Positive
    # means accretion outweighs destruction; negative means continents
    # are eroding. Target: roughly 0 for steady-state mass balance.
    initial_cont = sum(
        int((p.cell_mask & (p.crust == CRUST_CONTINENTAL)).sum())
        for p in plates if p.alive
    )
    accretion_rng = np.random.Generator(
        np.random.PCG64(seed ^ _ACCRETION_RNG_TAG)
    )
    tick_iter = tqdm(
        range(1, sim_config.n_ticks + 1),
        desc="sim", unit="tick", leave=True, dynamic_ncols=True)
    for tick in tick_iter:
        prev_owner = _global_owner(plates, gy, gx)
        _stamp_paint(plates, dt, cell_km)
        _rotate_plates(plates, dt, domain, gy, gx, cell_km)
        # Momentum exchange BEFORE damping: per-pair normal-direction
        # impulse equalises approach velocity along contact normals.
        # Damping then handles residual tangential / self-friction.
        total_collisions += _apply_momentum_exchange(
            plates, domain, gy, gx, cell_km, sim_config)
        _apply_velocity_damping(plates, sim_config)
        total_fusions += _apply_fusion(plates, sim_config)
        _resolve_contention(plates, gy, gx, sim_config, cell_km)
        _trailing_edge_fill(plates, prev_owner, sim_config, gy, gx)
        _apply_aging(plates, dt)
        total_accreted += _apply_co_accretion(
            plates, gy, gx, sim_config, cell_km, rng=accretion_rng,
        )
        total_hotspot_eruptions += _apply_hotspot_eruptions(
            plates, hotspots, tick,
            sim_config=sim_config, cell_km=cell_km, gy=gy, gx=gx,
            rng=hotspot_rng)
        total_absorbed += _apply_thinned_continental_absorption(
            plates, sim_config)
        if sim_config.erosion_period > 0 and tick % sim_config.erosion_period == 0:
            _apply_erosion(plates, sim_config)
        rel, redist, spawn = _cull_disconnected(plates, spawn_rng, sim_config)
        total_released += rel
        total_redistributed += redist
        total_spawned += spawn
        _mark_dead_small_plates(plates)

        if rift_rng.random() < sim_config.rift_prob_per_tick:
            if _rift_plate(plates, domain, divergence_kmpy,
                           cell_km, gy, gx, rift_rng, sim_config):
                # The rift just split a plate into two halves — each half
                # is already one connected blob, but cull again to be safe
                # (e.g. wrap-aware splits along the seam).
                rel, redist, spawn = _cull_disconnected(plates, spawn_rng, sim_config)
                total_released += rel
                total_redistributed += redist
                total_spawned += spawn
                _mark_dead_small_plates(plates)
                n_rifts += 1

        if tick % 10 == 0:
            alive = sum(1 for p in plates if p.alive)
            timeline.append((tick, alive))
        if capture_every > 0 and tick % capture_every == 0:
            capture(tick)

    n_unown = total_released - total_redistributed
    final_cont = sum(
        int((p.cell_mask & (p.crust == CRUST_CONTINENTAL)).sum())
        for p in plates if p.alive
    )
    net_cont = final_cont - initial_cont
    # Approximate destruction = accretion - net_change (since destroyed +
    # accreted = (final - initial) doesn't quite work — accreted_into-cells
    # actually shows up in final). Simpler: net change tells the story.
    print(
        f"  {n_rifts} rift events fired, "
        f"{total_released} cells culled "
        f"({total_redistributed} preserved as transfer-or-spawn, "
        f"{n_unown} lost = should be 0 under mass-invariant), "
        f"{total_spawned} new plates spawned from fragments, "
        f"{total_collisions} momentum-exchange events, "
        f"{total_fusions} plate fusions, "
        f"{total_accreted} continental cells accreted at C-O boundaries, "
        f"{total_absorbed} thinned continental cells absorbed, "
        f"{total_hotspot_eruptions} hotspot eruption-cells stamped "
        f"({n_prehistory} pre-history + "
        f"{total_hotspot_eruptions - n_prehistory} live) "
        f"across {len(hotspots)} hotspots"
    )
    print(
        f"  continental mass balance: initial={initial_cont} "
        f"final={final_cont} net={net_cont:+d} "
        f"(accretion contributed +{total_accreted}; "
        f"net = creation - destruction)"
    )

    # Bake buoyancy for the render.
    _apply_buoyancy_to_thickness(plates, sim_config)

    # Build the per-plate alpha-complex ONCE, at the end. It's only
    # consumed by polygons.png at render time — building it every
    # tick during the sim was wasted work.
    _build_polygons_for_render(plates, domain, cell_xy, cell_km, sim_config)
    owner, crust, age, thick = _flatten_state(plates, gy, gx)
    return (
        plates, owner, crust, age, thick, cell_km, timeline,
        frames, frames_thickness, frames_topography, hotspots)


def _flatten_state(
    plates: list[PolygonPlate], gy: int, gx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine per-plate paint into single (gy, gx) arrays for rendering
    + fingerprinting. Where a cell is claimed by multiple plates (rare,
    transient — should not happen at steady state) the lowest pid wins."""
    owner = np.full((gy, gx), -1, dtype=np.int64)
    crust = np.zeros((gy, gx), dtype=np.int8)
    age = np.zeros((gy, gx), dtype=np.float64)
    thick = np.zeros((gy, gx), dtype=np.float64)
    # Sort plates by pid descending so lowest pid is written last (wins).
    for p in sorted(plates, key=lambda q: -q.pid):
        if not p.alive:
            continue
        m = p.cell_mask
        owner[m] = p.pid
        crust[m] = p.crust[m]
        age[m] = p.age[m]
        thick[m] = p.thickness[m]
    return owner, crust, age, thick


def _fingerprint(owner, crust, age, thick) -> bytes:
    h = hashlib.sha256()
    for arr in (owner, crust, age, thick):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.digest()

