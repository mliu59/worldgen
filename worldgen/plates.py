"""Plate-tectonics initial-condition layer.

Generates the t=0 state for the dynamic tectonics simulation: seeds plates by
rejection sampling, classifies each as continental or oceanic, assigns a
random motion vector, and Voronoi-classifies every world hex to its initial
plate. The boundary classification (cc_convergent / oc_convergent / etc.) is
computed on the initial state and exposed for inspection, but the simulation
re-derives convergence locally per tick from live velocities.

Elevation is *not* assigned here. The dynamic tectonics simulation evolves
crust thickness and age over many Myr and produces the elevation field that
the elevation layer then modulates with fBm/ridged detail.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from worldgen.rng import RngHierarchy
from worldgen.hex import Hex
from tectonic_sim.noise import PerlinNoise2D
from worldgen.types import PlateConfig
from worldgen.world import map_half_extents_km, normalized_radial_position

PLATE_TYPE_CONTINENTAL = "continental"
PLATE_TYPE_OCEANIC = "oceanic"

BOUNDARY_CC_CONVERGENT = "cc_convergent"
BOUNDARY_OC_CONVERGENT = "oc_convergent"
BOUNDARY_OO_CONVERGENT = "oo_convergent"
BOUNDARY_DIVERGENT = "divergent"
BOUNDARY_TRANSFORM = "transform"


@dataclass(frozen=True, slots=True)
class Plate:
    """One tectonic plate at t=0.

    ``motion`` is the *physical* per-Myr velocity vector in cartesian km
    space — ``unit_vector × PlateConfig.motion_speed``. The tectonics
    simulator integrates this directly as km/Myr to advance the plate's
    centre over geological time.
    """

    id: int
    seed_hex: Hex
    type: str  # PLATE_TYPE_CONTINENTAL | PLATE_TYPE_OCEANIC
    motion: tuple[float, float]


@dataclass(frozen=True)
class PlateField:
    """Initial-condition output of the plates layer.

    ``hex_to_plate`` maps every hex to its t=0 plate id. ``distance_by_type``,
    ``boundary_type`` and ``distance_to_boundary_km`` are the initial-state
    classification of plate boundaries — useful for inspecting the seeded
    world before the simulation runs. The dynamic tectonics simulator
    produces its own final-state boundary classification.
    """

    plates: tuple[Plate, ...]
    hex_to_plate: dict[Hex, int]
    distance_by_type: dict[str, dict[Hex, float]]
    boundary_type: dict[Hex, str | None]
    distance_to_boundary_km: dict[Hex, float]


def _hex_to_xy(h: Hex) -> tuple[float, float]:
    """Same projection the elevation layer uses; kept in sync intentionally."""
    return 1.5 * h.q, math.sqrt(3.0) * (h.r + h.q / 2.0)


def _hex_distance_km(a: Hex, b: Hex, hex_size_km: float) -> float:
    """Approximate cartesian distance between two hex centers, in km."""
    ax, ay = _hex_to_xy(a)
    bx, by = _hex_to_xy(b)
    return math.hypot(ax - bx, ay - by) * hex_size_km


def _place_seeds(
    hexes: list[Hex],
    config: PlateConfig,
    hex_size_km: float,
    rng,
) -> list[Hex]:
    """Place plate seeds via rejection sampling.

    Iterates a shuffled hex list; accepts a hex as a seed if it is at least
    ``min_separation_km`` from every previously-accepted seed. If we cannot
    place ``count`` seeds before exhausting the hex list, the separation is
    relaxed and we try again — this keeps the function total rather than
    raising on a config the user can't easily diagnose.

    The world's half-extents (for normalising the seed radial bias) are
    derived from the input hex set's bounding box, so this layer never
    needs a separate world-shape parameter.
    """
    # Sort hexes by radial bias: positive bias prefers center, negative
    # prefers edge. Normalised pixel-distance-to-nearest-edge (0 at the
    # centre, 1 at any world edge) becomes the sort key, weighted by the
    # configured bias. Symmetric across x and y so wide-short and tall-thin
    # worlds behave the same.
    candidates = list(hexes)
    half_w, half_h = map_half_extents_km(hexes, hex_size_km)
    if config.seed_radial_bias != 0.0 and half_w > 0 and half_h > 0:
        def radial_key(h: Hex) -> float:
            d = normalized_radial_position(h, half_w, half_h, hex_size_km)
            jitter = rng.random() * 0.15  # small noise so ties break randomly
            # Positive bias → center first → key ~ d; negative bias → edge first → key ~ -d
            return config.seed_radial_bias * d + jitter
        candidates.sort(key=radial_key)
    else:
        rng.shuffle(candidates)

    separation = config.min_separation_km
    for _ in range(8):  # at most a few relaxations
        seeds: list[Hex] = []
        for h in candidates:
            if all(_hex_distance_km(h, s, hex_size_km) >= separation for s in seeds):
                seeds.append(h)
                if len(seeds) == config.count:
                    return seeds
        separation *= 0.7  # relax and try again
    # Final fallback: just take the first `count` shuffled candidates regardless.
    return candidates[: config.count]


def _classify_plate_type(continental_fraction: float, rng) -> str:
    return PLATE_TYPE_CONTINENTAL if rng.random() < continental_fraction else PLATE_TYPE_OCEANIC


def _random_unit_vector(rng) -> tuple[float, float]:
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return math.cos(theta), math.sin(theta)


def _assign_hex_to_plate(
    h: Hex,
    seeds_xy: list[tuple[int, float, float]],
    warp_x: PerlinNoise2D,
    warp_y: PerlinNoise2D,
    warp_freq: float,
    warp_strength_hex: float,
) -> int:
    """Return the plate id assigned to ``h`` — nearest warped seed."""
    x, y = _hex_to_xy(h)
    wx = warp_x.sample(x * warp_freq, y * warp_freq) * warp_strength_hex
    wy = warp_y.sample(x * warp_freq + 11.1, y * warp_freq + 5.3) * warp_strength_hex
    qx, qy = x + wx, y + wy

    best_id = seeds_xy[0][0]
    best_d2 = (qx - seeds_xy[0][1]) ** 2 + (qy - seeds_xy[0][2]) ** 2
    for pid, sx, sy in seeds_xy[1:]:
        d2 = (qx - sx) ** 2 + (qy - sy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_id = pid
    return best_id


def _classify_boundary(
    plate_a: Plate,
    plate_b: Plate,
    threshold: float,
) -> str:
    """Classify the boundary between two adjacent plates.

    The inter-plate normal is the unit vector from A's seed to B's seed. The
    sign of (B.motion - A.motion) · normal tells us whether the plates are
    closing (negative — convergent) or opening (positive — divergent) along
    their shared boundary. Magnitude below ``threshold`` is transform.
    """
    ax, ay = _hex_to_xy(plate_a.seed_hex)
    bx, by = _hex_to_xy(plate_b.seed_hex)
    nx, ny = bx - ax, by - ay
    n_norm = math.hypot(nx, ny)
    if n_norm == 0:
        return BOUNDARY_TRANSFORM
    nx, ny = nx / n_norm, ny / n_norm
    rel_x = plate_b.motion[0] - plate_a.motion[0]
    rel_y = plate_b.motion[1] - plate_a.motion[1]
    projection = rel_x * nx + rel_y * ny
    if projection > threshold:
        return BOUNDARY_DIVERGENT
    if projection < -threshold:
        # Convergent — refine by plate-type pair
        a_cont = plate_a.type == PLATE_TYPE_CONTINENTAL
        b_cont = plate_b.type == PLATE_TYPE_CONTINENTAL
        if a_cont and b_cont:
            return BOUNDARY_CC_CONVERGENT
        if not a_cont and not b_cont:
            return BOUNDARY_OO_CONVERGENT
        return BOUNDARY_OC_CONVERGENT
    return BOUNDARY_TRANSFORM


_BOUNDARY_TYPES: tuple[str, ...] = (
    BOUNDARY_CC_CONVERGENT,
    BOUNDARY_OC_CONVERGENT,
    BOUNDARY_OO_CONVERGENT,
    BOUNDARY_DIVERGENT,
    BOUNDARY_TRANSFORM,
)


def _bfs_distance_by_type(
    hexes: list[Hex],
    hex_to_plate: dict[Hex, int],
    plates: tuple[Plate, ...],
    threshold: float,
    hex_size_km: float,
) -> tuple[
    dict[str, dict[Hex, float]],
    dict[Hex, str | None],
    dict[Hex, float],
]:
    """Compute per-boundary-type distance maps and the per-hex nearest-type.

    For each boundary type, runs a multi-source BFS from all boundary hexes
    classified as that type. Returns:

    - ``distance_by_type[type][hex]`` — BFS distance in km to the nearest
      boundary of ``type``; ``math.inf`` if none exists in the world or none
      reachable.
    - ``boundary_type[hex]`` — the single type whose distance is smallest at
      this hex (the "dominant" type used for inspector / renderer).
    - ``distance_to_boundary_km[hex]`` — the minimum over all types.

    The previous design propagated a single boundary type via BFS, so a hex
    deep in a plate inherited *one* type's amplitude. At Y-junctions where
    boundaries of different types meet, adjacent hexes could inherit
    different types, producing a visible step in elevation (e.g. cc_conv
    +0.65 next to divergent −0.30). Computing per-type distance maps lets
    ``plate_elevation_bias`` sum smooth, decayed contributions from *all*
    nearby boundary types, removing those steps.
    """
    hex_set = set(hexes)
    plate_by_id = {p.id: p for p in plates}

    # 1. Classify every boundary hex by all boundary types it participates
    # in (a single hex on a triple-junction may sit on more than one type;
    # we record it as a source for each).
    sources_by_type: dict[str, list[Hex]] = {t: [] for t in _BOUNDARY_TYPES}
    dominant_type: dict[Hex, str | None] = {h: None for h in hexes}

    for h in hexes:
        pid = hex_to_plate[h]
        plate_a = plate_by_id[pid]
        types_here: set[str] = set()
        best_type: str | None = None
        best_priority = -1
        for nb in h.neighbors():
            if nb not in hex_set:
                continue
            nb_pid = hex_to_plate[nb]
            if nb_pid == pid:
                continue
            plate_b = plate_by_id[nb_pid]
            btype = _classify_boundary(plate_a, plate_b, threshold)
            types_here.add(btype)
            # Track the dominant type for interpretability.
            priority = _boundary_priority(btype)
            if priority > best_priority:
                best_priority = priority
                best_type = btype
        if types_here:
            dominant_type[h] = best_type
            for t in types_here:
                sources_by_type[t].append(h)

    # 2. Per-type BFS from each boundary set.
    step_km = hex_size_km * math.sqrt(3.0)
    distance_by_type: dict[str, dict[Hex, float]] = {}
    for t, sources in sources_by_type.items():
        d: dict[Hex, float] = {h: math.inf for h in hexes}
        if not sources:
            distance_by_type[t] = d
            continue
        queue: deque[Hex] = deque()
        for s in sources:
            d[s] = 0.0
            queue.append(s)
        while queue:
            h = queue.popleft()
            current = d[h]
            for nb in h.neighbors():
                if nb not in hex_set:
                    continue
                new_d = current + step_km
                if new_d < d[nb]:
                    d[nb] = new_d
                    queue.append(nb)
        distance_by_type[t] = d

    # 3. Per-hex "nearest boundary" view derived from the type maps.
    distance_to_boundary_km: dict[Hex, float] = {}
    boundary_type: dict[Hex, str | None] = {}
    for h in hexes:
        # Boundary hexes themselves keep their dominant classification; for
        # them the minimum-distance type is whichever they sit on, and the
        # priority rule above already picked the most expressive one.
        if dominant_type[h] is not None:
            boundary_type[h] = dominant_type[h]
            distance_to_boundary_km[h] = 0.0
            continue
        best_t: str | None = None
        best_d = math.inf
        for t in _BOUNDARY_TYPES:
            dt = distance_by_type[t][h]
            if dt < best_d:
                best_d = dt
                best_t = t
        boundary_type[h] = best_t
        distance_to_boundary_km[h] = best_d

    return distance_by_type, boundary_type, distance_to_boundary_km


def _boundary_priority(btype: str) -> int:
    """Tie-break for T-junctions: pick the more interpretable boundary."""
    return {
        BOUNDARY_CC_CONVERGENT: 5,
        BOUNDARY_OC_CONVERGENT: 4,
        BOUNDARY_OO_CONVERGENT: 3,
        BOUNDARY_DIVERGENT: 2,
        BOUNDARY_TRANSFORM: 1,
    }.get(btype, 0)


def generate_plates(
    hexes: Iterable[Hex],
    plate_config: PlateConfig,
    hex_size_km: float,
    rng: RngHierarchy,
) -> PlateField:
    """Build the full PlateField for the world.

    Pure function of (hex set, config, hex_size_km, seed). Each sub-step
    gets its own child RNG so reordering or adding plates later doesn't
    reshuffle earlier random draws. The world's geometry is taken from
    the hex set's bounding box; no separate shape parameter is needed.
    """
    hex_list = list(hexes)

    seeds_rng = rng.child("worldgen", "plates", "seeds")
    seeds = _place_seeds(hex_list, plate_config, hex_size_km, seeds_rng)

    plates: list[Plate] = []
    for i, seed_hex in enumerate(seeds):
        ptype_rng = rng.child("worldgen", "plates", "plate", i, "type")
        ptype = _classify_plate_type(plate_config.continental_fraction, ptype_rng)
        motion_rng = rng.child("worldgen", "plates", "plate", i, "motion")
        mx, my = _random_unit_vector(motion_rng)
        plates.append(Plate(
            id=i,
            seed_hex=seed_hex,
            type=ptype,
            motion=(mx * plate_config.motion_speed, my * plate_config.motion_speed),
        ))

    # Domain-warp noise for boundary irregularity.
    warp_x = PerlinNoise2D.from_rng(rng.child("worldgen", "plates", "warp_x"))
    warp_y = PerlinNoise2D.from_rng(rng.child("worldgen", "plates", "warp_y"))
    warp_freq = hex_size_km / plate_config.boundary_warp_wavelength_km
    warp_strength_hex = plate_config.boundary_warp_strength_km / hex_size_km

    seeds_xy: list[tuple[int, float, float]] = [
        (p.id, *_hex_to_xy(p.seed_hex)) for p in plates
    ]

    hex_to_plate: dict[Hex, int] = {
        h: _assign_hex_to_plate(
            h, seeds_xy, warp_x, warp_y, warp_freq, warp_strength_hex,
        )
        for h in hex_list
    }

    distance_by_type, boundary_type, distance_km = _bfs_distance_by_type(
        hex_list, hex_to_plate, tuple(plates),
        plate_config.convergence_threshold, hex_size_km,
    )

    return PlateField(
        plates=tuple(plates),
        hex_to_plate=hex_to_plate,
        distance_by_type=distance_by_type,
        boundary_type=boundary_type,
        distance_to_boundary_km=distance_km,
    )
