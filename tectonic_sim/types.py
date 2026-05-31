"""Shared data types for ``tectonic_sim``.

Public surface:

  - ``WorldRect`` — toroidal simulation domain in km
  - ``SimConfig`` — physics tunables (read by polygon_sim + worldgen bridge)
  - ``CRUST_CONTINENTAL`` / ``CRUST_OCEANIC`` — int8 codes
  - ``crust_type_code`` / ``crust_type_name`` — string ↔ int

Crust type encoding: integer codes ``CRUST_CONTINENTAL = 0`` and
``CRUST_OCEANIC = 1`` rather than strings, so the per-cell field can
live in an ``int8`` array. Helpers convert to/from strings at the
public boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

# Type alias for "scalar or numpy array of floats" — used in WorldRect's
# wrap helpers which work on both single coordinates and bulk arrays.
FloatLike = Union[float, np.ndarray]


# Crust type encoding for integer arrays.
CRUST_CONTINENTAL: int = 0
CRUST_OCEANIC: int = 1

_CRUST_TYPE_NAMES = ("continental", "oceanic")


def crust_type_name(code: int) -> str:
    """Map an integer crust code to its string name."""
    return _CRUST_TYPE_NAMES[code]


def crust_type_code(name: str) -> int:
    """Map a crust type name to its integer code. Raises on unknown."""
    if name == "continental":
        return CRUST_CONTINENTAL
    if name == "oceanic":
        return CRUST_OCEANIC
    raise ValueError(f"unknown crust_type {name!r}")


@dataclass(frozen=True)
class WorldRect:
    """Simulation domain in km, centred on (0, 0).

    The domain is **always** a torus — particles/cells that drift past
    one edge re-enter from the opposite edge, and cross-edge distance
    queries use the toroidal shortest-path metric. There is no other
    boundary mode.
    """

    width_km: float
    height_km: float

    @property
    def half_width_km(self) -> float:
        return self.width_km / 2.0

    @property
    def half_height_km(self) -> float:
        return self.height_km / 2.0

    @property
    def area_km2(self) -> float:
        return self.width_km * self.height_km

    # --- Toroidal geometry helpers ---

    def wrap_positions(self, positions_km: np.ndarray) -> np.ndarray:
        """Wrap an ``(N, 2)`` array of positions onto the toroidal domain.

        Coordinates are remapped to ``[-half_width, +half_width)`` ×
        ``[-half_height, +half_height)``. Returns a new array; input is
        not mutated.
        """
        out = np.empty_like(positions_km)
        out[:, 0] = (
            (positions_km[:, 0] + self.half_width_km) % self.width_km
            - self.half_width_km
        )
        out[:, 1] = (
            (positions_km[:, 1] + self.half_height_km) % self.height_km
            - self.half_height_km
        )
        return out

    def wrapped_delta_xy(
        self, dx: FloatLike, dy: FloatLike,
    ) -> tuple[FloatLike, FloatLike]:
        """Toroidal shortest-path delta for one or many ``(dx, dy)``."""
        wx = (dx + self.half_width_km) % self.width_km - self.half_width_km
        wy = (dy + self.half_height_km) % self.height_km - self.half_height_km
        return wx, wy

    def wrapped_distance_km(
        self,
        a_xy_km: np.ndarray,
        b_xy_km: np.ndarray,
    ) -> np.ndarray:
        """Toroidal Euclidean distance between paired points."""
        diff = a_xy_km - b_xy_km
        if diff.ndim == 1:
            dx, dy = self.wrapped_delta_xy(diff[0], diff[1])
            return float(np.hypot(dx, dy))
        wx, wy = self.wrapped_delta_xy(diff[:, 0], diff[:, 1])
        return np.hypot(wx, wy)


@dataclass(frozen=True)
class SimConfig:
    """Physics tunables for the rigid-polygon simulator.

    Loaded by ``config_loader.load_sim_config`` from a TOML table.
    Every field is required (no defaults at construction time so
    missing-config bugs surface at load, not as silent zeros
    downstream). All fields are threaded through the per-tick polygon-
    sim modules — there is no shadow set of module-level constants any
    more.

    Groups:

      - plate population (count, fraction, motion cap, seed bias)
      - sim duration (n_ticks, dt_myr)
      - crust thicknesses (continental, oceanic, rift)
      - half-space cooling (ridge_depth, subsidence_rate, max_ocean_depth)
      - continental isostasy (reference_thickness, factor) + sea level
      - collision (orogeny, folding_ratio, folding_displacement,
        subduction_arc)
      - velocity damping, erosion, snapshot capture
      - Voronoi seeding (warp_*, weight_*, init_thickness_*)
      - per-cell physics (init_angular_velocity, momentum_*, fusion_*,
        accretion_*, hotspot_*, rift_*, buoyancy_*, alpha_factor,
        min_continental_thickness, fragment_spawn_threshold)
    """

    # --- Plate population ---
    plate_count: int
    continental_fraction: float
    motion_speed_kmpy: float
    seed_radial_bias: float                       # 0 = uniform, >0 = centre, <0 = edge
    # Minimum plate-seed separation hint. Kept under the legacy name
    # for backward compatibility with the TOML; polygon_sim repurposes
    # it as a "min plate-seed separation" floor.
    particle_spacing_km: float

    # --- Sim duration ---
    n_ticks: int
    dt_myr: float

    # --- Crust thicknesses ---
    continental_thickness_km: float
    oceanic_thickness_km: float
    rift_thickness_km: float

    # --- Half-space cooling (oceanic floor depth) ---
    ridge_depth_km: float
    ridge_subsidence_rate: float
    max_ocean_depth_km: float

    # --- Continental isostasy ---
    continental_reference_thickness_km: float
    continental_isostasy_factor: float
    sea_level_km: float

    # --- Collision ---
    orogeny_uplift_per_overlap_km: float
    folding_ratio: float
    folding_displacement_km: float
    subduction_arc_uplift_km: float
    # Continental-continental fold-and-thrust belt. Each tick, the
    # contested-cell fold mass is distributed inland (opposite the
    # over-rider's velocity) across a band of depth
    # ``folding_belt_depth_km``, with weights decaying exponentially
    # with e-folding scale ``folding_belt_decay_km``. Setting depth ≤
    # cell size collapses the band to the legacy suture-only deposit.
    folding_belt_depth_km: float
    folding_belt_decay_km: float
    # Loser-side fold belt — narrower, sharper inland deposit on the
    # *down-going* plate's near-suture interior. Models the Himalayan
    # foothill / Lesser-Himalaya pattern: slices of the underthrusting
    # plate get scraped off and stacked along the suture on its own
    # side. ``folding_loser_side_ratio`` is the fraction of the loser's
    # cell thickness redeposited back onto the loser (in addition to
    # ``folding_ratio`` going to the over-rider). Sum of the two ratios
    # should be ≤ 1 to avoid creating mass; the remainder represents
    # crust "subducted to mantle". Belt starts one cell into the loser's
    # interior (the suture itself now belongs to the over-rider).
    folding_loser_side_ratio: float
    folding_belt_loser_depth_km: float
    folding_belt_loser_decay_km: float

    # --- Velocity damping ---
    velocity_damping_strength: float

    # --- Erosion / snapshot capture ---
    erosion_period: int
    erosion_strength: float
    snapshot_period_ticks: int

    # =====================================================================
    # Polygon-sim per-cell physics fields. All read by the matching
    # polygon_sim phase module and perturbable by
    # ``randomize_sim_config``.
    # =====================================================================
    init_speed_min_ratio: float
    plate_area_per_plate_km2: float
    init_angular_velocity_max_rad_per_myr: float
    angular_damping_multiplier: float
    momentum_restitution: float
    momentum_contact_boost: float
    fusion_overlap_threshold: float
    fusion_both_continental_only: bool
    accretion_prob_per_boundary_per_tick: float
    accretion_cells_per_event: int
    accretion_inland_offset_min_km: float
    accretion_inland_offset_max_km: float
    rift_prob_per_tick: float
    rift_min_plate_cells: int
    rift_divergence_ratio: float
    hotspot_density_per_km2: float
    hotspot_erupt_prob_per_tick: float
    hotspot_thickness_bump_km: float
    hotspot_island_radius_km: float
    hotspot_birth_stagger_ticks: int
    hotspot_lifespan_mean_ticks: float
    hotspot_lifespan_std_ticks: float

    # =====================================================================
    # Polygon-sim physics tunables that USED to live as module-level
    # constants in ``tectonic_sim.polygon_sim.types``. Moved here so
    # there's a single source of truth for everything tunable. The
    # polygon_sim physics functions read these via the SimConfig that
    # gets threaded through every per-tick phase.
    # =====================================================================

    # Cell-grid resolution. Polygon sim derives (gy, gx) from
    # (domain, target_cell_km). Smaller cell → finer grid + higher cost.
    target_cell_km: float
    # Scales the maximum plate translation speed relative to the
    # geometric cap (min(sim_w, sim_h) / total_time). 0.5 = at most
    # half the world over the run.
    translation_speed_ratio: float

    # Released-component handling: components ≥ threshold spawn as new
    # plates; smaller redistribute to neighbours.
    fragment_spawn_threshold: int

    # Alpha-complex circumradius cutoffs (× cell_km). Used by polygon
    # extraction and initial-state alpha-complex build.
    alpha_factor: float
    init_alpha_factor: float

    # Continental priority multiplier in per-cell contention. At 50, a
    # 1-cell continental plate has the same priority as a 50-cell
    # oceanic plate.
    crust_continental_weight: float

    # Pyplatec buoyancy bonus on young oceanic crust. Decays linearly
    # to zero at ``max_buoyancy_age_myr``.
    buoyancy_bonus_frac: float
    max_buoyancy_age_myr: float

    # Continental cells thinned below this threshold (km) are absorbed
    # by their over-rider via thickness transfer to a neighbour, and
    # the cell reverts to oceanic. 0 disables.
    min_continental_thickness_km: float

    # Initial plate-shape naturalisation (Methods 1+2: domain warp +
    # power weights). All values per tick or per draw.
    voronoi_warp_amplitude_km: float
    voronoi_warp_sigma_cells: float
    voronoi_warp_jaggedness: float
    voronoi_warp_jagged_sigma_cells: float
    voronoi_weight_sigma: float
    voronoi_weight_scale_km: float

    # Initial thickness variation overlays (per-plate baseline + per-cell
    # noise field). 0 → uniform 35/7 km per crust type.
    init_thickness_per_plate_sigma: float
    init_thickness_noise_amplitude_frac: float
    init_thickness_noise_sigma_cells: float
