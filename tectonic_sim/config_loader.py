"""Load a ``SimConfig`` from a TOML file or a pre-parsed table.

The module deliberately has *no* knowledge of where its config lives —
worldgen (or any other caller) is free to keep ``tectonic_sim.toml`` as a
standalone file or to inline the keys inside its own config. Two
entry points cover both cases:

  - ``load_sim_config_from_path(path)`` — read a TOML file, parse it.
  - ``load_sim_config(table)`` — parse a pre-loaded dict (the body of a
    TOML table the caller has already extracted).

Every field listed on ``SimConfig`` is required; missing keys raise at
parse time rather than defaulting silently downstream.
"""

from __future__ import annotations

from pathlib import Path

from tectonic_sim.types import SimConfig


_REQUIRED_KEYS: tuple[str, ...] = (
    "plate_count",
    "continental_fraction",
    "motion_speed_kmpy",
    "seed_radial_bias",
    "particle_spacing_km",
    "n_ticks",
    "dt_myr",
    "continental_thickness_km",
    "oceanic_thickness_km",
    "rift_thickness_km",
    "ridge_depth_km",
    "ridge_subsidence_rate",
    "max_ocean_depth_km",
    "continental_reference_thickness_km",
    "continental_isostasy_factor",
    "sea_level_km",
    # overlap_radius_km is *not* required — it's derived from
    # particle_spacing_km via SimConfig.overlap_radius_km.
    "orogeny_uplift_per_overlap_km",
    "folding_ratio",
    "folding_displacement_km",
    "subduction_arc_uplift_km",
    "folding_belt_depth_km",
    "folding_belt_decay_km",
    "folding_loser_side_ratio",
    "folding_belt_loser_depth_km",
    "folding_belt_loser_decay_km",
    "velocity_damping_strength",
    "erosion_period",
    "erosion_strength",
    "snapshot_period_ticks",
    # --- New features ported from rigid-polygon prototype (P1 of the port).
    "init_speed_min_ratio",
    "plate_area_per_plate_km2",
    "init_angular_velocity_max_rad_per_myr",
    "angular_damping_multiplier",
    "momentum_restitution",
    "momentum_contact_boost",
    "fusion_overlap_threshold",
    "fusion_both_continental_only",
    "accretion_prob_per_boundary_per_tick",
    "accretion_cells_per_event",
    "accretion_inland_offset_min_km",
    "accretion_inland_offset_max_km",
    "rift_prob_per_tick",
    "rift_min_plate_cells",
    "rift_divergence_ratio",
    "hotspot_density_per_km2",
    "hotspot_erupt_prob_per_tick",
    "hotspot_thickness_bump_km",
    "hotspot_island_radius_km",
    "hotspot_birth_stagger_ticks",
    "hotspot_lifespan_mean_ticks",
    "hotspot_lifespan_std_ticks",
    # --- Polygon-sim tunables migrated from polygon_sim/types.py ---
    "target_cell_km",
    "translation_speed_ratio",
    "fragment_spawn_threshold",
    "alpha_factor",
    "init_alpha_factor",
    "crust_continental_weight",
    "buoyancy_bonus_frac",
    "max_buoyancy_age_myr",
    "min_continental_thickness_km",
    "voronoi_warp_amplitude_km",
    "voronoi_warp_sigma_cells",
    "voronoi_warp_jaggedness",
    "voronoi_warp_jagged_sigma_cells",
    "voronoi_weight_sigma",
    "voronoi_weight_scale_km",
    "init_thickness_per_plate_sigma",
    "init_thickness_noise_amplitude_frac",
    "init_thickness_noise_sigma_cells",
)


def load_sim_config_from_path(path: Path) -> SimConfig:
    """Read a TOML file and parse it into a ``SimConfig``."""
    import tomllib

    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return load_sim_config(raw)


def load_sim_config(table: dict[str, object]) -> SimConfig:
    """Parse a TOML table into a ``SimConfig``.

    Validates that every required key is present. The simulation runs
    on a torus by default — there is no boundary-mode configuration
    field. Particles drifting past one edge re-enter from the opposite
    edge; every spatial query uses the toroidal shortest-path metric.
    """
    missing = [k for k in _REQUIRED_KEYS if k not in table]
    if missing:
        raise KeyError(
            f"tectonic_sim config missing required keys: {sorted(missing)}"
        )

    return SimConfig(
        plate_count=int(table["plate_count"]),  # type: ignore[arg-type]
        continental_fraction=float(table["continental_fraction"]),  # type: ignore[arg-type]
        motion_speed_kmpy=float(table["motion_speed_kmpy"]),  # type: ignore[arg-type]
        seed_radial_bias=float(table["seed_radial_bias"]),  # type: ignore[arg-type]
        particle_spacing_km=float(table["particle_spacing_km"]),  # type: ignore[arg-type]
        n_ticks=int(table["n_ticks"]),  # type: ignore[arg-type]
        dt_myr=float(table["dt_myr"]),  # type: ignore[arg-type]
        continental_thickness_km=float(table["continental_thickness_km"]),  # type: ignore[arg-type]
        oceanic_thickness_km=float(table["oceanic_thickness_km"]),  # type: ignore[arg-type]
        rift_thickness_km=float(table["rift_thickness_km"]),  # type: ignore[arg-type]
        ridge_depth_km=float(table["ridge_depth_km"]),  # type: ignore[arg-type]
        ridge_subsidence_rate=float(table["ridge_subsidence_rate"]),  # type: ignore[arg-type]
        max_ocean_depth_km=float(table["max_ocean_depth_km"]),  # type: ignore[arg-type]
        continental_reference_thickness_km=float(
            table["continental_reference_thickness_km"]  # type: ignore[arg-type]
        ),
        continental_isostasy_factor=float(table["continental_isostasy_factor"]),  # type: ignore[arg-type]
        sea_level_km=float(table["sea_level_km"]),  # type: ignore[arg-type]
        orogeny_uplift_per_overlap_km=float(
            table["orogeny_uplift_per_overlap_km"]  # type: ignore[arg-type]
        ),
        folding_ratio=float(table["folding_ratio"]),  # type: ignore[arg-type]
        folding_displacement_km=float(table["folding_displacement_km"]),  # type: ignore[arg-type]
        subduction_arc_uplift_km=float(table["subduction_arc_uplift_km"]),  # type: ignore[arg-type]
        folding_belt_depth_km=float(table["folding_belt_depth_km"]),  # type: ignore[arg-type]
        folding_belt_decay_km=float(table["folding_belt_decay_km"]),  # type: ignore[arg-type]
        folding_loser_side_ratio=float(table["folding_loser_side_ratio"]),  # type: ignore[arg-type]
        folding_belt_loser_depth_km=float(table["folding_belt_loser_depth_km"]),  # type: ignore[arg-type]
        folding_belt_loser_decay_km=float(table["folding_belt_loser_decay_km"]),  # type: ignore[arg-type]
        velocity_damping_strength=float(table["velocity_damping_strength"]),  # type: ignore[arg-type]
        erosion_period=int(table["erosion_period"]),  # type: ignore[arg-type]
        erosion_strength=float(table["erosion_strength"]),  # type: ignore[arg-type]
        snapshot_period_ticks=int(table["snapshot_period_ticks"]),  # type: ignore[arg-type]
        # --- Ported features (P1). Wiring lands in P2.
        init_speed_min_ratio=float(table["init_speed_min_ratio"]),  # type: ignore[arg-type]
        plate_area_per_plate_km2=float(
            table["plate_area_per_plate_km2"]  # type: ignore[arg-type]
        ),
        init_angular_velocity_max_rad_per_myr=float(
            table["init_angular_velocity_max_rad_per_myr"]  # type: ignore[arg-type]
        ),
        angular_damping_multiplier=float(
            table["angular_damping_multiplier"]  # type: ignore[arg-type]
        ),
        momentum_restitution=float(table["momentum_restitution"]),  # type: ignore[arg-type]
        momentum_contact_boost=float(table["momentum_contact_boost"]),  # type: ignore[arg-type]
        fusion_overlap_threshold=float(
            table["fusion_overlap_threshold"]  # type: ignore[arg-type]
        ),
        fusion_both_continental_only=bool(
            table["fusion_both_continental_only"]  # type: ignore[arg-type]
        ),
        accretion_prob_per_boundary_per_tick=float(
            table["accretion_prob_per_boundary_per_tick"]  # type: ignore[arg-type]
        ),
        accretion_cells_per_event=int(
            table["accretion_cells_per_event"]  # type: ignore[arg-type]
        ),
        accretion_inland_offset_min_km=float(
            table["accretion_inland_offset_min_km"]  # type: ignore[arg-type]
        ),
        accretion_inland_offset_max_km=float(
            table["accretion_inland_offset_max_km"]  # type: ignore[arg-type]
        ),
        rift_prob_per_tick=float(table["rift_prob_per_tick"]),  # type: ignore[arg-type]
        rift_min_plate_cells=int(
            table["rift_min_plate_cells"]  # type: ignore[arg-type]
        ),
        rift_divergence_ratio=float(
            table["rift_divergence_ratio"]  # type: ignore[arg-type]
        ),
        hotspot_density_per_km2=float(
            table["hotspot_density_per_km2"]  # type: ignore[arg-type]
        ),
        hotspot_erupt_prob_per_tick=float(
            table["hotspot_erupt_prob_per_tick"]  # type: ignore[arg-type]
        ),
        hotspot_thickness_bump_km=float(
            table["hotspot_thickness_bump_km"]  # type: ignore[arg-type]
        ),
        hotspot_island_radius_km=float(
            table["hotspot_island_radius_km"]  # type: ignore[arg-type]
        ),
        hotspot_birth_stagger_ticks=int(
            table["hotspot_birth_stagger_ticks"]  # type: ignore[arg-type]
        ),
        hotspot_lifespan_mean_ticks=float(
            table["hotspot_lifespan_mean_ticks"]  # type: ignore[arg-type]
        ),
        hotspot_lifespan_std_ticks=float(
            table["hotspot_lifespan_std_ticks"]  # type: ignore[arg-type]
        ),
        # --- Polygon-sim tunables (new) ---
        target_cell_km=float(table["target_cell_km"]),  # type: ignore[arg-type]
        translation_speed_ratio=float(
            table["translation_speed_ratio"]  # type: ignore[arg-type]
        ),
        fragment_spawn_threshold=int(
            table["fragment_spawn_threshold"]  # type: ignore[arg-type]
        ),
        alpha_factor=float(table["alpha_factor"]),  # type: ignore[arg-type]
        init_alpha_factor=float(table["init_alpha_factor"]),  # type: ignore[arg-type]
        crust_continental_weight=float(
            table["crust_continental_weight"]  # type: ignore[arg-type]
        ),
        buoyancy_bonus_frac=float(table["buoyancy_bonus_frac"]),  # type: ignore[arg-type]
        max_buoyancy_age_myr=float(
            table["max_buoyancy_age_myr"]  # type: ignore[arg-type]
        ),
        min_continental_thickness_km=float(
            table["min_continental_thickness_km"]  # type: ignore[arg-type]
        ),
        voronoi_warp_amplitude_km=float(
            table["voronoi_warp_amplitude_km"]  # type: ignore[arg-type]
        ),
        voronoi_warp_sigma_cells=float(
            table["voronoi_warp_sigma_cells"]  # type: ignore[arg-type]
        ),
        voronoi_warp_jaggedness=float(
            table["voronoi_warp_jaggedness"]  # type: ignore[arg-type]
        ),
        voronoi_warp_jagged_sigma_cells=float(
            table["voronoi_warp_jagged_sigma_cells"]  # type: ignore[arg-type]
        ),
        voronoi_weight_sigma=float(table["voronoi_weight_sigma"]),  # type: ignore[arg-type]
        voronoi_weight_scale_km=float(
            table["voronoi_weight_scale_km"]  # type: ignore[arg-type]
        ),
        init_thickness_per_plate_sigma=float(
            table["init_thickness_per_plate_sigma"]  # type: ignore[arg-type]
        ),
        init_thickness_noise_amplitude_frac=float(
            table["init_thickness_noise_amplitude_frac"]  # type: ignore[arg-type]
        ),
        init_thickness_noise_sigma_cells=float(
            table["init_thickness_noise_sigma_cells"]  # type: ignore[arg-type]
        ),
    )
