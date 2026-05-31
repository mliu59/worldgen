"""Polygon-based tectonic simulation.

Rigid-polygon plate model with per-cell paint grids. Worldgen's
production tectonics path goes through this package via
``worldgen.tectonics_cast.simulate_tectonics_via_continuous_sim``.

  - ``types``        — Hotspot, PolygonPlate, AlphaComplex + RNG tags
  - ``topology``     — torus-aware grid helpers + connected components
  - ``seeding``      — initial Voronoi cell-grid construction
  - ``kinematics``   — per-tick drift (stamping) + rotation
  - ``contention``   — per-cell ownership resolution + C-C folding
  - ``damping``      — translation + angular velocity damping
  - ``momentum``     — inelastic per-pair momentum exchange
  - ``fusion``       — small-into-big plate merge
  - ``accretion``    — C-O arc magmatism
  - ``hotspots``     — mantle plume eruptions
  - ``divergent``    — trailing-edge oceanic spawn
  - ``aging``        — aging, erosion, absorption, buoyancy
  - ``culling``      — connected-component fragment handling
  - ``polygons``     — alpha-complex construction + per-tick re-extraction
  - ``rifting``      — probabilistic plate splits
  - ``viz``          — all renderers + GIF helpers
  - ``simulate``     — top-level simulate_rigid_polygon orchestrator
  - ``isostasy``     — closed-form cell → signed elevation
"""

from __future__ import annotations

# The only symbols re-exported at package level are the ones worldgen's
# bridge / exporter actually consumes. Everything else stays private to
# polygon_sim; import the submodule directly if you need it.

from tectonic_sim.polygon_sim.simulate import simulate_rigid_polygon
from tectonic_sim.polygon_sim.viz import (
    _build_partition_image as build_partition_image,
    _build_crust_image as build_crust_image,
    _build_thickness_image as build_thickness_image,
    _build_topography_image as build_topography_image,
    _render_polygons as render_polygons_png,
    _save_drift_gif as save_drift_gif,
    _overlay_hotspots as overlay_hotspots,
)


__all__ = [
    "build_crust_image",
    "build_partition_image",
    "build_thickness_image",
    "build_topography_image",
    "overlay_hotspots",
    "render_polygons_png",
    "save_drift_gif",
    "simulate_rigid_polygon",
]
