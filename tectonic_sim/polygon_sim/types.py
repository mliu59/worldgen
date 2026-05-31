"""Polygon-sim data types and deterministic RNG tags.

This module used to carry the full set of physics tunables as
module-level ``_UPPERCASE`` constants. After the SimConfig migration
they all live in ``tectonic_sim.SimConfig`` (loaded from
``config/tectonic_sim.toml``). Every per-tick physics function reads
its tunables via the ``sim_config`` argument that gets threaded
through the per-tick pipeline.

What stays here:
  - The data types: ``Hotspot``, ``AlphaComplex``, ``PolygonPlate``.
  - The deterministic RNG seed-XOR tags. These are NOT tunables — they
    are arbitrary magic words used to derive independent RNG streams
    for each phase (so toggling one mechanic doesn't reshuffle the
    others' random draws).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------
# Deterministic per-phase RNG seed-XOR tags. Magic words used to derive
# independent ``np.random.Generator(PCG64(seed ^ TAG))`` streams. Each
# tag picks a distinct XOR-displaced seed so toggling one phase doesn't
# shift the others' random draws.
# ---------------------------------------------------------------------------

_RIFT_RNG_TAG: int = 0x21F7
_ROT_RNG_TAG: int = 0xA001
_VELOCITY_RNG_TAG: int = 0xB002
_SPAWN_RNG_TAG: int = 0xC003
_VORONOI_RNG_TAG: int = 0xE005
_ACCRETION_RNG_TAG: int = 0xD004
_HOTSPOT_RNG_TAG: int = 0xF006
_CONTINENTAL_RELIEF_RNG_TAG: int = 0x1007


# ---------------------------------------------------------------------------
# Data types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hotspot:
    """A mantle-frame volcanic hotspot (mantle plume).

    Position is fixed in the centred mantle-frame km coordinate system
    (0,0 at sim centre). Plates drift across it; eruptions stamp the
    cell currently above it.
    """
    position_xy_km: tuple[float, float]
    birth_tick: int
    lifespan_ticks: int

    def is_active(self, tick: int) -> bool:
        return self.birth_tick <= tick < self.birth_tick + self.lifespan_ticks


# (Delaunay triangulation in local frame, kept-triangle mask, ref point)
AlphaComplex = tuple[Delaunay, np.ndarray, np.ndarray]


@dataclass
class PolygonPlate:
    """Rigid-polygon plate state.

    The plate carries its own per-cell paint grids that travel with it
    via integer-shift stamping (translation) and centroid-relative
    rotation each tick. The polygon is the alpha-complex of the plate's
    owned-cell centres — derived from ``cell_mask`` each tick, used for
    visualisation and as the rigid-body conceptual model.
    """
    pid: int
    velocity_kmpy: np.ndarray            # (2,) float64
    accum: np.ndarray                    # (2,) float64 — sub-cell carry
    cell_mask: np.ndarray                # (gy, gx) bool
    crust: np.ndarray                    # (gy, gx) int8
    age: np.ndarray                      # (gy, gx) float64
    thickness: np.ndarray                # (gy, gx) float64
    polygon: AlphaComplex | None = None  # derived
    alive: bool = True
    angular_velocity_rad_per_myr: float = 0.0
