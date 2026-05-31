"""Alpha-complex polygons: construction + per-tick aliveness sweep.

Two responsibilities, kept in one module because they share the same
``cell_mask`` -> point-cloud bookkeeping:

  - ``_circumradius`` / ``_build_alpha_complex``
    — primitives for building an alpha-complex polygon from a point
    cloud in the plate's local (wrap-aware) frame.

  - ``_mark_dead_small_plates`` — cheap per-tick sweep that marks
    plates with fewer than 4 owned cells as not-alive. Called every
    tick from the sim loop.

  - ``_build_polygons_for_render`` — expensive Delaunay-based polygon
    construction over every alive plate. Called **once**, at the very
    end of the sim, just before ``_render_polygons`` consumes the
    result for ``polygons.png``. The polygon is not read by any
    per-tick code path, so building it 100× per run was pure waste.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from tectonic_sim.types import WorldRect

from tectonic_sim.polygon_sim.types import (
    AlphaComplex,
    PolygonPlate)


# ---------------------------------------------------------------------------
# Alpha-complex primitives.
# ---------------------------------------------------------------------------


def _circumradius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    la = np.linalg.norm(b - c, axis=1)
    lb = np.linalg.norm(a - c, axis=1)
    lc = np.linalg.norm(a - b, axis=1)
    area = 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    )
    out = np.full(area.shape, np.inf)
    nz = area > 1e-9
    out[nz] = (la[nz] * lb[nz] * lc[nz]) / (4.0 * area[nz])
    return out


def _build_alpha_complex(
    points: np.ndarray, domain: WorldRect, alpha: float) -> AlphaComplex | None:
    if points.shape[0] < 4:
        return None
    ref = points[0].copy()
    dx, dy = domain.wrapped_delta_xy(
        points[:, 0] - ref[0], points[:, 1] - ref[1])
    local = np.column_stack([dx, dy])
    try:
        tri = Delaunay(local)
    except Exception:
        return None
    simp = tri.simplices
    circ = _circumradius(
        local[simp[:, 0]], local[simp[:, 1]], local[simp[:, 2]])
    keep = circ < alpha
    if not keep.any():
        return None
    return tri, keep, ref


# ---------------------------------------------------------------------------
# Per-tick aliveness sweep (cheap).
# ---------------------------------------------------------------------------


def _mark_dead_small_plates(plates: list[PolygonPlate]) -> None:
    """Mark plates with fewer than 4 owned cells as not-alive.

    Cheap counterpart to the old ``_re_extract_polygons``: handles the
    aliveness-marking side effect every tick *without* doing the
    Delaunay-based polygon build, which is only needed for the final
    ``polygons.png`` render and is now deferred to ``_build_polygons_for_render``.
    """
    for p in plates:
        if not p.alive:
            continue
        if int(p.cell_mask.sum()) < 4:
            p.alive = False
            p.polygon = None


# ---------------------------------------------------------------------------
# End-of-sim polygon construction (expensive — runs once).
# ---------------------------------------------------------------------------


def _build_polygons_for_render(
    plates: list[PolygonPlate], domain: WorldRect,
    cell_xy: np.ndarray, cell_km: float, sim_config,
) -> None:
    """Build an alpha-complex polygon for every alive plate from its
    current owned-cell centres.

    The polygon is **not** consumed by any per-tick sim code — it only
    exists to feed the ``polygons.png`` renderer at export time. So
    this runs exactly once, after the simulation finishes, instead of
    every tick.

    Plates that survive culling and aliveness checks but produce a
    degenerate point cloud (e.g. all collinear cells) get their
    ``polygon`` left as ``None``; the renderer skips them.
    """
    alpha = sim_config.alpha_factor * cell_km
    for p in plates:
        if not p.alive:
            p.polygon = None
            continue
        sel = p.cell_mask.ravel()
        if int(sel.sum()) < 4:
            p.polygon = None
            continue
        pts = cell_xy[sel]
        p.polygon = _build_alpha_complex(pts, domain, alpha)
