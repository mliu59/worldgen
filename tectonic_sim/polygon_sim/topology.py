"""Torus-aware grid helpers."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as _ndi_label

from tectonic_sim.types import WorldRect


# ---------------------------------------------------------------------------


def _torus_components(mask: np.ndarray) -> np.ndarray:
    """Connected-component labelling that respects torus topology.

    ``scipy.ndimage.label`` does not know about wrap, so a blob that
    straddles the seam gets two labels. We post-process by walking the
    seams and union-finding labels that touch across.

    Returns a (gy, gx) int array: 0 = not in mask, 1..N = components.
    """
    lbl, n = _ndi_label(mask)
    if n <= 1:
        return lbl
    gy, gx = mask.shape
    parent = np.arange(n + 1)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # East-west seam.
    for y in range(gy):
        if mask[y, 0] and mask[y, gx - 1]:
            union(int(lbl[y, 0]), int(lbl[y, gx - 1]))
    # North-south seam.
    for x in range(gx):
        if mask[0, x] and mask[gy - 1, x]:
            union(int(lbl[0, x]), int(lbl[gy - 1, x]))

    roots = np.array([find(i) for i in range(n + 1)], dtype=np.int64)
    return roots[lbl]


# ---------------------------------------------------------------------------
# Grid setup.
# ---------------------------------------------------------------------------


def _grid_dims(domain: WorldRect, sim_config) -> tuple[int, int, float]:
    """Grid resolution derived from the domain at fixed cell_km.

    Cell size comes from ``sim_config.target_cell_km`` so changing the
    sim domain only changes the grid count, not the physics resolution.
    """
    cell_km = sim_config.target_cell_km
    gx = int(round(domain.width_km / cell_km))
    gy = int(round(domain.height_km / cell_km))
    return gy, gx, cell_km


def _cell_centres(gy: int, gx: int, cell_km: float) -> np.ndarray:
    """Cell-centre coordinates in the centred [-half_w, +half_w] frame."""
    xs = (np.arange(gx) + 0.5) * cell_km - 0.5 * gx * cell_km
    ys = (np.arange(gy) + 0.5) * cell_km - 0.5 * gy * cell_km
    cx, cy = np.meshgrid(xs, ys)
    return np.column_stack([cx.ravel(), cy.ravel()])

