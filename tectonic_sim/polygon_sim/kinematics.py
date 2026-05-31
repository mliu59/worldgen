"""Drift (stamping translation) + rotation.

The alpha-complex ``polygon`` is built only at the end of the sim by
``_build_polygons_for_render`` from the final ``cell_mask`` — there's
no per-tick polygon to drift here, so this module only moves the cell
grids and lets the polygon be derived once from the final position.
"""

from __future__ import annotations

import numpy as np

from tectonic_sim.types import WorldRect

from tectonic_sim.polygon_sim.types import PolygonPlate


# ---------------------------------------------------------------------------


def _stamp_paint(plates: list[PolygonPlate], dt: float, cell_km: float) -> None:
    """Roll each plate's mask + paint by an integer cell shift derived
    from ``velocity * dt + accum``. Sub-cell remainder kept in ``accum``
    so a slow plate still drifts eventually.
    """
    for p in plates:
        if not p.alive:
            continue
        ax = float(p.accum[0] + p.velocity_kmpy[0] * dt)
        ay = float(p.accum[1] + p.velocity_kmpy[1] * dt)
        sx = int(round(ax / cell_km))
        sy = int(round(ay / cell_km))
        p.accum = np.array([ax - sx * cell_km, ay - sy * cell_km],
                           dtype=np.float64)
        if sx == 0 and sy == 0:
            continue
        p.cell_mask = np.roll(p.cell_mask, shift=(sy, sx), axis=(0, 1))
        p.crust = np.roll(p.crust, shift=(sy, sx), axis=(0, 1))
        p.age = np.roll(p.age, shift=(sy, sx), axis=(0, 1))
        p.thickness = np.roll(p.thickness, shift=(sy, sx), axis=(0, 1))


def _rotate_plates(
    plates: list[PolygonPlate], dt: float, domain: WorldRect,
    gy: int, gx: int, cell_km: float) -> None:
    """Rotate each alive plate's mask + paint about its wrap-aware
    centroid by ``angular_velocity * dt``. Discrete rotation via
    nearest-neighbour inverse mapping — deterministic.

    Note: per-tick angles are small (typically a fraction of a degree),
    so NN resampling preserves the mask shape well. For large angles a
    bilinear-then-threshold would be smoother, but NN is fine here and
    keeps the boolean mask boolean without extra work.
    """
    half_w = 0.5 * gx * cell_km
    half_h = 0.5 * gy * cell_km

    # Target-grid km coordinates, computed once (same for all plates).
    yy, xx = np.indices((gy, gx))
    target_kx = (xx + 0.5) * cell_km - half_w
    target_ky = (yy + 0.5) * cell_km - half_h

    for p in plates:
        if not p.alive or not p.cell_mask.any():
            continue
        angle = float(p.angular_velocity_rad_per_myr) * dt
        if abs(angle) < 1e-9:
            continue

        # Wrap-aware centroid via circular mean: pick the first owned
        # cell as reference, take wrapped deltas, mean, re-wrap.
        ys, xs = np.where(p.cell_mask)
        c_kx = (xs + 0.5) * cell_km - half_w
        c_ky = (ys + 0.5) * cell_km - half_h
        ref_x, ref_y = float(c_kx[0]), float(c_ky[0])
        dx, dy = domain.wrapped_delta_xy(c_kx - ref_x, c_ky - ref_y)
        cent_x = ref_x + float(dx.mean())
        cent_y = ref_y + float(dy.mean())
        cent_x = ((cent_x + domain.half_width_km) % domain.width_km
                  ) - domain.half_width_km
        cent_y = ((cent_y + domain.half_height_km) % domain.height_km
                  ) - domain.half_height_km

        # For each target cell, find the source cell via INVERSE rotation
        # (rotate the target coord back by -angle, look up that source).
        rel_x, rel_y = domain.wrapped_delta_xy(
            target_kx - cent_x, target_ky - cent_y)
        cos_a = float(np.cos(-angle))
        sin_a = float(np.sin(-angle))
        src_rel_x = cos_a * rel_x - sin_a * rel_y
        src_rel_y = sin_a * rel_x + cos_a * rel_y
        src_kx = cent_x + src_rel_x
        src_ky = cent_y + src_rel_y
        # Wrap into centred domain, then convert to grid indices.
        src_kx = ((src_kx + domain.half_width_km) % domain.width_km
                  ) - domain.half_width_km
        src_ky = ((src_ky + domain.half_height_km) % domain.height_km
                  ) - domain.half_height_km
        src_xi = (np.floor(
            (src_kx + domain.half_width_km) / cell_km
        ).astype(np.int64)) % gx
        src_yi = (np.floor(
            (src_ky + domain.half_height_km) / cell_km
        ).astype(np.int64)) % gy

        # Resample mask + paint from source indices.
        p.cell_mask = p.cell_mask[src_yi, src_xi]
        p.crust = p.crust[src_yi, src_xi].astype(np.int8)
        p.age = p.age[src_yi, src_xi]
        p.thickness = p.thickness[src_yi, src_xi]

