"""Polygon-sim visualisations (partition, crust, thickness, topo, polygons, GIFs)."""

from __future__ import annotations

import colorsys

import numpy as np

from tectonic_sim.types import CRUST_CONTINENTAL, CRUST_OCEANIC

from tectonic_sim.polygon_sim.isostasy import particle_elevation_km
from tectonic_sim.polygon_sim.types import Hotspot


# Renderers.
# ---------------------------------------------------------------------------


def _color_for(pid: int) -> tuple[int, int, int]:
    h = (pid * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.92)
    return int(r * 255), int(g * 255), int(b * 255)


def _overlay_hotspots(
    img, hotspots: list[Hotspot], tick: int,
    *, cell_km: float, gy: int, gx: int, upscale: int,
    x0_cells: int = 0, y0_cells: int = 0,
    only_active: bool = True) -> None:
    """Draw a marker at every hotspot position on ``img``.

    Active hotspots get a filled red circle with black outline.
    Extinct hotspots (when ``only_active=False``) get a hollow grey
    circle so you can still see where the plume USED to be.

    Coordinate mapping: hotspots live in the centred [-half_w, +half_w]
    km frame; this function shifts to the [0, gx) grid frame, applies
    the crop offset ``(x0_cells, y0_cells)`` (left/top edge of the
    crop in sim-cell units, or 0 if the image isn't cropped), then
    scales by ``upscale`` for the upsampled-pixel image.
    """
    from PIL import ImageDraw, ImageFont
    if not hotspots:
        return
    draw = ImageDraw.Draw(img)
    # Marker is a small red filled disk + ring for active, hollow grey
    # ring for extinct. Tuned to be visible at upscale=4 (sim GIF frames)
    # and upscale=6 (final PNGs).
    r = max(6, upscale * 2)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(11, upscale * 2))
    except OSError:
        font = ImageFont.load_default()
    for idx, h in enumerate(hotspots):
        is_active = h.is_active(tick)
        if only_active and not is_active:
            continue
        # Convert km position → continuous cell coords (sim frame).
        sim_cx = (h.position_xy_km[0] + 0.5 * gx * cell_km) / cell_km
        sim_cy = (h.position_xy_km[1] + 0.5 * gy * cell_km) / cell_km
        # Shift to cropped grid, then upsample to pixels.
        px = (sim_cx - x0_cells) * upscale
        py = (sim_cy - y0_cells) * upscale
        if not (-r <= px < img.width + r and -r <= py < img.height + r):
            continue
        if is_active:
            # Red filled disk with black ring + thin white halo for
            # contrast against any background (continent or ocean).
            draw.ellipse(
                [(px - r - 1, py - r - 1), (px + r + 1, py + r + 1)],
                outline=(255, 255, 255), width=1)
            draw.ellipse(
                [(px - r, py - r), (px + r, py + r)],
                outline=(0, 0, 0), fill=(230, 30, 30), width=2)
        else:
            draw.ellipse(
                [(px - r, py - r), (px + r, py + r)],
                outline=(80, 80, 80), fill=None, width=2)
        # Hotspot index label to the upper-right of the marker.
        draw.text(
            (px + r + 2, py - r),
            f"H{idx}",
            fill=(0, 0, 0),
            font=font,
            stroke_width=2, stroke_fill=(255, 255, 255))


def _build_partition_image(owner, caption, upscale=6, draw_edges=True,
                           draw_labels=True):
    """Construct a partition view as a PIL Image (no save).

    When ``draw_labels`` is True, each plate's integer id is drawn at the
    plate's centroid. Centroids are computed via torus-aware circular
    mean so a plate that wraps the seam still gets a label inside its
    own footprint (not on the opposite side of the world).
    """
    from PIL import Image, ImageDraw, ImageFont
    gy, gx = owner.shape
    rgb = np.zeros((gy, gx, 3), dtype=np.uint8)
    for pid in np.unique(owner):
        if pid < 0:
            rgb[owner == pid] = (40, 40, 40)
        else:
            rgb[owner == pid] = _color_for(int(pid))
    big = np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)
    if draw_edges:
        black = (15, 15, 15)
        vy, vx = np.where(owner[:, :-1] != owner[:, 1:])
        for y, x in zip(vy, vx):
            px = (x + 1) * upscale
            big[y * upscale:(y + 1) * upscale, px - 1:px + 1] = black
        hy, hx = np.where(owner[:-1, :] != owner[1:, :])
        for y, x in zip(hy, hx):
            py = (y + 1) * upscale
            big[py - 1:py + 1, x * upscale:(x + 1) * upscale] = black
    img = Image.fromarray(big, "RGB")
    draw = ImageDraw.Draw(img)
    if draw_labels:
        # Try a slightly larger TrueType font; fall back to PIL's bitmap
        # default if no system font is found.
        try:
            label_font = ImageFont.truetype("arial.ttf", size=max(12, upscale * 2))
        except OSError:
            label_font = ImageFont.load_default()
        for pid in np.unique(owner):
            if pid < 0:
                continue
            ys, xs = np.where(owner == pid)
            if xs.size == 0:
                continue
            # Torus-aware circular mean: treat each axis as an angle.
            ax = 2.0 * np.pi * (xs.astype(np.float64) / gx)
            ay = 2.0 * np.pi * (ys.astype(np.float64) / gy)
            cx = (np.arctan2(np.sin(ax).mean(), np.cos(ax).mean())
                  * gx / (2.0 * np.pi)) % gx
            cy = (np.arctan2(np.sin(ay).mean(), np.cos(ay).mean())
                  * gy / (2.0 * np.pi)) % gy
            # Snap to the nearest cell that actually belongs to this
            # plate — protects against U-shaped plates whose circular
            # mean lands in a neighbour's territory.
            d2 = ((xs - cx) % gx) ** 2 + ((ys - cy) % gy) ** 2
            d2 = np.minimum(d2, ((cx - xs) % gx) ** 2 + ((cy - ys) % gy) ** 2)
            best = int(np.argmin(d2))
            tx = (xs[best] + 0.5) * upscale
            ty = (ys[best] + 0.5) * upscale
            # Contrast text against the plate's HSV value (~0.92 → light
            # plate, so use black ink with white halo for readability).
            text = str(int(pid))
            draw.text(
                (tx, ty), text,
                fill=(0, 0, 0), font=label_font,
                anchor="mm",
                stroke_width=2, stroke_fill=(255, 255, 255))
    draw.text((6, 6), caption, fill=(0, 0, 0))
    return img


def _render_partition(owner, out_path, caption, upscale=6):
    _build_partition_image(owner, caption, upscale).save(out_path)


def _build_crust_image(owner, crust, age, thick, max_age, caption, upscale=6):
    """Construct a crust view as a PIL Image (no save)."""
    from PIL import Image, ImageDraw
    gy, gx = owner.shape
    rgb = np.zeros((gy, gx, 3), dtype=np.uint8)
    # Unowned cells have crust=0 (= CRUST_CONTINENTAL default) but no
    # legitimate continental crust. Mask them by owner first.
    owned = owner != -1
    cont = (crust == CRUST_CONTINENTAL) & owned
    ocean = (crust == CRUST_OCEANIC) & owned
    nil = ~owned
    if cont.any():
        base = np.array([196, 170, 120], dtype=np.float64)
        peak = np.array([130, 90, 50], dtype=np.float64)
        t_lo = thick[cont].min() if cont.any() else 0.0
        t_hi = thick[cont].max() if cont.any() else 1.0
        denom = max(1e-6, t_hi - t_lo)
        t_norm = np.clip((thick - t_lo) / denom, 0.0, 1.0)
        for c in range(3):
            rgb[..., c] = np.where(
                cont,
                (base[c] + (peak[c] - base[c]) * t_norm).astype(np.uint8),
                rgb[..., c])
    if ocean.any():
        if max_age > 0:
            frac = np.clip(age / max_age, 0.0, 1.0)
        else:
            frac = np.zeros_like(age)
        r = (90 * (1 - frac)).astype(np.uint8)
        g = (200 - 150 * frac).astype(np.uint8)
        b = (255 - 90 * frac).astype(np.uint8)
        rgb[ocean] = np.stack([r, g, b], axis=-1)[ocean]
    if nil.any():
        rgb[nil] = (60, 60, 60)
    big = np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)
    img = Image.fromarray(big, "RGB")
    ImageDraw.Draw(img).text((6, 6), caption, fill=(0, 0, 0))
    return img


def _render_crust(owner, crust, age, thick, max_age, out_path, caption,
                  upscale=6):
    _build_crust_image(
        owner, crust, age, thick, max_age, caption, upscale).save(out_path)


# 8-stop topography colormap (km elevation → RGB).
# Standard atlas-style: dark navy abyssal, blue ocean, sandy shore,
# green plains, tan hills, brown mountains, white snow peaks.
_TOPO_STOPS = np.array([
    (-6.0, 10, 10, 50),       # abyssal trench: dark navy
    (-3.0, 40, 80, 160),      # deep ocean: ocean blue
    (-0.8, 100, 160, 220),    # continental shelf: light blue
    (0.0, 220, 200, 160),     # coastline: sandy
    (0.4, 90, 160, 80),       # plains: green
    (1.2, 180, 150, 80),      # hills: tan
    (3.0, 140, 90, 60),       # mountains: brown
    (5.0, 255, 255, 255),     # snow / high peaks: white
], dtype=np.float64)


def _build_topography_image(
    owner, crust, age, thick, sim_config, caption, upscale=6):
    """Render signed elevation (km above/below sea level) using a
    classic land/sea topographic colormap.

    Elevation is derived per cell from ``particle_elevation_km``
    (continental: isostatic excess from reference thickness; oceanic:
    half-space cooling depth from age). Sea level is ``sim_config
    .sea_level_km`` (typically 0).
    """
    from PIL import Image, ImageDraw
    gy, gx = owner.shape
    rgb = np.zeros((gy, gx, 3), dtype=np.uint8)
    owned = owner != -1
    if owned.any():
        elev = particle_elevation_km(
            crust.ravel(), thick.ravel(), age.ravel(), sim_config).reshape(gy, gx)
        # Vectorised lookup in the 8-stop colormap.
        stops_e = _TOPO_STOPS[:, 0]
        stops_c = _TOPO_STOPS[:, 1:4]
        e_clamped = np.clip(elev, stops_e[0], stops_e[-1])
        idx = np.searchsorted(stops_e, e_clamped, side="right") - 1
        idx = np.clip(idx, 0, len(stops_e) - 2)
        e0 = stops_e[idx]
        e1 = stops_e[idx + 1]
        frac = (e_clamped - e0) / np.maximum(e1 - e0, 1e-9)
        c0 = stops_c[idx]
        c1 = stops_c[idx + 1]
        col = (c0 + (c1 - c0) * frac[..., None]).astype(np.uint8)
        for c in range(3):
            rgb[..., c] = np.where(owned, col[..., c], rgb[..., c])
    nil = ~owned
    if nil.any():
        rgb[nil] = (60, 60, 60)
    big = np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)
    img = Image.fromarray(big, "RGB")
    ImageDraw.Draw(img).text((6, 6), caption, fill=(255, 255, 255))
    return img


def _render_topography(
    owner, crust, age, thick, sim_config, out_path, caption, upscale=6):
    _build_topography_image(
        owner, crust, age, thick, sim_config, caption, upscale).save(out_path)


# 5-stop colormap for crust thickness. Stops correspond to:
#   0 km  → deep navy   (unowned background already painted gray)
#  12 km  → cyan        (typical oceanic thickness, with young-ridge bonus)
#  25 km  → green       (transition / thinned continental)
#  40 km  → gold        (typical continental thickness)
#  60+ km → red         (orogeny — folded mountain ranges)
_THICKNESS_STOPS_KM = np.array([0.0, 12.0, 25.0, 40.0, 60.0], dtype=np.float64)
_THICKNESS_STOPS_RGB = np.array([
    (10, 30, 100),
    (60, 180, 200),
    (100, 200, 100),
    (240, 200, 80),
    (220, 60, 60),
], dtype=np.float64)


def _build_thickness_image(owner, thickness, caption, upscale=6):
    """Crust-thickness heatmap using the 5-stop colormap above.

    Unowned cells (owner < 0) render as dark gray. Owned cells are
    coloured by their thickness in km via piecewise-linear interpolation
    on the fixed stops 0/12/25/40/60+ km.
    """
    from PIL import Image, ImageDraw

    gy, gx = thickness.shape
    rgb = np.zeros((gy, gx, 3), dtype=np.uint8)
    owned = owner != -1
    nil = ~owned

    if owned.any():
        # Vectorised piecewise-linear lookup across the 5 stops.
        t = np.clip(thickness, 0.0, _THICKNESS_STOPS_KM[-1])
        # Find the segment index for each cell: largest k with stops[k] <= t.
        # For t at or below stops[0] it's 0; at or above stops[-1] it's n-2.
        idx = np.searchsorted(_THICKNESS_STOPS_KM, t, side="right") - 1
        idx = np.clip(idx, 0, len(_THICKNESS_STOPS_KM) - 2)
        seg_lo = _THICKNESS_STOPS_KM[idx]
        seg_hi = _THICKNESS_STOPS_KM[idx + 1]
        frac = (t - seg_lo) / np.maximum(seg_hi - seg_lo, 1e-9)
        c0 = _THICKNESS_STOPS_RGB[idx]
        c1 = _THICKNESS_STOPS_RGB[idx + 1]
        col = c0 + (c1 - c0) * frac[..., None]
        col_u = col.astype(np.uint8)
        for c in range(3):
            rgb[..., c] = np.where(owned, col_u[..., c], rgb[..., c])

    if nil.any():
        rgb[nil] = (60, 60, 60)

    big = np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)
    img = Image.fromarray(big, "RGB")
    ImageDraw.Draw(img).text((6, 6), caption, fill=(255, 255, 255))
    return img


def _combine_frame(left, right, header, padding=10, header_h=28):
    """Side-by-side composition with a header strip."""
    from PIL import Image, ImageDraw
    lw, lh = left.size
    rw, rh = right.size
    w = lw + padding + rw
    h = max(lh, rh) + header_h
    img = Image.new("RGB", (w, h), (240, 240, 240))
    img.paste(left, (0, header_h))
    img.paste(right, (lw + padding, header_h))
    ImageDraw.Draw(img).text((6, 6), header, fill=(0, 0, 0))
    return img


def _save_drift_gif(frames, out_path, fps=10):
    """Save a list of PIL Images as an animated GIF."""
    if not frames:
        return
    duration_ms = max(1, int(round(1000 / fps)))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True)


def _render_polygons(domain, plates, out_path, caption, upscale=6):
    """Two layers in one image:

      1. **Cell-mask fill** (light tint of each plate's id-color): shows
         which cells each plate currently *owns* — the ground truth of
         the partition.
      2. **Alpha-complex boundary** (bold saturated lines in the same
         id-color): shows what the *rigid polygon* (Qhull-deterministic
         alpha-complex of the owned-cell centres) abstracts that mask
         to. The polygon is what drives priority-based partition
         decisions in the rigid-polygon model.

    Where the boundary hugs the cell-mask edge, the polygon faithfully
    represents the plate. Where the boundary cuts inside the fill, the
    alpha-filter rejected some interior triangles (concavities / sparse
    regions).

    Seam handling: edges live in the plate's wrap-aware LOCAL frame
    anchored at ``ref = tri.points[0]`` in absolute km. Two distinct
    seam issues need to be handled:

      1. **Canonical-domain wrap of edge endpoints.** ``ref + tri.points``
         may land outside ``[-half_w, +half_w] × [-half_h, +half_h]``.
         We render via 3×3 torus tiling: draw nine copies offset by every
         ``(dx, dy)`` in ``{-full_w, 0, +full_w} × {-full_h, 0, +full_h}``,
         with a per-copy bbox check that skips copies whose translated
         edge can't intersect the canvas. PIL clips the rest at the canvas
         edges, so a seam-crossing edge shows up as its two halves on
         opposite sides of the image.
      2. **Local-frame wrap boundary.** When a plate's cells span more
         than half the torus along some axis, ``wrapped_delta_xy`` puts
         the far-side cells at the extreme of the local frame (local
         coords near ±half_w / ±half_h). The Delaunay can't bridge the
         resulting gap, so the alpha-complex ends up with a false "wall"
         of boundary edges right at the local-frame seam — visible as
         long straight grid-axis-aligned streaks in earlier renders. We
         skip an edge if both endpoints sit within ``cell_km`` of the
         same local-frame edge (same sign, same axis): such edges trace
         the artifactual wall, not a real plate boundary.
    """
    from PIL import Image, ImageDraw

    alive = [p for p in plates if p.alive]
    if not alive:
        img = Image.new("RGB", (640, 640), (248, 248, 248))
        ImageDraw.Draw(img).text((6, 6), caption + "  (no alive plates)",
                                 fill=(20, 20, 20))
        img.save(out_path)
        return

    gy, gx = alive[0].cell_mask.shape

    # Layer 1: cell-mask fill at the grid resolution, then upscale.
    rgb = np.full((gy, gx, 3), 248, dtype=np.uint8)
    for p in alive:
        color = _color_for(p.pid)
        # Lerp 65% toward white for a light tint.
        light = tuple(int(c + (255 - c) * 0.65) for c in color)
        rgb[p.cell_mask] = light
    big = np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)
    img = Image.fromarray(big, "RGB")
    draw = ImageDraw.Draw(img)

    # Layer 2: alpha-complex boundary overlay. Pixel-space mapping must
    # match the fill — grid row 0 lives at ky = -half_h (see
    # ``_cell_centres``), and numpy/PIL places grid row 0 at the TOP of
    # the image, so low ky = top of image. No y inversion here; the
    # earlier version's ``(half_h - ky)`` flipped the outlines vertically
    # relative to the fill, draining the upper half of the render of
    # polygon outlines (the upper-fill plates' outlines ended up drawn
    # at the bottom of the image, on top of the lower fills).
    px_per_km_x = (gx * upscale) / domain.width_km
    px_per_km_y = (gy * upscale) / domain.height_km

    def to_px(kx: float, ky: float) -> tuple[float, float]:
        return (
            (kx + domain.half_width_km) * px_per_km_x,
            (ky + domain.half_height_km) * px_per_km_y)

    half_w = domain.half_width_km
    half_h = domain.half_height_km
    full_w = domain.width_km
    full_h = domain.height_km
    # Inferred from the sim grid the cell_mask lives on (the caller
    # builds the alpha-complex on a grid of cell centres spaced cell_km
    # apart, so domain.width_km / gx is exactly that spacing).
    cell_km = domain.width_km / gx
    # Skip-an-edge tolerance: any boundary edge whose both endpoints sit
    # within this many km of the same local-frame edge is a false "wall"
    # along the wrap boundary — see the docstring.
    seam_tol_km = 1.0 * cell_km

    # 3×3 torus tiling: each polygon edge is drawn at every offset in
    # this product set. Only copies whose translated bounding box
    # overlaps the canonical [-half, +half] × [-half, +half] canvas
    # actually get a line draw (PIL clips the partial-overlap cases).
    offsets = (-full_w, 0.0, full_w)
    voffsets = (-full_h, 0.0, full_h)

    for p in alive:
        if p.polygon is None:
            continue
        tri, keep, ref = p.polygon
        edge_count: dict[tuple[int, int], int] = {}
        for t in tri.simplices[keep]:
            for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                e = (u, v) if u < v else (v, u)
                edge_count[e] = edge_count.get(e, 0) + 1
        color = _color_for(p.pid)
        for (u, v), cnt in edge_count.items():
            if cnt != 1:
                continue
            # Local-frame endpoints (i.e. relative to ``ref``). Use these
            # for the local-seam test: an edge along the wrap boundary
            # has both endpoints with the same coord saturated near
            # ±half_w (or ±half_h).
            lu = tri.points[u]
            lv = tri.points[v]
            lu_x, lu_y = float(lu[0]), float(lu[1])
            lv_x, lv_y = float(lv[0]), float(lv[1])
            if (
                (lu_x > half_w - seam_tol_km and lv_x > half_w - seam_tol_km)
                or (lu_x < -half_w + seam_tol_km and lv_x < -half_w + seam_tol_km)
                or (lu_y > half_h - seam_tol_km and lv_y > half_h - seam_tol_km)
                or (lu_y < -half_h + seam_tol_km and lv_y < -half_h + seam_tol_km)
            ):
                continue
            # Absolute-frame endpoints (no wrap). May lie outside the
            # canonical [-half, +half] domain; the tiling loop below
            # handles that.
            pu_x, pu_y = float(ref[0]) + lu_x, float(ref[1]) + lu_y
            pv_x, pv_y = float(ref[0]) + lv_x, float(ref[1]) + lv_y
            min_x, max_x = (pu_x, pv_x) if pu_x <= pv_x else (pv_x, pu_x)
            min_y, max_y = (pu_y, pv_y) if pu_y <= pv_y else (pv_y, pu_y)
            for dx in offsets:
                if min_x + dx > half_w or max_x + dx < -half_w:
                    continue
                for dy in voffsets:
                    if min_y + dy > half_h or max_y + dy < -half_h:
                        continue
                    draw.line(
                        [to_px(pu_x + dx, pu_y + dy),
                         to_px(pv_x + dx, pv_y + dy)],
                        fill=color, width=3)

    draw.text(
        (6, 6),
        caption + "  | fill=cell_mask, outline=alpha-complex polygon",
        fill=(20, 20, 20))
    img.save(out_path)


