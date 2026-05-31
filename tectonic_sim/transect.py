"""1D transect: sample sim state along a line between two km points.

Given a polygon-sim final state and two sim-centred km endpoints
``p1=(x1, y1)``, ``p2=(x2, y2)``, this module samples a configurable
number of points along the wrap-aware shortest segment from ``p1`` to
``p2`` and returns the per-sample owner / crust / age / thickness /
elevation as parallel 1D arrays.

A two-panel PIL renderer turns the result into ``transect.png``:

  - **Top panel** — signed elevation (km) vs along-path distance, with
    a dashed horizontal line at ``sea_level_km`` and vertical thin
    lines wherever owner changes between adjacent samples.
  - **Bottom panel** — crust thickness (km) vs distance, same x-axis,
    same plate-boundary markers.

Both panels use a per-plate hue that matches the ``partition.png``
palette (``tectonic_sim.polygon_sim.viz._color_for``), so transect
colours line up visually with the partition view.

Sampling is **nearest-neighbour** in cell space — same convention
worldgen's per-hex sampler uses. That means plate boundaries appear as
honest step changes, not smoothed-over interpolation. The segment is
wrap-aware: a transect crossing the toroidal seam Just Works because
the displacement from ``p1`` to ``p2`` uses ``WorldRect.wrapped_delta_xy``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tectonic_sim.io import SimState, load_state
from tectonic_sim.polygon_sim.viz import _color_for


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransectResult:
    """Parallel 1D arrays sampled along a wrap-aware segment.

    All ``(n_samples,)`` shape unless noted.

      - ``distance_km`` — cumulative along-path distance, starts at 0.0
      - ``x_km``, ``y_km`` — wrapped sample coordinates in the sim's
        centred frame (``[-half_w, +half_w)`` × similar in y)
      - ``owner`` (int32) — plate id; ``-1`` for unowned cells
      - ``crust`` (int8) — ``CRUST_CONTINENTAL`` / ``CRUST_OCEANIC``
      - ``thickness_km``, ``age_myr``, ``elevation_km`` (float64)
      - ``sea_level_km`` — scalar, copied from the source state
      - ``p1_km``, ``p2_km`` — the original endpoint coords (for the
        renderer's title; carry through unchanged)
    """

    distance_km: np.ndarray
    x_km: np.ndarray
    y_km: np.ndarray
    owner: np.ndarray
    crust: np.ndarray
    thickness_km: np.ndarray
    age_myr: np.ndarray
    elevation_km: np.ndarray
    sea_level_km: float
    p1_km: tuple[float, float]
    p2_km: tuple[float, float]


def sample_transect(
    state: SimState,
    p1: tuple[float, float],
    p2: tuple[float, float],
    n_samples: int = 400,
) -> TransectResult:
    """Sample ``state`` along the wrap-aware segment ``p1 → p2``.

    Both endpoints are in sim-centred km. ``n_samples`` includes both
    endpoints (so the t-parameter steps through ``linspace(0, 1, n)``).
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")

    domain = state.sim_domain
    cell_km = state.cell_km
    gy, gx = state.owner.shape

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    # The segment is interpreted as the **direct Cartesian line** from
    # p1 to p2 — exactly the path a user sees when picking endpoints off
    # a partition map. Toroidal wrap is applied per-sample at cell
    # lookup time, so a segment that physically crosses the seam still
    # reads the correct cell. We deliberately do NOT pick the toroidal-
    # shortest path here: the user typed specific endpoints, the line
    # they want is the visible one.
    dx = x2 - x1
    dy = y2 - y1

    t = np.linspace(0.0, 1.0, n_samples)
    raw_x = x1 + dx * t
    raw_y = y1 + dy * t

    # Cumulative along-path distance: t * |segment_length|.
    seg_len = float(np.hypot(dx, dy))
    distance_km = t * seg_len

    # Wrap to the centred sim frame, then convert to cell indices.
    half_w = domain.half_width_km
    half_h = domain.half_height_km
    wx = (raw_x + half_w) % domain.width_km - half_w
    wy = (raw_y + half_h) % domain.height_km - half_h

    # Cell index in grid frame. Use floor on (centred + half) / cell_km
    # so the math matches the worldgen hex sampler exactly.
    cx = np.floor((wx + half_w) / cell_km).astype(np.int64) % gx
    cy = np.floor((wy + half_h) / cell_km).astype(np.int64) % gy

    owner = state.owner[cy, cx].astype(np.int32)
    crust = state.crust[cy, cx].astype(np.int8)
    age = state.age[cy, cx].astype(np.float64)
    thick = state.thickness[cy, cx].astype(np.float64)
    elev = state.elevation_km(crust, thick, age)

    return TransectResult(
        distance_km=distance_km,
        x_km=wx,
        y_km=wy,
        owner=owner,
        crust=crust,
        thickness_km=thick,
        age_myr=age,
        elevation_km=elev,
        sea_level_km=state.sea_level_km,
        p1_km=(x1, y1),
        p2_km=(x2, y2),
    )


# ---------------------------------------------------------------------------
# Rendering (PIL — no matplotlib)
# ---------------------------------------------------------------------------


# Visual layout (pixels). Two stacked panels with a small gap.
_W = 1400
_PANEL_H = 280
_GAP = 60
_MARGIN_L = 90
_MARGIN_R = 30
_MARGIN_T = 50
_MARGIN_B = 50
_H = _MARGIN_T + 2 * _PANEL_H + _GAP + _MARGIN_B  # total height


def render_transect(
    result: TransectResult,
    path: Path,
    *,
    title: str | None = None,
) -> None:
    """Render a two-panel transect figure to ``path`` (PNG via PIL).

    Top panel: signed elevation (km) with dashed sea-level reference.
    Bottom panel: crust thickness (km).
    Both: line plot coloured per plate, with vertical thin lines at
    owner-change boundaries and "P{pid}" labels over the longest run of
    each plate.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (_W, _H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=12)
        big_font = ImageFont.truetype("arial.ttf", size=14)
    except OSError:
        font = ImageFont.load_default()
        big_font = font

    # Title bar.
    if title is None:
        x1, y1 = result.p1_km
        x2, y2 = result.p2_km
        title = (
            f"transect  ({x1:+.0f}, {y1:+.0f}) → ({x2:+.0f}, {y2:+.0f}) km   "
            f"length = {float(result.distance_km[-1]):.0f} km   "
            f"samples = {len(result.distance_km)}"
        )
    draw.text((_MARGIN_L, 12), title, fill=(20, 20, 20), font=big_font)

    # Panel rectangles.
    elev_top = _MARGIN_T
    elev_bot = elev_top + _PANEL_H
    thk_top = elev_bot + _GAP
    thk_bot = thk_top + _PANEL_H
    plot_left = _MARGIN_L
    plot_right = _W - _MARGIN_R

    # Y-ranges. Lock thickness to start at 0; elevation auto with a
    # comfortable margin and pinned to include sea level.
    e_lo, e_hi = _padded_range(
        result.elevation_km, pad_frac=0.10, include=result.sea_level_km,
    )
    t_lo, t_hi = _padded_range(
        result.thickness_km, pad_frac=0.05, lo_floor=0.0,
    )

    # Plate boundary indices: where owner changes between adjacent samples.
    owner = result.owner
    boundary_idx = np.where(np.diff(owner) != 0)[0]  # idx i means change at i→i+1

    # Plate run-length encoding for colouring + labels.
    runs = _run_length_encode(owner)  # list of (pid, start_idx, end_idx)

    # X-axis mapping.
    d = result.distance_km
    d_lo = float(d[0])
    d_hi = float(d[-1])
    d_span = max(d_hi - d_lo, 1e-9)

    def x_of(i: int) -> float:
        return plot_left + (d[i] - d_lo) / d_span * (plot_right - plot_left)

    # Render each panel.
    _draw_panel(
        draw, font, big_font,
        top=elev_top, bot=elev_bot, left=plot_left, right=plot_right,
        y_lo=e_lo, y_hi=e_hi, y_label="elevation (km)",
        x_lo=d_lo, x_hi=d_hi, x_label="along-path distance (km)",
        runs=runs, boundary_idx=boundary_idx,
        d=d, y_samples=result.elevation_km,
        sea_level_km=result.sea_level_km,
        show_sea_level=True,
        label_plates=True,
    )
    _draw_panel(
        draw, font, big_font,
        top=thk_top, bot=thk_bot, left=plot_left, right=plot_right,
        y_lo=t_lo, y_hi=t_hi, y_label="thickness (km)",
        x_lo=d_lo, x_hi=d_hi, x_label="along-path distance (km)",
        runs=runs, boundary_idx=boundary_idx,
        d=d, y_samples=result.thickness_km,
        sea_level_km=None,
        show_sea_level=False,
        label_plates=False,
    )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _draw_panel(
    draw, font, big_font,
    *,
    top: int, bot: int, left: int, right: int,
    y_lo: float, y_hi: float, y_label: str,
    x_lo: float, x_hi: float, x_label: str,
    runs: list[tuple[int, int, int]],
    boundary_idx: np.ndarray,
    d: np.ndarray,
    y_samples: np.ndarray,
    sea_level_km: float | None,
    show_sea_level: bool,
    label_plates: bool,
) -> None:
    """Draw one panel of the figure. ``runs`` is a list of
    ``(pid, start_sample_idx, end_sample_idx)`` so the line can be
    drawn per-plate-segment with the right colour."""
    panel_w = right - left
    panel_h = bot - top
    y_span = max(y_hi - y_lo, 1e-9)
    x_span = max(x_hi - x_lo, 1e-9)

    # Panel background + frame.
    draw.rectangle([(left, top), (right, bot)], fill=(252, 252, 252),
                   outline=(80, 80, 80), width=1)

    # Horizontal gridlines (5 ticks + y-label per tick).
    n_y_ticks = 5
    for i in range(n_y_ticks + 1):
        v = y_lo + (y_hi - y_lo) * i / n_y_ticks
        py = bot - (v - y_lo) / y_span * panel_h
        draw.line([(left, py), (right, py)], fill=(225, 225, 225), width=1)
        draw.text((left - 6, py - 7), f"{v:+.1f}",
                  fill=(80, 80, 80), font=font, anchor="ra")

    # X-axis ticks (5 ticks + km label).
    n_x_ticks = 5
    for i in range(n_x_ticks + 1):
        v = x_lo + (x_hi - x_lo) * i / n_x_ticks
        px = left + (v - x_lo) / x_span * panel_w
        draw.line([(px, bot), (px, bot + 4)], fill=(120, 120, 120), width=1)
        draw.text((px, bot + 6), f"{v:.0f}",
                  fill=(80, 80, 80), font=font, anchor="ma")

    # Axis labels.
    draw.text((left + panel_w / 2, bot + 28),
              x_label, fill=(60, 60, 60), font=font, anchor="ma")
    draw.text((left - 70, top + panel_h / 2),
              y_label, fill=(60, 60, 60), font=font, anchor="lm")

    # Sea-level reference line (dashed).
    if show_sea_level and sea_level_km is not None:
        sl = float(sea_level_km)
        if y_lo <= sl <= y_hi:
            py = bot - (sl - y_lo) / y_span * panel_h
            _dashed_hline(draw, left, right, py,
                          fill=(40, 40, 200), width=1, dash=6)
            draw.text((right - 6, py - 14),
                      f"sea level ({sl:+.1f} km)",
                      fill=(40, 40, 200), font=font, anchor="ra")

    # Plate-boundary vertical lines.
    for bi in boundary_idx:
        # boundary between sample bi and bi+1; draw at midpoint distance.
        d_mid = 0.5 * (float(d[bi]) + float(d[bi + 1]))
        px = left + (d_mid - x_lo) / x_span * panel_w
        draw.line([(px, top), (px, bot)], fill=(170, 170, 170), width=1)

    # Per-plate-segment polyline.
    for pid, s, e in runs:
        if e - s < 1:
            # Single-sample run — skip the line; the boundary markers
            # already show where it lives.
            continue
        rgb = _color_for_pid(int(pid))
        pts = []
        for k in range(s, e + 1):
            px = left + (float(d[k]) - x_lo) / x_span * panel_w
            py = bot - (float(y_samples[k]) - y_lo) / y_span * panel_h
            pts.append((px, py))
        if len(pts) >= 2:
            draw.line(pts, fill=rgb, width=2)

    # Plate labels (top panel only).
    if label_plates:
        for pid, s, e in runs:
            if pid < 0:
                continue
            d_mid = 0.5 * (float(d[s]) + float(d[e]))
            px = left + (d_mid - x_lo) / x_span * panel_w
            rgb = _color_for_pid(int(pid))
            # Label at the top of the panel just inside the frame.
            draw.text((px, top + 4), f"P{int(pid)}",
                      fill=rgb, font=big_font, anchor="ma",
                      stroke_width=2, stroke_fill=(255, 255, 255))


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _color_for_pid(pid: int) -> tuple[int, int, int]:
    """Per-plate colour matching the partition.png palette.

    Unowned cells (``pid == -1``) get a neutral dark grey.
    """
    if pid < 0:
        return (90, 90, 90)
    return _color_for(int(pid))


def _padded_range(
    arr: np.ndarray, *,
    pad_frac: float = 0.05,
    include: float | None = None,
    lo_floor: float | None = None,
) -> tuple[float, float]:
    """Return a (lo, hi) y-range with proportional padding, optionally
    pinned to include a scalar (sea level) and/or floored at ``lo_floor``.
    """
    lo = float(arr.min())
    hi = float(arr.max())
    if include is not None:
        lo = min(lo, float(include))
        hi = max(hi, float(include))
    span = max(hi - lo, 1e-3)
    pad = span * pad_frac
    lo -= pad
    hi += pad
    if lo_floor is not None:
        lo = max(lo, float(lo_floor))
    return lo, hi


def _dashed_hline(
    draw, left: float, right: float, y: float,
    *, fill, width: int, dash: int,
) -> None:
    """Draw a dashed horizontal line via short segments."""
    x = left
    on = True
    while x < right:
        x_end = min(x + dash, right)
        if on:
            draw.line([(x, y), (x_end, y)], fill=fill, width=width)
        x = x_end
        on = not on


def _run_length_encode(arr: np.ndarray) -> list[tuple[int, int, int]]:
    """RLE of a 1D integer array.

    Returns a list of ``(value, start_idx, end_idx_inclusive)``. Used to
    colour each plate's line segment with its own hue and place plate
    labels at the centre of the longest contiguous run.
    """
    if arr.size == 0:
        return []
    runs: list[tuple[int, int, int]] = []
    start = 0
    cur = int(arr[0])
    for i in range(1, arr.size):
        v = int(arr[i])
        if v != cur:
            runs.append((cur, start, i - 1))
            start = i
            cur = v
    runs.append((cur, start, arr.size - 1))
    return runs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_xy(s: str, *, name: str) -> tuple[float, float]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--{name} expected 'x,y' (two comma-separated km values), "
            f"got {s!r}"
        )
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--{name} could not parse {s!r} as 'x,y' km floats: {exc}"
        )


def main(argv: list[str] | None = None) -> None:
    """``python -m tectonic_sim.transect`` entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m tectonic_sim.transect",
        description=(
            "Sample a 1D transect through a tectonic_sim state.npz and "
            "render a two-panel (elevation, thickness) PNG."
        ),
    )
    parser.add_argument(
        "state", type=Path,
        help="Path to state.npz (written by worldgen export).",
    )
    parser.add_argument(
        "--p1", required=True,
        help="Start point as 'x,y' in sim-centred km (e.g. '-400,0').",
    )
    parser.add_argument(
        "--p2", required=True,
        help="End point as 'x,y' in sim-centred km (e.g. '400,0').",
    )
    parser.add_argument(
        "--n-samples", type=int, default=400,
        help="Number of samples along the segment (default: 400).",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title", default=None,
        help="Optional override for the figure title.",
    )
    args = parser.parse_args(argv)

    p1 = _parse_xy(args.p1, name="p1")
    p2 = _parse_xy(args.p2, name="p2")

    state = load_state(args.state)
    result = sample_transect(state, p1, p2, n_samples=args.n_samples)
    render_transect(result, args.out, title=args.title)
    print(
        f"transect: {len(result.distance_km)} samples, "
        f"length {float(result.distance_km[-1]):.1f} km -> {args.out}"
    )


if __name__ == "__main__":
    main()
