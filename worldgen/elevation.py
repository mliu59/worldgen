"""Elevation layer: tectonic baseline blended with fBm/ridged noise.

The macro structure (continents, basins, mountain belts) comes from the
time-stepped tectonics simulation — ``LithosphereState.elevation_km``,
already in physical km. We normalize it against ``max_elevation_km`` so the
downstream pipeline (lapse rate, biome thresholds) keeps its usual
[-1, 1]-ish range with sea level at 0.

fBm + ridged noise (with domain warp) provides small-scale texture on top:
crinkle within mountain ranges, basin-scale variation in plains, etc. The
``tectonic_blend_weight`` knob controls the mix — 1.0 is pure tectonics,
0.0 is pure noise (useful for debugging the noise field alone).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from worldgen._log import progress
from worldgen.rng import RngHierarchy
from worldgen.hex import Hex
from tectonic_sim.noise import PerlinNoise2D, fbm, ridged_fbm
from worldgen.tectonics import LithosphereState
from worldgen.types import ElevationLayer, WorldgenConfig


def _hex_to_xy(h: Hex) -> tuple[float, float]:
    """Axial → flat cartesian for noise sampling.

    Uses the same flat-top-hex pixel mapping the renderer uses, normalized so
    units are roughly "hexes" rather than pixels.
    """
    x = 1.5 * h.q
    y = math.sqrt(3.0) * (h.r + h.q / 2.0)
    return x, y


def compute(
    hexes: Iterable[Hex],
    config: WorldgenConfig,
    rng: RngHierarchy,
    lithosphere: LithosphereState,
) -> ElevationLayer:
    """Compute the normalized elevation field and the sea-level threshold (0).

    The lithosphere's per-hex elevation_km (signed, km) is shifted so the
    configured ``sea_level_km`` becomes 0 and normalized by
    ``max_elevation_km`` so peaks sit near +1 and abyssal floors near -1.
    Noise is then blended in via ``tectonic_blend_weight``.
    """
    noise_base = PerlinNoise2D.from_rng(rng.child("worldgen", "elevation", "base"))
    noise_ridge = PerlinNoise2D.from_rng(rng.child("worldgen", "elevation", "ridge"))
    noise_warp_x = PerlinNoise2D.from_rng(rng.child("worldgen", "elevation", "warp_x"))
    noise_warp_y = PerlinNoise2D.from_rng(rng.child("worldgen", "elevation", "warp_y"))

    sea_level_km = config.tectonics.sea_level_km
    blend = config.tectonic_blend_weight

    elevation: dict[Hex, float] = {}
    hex_list = list(hexes)

    for h in progress(hex_list, desc="elevation", total=len(hex_list)):
        x, y = _hex_to_xy(h)

        # Domain warp: bend coordinates by a low-frequency noise field.
        wx = noise_warp_x.sample(x * config.warp_frequency, y * config.warp_frequency)
        wy = noise_warp_y.sample(x * config.warp_frequency + 13.7,
                                 y * config.warp_frequency + 7.3)
        x_w = x + wx * config.warp_strength
        y_w = y + wy * config.warp_strength

        # Base noise: fBm in roughly [-1, 1].
        noise = fbm(
            noise_base, x_w, y_w,
            octaves=config.noise_octaves,
            lacunarity=config.noise_lacunarity,
            persistence=config.noise_persistence,
            base_frequency=config.noise_base_frequency,
        )

        # Ridged contribution, gated by base elevation so ridges only appear
        # in already-high terrain (avoids ridged crinkles in ocean basins).
        if noise > config.ridge_threshold:
            gate = min(1.0, (noise - config.ridge_threshold) / (1.0 - config.ridge_threshold))
            ridge = ridged_fbm(
                noise_ridge, x_w, y_w,
                octaves=config.ridge_octaves,
                lacunarity=config.noise_lacunarity,
                persistence=config.noise_persistence,
                base_frequency=config.noise_base_frequency * 1.5,
            )
            noise = noise + gate * ridge * config.ridge_amplitude

        baseline_norm = (
            (lithosphere.elevation_km[h] - sea_level_km) / config.max_elevation_km
        )
        # Convex blend: tectonics weight `blend`, noise weight `1 - blend`.
        elevation[h] = blend * baseline_norm + (1.0 - blend) * noise

    # Sea level is baked into the normalization above (0 ≡ sea_level_km).
    sea_level = 0.0

    # Clip peak land elevation to ≤ 1.0 so downstream layers (lapse rate,
    # biome elevation thresholds) keep their calibration. Tectonic collisions
    # can produce columns thicker than ``max_elevation_km`` suggests.
    sorted_vals = sorted(elevation.values())
    max_above = max(0.0, sorted_vals[-1] - sea_level)
    if max_above > 1.0:
        scale = 1.0 / max_above
        for h, v in elevation.items():
            delta = v - sea_level
            if delta > 0:
                elevation[h] = sea_level + delta * scale

    return ElevationLayer(elevation=elevation, sea_level=sea_level)
