"""Deterministic 2D Perlin noise — scalar and numpy-vectorized.

Seeded from either ``random.Random`` or ``np.random.Generator`` produced
by the respective hierarchies in ``worldgen.rng`` / ``tectonic_sim.rng``.
No global state. Used for:

  * scalar ``sample`` + ``fbm`` / ``ridged_fbm`` — per-hex elevation
    detail in ``worldgen.elevation``;
  * vectorized ``sample_grid`` + ``fbm_grid`` — full-grid stamping in
    ``tectonic_sim.polygon_sim.seeding`` (continental relief at sim t=0).

Both paths share the same permutation table and gradient lookup, so a
PerlinNoise2D built from a given seed produces identical values whether
sampled scalar-by-scalar or as a grid.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np


def _fade(t: float) -> float:
    """Quintic fade curve (Perlin's improved interpolant)."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _fade_np(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# Unit gradient vectors for the 8 octants — classic Perlin 2D grads.
_GRADS_2D: tuple[tuple[float, float], ...] = (
    (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
    (math.sqrt(0.5), math.sqrt(0.5)),
    (-math.sqrt(0.5), math.sqrt(0.5)),
    (math.sqrt(0.5), -math.sqrt(0.5)),
    (-math.sqrt(0.5), -math.sqrt(0.5)),
)
_GRADS_2D_NP = np.array(_GRADS_2D, dtype=np.float64)


@dataclass(frozen=True)
class PerlinNoise2D:
    """2D Perlin noise with a seeded permutation table.

    Output range is approximately ``[-1, 1]``.
    """

    perm: tuple[int, ...]  # length 512 (doubled to avoid modulo on lookup)

    @staticmethod
    def from_rng(rng: random.Random | np.random.Generator) -> "PerlinNoise2D":
        """Construct from either a ``random.Random`` or ``np.random.Generator``.

        Both paths produce a permutation of ``[0, 256)`` derived from the
        rng's own shuffle; the same rng seed gives the same table within
        each rng-flavour family. Cross-flavour reproducibility is not
        guaranteed (random.Random and numpy use different shuffle
        algorithms).
        """
        table_list = list(range(256))
        if isinstance(rng, np.random.Generator):
            arr = np.array(table_list, dtype=np.int64)
            rng.shuffle(arr)
            table = [int(x) for x in arr]
        else:
            rng.shuffle(table_list)
            table = table_list
        return PerlinNoise2D(perm=tuple(table + table))

    # ----- Scalar path --------------------------------------------------

    def _grad(self, ix: int, iy: int, dx: float, dy: float) -> float:
        h = self.perm[(ix + self.perm[iy & 255]) & 255] & 7
        gx, gy = _GRADS_2D[h]
        return gx * dx + gy * dy

    def sample(self, x: float, y: float) -> float:
        x0 = math.floor(x)
        y0 = math.floor(y)
        xf = x - x0
        yf = y - y0
        u = _fade(xf)
        v = _fade(yf)

        n00 = self._grad(x0, y0, xf, yf)
        n10 = self._grad(x0 + 1, y0, xf - 1.0, yf)
        n01 = self._grad(x0, y0 + 1, xf, yf - 1.0)
        n11 = self._grad(x0 + 1, y0 + 1, xf - 1.0, yf - 1.0)

        nx0 = _lerp(n00, n10, u)
        nx1 = _lerp(n01, n11, u)
        return _lerp(nx0, nx1, v)

    # ----- Vectorized path ----------------------------------------------

    def sample_grid(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Sample at every paired (x[i], y[i]). Inputs same shape; output
        the same shape and dtype ``float64``.

        Numerically identical to scalar ``sample`` at every point —
        verified by ``tests/test_noise``.
        """
        perm = np.asarray(self.perm, dtype=np.int64)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        xf = (x - x0).astype(np.float64)
        yf = (y - y0).astype(np.float64)
        u = _fade_np(xf)
        v = _fade_np(yf)

        def _grad_arr(ix: np.ndarray, iy: np.ndarray,
                      dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
            # Mirror the scalar path's lookup: perm[(ix + perm[iy & 255]) & 255] & 7
            h = perm[(ix + perm[iy & 255]) & 255] & 7
            return _GRADS_2D_NP[h, 0] * dx + _GRADS_2D_NP[h, 1] * dy

        n00 = _grad_arr(x0,     y0,     xf,         yf)
        n10 = _grad_arr(x0 + 1, y0,     xf - 1.0,   yf)
        n01 = _grad_arr(x0,     y0 + 1, xf,         yf - 1.0)
        n11 = _grad_arr(x0 + 1, y0 + 1, xf - 1.0,   yf - 1.0)

        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        return nx0 + v * (nx1 - nx0)


# ---------------------------------------------------------------------------
# fBm — scalar + vectorized
# ---------------------------------------------------------------------------


def fbm(
    noise: PerlinNoise2D,
    x: float,
    y: float,
    *,
    octaves: int,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    base_frequency: float = 1.0,
) -> float:
    """Fractal Brownian motion: sum of noise octaves with rising frequency
    and falling amplitude. Output approximately ``[-1, 1]``.
    """
    total = 0.0
    amplitude = 1.0
    frequency = base_frequency
    norm = 0.0
    for _ in range(octaves):
        total += amplitude * noise.sample(x * frequency, y * frequency)
        norm += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / norm if norm > 0 else 0.0


def fbm_grid(
    noise: PerlinNoise2D,
    x: np.ndarray,
    y: np.ndarray,
    *,
    octaves: int,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    base_frequency: float = 1.0,
) -> np.ndarray:
    """Vectorized fBm — see ``fbm`` for the semantic. Inputs same shape,
    output same shape, dtype ``float64``. Output is the per-octave
    amplitude-weighted average, so the range is approximately ``[-1, 1]``
    independent of ``octaves``.
    """
    total = np.zeros_like(x, dtype=np.float64)
    amplitude = 1.0
    frequency = base_frequency
    norm = 0.0
    for _ in range(octaves):
        total += amplitude * noise.sample_grid(x * frequency, y * frequency)
        norm += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    if norm > 0.0:
        total /= norm
    return total


def ridged_fbm(
    noise: PerlinNoise2D,
    x: float,
    y: float,
    *,
    octaves: int,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    base_frequency: float = 1.0,
) -> float:
    """Ridged multifractal: ``1 - |noise|`` per octave, with each octave
    modulated by the previous. Output approximately ``[0, 1]`` with sharp
    ridge-lines at zero-crossings of the underlying noise.
    """
    total = 0.0
    amplitude = 1.0
    frequency = base_frequency
    weight = 1.0
    norm = 0.0
    for _ in range(octaves):
        n = noise.sample(x * frequency, y * frequency)
        n = 1.0 - abs(n)
        n *= n  # sharpen
        n *= weight
        weight = min(1.0, n * 2.0)  # gate higher octaves by lower ones
        total += amplitude * n
        norm += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / norm if norm > 0 else 0.0
