"""Tests for the deterministic Perlin noise module."""

from __future__ import annotations

import random

import numpy as np
import pytest

from tectonic_sim.noise import PerlinNoise2D, fbm, fbm_grid, ridged_fbm


@pytest.fixture
def noise() -> PerlinNoise2D:
    return PerlinNoise2D.from_rng(random.Random(0))


def test_noise_is_bounded(noise: PerlinNoise2D) -> None:
    """Perlin output should be approximately in [-1, 1]."""
    rng = random.Random(1)
    samples = [noise.sample(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(2000)]
    assert max(samples) <= 1.01
    assert min(samples) >= -1.01


def test_noise_zero_at_integer_lattice(noise: PerlinNoise2D) -> None:
    """At integer coordinates the gradient dot products vanish → noise = 0."""
    for ix in range(-5, 6):
        for iy in range(-5, 6):
            assert abs(noise.sample(float(ix), float(iy))) < 1e-9


def test_noise_continuous(noise: PerlinNoise2D) -> None:
    """Noise should be C0-continuous: nearby points have nearby values."""
    rng = random.Random(2)
    for _ in range(100):
        x = rng.uniform(-20, 20)
        y = rng.uniform(-20, 20)
        v0 = noise.sample(x, y)
        v1 = noise.sample(x + 0.001, y)
        # Gradient is bounded for Perlin; small step → small change.
        assert abs(v0 - v1) < 0.05


def test_noise_seed_reproducible() -> None:
    """Same seed → same noise field, byte-for-byte."""
    n1 = PerlinNoise2D.from_rng(random.Random(123))
    n2 = PerlinNoise2D.from_rng(random.Random(123))
    for x in (-3.7, 0.0, 4.4, 11.1):
        for y in (-2.2, 1.5, 8.8):
            assert n1.sample(x, y) == n2.sample(x, y)


def test_noise_seed_diverges() -> None:
    """Different seeds → different noise fields.

    Sample at non-integer points, since Perlin vanishes at integer lattice
    coordinates regardless of seed.
    """
    n1 = PerlinNoise2D.from_rng(random.Random(1))
    n2 = PerlinNoise2D.from_rng(random.Random(2))
    diff = sum(abs(n1.sample(x + 0.37, x * 0.5 + 0.21) - n2.sample(x + 0.37, x * 0.5 + 0.21))
               for x in range(40))
    assert diff > 1.0


def test_fbm_bounded(noise: PerlinNoise2D) -> None:
    rng = random.Random(3)
    samples = [
        fbm(noise, rng.uniform(-50, 50), rng.uniform(-50, 50),
            octaves=6, base_frequency=0.05)
        for _ in range(500)
    ]
    assert max(samples) <= 1.01
    assert min(samples) >= -1.01


def test_fbm_one_octave_matches_single_noise(noise: PerlinNoise2D) -> None:
    """fBm with 1 octave at frequency f equals a single noise(x*f, y*f) call."""
    for x, y in [(0.3, 0.4), (1.7, -2.1), (5.5, 5.5)]:
        direct = noise.sample(x * 1.5, y * 1.5)
        via_fbm = fbm(noise, x, y, octaves=1, base_frequency=1.5)
        assert abs(direct - via_fbm) < 1e-12


def test_fbm_octaves_yield_different_output(noise: PerlinNoise2D) -> None:
    """6-octave fBm should differ measurably from 1-octave fBm at the same point."""
    diff = 0.0
    for x in range(0, 200):
        a = fbm(noise, x * 0.1 + 0.13, 0.7, octaves=1, base_frequency=0.5)
        b = fbm(noise, x * 0.1 + 0.13, 0.7, octaves=6, base_frequency=0.5)
        diff += abs(a - b)
    assert diff > 0.5


def test_ridged_fbm_non_negative(noise: PerlinNoise2D) -> None:
    """Ridged multifractal outputs are ``1 - |noise|`` style — non-negative."""
    rng = random.Random(5)
    for _ in range(500):
        x = rng.uniform(-50, 50)
        y = rng.uniform(-50, 50)
        v = ridged_fbm(noise, x, y, octaves=4, base_frequency=0.05)
        assert v >= 0.0


# ----- Vectorized path ---------------------------------------------------


def test_sample_grid_matches_scalar(noise: PerlinNoise2D) -> None:
    """sample_grid must produce the same values as scalar sample, pointwise.

    This is the contract that lets the seeding stage use the fast grid
    path without diverging from the scalar elevation layer's noise.
    """
    rng = random.Random(7)
    xs = np.array([rng.uniform(-30, 30) for _ in range(50)], dtype=np.float64)
    ys = np.array([rng.uniform(-30, 30) for _ in range(50)], dtype=np.float64)
    grid = noise.sample_grid(xs, ys)
    for i, (x, y) in enumerate(zip(xs, ys)):
        assert abs(float(grid[i]) - noise.sample(float(x), float(y))) < 1e-12


def test_sample_grid_2d_shape(noise: PerlinNoise2D) -> None:
    """sample_grid preserves 2D input shape."""
    xs = np.linspace(-5.0, 5.0, 13)
    ys = np.linspace(-5.0, 5.0, 17)
    gx, gy = np.meshgrid(xs, ys)
    out = noise.sample_grid(gx, gy)
    assert out.shape == (17, 13)
    assert out.dtype == np.float64


def test_fbm_grid_matches_scalar(noise: PerlinNoise2D) -> None:
    """fbm_grid result equals scalar fbm at the same points."""
    rng = random.Random(9)
    xs = np.array([rng.uniform(-50, 50) for _ in range(30)], dtype=np.float64)
    ys = np.array([rng.uniform(-50, 50) for _ in range(30)], dtype=np.float64)
    grid = fbm_grid(noise, xs, ys, octaves=4, base_frequency=0.05)
    for i, (x, y) in enumerate(zip(xs, ys)):
        scalar = fbm(noise, float(x), float(y), octaves=4, base_frequency=0.05)
        assert abs(float(grid[i]) - scalar) < 1e-12


def test_fbm_grid_bounded() -> None:
    """fbm_grid output stays in approximately [-1, 1] across many octaves."""
    n = PerlinNoise2D.from_rng(random.Random(11))
    rng = np.random.default_rng(11)
    xs = rng.uniform(-100, 100, 2000)
    ys = rng.uniform(-100, 100, 2000)
    out = fbm_grid(n, xs, ys, octaves=6, base_frequency=0.05)
    assert out.max() <= 1.01
    assert out.min() >= -1.01


def test_perlin_from_numpy_generator() -> None:
    """PerlinNoise2D.from_rng accepts np.random.Generator and is deterministic."""
    gen1 = np.random.Generator(np.random.PCG64(42))
    gen2 = np.random.Generator(np.random.PCG64(42))
    n1 = PerlinNoise2D.from_rng(gen1)
    n2 = PerlinNoise2D.from_rng(gen2)
    assert n1.perm == n2.perm
    # Should produce non-trivial noise.
    sample = n1.sample(2.7, 3.3)
    assert abs(sample) > 0.0
