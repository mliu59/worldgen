"""Tests for the continental-relief Perlin noise feature in seeding.

Continental relief perturbs initial continental cell thickness with a
zero-mean (per plate) Perlin fBm field in physical km, producing varied
continental interiors — shelves, inland basins, straits, archipelagos —
after sea-level sampling.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tectonic_sim.config_loader import load_sim_config_from_path
from tectonic_sim.polygon_sim.seeding import _initial_state
from tectonic_sim.types import CRUST_CONTINENTAL, WorldRect


@pytest.fixture
def sim_config():
    from pathlib import Path
    return load_sim_config_from_path(Path("config/tectonic_sim.toml"))


@pytest.fixture
def domain() -> WorldRect:
    # Large enough that the 800 km wavelength noise has multiple wavelengths
    # across the domain — gets a meaningful relief field, not a single tilt.
    return WorldRect(width_km=2000.0, height_km=2000.0)


def _continental_thickness(plates) -> np.ndarray:
    """All continental cell thicknesses across all plates."""
    chunks = []
    for p in plates:
        m = (p.crust == CRUST_CONTINENTAL) & p.cell_mask
        if m.any():
            chunks.append(p.thickness[m])
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)


def _per_plate_continental_means(plates) -> dict[int, float]:
    """Mean continental thickness per plate id."""
    out: dict[int, float] = {}
    for p in plates:
        m = (p.crust == CRUST_CONTINENTAL) & p.cell_mask
        if m.any():
            out[p.pid] = float(p.thickness[m].mean())
    return out


def test_continental_relief_adds_variability(sim_config, domain) -> None:
    """Turning on relief widens the per-cell thickness distribution."""
    cfg_off = replace(sim_config, continental_relief_amplitude_km=0.0)
    cfg_on  = sim_config  # config default = 6 km amplitude

    plates_off, _ = _initial_state(domain, cfg_off, seed=42)
    plates_on,  _ = _initial_state(domain, cfg_on,  seed=42)

    t_off = _continental_thickness(plates_off)
    t_on  = _continental_thickness(plates_on)

    # OFF still has per-plate scalar variation (init_thickness_per_plate_sigma),
    # so its std is non-zero — but the per-cell field should add substantially
    # more variability on top of that.
    assert t_on.std() > t_off.std() + 1.0, (
        f"relief should add ≥1 km of std; off={t_off.std():.2f}, on={t_on.std():.2f}"
    )

    # Mean unchanged because relief is zero-mean per plate, but per-plate
    # multipliers are seeded independently so OFF vs ON could differ slightly
    # in plate composition; loose check.
    assert abs(t_on.mean() - t_off.mean()) < 0.5


def test_continental_relief_is_zero_mean_per_plate(sim_config, domain) -> None:
    """Each plate's average continental thickness is unchanged by relief
    — the mass-conservation contract on per-plate scale.
    """
    cfg_off = replace(sim_config, continental_relief_amplitude_km=0.0)
    cfg_on  = sim_config

    plates_off, _ = _initial_state(domain, cfg_off, seed=42)
    plates_on,  _ = _initial_state(domain, cfg_on,  seed=42)

    means_off = _per_plate_continental_means(plates_off)
    means_on  = _per_plate_continental_means(plates_on)

    common = set(means_off) & set(means_on)
    assert common, "expected at least one continental plate shared between runs"
    # Plate ownership/geometry is identical across runs (same seed, relief
    # noise only touches thickness), so per-plate means should match
    # numerically to within float epsilon.
    for pid in common:
        assert abs(means_off[pid] - means_on[pid]) < 1e-9, (
            f"plate {pid}: off={means_off[pid]:.6f} on={means_on[pid]:.6f}"
        )


def test_continental_relief_only_affects_continental_cells(
    sim_config, domain,
) -> None:
    """Oceanic cells must be untouched by the relief field — only
    continental thicknesses change between off/on.
    """
    cfg_off = replace(sim_config, continental_relief_amplitude_km=0.0)
    cfg_on  = sim_config

    plates_off, _ = _initial_state(domain, cfg_off, seed=42)
    plates_on,  _ = _initial_state(domain, cfg_on,  seed=42)

    # Pair plates by pid and compare oceanic thicknesses.
    off_by_pid = {p.pid: p for p in plates_off}
    for p_on in plates_on:
        if p_on.pid not in off_by_pid:
            continue
        p_off = off_by_pid[p_on.pid]
        # Oceanic-cell mask that's identical in both (since geometry is fixed
        # by seed): use the off-run's mask.
        ocean_mask = (p_off.crust != CRUST_CONTINENTAL) & p_off.cell_mask
        if not ocean_mask.any():
            continue
        np.testing.assert_array_equal(
            p_off.thickness[ocean_mask],
            p_on.thickness[ocean_mask],
            err_msg=f"plate {p_on.pid}: oceanic thicknesses changed under relief",
        )


def test_continental_relief_deterministic(sim_config, domain) -> None:
    """Same (config, seed) → byte-identical thickness arrays."""
    plates_a, _ = _initial_state(domain, sim_config, seed=7)
    plates_b, _ = _initial_state(domain, sim_config, seed=7)
    for pa, pb in zip(plates_a, plates_b):
        np.testing.assert_array_equal(pa.thickness, pb.thickness)


def test_continental_relief_diverges_on_seed_change(sim_config, domain) -> None:
    """Different seeds → meaningfully different continental-relief fields.

    The RNG hierarchy for the relief noise uses ``seed ^ TAG``, so two
    distinct seeds must produce different perm tables and thus a
    different perturbation field.
    """
    plates_a, _ = _initial_state(domain, sim_config, seed=1)
    plates_b, _ = _initial_state(domain, sim_config, seed=2)
    t_a = _continental_thickness(plates_a)
    t_b = _continental_thickness(plates_b)
    # Different plate placements → can't compare cell-by-cell. Verify
    # std overlap is plausible but the global statistical fingerprint
    # differs (max/min not identical).
    assert (t_a.max() != t_b.max()) or (t_a.min() != t_b.min())
