"""Tests for the 1D transect tool + state.npz roundtrip.

Builds a synthetic raw_snapshot dict by hand (3 plates on a 40x30 cell
grid, deterministic) so the tests don't depend on running the full
polygon sim. The shape and dtypes match what worldgen builds in
``tectonics_cast.simulate_tectonics_via_continuous_sim``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tectonic_sim import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    SimConfig,
    WorldRect,
    load_state,
    save_state,
)
from tectonic_sim.transect import (
    TransectResult,
    render_transect,
    sample_transect,
)


# ---------------------------------------------------------------------------
# Synthetic state fixture
# ---------------------------------------------------------------------------


def _make_synthetic_snapshot(default_sim_config: SimConfig) -> dict:
    """Build a raw_snapshot dict on a 40x30 cell grid with 3 vertical
    plate stripes (pids 0, 1, 2). Plate 0 = continental, 1 = oceanic,
    2 = continental — picks up the elevation discontinuity at plate
    boundaries so the transect samples actually vary.
    """
    gy, gx = 30, 40
    cell_km = 25.0
    owner = np.zeros((gy, gx), dtype=np.int32)
    crust = np.full((gy, gx), CRUST_CONTINENTAL, dtype=np.int8)
    age = np.zeros((gy, gx), dtype=np.float64)
    thick = np.full((gy, gx), 35.0, dtype=np.float64)

    # Three vertical stripes.
    owner[:, : gx // 3] = 0           # continental, thick
    owner[:, gx // 3 : 2 * gx // 3] = 1   # oceanic
    owner[:, 2 * gx // 3 :] = 2       # continental, thin

    crust[:, gx // 3 : 2 * gx // 3] = CRUST_OCEANIC
    thick[:, : gx // 3] = 50.0           # plate 0 thicker → high elevation
    thick[:, gx // 3 : 2 * gx // 3] = 7.0  # plate 1 oceanic thickness
    thick[:, 2 * gx // 3 :] = 30.0       # plate 2 thinner

    age[:, gx // 3 : 2 * gx // 3] = 50.0  # oceanic age for subsidence

    sim_domain = WorldRect(width_km=gx * cell_km, height_km=gy * cell_km)
    return {
        "kind": "polygon_sim",
        "owner": owner,
        "crust": crust,
        "age": age,
        "thickness": thick,
        "cell_km": cell_km,
        "sim_domain": sim_domain,
        "sim_config": default_sim_config,
    }


# ---------------------------------------------------------------------------
# state.npz roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(
    tmp_path: Path, default_sim_config: SimConfig,
) -> None:
    snap = _make_synthetic_snapshot(default_sim_config)
    save_state(tmp_path / "state.npz", snap)
    state = load_state(tmp_path / "state.npz")
    np.testing.assert_array_equal(state.owner, snap["owner"])
    np.testing.assert_array_equal(state.crust, snap["crust"])
    np.testing.assert_array_equal(state.age, snap["age"])
    np.testing.assert_array_equal(state.thickness, snap["thickness"])
    assert state.cell_km == snap["cell_km"]
    assert state.sim_domain.width_km == snap["sim_domain"].width_km
    assert state.sim_domain.height_km == snap["sim_domain"].height_km
    assert state.sea_level_km == default_sim_config.sea_level_km
    # Isostasy scalars preserved.
    assert (state.continental_reference_thickness_km
            == default_sim_config.continental_reference_thickness_km)
    assert (state.continental_isostasy_factor
            == default_sim_config.continental_isostasy_factor)


def test_save_rejects_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unrecognised raw_snapshot kind"):
        save_state(tmp_path / "state.npz", {"kind": "particle_sim"})


# ---------------------------------------------------------------------------
# Transect sampling
# ---------------------------------------------------------------------------


def _build_state(default_sim_config, tmp_path):
    snap = _make_synthetic_snapshot(default_sim_config)
    save_state(tmp_path / "state.npz", snap)
    return load_state(tmp_path / "state.npz")


def test_transect_endpoints_match_input(
    tmp_path: Path, default_sim_config: SimConfig,
) -> None:
    state = _build_state(default_sim_config, tmp_path)
    p1 = (-400.0, -100.0)
    p2 = (300.0, 200.0)
    result = sample_transect(state, p1, p2, n_samples=50)
    # First and last samples should sit at exactly p1 / p2 (wrap-aware
    # path has no seam crossing here since |dx|<sim.w/2 and |dy|<sim.h/2).
    assert result.x_km[0] == pytest.approx(p1[0])
    assert result.y_km[0] == pytest.approx(p1[1])
    assert result.x_km[-1] == pytest.approx(p2[0])
    assert result.y_km[-1] == pytest.approx(p2[1])
    # Distance monotone non-decreasing, starts at 0.
    assert result.distance_km[0] == 0.0
    assert np.all(np.diff(result.distance_km) >= 0.0)


def test_transect_uses_direct_cartesian_path(
    tmp_path: Path, default_sim_config: SimConfig,
) -> None:
    """The segment between p1 and p2 is the literal Cartesian line —
    NOT the toroidal-shortest path. Picking endpoints visible on the
    partition map should give the visible line, not a wrap.
    """
    state = _build_state(default_sim_config, tmp_path)
    w = state.sim_domain.width_km
    # Span almost the full sim width along x. Direct distance = w - 100.
    # Toroidal-shortest would be 100 km (wrap), so this test fails if
    # the implementation accidentally takes the shortest path.
    p1 = (-w / 2 + 50.0, 0.0)
    p2 = (+w / 2 - 50.0, 0.0)
    result = sample_transect(state, p1, p2, n_samples=50)
    expected_len = w - 100.0
    assert float(result.distance_km[-1]) == pytest.approx(
        expected_len, rel=1e-6)


def test_transect_crossing_seam_still_reads_cells(
    tmp_path: Path, default_sim_config: SimConfig,
) -> None:
    """A segment whose endpoints sit outside the centred frame should
    still produce sane per-sample owner reads — the cell lookup wraps
    even though the segment itself is a raw Cartesian line.
    """
    state = _build_state(default_sim_config, tmp_path)
    w = state.sim_domain.width_km
    # Endpoints deliberately outside ±half_w so wrap kicks in at lookup.
    p1 = (-w, 0.0)
    p2 = (+w, 0.0)
    result = sample_transect(state, p1, p2, n_samples=100)
    # Every sample's owner must be a valid plate id present in the grid.
    valid = set(int(v) for v in np.unique(state.owner))
    seen = set(int(v) for v in np.unique(result.owner))
    assert seen.issubset(valid)


def test_transect_nearest_matches_grid(
    tmp_path: Path, default_sim_config: SimConfig,
) -> None:
    """Every sample's reported owner/crust/thickness should equal a
    direct lookup at its (cx, cy) cell index."""
    state = _build_state(default_sim_config, tmp_path)
    p1 = (-300.0, -50.0)
    p2 = (300.0, 50.0)
    result = sample_transect(state, p1, p2, n_samples=200)

    gy, gx = state.owner.shape
    half_w = state.sim_domain.half_width_km
    half_h = state.sim_domain.half_height_km
    for k in range(result.x_km.size):
        cx = int(np.floor((result.x_km[k] + half_w) / state.cell_km)) % gx
        cy = int(np.floor((result.y_km[k] + half_h) / state.cell_km)) % gy
        assert int(result.owner[k]) == int(state.owner[cy, cx])
        assert int(result.crust[k]) == int(state.crust[cy, cx])
        assert float(result.thickness_km[k]) == pytest.approx(
            float(state.thickness[cy, cx]))


def test_transect_determinism(
    tmp_path: Path, default_sim_config: SimConfig,
) -> None:
    state = _build_state(default_sim_config, tmp_path)
    p1 = (-200.0, 30.0)
    p2 = (400.0, -10.0)
    a = sample_transect(state, p1, p2, n_samples=128)
    b = sample_transect(state, p1, p2, n_samples=128)
    np.testing.assert_array_equal(a.owner, b.owner)
    np.testing.assert_array_equal(a.crust, b.crust)
    np.testing.assert_array_equal(a.thickness_km, b.thickness_km)
    np.testing.assert_array_equal(a.elevation_km, b.elevation_km)
    np.testing.assert_array_equal(a.distance_km, b.distance_km)


def test_transect_picks_up_plate_boundaries(
    tmp_path: Path, default_sim_config: SimConfig,
) -> None:
    """A horizontal transect across the 3-stripe synthetic world must
    visit all three plate IDs and produce 2 boundary transitions.

    NOTE: endpoints must stay within ``±half_width`` of each other so
    ``wrapped_delta_xy`` picks the direct (centre-crossing) path rather
    than the wrap-around path which would skip the middle stripe.
    """
    state = _build_state(default_sim_config, tmp_path)
    half_w = state.sim_domain.half_width_km
    # Direct distance |p2 - p1| < half_width keeps us on the non-wrap path.
    span = 0.45 * state.sim_domain.width_km  # < half_width
    p1 = (-span / 2.0, 0.0)
    p2 = (+span / 2.0, 0.0)
    assert abs(p2[0] - p1[0]) < half_w  # sanity: stays on direct path
    result = sample_transect(state, p1, p2, n_samples=300)
    unique_pids = set(int(p) for p in np.unique(result.owner))
    assert {0, 1, 2}.issubset(unique_pids)
    # Owner transitions count: should be 2 (plate 0→1, 1→2).
    transitions = int(np.sum(np.diff(result.owner) != 0))
    assert transitions == 2


def test_transect_elevation_responds_to_sea_level(
    tmp_path: Path, default_sim_config: SimConfig,
) -> None:
    """Plate 0 (thick continental) should sit above sea level; plate 1
    (oceanic with substantial age) should sit below.

    Endpoints stay within ±half_width of each other so the segment
    takes the direct path through all three stripes (not wrap-around).
    """
    state = _build_state(default_sim_config, tmp_path)
    span = 0.45 * state.sim_domain.width_km
    p1 = (-span / 2.0, 0.0)
    p2 = (+span / 2.0, 0.0)
    result = sample_transect(state, p1, p2, n_samples=200)
    p0_mask = result.owner == 0
    p1_mask = result.owner == 1
    assert p0_mask.any() and p1_mask.any()
    assert float(result.elevation_km[p0_mask].mean()) > state.sea_level_km
    assert float(result.elevation_km[p1_mask].mean()) < state.sea_level_km


def test_transect_render_writes_png(
    tmp_path: Path, default_sim_config: SimConfig,
) -> None:
    """Smoke test that the renderer produces a non-trivial PNG."""
    state = _build_state(default_sim_config, tmp_path)
    span = 0.45 * state.sim_domain.width_km
    p1 = (-span / 2.0, 0.0)
    p2 = (+span / 2.0, 0.0)
    result = sample_transect(state, p1, p2, n_samples=200)
    out = tmp_path / "transect.png"
    render_transect(result, out)
    assert out.exists()
    # Non-trivial: > 1 KB.
    assert out.stat().st_size > 1024


def test_transect_rejects_tiny_n_samples(
    tmp_path: Path, default_sim_config: SimConfig,
) -> None:
    state = _build_state(default_sim_config, tmp_path)
    with pytest.raises(ValueError, match="n_samples"):
        sample_transect(state, (0.0, 0.0), (10.0, 0.0), n_samples=1)
