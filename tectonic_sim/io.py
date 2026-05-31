"""On-disk persistence for a polygon-sim final state.

The polygon sim returns its result in memory as a bundle of numpy arrays
(``owner``, ``crust``, ``age``, ``thickness``) plus geometry scalars
(``cell_km``, ``sim_domain``) and the isostasy-relevant subset of
``SimConfig`` (sea level, continental reference + isostasy factor,
ridge depth + subsidence, max ocean depth). Only that minimal subset
is needed for any analysis tool that wants to read crust thickness,
plate ownership, or signed elevation along an arbitrary path.

``save_state`` writes a single ``.npz`` containing those arrays + scalars.
``load_state`` reads it back as a frozen ``SimState`` dataclass that the
transect tool (and any future analysis tool) consumes directly.

The full ``SimConfig`` is not persisted here — only the scalars
``particle_elevation_km`` actually reads. If a downstream tool needs
more sim-config context, it can load ``config.json`` from the worldgen
export folder; that's the source of truth for reproducibility, not this
state file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tectonic_sim.polygon_sim.isostasy import particle_elevation_km
from tectonic_sim.types import WorldRect


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimState:
    """Final state of a polygon-sim run, as needed by analysis tools.

    Per-cell arrays all share shape ``(gy, gx)``:

      - ``owner`` (int32)    — plate id per cell; ``-1`` = unowned
      - ``crust`` (int8)     — ``CRUST_CONTINENTAL`` / ``CRUST_OCEANIC``
      - ``age`` (float64)    — crust age in Myr
      - ``thickness`` (float64) — crust thickness in km

    Geometry:

      - ``cell_km``         — physical cell width in km
      - ``sim_domain``      — toroidal sim domain (width, height in km)

    Isostasy scalars (the subset of ``SimConfig`` that
    ``particle_elevation_km`` actually consumes):

      - ``sea_level_km``
      - ``continental_reference_thickness_km``
      - ``continental_isostasy_factor``
      - ``ridge_depth_km``
      - ``ridge_subsidence_rate``
      - ``max_ocean_depth_km``
    """

    owner: np.ndarray
    crust: np.ndarray
    age: np.ndarray
    thickness: np.ndarray
    cell_km: float
    sim_domain: WorldRect
    sea_level_km: float
    continental_reference_thickness_km: float
    continental_isostasy_factor: float
    ridge_depth_km: float
    ridge_subsidence_rate: float
    max_ocean_depth_km: float

    @property
    def gy(self) -> int:
        return int(self.owner.shape[0])

    @property
    def gx(self) -> int:
        return int(self.owner.shape[1])

    def elevation_km(
        self,
        crust: np.ndarray,
        thickness: np.ndarray,
        age: np.ndarray,
    ) -> np.ndarray:
        """Signed elevation via the same isostasy mapping the sim itself
        uses. Inputs are arbitrary-shape (must match each other).
        """
        # particle_elevation_km duck-types on its sim_config arg; we
        # only need the 5 isostasy scalars.
        iso = SimpleNamespace(
            continental_reference_thickness_km=
                self.continental_reference_thickness_km,
            continental_isostasy_factor=self.continental_isostasy_factor,
            ridge_depth_km=self.ridge_depth_km,
            ridge_subsidence_rate=self.ridge_subsidence_rate,
            max_ocean_depth_km=self.max_ocean_depth_km,
        )
        return particle_elevation_km(crust, thickness, age, iso)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


_NPZ_KEYS = (
    "owner", "crust", "age", "thickness",
    "cell_km", "sim_width_km", "sim_height_km",
    "sea_level_km",
    "continental_reference_thickness_km",
    "continental_isostasy_factor",
    "ridge_depth_km",
    "ridge_subsidence_rate",
    "max_ocean_depth_km",
)


def save_state(path: Path, raw_snapshot: dict) -> None:
    """Persist a polygon-sim raw_snapshot dict to ``path`` as ``.npz``.

    Expects ``raw_snapshot`` in the shape worldgen builds it in
    ``tectonics_cast.simulate_tectonics_via_continuous_sim``:
    keys ``owner``, ``crust``, ``age``, ``thickness``, ``cell_km``,
    ``sim_domain``, ``sim_config``. Other keys (``plates``, ``frames``,
    ``hotspots``) are ignored — they're either non-numeric or
    visualisation-only.
    """
    if raw_snapshot.get("kind") != "polygon_sim":
        raise ValueError(
            f"unrecognised raw_snapshot kind {raw_snapshot.get('kind')!r}; "
            "expected 'polygon_sim'."
        )
    sim_domain: WorldRect = raw_snapshot["sim_domain"]
    cfg = raw_snapshot["sim_config"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        owner=raw_snapshot["owner"].astype(np.int32, copy=False),
        crust=raw_snapshot["crust"].astype(np.int8, copy=False),
        age=raw_snapshot["age"].astype(np.float64, copy=False),
        thickness=raw_snapshot["thickness"].astype(np.float64, copy=False),
        cell_km=np.float64(raw_snapshot["cell_km"]),
        sim_width_km=np.float64(sim_domain.width_km),
        sim_height_km=np.float64(sim_domain.height_km),
        sea_level_km=np.float64(cfg.sea_level_km),
        continental_reference_thickness_km=np.float64(
            cfg.continental_reference_thickness_km),
        continental_isostasy_factor=np.float64(
            cfg.continental_isostasy_factor),
        ridge_depth_km=np.float64(cfg.ridge_depth_km),
        ridge_subsidence_rate=np.float64(cfg.ridge_subsidence_rate),
        max_ocean_depth_km=np.float64(cfg.max_ocean_depth_km),
    )


def load_state(path: Path) -> SimState:
    """Read a state file written by ``save_state`` and return a SimState."""
    path = Path(path)
    with np.load(path) as data:
        missing = [k for k in _NPZ_KEYS if k not in data.files]
        if missing:
            raise ValueError(
                f"state file {path} missing keys: {missing}; "
                f"present: {list(data.files)}"
            )
        return SimState(
            owner=np.asarray(data["owner"], dtype=np.int32),
            crust=np.asarray(data["crust"], dtype=np.int8),
            age=np.asarray(data["age"], dtype=np.float64),
            thickness=np.asarray(data["thickness"], dtype=np.float64),
            cell_km=float(data["cell_km"]),
            sim_domain=WorldRect(
                width_km=float(data["sim_width_km"]),
                height_km=float(data["sim_height_km"]),
            ),
            sea_level_km=float(data["sea_level_km"]),
            continental_reference_thickness_km=float(
                data["continental_reference_thickness_km"]),
            continental_isostasy_factor=float(
                data["continental_isostasy_factor"]),
            ridge_depth_km=float(data["ridge_depth_km"]),
            ridge_subsidence_rate=float(data["ridge_subsidence_rate"]),
            max_ocean_depth_km=float(data["max_ocean_depth_km"]),
        )
