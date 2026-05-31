"""tectonic_sim — rigid-polygon plate-tectonics simulator.

Pure-physics module. Knows nothing about hex grids; everything happens
in floating-point km space on a 2D toroidal cell grid. Worldgen calls
``simulate_rigid_polygon(...)`` from ``tectonic_sim.polygon_sim`` to get
the final plate state + cell grids and then samples them at hex centres.

The legacy particle-cloud sim was deleted — the rigid-polygon
representation is the only supported model going forward.

Public surface:

  - ``WorldRect, SimConfig``              — shared data types
  - ``load_sim_config``                   — TOML → SimConfig
  - ``randomize_sim_config``              — param_temperature perturbation
  - ``CRUST_CONTINENTAL, CRUST_OCEANIC``  — crust-type integer codes
  - ``crust_type_code, crust_type_name``  — string↔int helpers
  - ``tectonic_sim.polygon_sim``          — the actual simulator
"""

from __future__ import annotations

from tectonic_sim.config_loader import (
    load_sim_config,
    load_sim_config_from_path,
)
from tectonic_sim.io import (
    SimState,
    load_state,
    save_state,
)
from tectonic_sim.randomization import (
    FieldRandomizer,
    randomize_dataclass_fields,
    randomize_sim_config,
)
from tectonic_sim.types import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    SimConfig,
    WorldRect,
    crust_type_code,
    crust_type_name,
)

__all__ = [
    "CRUST_CONTINENTAL",
    "CRUST_OCEANIC",
    "FieldRandomizer",
    "SimConfig",
    "SimState",
    "WorldRect",
    "crust_type_code",
    "crust_type_name",
    "load_sim_config",
    "load_sim_config_from_path",
    "load_state",
    "randomize_dataclass_fields",
    "randomize_sim_config",
    "save_state",
]
