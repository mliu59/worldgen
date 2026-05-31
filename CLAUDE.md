# CLAUDE.md — Worldgen

This file is the project's memory and design contract. Read it fully before making any changes. When in doubt, the principles here override your defaults.

When user instructions conflict with the principles here, raise them for review by asking clarification questions and pointing out discrepancies. 

Technical implementation details should be self-documenting in the code itself. DO NOT add specifics of computational methods here. This memory document should only serve as a CONCISE high level overview of the concepts involved in this package. 

DO NOT use this document as an active tracker for temporary status. Any temporary notes regarding the completion/pending status of tasks should go into a separate folder. Use `.claude`

## Project mission

`worldgen` is a deterministic, layered hex-grid world generator: from a `(config, seed)` pair it produces a `GeneratedWorld` containing per-hex elevation, sea/coast/lake/river flags, temperature, precipitation, and biome — plus the intermediate layer outputs needed for testing and rendering. The world is a rectangular footprint set by `[worldgen.world] width_km, height_km` (`WorldShape` in `worldgen.types`) — not a hex radius.

The package has no runtime dependencies; the optional `[preview]` extra adds Pillow for PNG rendering.

## Three commitments that override convenience

These are non-negotiable and drive most architectural decisions:

1. **Determinism / simulatability** — every output is a pure function of `(config, seed)`. No `random.random()`, no `time.time()`, no unordered iteration over dicts that affects results. Each layer derives its child RNG by hashing the layer name into `RngHierarchy`, so adding a new layer or reordering layers within `pipeline.generate` never reshuffles existing seeds.

2. **Interpretability** — the full result is inspectable. `GeneratedWorld` exposes both per-hex `HexData` and every intermediate layer (`ElevationLayer`, `SeaLayer`, `ClimateLayer`, `HydrologyLayer`, `PlateField | None`). Anything that affects the final map can be examined directly; nothing is hidden behind aggregated outputs.

3. **Testability** — every deterministic function gets a unit test. Tests live under `tests/` and run without any heavy dependency (Pillow is only needed for `preview.py`, never imported by the layers themselves). Snapshot-style tests pin the final world hash for a fixed `(config, seed)`; any change to outputs should be a conscious decision, not a surprise.

If a design choice trades against any of these three, flag it and ask before proceeding.

## Architectural principles

**Pure functions, layer by layer.** Each layer is `compute(prior_layer_outputs, config, rng_child) → LayerOutput`. Mutation is fine inside a layer; the layer's inputs are treated as immutable. `pipeline.generate` is the only orchestrator.

**Configuration over code.** Coefficients (noise frequencies, lapse rates, river thresholds, crop envelopes, deposit cluster parameters) live in TOML configs in `config`, not in code. The engine should be the same regardless of which world parameters are loaded.

**No backwards compatibility.** When changing a feature, *change it*. Don't add fallback paths, default values to paper over missing fields, optional flags toggling old vs new behavior, or compatibility shims. If old call sites break, fix them.

**Explicit data structures.** All state objects are `@dataclass(frozen=True)` in `worldgen/types.py`. No bare dicts for anything with a stable schema. No default values on dataclass fields whose meaning is non-trivial — missing-data bugs should surface at construction, not as silent zeros downstream.

**Module boundaries:**

```
worldgen/
├── hex.py             — Hex coordinate primitive (axial + spiral iterator)
├── terrain.py         — TERRAIN_NAMES (the canonical biome name tuple)
├── rng.py             — RngHierarchy (sha256-keyed child RNGs)
├── types.py           — WorldgenConfig + per-layer dataclasses + HexData
├── config_loader.py   — TOML → WorldgenConfig
├── pipeline.py        — orchestrator; returns GeneratedWorld
├── plates.py          — L0a: t=0 plate seed placement + boundary classification
├── tectonics.py       — L0b: per-hex result types + adaptor that delegates
│                         to tectonics_cast
├── tectonics_cast.py  — Bridge: runs `tectonic_sim.polygon_sim`, samples
│                         its final cell grid onto worldgen hexes
├── elevation.py       — L1: tectonic baseline + fBm/ridged detail + analytic mask
├── sea.py             — L2: ocean/coast mask
├── climate.py         — L3+L4: temperature + precipitation (wind sweep, orographic)
├── hydrology.py       — L5: priority-flood + D6 flow accumulation → rivers/lakes
├── ocean.py           — L2.5: gyre-based currents + continentality (Tier 2)
├── biome.py           — L6: Whittaker(T, P) + elevation/coast/water overrides
├── preview.py         — Library: PNG renderer (requires Pillow), used by export
└── export.py          — Public: WorldSnapshot + serialize/save/load +
                         export_world (snapshot + per-layer PNGs +
                         tectonic_sim visualisation artefacts)
```

Tests under `tests/` mirror the package; `conftest.py` exposes `default_worldgen_config`, `small_world` (120×120 km, seed 42), and `medium_world` (300×300 km, seed 42) session-scoped fixtures.

## World generation pipeline

Layer order — each layer is a pure function of all earlier layers' outputs plus its own seeded child RNG. Adding a new layer or a new entry within a layer never reshuffles existing seeds.

```
seed + config
  ↓ L0a plates         Noise-applied weighted voronoi plate seeding
  ↓ L0b tectonics      time-stepped sim (n_ticks × dt_myr of geological time)
  ↓ L1  elevation      tectonic baseline (km) + fBm/ridged detail
  ↓ L2  sea level      ocean/coast mask (absolute sea_level_km)
  ↓ L2.5 ocean         gyre-based currents + continentality (Tier 2)
  ↓ L3  temperature    latitude band + lapse + ocean-current anomaly
  ↓ L4  precipitation  prevailing-wind sweep + orographic uplift,
                        floor damped by continentality
  ↓ L5  hydrology      priority-flood + ε-tilt → D6 flow accum → rivers, lakes
  ↓ L6  biome          Whittaker(T, P) lookup + elevation/coast/water overrides
GeneratedWorld
```

## Ocean layer (Tier 2 climate)

`worldgen/ocean.py` runs between sea and climate, producing per-hex
current directions and temperature anomalies as an annual-mean
snapshot. It captures three coupled effects:

- **Gyres.** Each connected ocean basin is split into Coriolis-driven
  rotating cells, one or two per hemisphere.
- **Boundary-current anomaly.** Each ocean hex inherits a temperature
  offset from the latitudinal context sampled upstream along its gyre
  flow — the mechanism behind warm western-boundary and cold eastern-
  boundary currents.
- **Coastal pickup + continentality.** Land hexes within reach of the
  ocean inherit a damped fraction of the local ocean anomaly; the
  precipitation floor is damped by distance-to-ocean so deep continental
  interiors don't carry an unrealistic moisture carpet.

This is a single-pass approximation. Seasonality, pressure-field
advection, and vegetation feedback are not modelled.

## Tectonics layer

`worldgen.tectonics.simulate_tectonics` is a thin worldgen adaptor over
the rigid-polygon `tectonic_sim.polygon_sim` package. It delegates to
`worldgen.tectonics_cast.simulate_tectonics_via_continuous_sim`, which
runs the polygon sim on a domain larger than the worldgen world (so
plates have room to drift), samples the final cell grid at every
world-hex centre, and packages the raw polygon-sim output onto
`LithosphereState.raw_snapshot` for export-time renderers.

**`tectonic_sim` has no crop concept.** It outputs the full sim domain
— per-tick frames and the final `(owner, crust, age, thickness)` ndarrays
all cover the entire torus the sim runs on. Worldgen does its own
sampling: `_sample_polygon_at_hex_centres` reads each world-hex's
mantle-frame `(x, y)` directly from the full sim grid (wrap modulo
puts the hex into the sim's central region by construction). The
`tectonic_sim_views/` PNGs and GIFs that worldgen exports render the
full sim — including the buffer area outside the world rectangle —
so plate drift, sutures, and hotspots remain visible everywhere. The
worldgen-specific per-hex outputs (`layers/elevation.png`, etc.) stay
world-sized because they're built from the hex set.

The export also writes `tectonic_sim_views/state.npz` — the raw sim
arrays (`owner`, `crust`, `age`, `thickness`) + cell geometry + the
isostasy scalars needed to derive signed elevation. `tectonic_sim.io`
(`save_state` / `load_state` / `SimState`) is the canonical reader.
Offline analysis tools like `tectonic_sim.transect` consume this file
directly; they never re-run the sim.

**1D transect tool** (`python -m tectonic_sim.transect state.npz
--p1=x1,y1 --p2=x2,y2 --out path.png`) samples plate ownership, crust
thickness, age, and signed elevation along an arbitrary Cartesian
segment in sim-centred km. The segment is interpreted literally
(direct line, not toroidal-shortest); cell lookups wrap so a segment
that runs off one edge still reads sensible values. The rendered PNG
has two stacked panels — elevation with a sea-level reference line on
top, thickness below — coloured per plate with the same palette as
`partition.png`.

All `tectonic_sim_views/` PNGs (partition / crust / thickness /
topography / polygons) carry km tick marks + labels along the bottom
and left edges (`_overlay_km_axes` in `viz.py`). The km coordinates
read off these axes feed directly into the transect tool's
`--p1`/`--p2` arguments — no mental conversion from cell indices.

The polygon sim itself models plate kinematics, contention, fusion,
rifting, accretion, hotspot volcanism, erosion, and connected-component
culling as a per-tick pipeline. See `tectonic_sim/polygon_sim/` for the
authoritative implementation; the worldgen side does not know or care
about the per-tick order.

**Continental collisions build asymmetric inland fold-belts on BOTH
plates.** C-C contention distributes folded mass into two belts:

- **Over-rider side** — wide, broad. Inland on the winner's
  `−velocity` direction, ~120 km depth with 35 km decay. Tibet-plateau
  analogue. Knobs: `folding_belt_depth_km`, `folding_belt_decay_km`.
- **Loser side** — narrow, sharp. Inland on the loser's own
  `−velocity` direction (= opposite, propagates the other way from the
  suture), starting one cell into the loser's interior, ~50 km depth
  with 15 km decay. Himalayan-foothill analogue. Knobs:
  `folding_loser_side_ratio`, `folding_belt_loser_depth_km`,
  `folding_belt_loser_decay_km`.

Mass accounting per fold event: over-rider gains `folding_ratio · t`,
loser gains `folding_loser_side_ratio · t`, mantle absorbs the rest.
At triple junctions, each loser contributes independently into its
own continent. Belt cells running off-plate or onto oceanic crust
drop their mass (models belt running into ocean).

**Continental relief is seeded as Perlin fBm noise at t=0.** Continental
cells get a zero-mean (per-plate) thickness perturbation in physical
km at continent-scale wavelengths. After sea-level sampling, the thin
spots become shelves / inland seas / straits and the thick spots
stand proud — what would otherwise be featureless plate interiors
turn into varied continents with shelf systems, archipelagos, and
inland basins. Knobs: `continental_relief_amplitude_km`,
`continental_relief_wavelength_km`, `continental_relief_octaves`,
`continental_relief_persistence`. 0 amplitude disables. Combines
naturally with the decoupled sea-level knob — one sim run yields a
family of geographies as `sea_level_km` sweeps.

**Boundary mode is torus-only.** Every spatial query inside
`tectonic_sim` uses the toroidal shortest-path metric.

**Single source of truth for configuration.** `config/tectonic_sim.toml`
holds every tunable for the polygon sim. Worldgen reads it directly
(path from `[worldgen].tectonic_sim_config` in `worldgen.toml`) and
assigns the resulting `SimConfig` to `WorldgenConfig.tectonics`.
`TectonicsConfig` in `worldgen.types` is a transparent alias for
`SimConfig`; there is no second source of tectonic-sim tunables.

**Sea level is decoupled from crust dynamics.** `sea_level_km` is a
passive sampling threshold + elevation-render colormap midpoint;
nothing in the per-tick polygon sim reads it. Cells carry thickness,
age, crust type, and plate id — not "above water." Isostasy returns
signed elevation in km from a mantle reference, and sea level is just
the water line on that signed axis. The payoff is that two natural
sweeps come for free:

- **Hold the world fixed, vary sea level.** Raising the threshold
  instantly converts shallow continental hexes to epicontinental sea —
  no sim rerun.
- **Hold sea level fixed, vary tectonic parameters.** Plate dynamics
  alone reshape geography; the water line stays at the same physical
  reference so before/after maps are directly comparable.

## `param_temperature` — physics-parameter exploration

`WorldgenConfig.param_temperature` is a top-level hyperparameter that
controls how much the run perturbs subsystem physics around its
configured baseline. `0` is fully deterministic; larger values produce a
parameter draw whose spread is proportional to the temperature.

Today only the tectonics bridge consumes it — by feeding the loaded
`SimConfig` through `tectonic_sim.randomize_sim_config`. The hook is
intended to extend to other subsystems (climate priors, ocean
coefficients, …) when the need arises. Randomization is decoupled from
the simulator's own RNG hierarchy so changing the temperature doesn't
reshuffle the base seed.

## Map latitude window

`hex_size_km` (the physical resolution) is **independent** of where the
map sits on the planet. `[worldgen.climate]` carries `map_lat_min` /
`map_lat_max` — the geographic latitudes the map's r-axis covers. The
planet's overall climate is anchored by `equator_temp_c` and
`polar_temp_c`; the map samples a slice of that gradient through its
lat window, with Earth-like wind bands.

The km extent of the map is set independently by `world.width_km` /
`world.height_km`, so the same physical map can span any latitude range
— a 1000-km map can equally well represent 1° or 60° of latitude.

**Physical-unit scaling.** Scale-dependent generator parameters live in
physical units (km, km², mm/km-of-land-fetch) and are converted to
per-hex units via `hex_size_km` at use time. Changing `hex_size_km`
automatically rescales noise frequency, wind reach, river thresholds,
and precipitation rates — the same physical world looks the same at any
chosen resolution.

The per-hex output schema lives in `HexData` (`worldgen/types.py`).

## Export

Two public endpoints in `worldgen.export` (re-exported from the package root):

- `serialize_world(world) -> WorldSnapshot` — pure projection of a
  `GeneratedWorld` into a generic, JSON-friendly container. No side effects;
  no timestamp / seed injected here. Identical worlds → equal snapshots.
- `export_world(config, seed, output_root, ...) -> Path` — generates,
  serializes to `snapshot.json`, and renders one PNG per layer under
  `<output_root>/seed<seed>_<W>x<H>km_<YYYYMMDD-HHMMSS>/layers/`. The
  per-export folder name carries the seed, world footprint (W×H from
  `config.world`), and timestamp; the snapshot's `metadata` carries
  seed, timestamp, schema_version, world_width_km, world_height_km,
  hex_size_km.

`save_snapshot` / `load_snapshot` are JSON file I/O helpers; `WorldSnapshot`
itself is format-agnostic via `to_dict` / `from_dict`.

**`WorldSnapshot` is deliberately a generic data container.** All three fields
are open-ended `dict[str, Any]` / `list[dict[str, Any]]`:

| Field | Shape | Holds |
|---|---|---|
| `metadata` | `dict[str, Any]` | run parameters, schema version, anything caller-injected |
| `hexes` | `list[dict[str, Any]]` | one flat record per hex (q, r + every HexData field as primitives) |
| `layers` | `dict[str, dict[str, Any]]` | layer-name → layer-level (non-per-hex) data |

Adding a new per-hex field, a new intermediate layer, or new metadata keys
does **not** require changing `WorldSnapshot`'s schema — the new data slots
into the existing dicts. This is the extension point as the simulator grows.

## Export CLI

`python -m worldgen` is the canonical entry point — it generates a world,
serializes it to `snapshot.json`, and renders every available layer as a
PNG (plus a per-plate `plates/plate_NN.png` and a drift `drift.gif`).

```
python -m worldgen --seed 42 --out exports/
python -m worldgen --seed 42 --out exports/ --stop-after climate
python -m worldgen --seed 42 --out exports/ -q   # silence logs
python -m worldgen --seed 42 --out exports/ --profile
python -m worldgen --seed 42 --config path/to/custom.toml --out exports/
```

World dimensions come from `[worldgen.world] width_km, height_km` in the
TOML — there is no `--radius` / `--width` / `--height` CLI override; if
you want a different footprint, point `--config` at a TOML with the
shape you want.

`--stop-after STEP` halts the pipeline after the named step (one of
`PIPELINE_STEPS`); only the layers whose source data is populated are
rendered. Default logging is DEBUG (per-layer timings + progress bars);
`-q`/`--quiet` drops to WARNING. Requires the `[preview]` extra
(Pillow). `worldgen/preview.py` is a pure library; `python -m worldgen`
is the only CLI.

`--profile` re-execs the run under py-spy and records a wall-clock
flamegraph (SVG) into `<out>/profiles/profile_<timestamp>_s<seed>.svg`
alongside the export folder; pair with `--profile-sample-rate HZ`
(default 200) to tune sampling density. Requires the `[dev]` extra
(py-spy on PATH or installed in the active environment).

## Demos

`demos/` is intentionally near-empty (only `__init__.py` and a
gitignored `output/`). All visual inspection and profiling runs
through the canonical `python -m worldgen` entry point: polygon-sim
renders land in `<export>/tectonic_sim_views/`, profiling is the
`--profile` flag (see Export CLI).

Add new demos here only when they exercise something the main CLI
genuinely can't (e.g. multi-seed parameter sweeps across processes);
otherwise prefer extending `python -m worldgen`.

## Conventions

**Python.** 3.11+. Modern type hints (`list[int]`, `X | None`, etc.). Code should pass `mypy --strict` on `worldgen/`.

**Formatting & linting.** `ruff` for both. Config in `pyproject.toml`.

**Testing.** `pytest`. Tests in `tests/` mirror `worldgen/`. Use fixtures for common world setups.

**Naming.**
- `layer` = one pipeline stage; `LayerOutput` = its frozen dataclass result
- `hex` (lowercase) = a `Hex` coordinate; `hexes` = an iterable of them
- `gen` / `world` = a `GeneratedWorld`
- `cfg` = a `WorldgenConfig`

**RNG discipline.** Single root seed → `RngHierarchy(seed)`. Layers call `rng.child("layer_name")` (or a more specific tuple) to get a `random.Random`. Never call the root RNG directly. Never seed from `time.time()`.

**Errors.** Fail loudly. If a config is missing a required field, raise. No silent defaults.

## Anti-patterns to avoid

- Hidden randomness outside `RngHierarchy`.
- Magic numbers in code. Coefficients go in `config/worldgen.toml`.
- Backwards-compat shims when changing a feature. No fallback paths, no "if not configured, do the old thing" branches.
- Premature abstraction. Don't build a plugin system for layers; we have a handful.
- Premature optimization. Clarity wins for v0. Profile before tuning.
- Layer-to-layer reaching past the explicit `pipeline.generate` wiring (e.g. importing `hydrology` from inside `climate`). New cross-layer information must flow through a typed layer output.

## Behavioral patterns to follow

### 1. Think Before Coding

Before implementing:
- State your assumptions. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so.

### 2. Simplicity First

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" that wasn't requested.

### 3. Surgical Changes

- Touch only what you must.
- Don't refactor adjacent code that isn't broken.
- Match existing style.

### 4. Goal-Driven Execution

- Define success criteria.
- For multi-step tasks, state the plan as steps with verification points.
