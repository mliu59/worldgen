# Terrain Generation — Methods and Notes

## Goal

Replace the original IID terrain sampler with a layered, physically-motivated
pipeline that produces continents, linear mountain ranges, dendritic river
networks, rain-shadow deserts, and latitudinal climate banding — while
remaining fully deterministic given `(seed, config)`.

The default hex resolution is **5 km × 5 km** (≈ 21.65 km² per hex,
flat-top). At radius 80 that is ≈ 485,000 km² — about the size of Spain.
Feature sizes (range lengths, river basins, biome belts) are configured in
**physical units (km, km², mm/km)** rather than per-hex constants, so
``hex_size_km`` can be changed in the config and the generated world will
look the same at the new resolution. See the "Hex resolution" section below
for details.

## Pipeline

The world is built as a sequence of pure-functional layers. Each layer is a
function of all earlier layers' outputs plus its own seeded child-RNG. Layers
do not mutate prior state; new state objects are returned. This matches the
project's three commitments: simulatability, interpretability, testability.

The pre-existing axial hex-coordinate math (`Hex.distance`, `neighbors`,
`ring`, `spiral`) used throughout the pipeline follows Patel's hexagonal
grid reference ([Patel 2013](#ref-patel-hex)).

```
seed + config
  ↓
[L1] elevation        fBm + ridged multifractal + domain warp + radial falloff
  ↓
[L2] sea level        quantile threshold to hit target land_fraction; coast tag
  ↓
[L3] temperature      latitude band + elevation lapse (6.5 °C/km) + small noise
  ↓
[L4] precipitation    prevailing-wind moisture sweep with orographic uplift
  ↓
[L5] hydrology        priority-flood + ε-tilt → D6 flow accum → rivers, lakes
  ↓
[L6] biome            Whittaker(T, P) lookup with elevation/coast/water overrides
  ↓
[L7] resources        per-hex crop suitability (FAO-style envelopes)
                      + natural deposits (per-resource clustered noise)
  ↓
GeneratedWorld → HexGrid (engine-compatible)
```

Each layer:

### L1 — Elevation

Four components, summed:

1. **fBm Perlin noise** (6 octaves, lacunarity 2.0, persistence 0.5,
   base frequency ≈ 1/50 hexes). The dominant scale gives a feature size
   around 50 hexes (≈ 250 km), comparable to a real sub-continental basin.
   Perlin noise was introduced by Ken Perlin and refined in the 2002
   "improved noise" paper with the quintic fade curve used here.
   ([Perlin 1985](#ref-perlin85); [Perlin 2002](#ref-perlin02)). Parameter
   choices and the fBm recipe follow Amit Patel's *Making maps with noise
   functions* tutorial ([Patel 2015](#ref-patel-noise)).
2. **Domain warp** (Perlin field, strength ≈ 5 hexes). Bends the noise
   coordinates so coastlines and basins are sinuous, not symmetric blobs.
   The trick is canonically described in Iñigo Quílez's *Domain warping*
   article ([Quílez 2008](#ref-quilez-warp)).
3. **Ridged multifractal noise** (5 octaves, gated by base elevation).
   Each octave outputs `(1 - |noise|)²`, weighted by the magnitude of the
   previous octave so ridges accumulate detail along sharp crests. The
   formulation is Musgrave, Kolb & Mace's ridged multifractal
   ([Musgrave et al. 1989](#ref-musgrave89)), in the practical form
   popularised by *libnoise* ([Bevins, *libnoise* Tutorial 5](#ref-libnoise5)).
   The "gate higher octaves with lower ones" trick comes from the same
   tutorial. Ridges are applied only above a base-elevation quantile so
   they cluster in already-mountainous regions instead of smearing across
   plains.
4. **Radial falloff** toward the world edge so the world is a continent /
   island world rather than wrap-around noise. This is the standard island
   mask in Patel's mapgen2 and Azgaar's heightmap pipeline
   ([Patel 2010](#ref-patel-mapgen2); [Azgaar 2017a](#ref-azgaar-heightmap)).

Sea level is chosen by **quantile**: the threshold below which is ocean is
set so that exactly `land_fraction` of hexes end up as land. This guarantees
stable land area independent of noise variance — same approach as mapgen2
and Azgaar's generator.

### L2 — Sea / coast

Ocean if elevation below the sea-level threshold. Coast = land with at least
one ocean neighbor. Trivial but isolating this as a layer means later layers
can ask "is this hex ocean?" without recomputing.

### L3 — Temperature

```
T = T_eq + (T_pole - T_eq) * |latitude|^1.3 - elev_km * lapse_rate + small_noise
```

`latitude = r / radius` in `[-1, 1]`. The latitude exponent 1.3 widens the
equatorial warm band slightly relative to a cosine. Elevation contribution
is in km (normalized elevation × `max_elevation_km`, default 4.5 km).

The 6.5 °C/km lapse rate is the standard tropospheric environmental lapse
rate from atmospheric science ([NOAA NWS Glossary](#ref-nws-lapse)). The
elevation-plus-latitude model is the simplification used in essentially all
procedural climate pipelines for games and simulations, including Azgaar's
*Climate, temperature and humidity* approach ([Azgaar 2017b](#ref-azgaar-climate))
and Dwarf Fortress's world generator ([Adams 2015](#ref-adams-df)).

### L4 — Precipitation

The most physically-grounded layer and the one that does most of the work to
make the biome map look right.

Each hex's annual mean precipitation is computed by **integrating along the
prevailing wind path** ending at that hex:

1. Determine prevailing wind direction from latitude — trade easterlies near
   the equator, westerlies in the temperate band, polar easterlies near the
   poles. The three-cell model dates back to Hadley's 1735 hypothesis as
   extended by Ferrel in 1856; modern presentations are in any introductory
   atmospheric-science text ([Hadley 1735](#ref-hadley35);
   [Lutgens & Tarbuck 2018](#ref-lutgens18)).
2. Walk along the wind vector from far upwind toward the target hex
   (max steps ≈ world diameter). At each step:
   - **Ocean step:** add `pickup × warmth(temperature)` to a moisture
     "tank" carried by the parcel.
   - **Land step:** deposit a fraction (`precip_loss_per_land`) of the
     current moisture plus an **orographic bonus** proportional to the
     positive elevation gradient relative to the previous step. The
     orographic term is what produces wet windward slopes and dry leeward
     interiors (rain shadow). The linear-theory justification for
     `precip ∝ ∂h/∂x` over windward terrain is in Smith's seminal
     orographic-precipitation paper ([Smith 1979](#ref-smith79)) and
     summarised in Roe's review ([Roe 2005](#ref-roe05)).
3. The precipitation at the target hex = the deposit on its own step + a
   small baseline + low-amplitude fBm noise.

This per-hex moisture-budget approach is essentially the simplified
procedural model used by Azgaar's *Biomes* article
([Azgaar 2017c](#ref-azgaar-biomes)) and Dwarf Fortress's "orographic
precipitation" mode ([Adams 2015](#ref-adams-df); [DF Wiki —
Advanced world gen](#ref-df-wiki)).

Tuning at 5 km/hex: the moisture tank fills over ~10 ocean steps (≈ 50 km
of upwind ocean) and is consumed by ~10–20 land steps. Coastal windward
hexes therefore receive 1000–2200 mm/yr; interiors with no upwind ocean
or downwind of a major range fall below 250 mm/yr → desert.

### L5 — Hydrology

Sink-fill via **priority-flood with epsilon tilt**
([Barnes, Lehman & Mulla 2014](#ref-barnes14)). All ocean hexes are seeded
as outlets; the algorithm pulls hexes off a min-heap keyed by
`(elevation, q, r)` (deterministic tiebreakers) and assigns each land hex
a fill elevation of `max(natural_elev, parent_fill + ε)`. The ε-tilt
guarantees a strict downhill chain back to the ocean — without it,
priority-flood leaves perfectly flat plateaus and lake interiors where flow
direction is undefined and rivers stall. The ε-improvement is variant
"Improved Priority-Flood" in the same paper (Barnes 2014, §3.3).

Then:
- **D6 flow direction:** the steepest of six neighbors of the filled DEM.
  D-N flow routing on a regular grid is the classic O'Callaghan & Mark
  formulation ([O'Callaghan & Mark 1984](#ref-ocm84)). On a hex grid the
  six-neighbor variant avoids the well-known cardinal/diagonal orientation
  bias of D8 on square grids — see Shelef & Hilley's analysis of grid
  artifacts in flow routing ([Shelef & Hilley 2013](#ref-shelef13)) and the
  comparative study by Schwanghart & Heckmann
  ([Schwanghart & Heckmann 2012](#ref-schwanghart12)).
- **Flow accumulation:** topological sort by filled elevation descending;
  push each land hex's 1-unit contribution downstream. The single-pass
  topo-sort approach to accumulation is Mark's
  ([Mark 1988](#ref-mark88)).
- **Rivers:** hexes with accumulated drainage above
  `river_drainage_threshold` become river hexes. The drainage-area
  threshold for channel initiation traces back to
  [Montgomery & Dietrich 1988](#ref-montgomery88) and is the standard rule
  in DEM-based river extraction. Default threshold ≈ 60 upstream hexes
  (≈ 1300 km² of catchment) which corresponds roughly to a stream visible
  on a continental map.
- **Lakes:** hexes whose fill elevation exceeds natural elevation by at
  least `lake_min_depth`. Small puddles below this threshold are ignored.

### L6 — Biome

The classification table is from Robert Whittaker's biome diagram
([Whittaker 1975](#ref-whittaker75)), which plots terrestrial biomes on the
two axes of mean annual temperature and mean annual precipitation. Holdridge
life zones ([Holdridge 1947](#ref-holdridge47)) are an alternative that adds
potential evapotranspiration — overkill for this pipeline.

Whittaker (temperature × precipitation) lookup with overrides:

1. **Water:** ocean / deep_ocean (depth > 0.15) / coast / lake.
2. **Hydrology:** river hexes override their lowland biome.
3. **Elevation:** above the configured land-elevation quantile, hexes become
   hills → mountain → snow_peak (snow only when cold).
4. **Whittaker:** otherwise, look up temperature × precipitation in a small
   table to produce `tundra | taiga | plains | grassland | savanna |
   temperate_forest | jungle | desert`.

The Whittaker thresholds are tuned so:
- < −2 °C → tundra (regardless of moisture)
- cold (−2…8 °C): dry → tundra, wet → taiga
- temperate (8…22 °C): dry → desert, moderate → plains, wet → temperate_forest
- warm (> 22 °C): dry → desert, moderate → savanna, wet → jungle

This is the same coarse lookup Azgaar uses in his biome step
([Azgaar 2017c](#ref-azgaar-biomes)), discretised to a small table rather
than a continuous fit.

### L7 — Resources

Two parallel sub-systems run after biome assignment, both driven by their
own seeded child-RNGs derived from `("worldgen", "resources", <name>)` so
adding a new crop or resource does not reshuffle existing ones (protecting
snapshot tests).

#### Crop suitability (FAO Ecocrop-style envelopes)

For each crop in `[worldgen.crops.*]` and each land hex, suitability is
computed as

```
suitability = trapezoid(T) · trapezoid(P_eff) · biome_compat(biome)
              · (1 + river_bonus | river_adjacent_bonus | coast_bonus)
```

where each trapezoid is the four-point membership function
`(abs_min, opt_min, opt_max, abs_max)` standard in FAO Ecocrop
([FAO Ecocrop](#ref-fao-ecocrop)). `P_eff` adds an "irrigation equivalent"
to the local rainfall when the hex is itself a river or has a water
neighbor, which is how paddy rice can grow in otherwise-dry tropics. Any
factor zero produces zero suitability; the score is clamped to `[0, 1]`.

Eight crops ship in the default config: **wheat, rice, maize, barley,
millet, cotton, olive, potato**. Their envelopes follow FAO crop water
profiles and the Global Agro-Ecological Zones (GAEZ) model
([FAO 2018 — Wheat](#ref-fao-wheat); [FAO 2018 — Maize](#ref-fao-maize);
[FAO 2018 — Cotton](#ref-fao-cotton); [FAO 2018 — Olive](#ref-fao-olive);
[FAO 2018 — Potato](#ref-fao-potato); [GAEZ v4](#ref-gaez4)). New crops
are added by writing a `[worldgen.crops.<name>]` table — no code changes
needed.

#### Natural resource deposits

Each resource has:
- A **host biome set** (e.g. iron in `mountain, hills, plains`; salt in
  `desert, coast, plains`).
- **Elevation / temperature / precipitation** gates so deposits that
  geologically require specific conditions don't appear outside them
  (salt only below 600 mm/yr; coal only below 0.4 normalised elevation).
- A **feature wavelength in km** so deposits form *districts* of the right
  physical size — copper porphyries cluster in ~150 km belts, salt pans
  in ~50 km basins, coal in ~250 km basins.
- An **abundance** parameter — the top fraction of eligible hexes that
  host a deposit, calibrated against the noise field.
- A **mean quantity** plus optional elevation bonus, controlling per-cell
  yield.

Distribution algorithm per resource:

1. Build the eligibility mask (biome + elev + climate).
2. Sample a 3-octave fBm Perlin field (per-resource child-RNG, frequency
   `hex_size_km / feature_wavelength_km` so deposit-district size scales
   with `hex_size_km`).
3. Threshold the field at the quantile that retains the configured
   `abundance` fraction of eligible hexes.
4. Where the field exceeds the threshold, emit a deposit with quantity
   `mean_quantity × (0.5 + 0.5 · noise_strength) × (1 + elev_bonus · elevation)`.

Ten resources ship in the default config, in five categories:

| Category    | Resources              | Geological / ecological setting |
|---|---|---|
| ore         | iron, copper, tin, gold | Mountains and hills (orogenic + porphyry); gold also placer in rivers. ([Wikipedia — banded iron formation](#ref-bif); [USGS — porphyry copper](#ref-usgs-cu); [Lehmann 2021](#ref-tin); [Wikipedia — orogenic gold](#ref-gold)) |
| fuel        | coal                   | Sedimentary basins in low-elevation plains / forest belts. ([Britannica — origin of coal](#ref-coal)) |
| evaporite   | salt                   | Arid plains, deserts, coastal sabkhas — climate-gated `P < 600 mm`. ([Wikipedia — evaporite](#ref-evaporite); [Wikipedia — sabkha](#ref-sabkha)) |
| building    | stone                  | Mountains and hills (granite, marble, limestone). |
| sedimentary | clay                   | River floodplains, lakes, low-elevation plains. |
| timber      | softwood, hardwood     | Taiga (softwood) vs temperate / jungle (hardwood); temperature-gated. |

Deposit quantity, host biomes, elevation/climate bounds, feature
wavelength, and abundance are all per-resource config knobs in
`[worldgen.resources.<name>]`. Adding a new resource is a config-only
change.

### Visual examples — crops

Wheat (temperate cool, moderate rain) clusters in mid-latitude plains and
grassland with rain shadow excluded; rice (warm, wet, river/irrigation)
follows river valleys in the warm zones, including arid ones that get the
irrigation bonus:

| Wheat suitability | Rice suitability |
|---|---|
| ![](crops/seed42_r80_crop_wheat.png) | ![](crops/seed42_r80_crop_rice.png) |

| Barley (cool tolerant) | Cotton (warm, lowland) |
|---|---|
| ![](crops/seed42_r80_crop_barley.png) | ![](crops/seed42_r80_crop_cotton.png) |

| Maize | Millet (hot, drought-tolerant) |
|---|---|
| ![](crops/seed42_r80_crop_maize.png) | ![](crops/seed42_r80_crop_millet.png) |

| Olive (Mediterranean) | Potato (cool, highland-tolerant) |
|---|---|
| ![](crops/seed42_r80_crop_olive.png) | ![](crops/seed42_r80_crop_potato.png) |

### Visual examples — resources

Iron in the central orogenic belt and northern uplands; salt in the dry
rain-shadow interior and coastal arid strips:

| Iron districts (ore) | Salt pans (evaporite) |
|---|---|
| ![](resources/seed42_r80_resource_iron.png) | ![](resources/seed42_r80_resource_salt.png) |

| Coal (fuel) | Copper (ore) |
|---|---|
| ![](resources/seed42_r80_resource_coal.png) | ![](resources/seed42_r80_resource_copper.png) |

| Hardwood timber | Softwood timber |
|---|---|
| ![](resources/seed42_r80_resource_timber_hardwood.png) | ![](resources/seed42_r80_resource_timber_softwood.png) |

All resource deposits at once, color-coded by category:

![all resources](resources/all_resources.png)

## Determinism

All randomness flows through `RngHierarchy.child(*keys)`. Each layer derives
its own child RNG by hashing its name; within a layer, sub-RNGs hash in
the noise field name (`"base"`, `"ridge"`, `"warp_x"`, etc.). The Perlin
permutation table is generated from the child RNG and is the only seeded
state inside the noise sampler. Result: byte-identical output for
`(seed, config, radius)` across runs — covered by
`tests/worldgen/test_pipeline.py::test_pipeline_deterministic`.

## Configuration

All coefficients live in `config/default.toml` under `[worldgen.*]`. The
biome thresholds, precipitation pickup/loss constants, ridge amplitude,
falloff strength, and quantile sea level are all tunable without code
changes.

## Sample worlds

Four random seeds, all generated with the production config (radius 80,
≈ 19,400 hexes, ≈ 485,000 km²).

### Seed 42 — base example

Biome map (left) and composite with hillshade + river overlay (right):

| ![seed42 biome](seed42_biome.png) | ![seed42 composite](seed42_composite.png) |
|---|---|

Climate diagnostics for the same world:

| Elevation | Temperature | Precipitation |
|---|---|---|
| ![](seed42_elevation.png) | ![](seed42_temperature.png) | ![](seed42_precipitation.png) |

Notable features visible in seed 42:
- A central east–west **mountain spine** with snow peaks
- A **rain-shadow band** of savanna / desert immediately downwind (east) of
  the spine
- Dendritic **river networks** draining from the spine to both coasts
- A **tundra plateau** in the north-east
- A **monsoon-style wet east coast** in the southern tropical latitudes

### Seed 7 — single-continent variant

| ![seed7 biome](seed7_biome.png) | ![seed7 composite](seed7_composite.png) |
|---|---|

A more compact landmass with a dramatic snow-capped peninsula in the south
and broad inland taiga.

### Seed 1337 — fragmented archipelago

| ![seed1337 biome](seed1337_biome.png) | ![seed1337 composite](seed1337_composite.png) |
|---|---|

Multiple smaller landmasses with inland seas and several alpine ranges.

### Seed 2026 — current calendar year

| ![seed2026 biome](seed2026_biome.png) | ![seed2026 composite](seed2026_composite.png) |
|---|---|

## What works

- Continents look right at this scale (200–500 km long, sinuous coasts).
- Mountain ranges are *linear* (not random peaks), thanks to ridged
  multifractal + gating.
- Snow caps only on the highest peaks of cold zones.
- Rivers form dendritic networks reaching the ocean (verified by an
  invariant test: every river hex's flow chain terminates at ocean).
- Rain shadow is visible whenever a major range crosses a wind direction.
- Latitudinal climate banding is clear: polar tundra → subarctic taiga →
  temperate plains → tropical savanna/jungle.
- Determinism is enforced by tests.

## Known limitations

- **Plate tectonics** is not modelled. The research survey identified
  tectonic-plate boundary uplift as the most realistic way to get
  geologically-plausible range *arcs* (subduction zones produce curved
  ranges). I left this for a later phase. Current ranges follow ridged-noise
  ridgelines, which look plausible at first glance but lack the
  characteristic curvature of, say, the Andes or the Aleutian arc.
- **Erosion** is also not modelled. A hydraulic erosion pass would smooth
  windward slopes and sharpen valleys. Current mountains have somewhat
  blocky outlines because they are pure noise rather than noise-then-erode.
- **River widening** at confluences and **delta formation** at river mouths
  are not modelled — a single-hex river hex looks the same whether it is a
  headwater stream or the main stem of a major basin. Width information
  *is* available in the flow accumulation field, used by the preview
  renderer to scale line widths.
- **Wind direction** is a simple latitude-banded model (Earth's three-cell
  pattern). It does not respond to continental heating gradients or
  seasonal shifts. For an annual mean this is adequate, but a model with
  seasonal winds (monsoon physics) would produce more nuanced precipitation
  patterns.
- **Polar overrides:** very high-latitude hexes are simply tundra. There is
  no permafrost, sea ice, or glacier model.
- **Soil chemistry** is not modelled. Crop suitability is driven by
  temperature, precipitation, and biome only — no pH, no nitrogen, no clay
  vs sand differentiation. This is good enough at the resolution and
  abstraction level of the simulation, but real agronomy depends heavily on
  soil chemistry that biome categories cannot capture.
- **Placer gold derivative from lode gold** is not modelled — gold has a
  `host_biomes` list that includes `river`, but the spatial noise field for
  river placers is independent of the upstream lode noise field. A proper
  model would propagate gold deposits downstream from mountain sources.
- **Salt domes** beneath plains (former evaporite basins now buried) are
  not modelled — the salt eligibility mask uses *current* climate (P < 600
  mm), not paleoclimate.

## Hex resolution

The pipeline is resolution-independent. The config exposes
``worldgen.hex_size_km`` (default 5.0 km), and the scale-dependent generator
parameters are stored in physical units:

| Physical-unit config field | Derived per-hex value | Default value |
|---|---|---|
| ``feature_wavelength_km`` | ``noise_base_frequency = hex_size / wavelength`` | 250 km |
| ``warp_wavelength_km`` | ``warp_frequency`` | 333 km |
| ``warp_strength_km`` | ``warp_strength`` in hex coords | 25 km |
| ``wind_reach_km`` | ``wind_reach_hexes`` | 1000 km |
| ``precip_pickup_per_ocean_km`` | per-hex pickup (mm) | 120 mm/km |
| ``precip_loss_per_km`` | per-hex loss = ``1 − exp(−r · hex_size)`` | 0.066 / km |
| ``river_drainage_threshold_km2`` | upstream-hex count | 1300 km² |
| ``hex_area_km2`` (computed) | hex area = ``(√3/2)·hex_size_km²`` | 21.65 km² |

Changing ``hex_size_km`` automatically rescales noise frequency, wind reach,
river thresholds, and precipitation rates. The same world (same seed, same
physical parameters) rendered at different hex sizes should produce
recognizably the *same continent shape* — only the cell granularity differs.

### Visual demonstration

The same physical world (seed 42, 400 km radius) rendered at three hex sizes:

| 2.5 km/hex (radius 160, 77 k hexes) | 5 km/hex (radius 80, 19 k hexes) | 10 km/hex (radius 40, 5 k hexes) |
|---|---|---|
| ![](scale_2_5km.png) | ![](scale_5km.png) | ![](scale_10km.png) |

All three share the same central east–west mountain spine, the same large
peninsulas, and the same snow-cap positions. The 2.5 km version resolves
finer coastline detail and more individual rivers; the 10 km version
preserves the overall geography with chunkier biome regions.

## Engine integration note

This change replaces the IID terrain sampler and introduces ten new terrain
types (`deep_ocean`, `ocean`, `coast`, `lake`, `river`, `plains`, `grassland`,
`savanna`, `desert`, `tundra`, `temperate_forest`, `taiga`, `jungle`, `hills`,
`mountain`, `snow_peak`). The simulation engine reads terrain by name and
all per-terrain coefficients (yields, movement_cost, expansion costs) are
defined for all 16 names in `default.toml`.

The engine no longer hard-codes ``Hex(0, 0)`` as the starting position.
``sim/engine/runner.py::find_initial_settlement_area`` ranks all land hexes
by the *geometric mean* of subsistence-good yields in their seven-hex
neighborhood (hex + 6 immediate neighbors) and chooses the highest-scoring
hex closest to the world center. The settlement then claims that hex plus,
if the start hex does not produce all subsistence goods itself, one
adjacent hex that supplies each missing good (typically a coast or river
neighbor for fish).

This makes the simulation robust to whatever terrain the generator
produces at the center, while remaining fully deterministic — the chosen
hex is a pure function of ``(seed, config)``. All 68 engine integration
tests and all 35 terrain tests pass together.

## References

### Noise and procedural elevation

<a id="ref-perlin85"></a>
**Perlin, K. (1985).** "An image synthesizer." *SIGGRAPH Computer Graphics*,
19(3): 287–296. — The original gradient-noise paper.
[doi:10.1145/325165.325247](https://doi.org/10.1145/325165.325247)

<a id="ref-perlin02"></a>
**Perlin, K. (2002).** "Improving noise." *ACM Transactions on Graphics*,
21(3): 681–682. — The improved-Perlin paper introducing the quintic fade
curve `6t⁵ − 15t⁴ + 10t³` used in `sim/world/noise.py`.
[doi:10.1145/566654.566636](https://doi.org/10.1145/566654.566636)

<a id="ref-musgrave89"></a>
**Musgrave, F. K., Kolb, C. E., & Mace, R. S. (1989).** "The synthesis and
rendering of eroded fractal terrains." *SIGGRAPH '89*, 41–50. — Origin of
ridged-multifractal noise.
[doi:10.1145/74333.74337](https://doi.org/10.1145/74333.74337)

<a id="ref-quilez-warp"></a>
**Quílez, I. (2008).** *Domain warping.* — Canonical writeup of
`noise(p + noise(p))` for distorting feature shapes.
[iquilezles.org/articles/warp](https://iquilezles.org/articles/warp/)

<a id="ref-libnoise5"></a>
**Bevins, J. (2007).** *libnoise — Tutorial 5: Creating mountainous terrain.*
— The practical "gate higher octaves by lower ones" ridged-multifractal
recipe used in `ridged_fbm`.
[libnoise.sourceforge.net/tutorials/tutorial5.html](https://libnoise.sourceforge.net/tutorials/tutorial5.html)

<a id="ref-patel-noise"></a>
**Patel, A. (2015).** *Making maps with noise functions.* — Parameter
recipes for fBm + falloff in a procedural-map context.
[redblobgames.com/maps/terrain-from-noise](https://www.redblobgames.com/maps/terrain-from-noise/)

<a id="ref-patel-mapgen2"></a>
**Patel, A. (2010).** *Polygonal map generation for games (mapgen2).* —
Voronoi continents, downhill rivers, biome lookup; the layered approach
used here is patterned after this.
[redblobgames.com/maps/mapgen2](https://www.redblobgames.com/maps/mapgen2/)

<a id="ref-patel-mapgen4"></a>
**Patel, A. (2018).** *mapgen4.* — Paint-driven elevation followed by
simulated wind / rainfall / rivers. The "wind sweep then orographic" idea
in L4 mirrors this article's approach.
[redblobgames.com/maps/mapgen4](https://www.redblobgames.com/maps/mapgen4/)

<a id="ref-patel-hex"></a>
**Patel, A. (2013).** *Hexagonal grids.* — Reference for axial and cube
coordinates, neighbors, rings, spirals; pre-existing `sim/world/hex.py`
follows this directly.
[redblobgames.com/grids/hexagons](https://www.redblobgames.com/grids/hexagons/)

<a id="ref-azgaar-heightmap"></a>
**Azgaar (2017a).** *Fantasy Map Generator: heightmap.* — Quantile sea
level, blob-based island heightmap, the "edge-low / center-high" mask.
[azgaar.wordpress.com/2017/04/01/heightmap/](https://azgaar.wordpress.com/2017/04/01/heightmap/)

<a id="ref-azgaar-climate"></a>
**Azgaar (2017b).** *Climate (temperature and humidity).*
[azgaar.wordpress.com/2017/05/08/temperature-and-precipitation/](https://azgaar.wordpress.com/2017/05/08/temperature-and-precipitation/)

<a id="ref-azgaar-biomes"></a>
**Azgaar (2017c).** *Biomes generation and rendering.* — Whittaker-table
lookup over (T, P).
[azgaar.wordpress.com/2017/06/30/biomes-generation-and-rendering/](https://azgaar.wordpress.com/2017/06/30/biomes-generation-and-rendering/)

<a id="ref-mewo2"></a>
**O'Leary, M. (2016).** *Generating fantasy maps.* — Voronoi-mesh erosion
and river extraction; used as background reading though this pipeline is
hex-based rather than Voronoi-based.
[mewo2.com/notes/terrain](http://mewo2.com/notes/terrain/)

### Atmospheric science (temperature, wind, precipitation)

<a id="ref-hadley35"></a>
**Hadley, G. (1735).** "Concerning the cause of the general trade-winds."
*Philosophical Transactions of the Royal Society*, 39: 58–62. — Original
proposal of the equator-to-pole circulation cell.

<a id="ref-lutgens18"></a>
**Lutgens, F. K. & Tarbuck, E. J. (2018).** *The Atmosphere: An Introduction
to Meteorology* (14th ed.), Pearson. — Modern textbook reference for the
three-cell model, lapse rate, and orographic precipitation used in L3/L4.

<a id="ref-nws-lapse"></a>
**NOAA NWS (n.d.).** *Glossary — Environmental Lapse Rate.* — Source of
the 6.5 °C/km figure.
[forecast.weather.gov/glossary.php?word=environmental+lapse+rate](https://forecast.weather.gov/glossary.php?word=environmental+lapse+rate)

<a id="ref-smith79"></a>
**Smith, R. B. (1979).** "The influence of mountains on the atmosphere."
*Advances in Geophysics*, 21: 87–230. — Foundational paper on orographic
precipitation; justifies the `precip ∝ ∂h/∂x` term used in L4.
[doi:10.1016/S0065-2687(08)60262-9](https://doi.org/10.1016/S0065-2687(08)60262-9)

<a id="ref-roe05"></a>
**Roe, G. H. (2005).** "Orographic precipitation." *Annual Review of Earth
and Planetary Sciences*, 33: 645–671. — Modern review summarising the
linear theory; readable companion to Smith 1979.
[doi:10.1146/annurev.earth.33.092203.122541](https://doi.org/10.1146/annurev.earth.33.092203.122541)

### Hydrology / DEM analysis

<a id="ref-ocm84"></a>
**O'Callaghan, J. F. & Mark, D. M. (1984).** "The extraction of drainage
networks from digital elevation data." *Computer Vision, Graphics, and
Image Processing*, 28(3): 323–344. — The D8 single-flow-direction
algorithm; L5 uses its hex analogue, D6.
[doi:10.1016/S0734-189X(84)80011-0](https://doi.org/10.1016/S0734-189X(84)80011-0)

<a id="ref-mark88"></a>
**Mark, D. M. (1988).** "Network models in geomorphology." In
*Modelling Geomorphological Systems*, ed. M. G. Anderson, Wiley:
73–97. — The topological-sort flow-accumulation approach used in
`hydrology.py`.

<a id="ref-montgomery88"></a>
**Montgomery, D. R. & Dietrich, W. E. (1988).** "Where do channels begin?"
*Nature*, 336: 232–234. — Drainage-area threshold for channel initiation;
basis for the `river_drainage_threshold_km2` parameter.
[doi:10.1038/336232a0](https://doi.org/10.1038/336232a0)

<a id="ref-barnes14"></a>
**Barnes, R., Lehman, C. & Mulla, D. (2014).** "Priority-flood: An
optimal depression-filling and watershed-labeling algorithm for digital
elevation models." *Computers & Geosciences*, 62: 117–127. — The sink-fill
algorithm and the ε-tilt variant used in `_priority_flood`.
[doi:10.1016/j.cageo.2013.04.024](https://doi.org/10.1016/j.cageo.2013.04.024)
([arXiv:1511.04463](https://arxiv.org/abs/1511.04463))

<a id="ref-shelef13"></a>
**Shelef, E. & Hilley, G. E. (2013).** "Impact of flow routing on
catchment area calculations, slope estimates, and stream channel
locations." *Computers & Geosciences*, 53: 1–9. — Quantifies the
orientation bias of D8 on square grids that hex grids avoid.
[doi:10.1016/j.cageo.2012.04.030](https://doi.org/10.1016/j.cageo.2012.04.030)

<a id="ref-schwanghart12"></a>
**Schwanghart, W. & Heckmann, T. (2012).** "Fuzzy delineation of
drainage basins through probabilistic interpretation of diverging flow
algorithms." *Environmental Modelling & Software*, 33: 106–113. —
Comparative study of D8/D-infinity flow routing, including grid
sensitivity.
[doi:10.1016/j.envsoft.2012.01.016](https://doi.org/10.1016/j.envsoft.2012.01.016)

### Agronomy / crop envelopes

<a id="ref-fao-ecocrop"></a>
**FAO Ecocrop (n.d.).** *Crop Environmental Requirements database.* —
Source of the four-point `(T_abs_min, T_opt_min, T_opt_max, T_abs_max)`
trapezoidal envelope schema used by L7's `_trapezoid` function and the
crop-suitability formula.
[ecocrop.apps.fao.org/ecocrop/srv/en/home](https://ecocrop.apps.fao.org/ecocrop/srv/en/home)

<a id="ref-fao-wheat"></a>
**FAO Land & Water (2018).** *Crop Information — Wheat.*
[fao.org/land-water/databases-and-software/crop-information/wheat/en/](https://www.fao.org/land-water/databases-and-software/crop-information/wheat/en/)

<a id="ref-fao-maize"></a>
**FAO Land & Water (2018).** *Crop Information — Maize.*
[fao.org/land-water/databases-and-software/crop-information/maize/en/](https://www.fao.org/land-water/databases-and-software/crop-information/maize/en/)

<a id="ref-fao-cotton"></a>
**FAO Land & Water (2018).** *Crop Information — Cotton.*
[fao.org/land-water/databases-and-software/crop-information/cotton/en/](https://www.fao.org/land-water/databases-and-software/crop-information/cotton/en/)

<a id="ref-fao-olive"></a>
**FAO Land & Water (2018).** *Crop Information — Olive.*
[fao.org/land-water/databases-and-software/crop-information/olive/en/](https://www.fao.org/land-water/databases-and-software/crop-information/olive/en/)

<a id="ref-fao-potato"></a>
**FAO Land & Water (2018).** *Crop Information — Potato.*
[fao.org/land-water/databases-and-software/crop-information/potato/en/](https://www.fao.org/land-water/databases-and-software/crop-information/potato/en/)

<a id="ref-gaez4"></a>
**FAO & IIASA (2021).** *Global Agro-Ecological Zones v4 (GAEZ).* —
Crop-suitability methodology and global maps; basis for cross-checking
the envelope parameters.
[gaez.fao.org](https://gaez.fao.org/)

### Economic geology / resource deposits

<a id="ref-bif"></a>
**Wikipedia.** *Banded iron formation.* — Precambrian iron-source rocks;
basis for the `iron` deposit host set (mountain / hills / plains over old
shields).
[en.wikipedia.org/wiki/Banded_iron_formation](https://en.wikipedia.org/wiki/Banded_iron_formation)

<a id="ref-usgs-cu"></a>
**Singer, D. A., et al. (2008).** *Preliminary Quantitative Model of
Porphyry Copper Deposits.* USGS Open-File Report 2008–1321. — Source for
copper-deposit spatial pattern (~150 km belts along arcs).
[pubs.usgs.gov/of/2008/1321/pdf/OF081321_508.pdf](https://pubs.usgs.gov/of/2008/1321/pdf/OF081321_508.pdf)

<a id="ref-tin"></a>
**Lehmann, B. (2021).** "Formation of tin ore deposits: A reassessment."
*Lithos*, 402–403: 105756. — Tin-belt geography; basis for the very low
`abundance = 0.02` default.
[doi:10.1016/j.lithos.2020.105756](https://doi.org/10.1016/j.lithos.2020.105756)

<a id="ref-gold"></a>
**Wikipedia.** *Orogenic gold deposit.* — Gold occurs in mountain
metamorphic belts and as placers downstream; basis for the
`host_biomes = ["mountain", "hills", "river"]` setting.
[en.wikipedia.org/wiki/Orogenic_gold_deposit](https://en.wikipedia.org/wiki/Orogenic_gold_deposit)

<a id="ref-coal"></a>
**Britannica.** *Coal — Origin of coal.* — Sedimentary basins from
ancient peatland; basis for the `host_biomes = ["plains", "hills",
"temperate_forest", "taiga"]` plus low-elevation cap.
[britannica.com/science/coal-fossil-fuel/Origin-of-coal](https://www.britannica.com/science/coal-fossil-fuel/Origin-of-coal)

<a id="ref-evaporite"></a>
**Wikipedia.** *Evaporite.* — Geology of salt formation in arid basins.
[en.wikipedia.org/wiki/Evaporite](https://en.wikipedia.org/wiki/Evaporite)

<a id="ref-sabkha"></a>
**Wikipedia.** *Sabkha.* — Coastal evaporite environment; justifies
including `coast` in the salt host biomes.
[en.wikipedia.org/wiki/Sabkha](https://en.wikipedia.org/wiki/Sabkha)

### Biome classification

<a id="ref-whittaker75"></a>
**Whittaker, R. H. (1975).** *Communities and Ecosystems* (2nd ed.),
Macmillan. — Origin of the temperature × precipitation biome diagram
used in L6.

<a id="ref-holdridge47"></a>
**Holdridge, L. R. (1947).** "Determination of world plant formations
from simple climatic data." *Science*, 105(2727): 367–368. — Alternative
biome classification using potential evapotranspiration; mentioned as a
fancier option not used here.
[doi:10.1126/science.105.2727.367](https://doi.org/10.1126/science.105.2727.367)

### Game-development references

<a id="ref-adams-df"></a>
**Adams, T. (2015).** "Simulation principles from Dwarf Fortress." In
*Game AI Pro 2: Collected Wisdom of Game AI Professionals*, ed. S.
Rabin, CRC Press, ch. 41. — Describes DF's elevation / rainfall /
temperature / drainage / vegetation channel approach.
[gameaipro.com — chapter 41 PDF](http://www.gameaipro.com/GameAIPro2/GameAIPro2_Chapter41_Simulation_Principles_from_Dwarf_Fortress.pdf)

<a id="ref-df-wiki"></a>
**Dwarf Fortress wiki (n.d.).** *Advanced world generation.* — Practical
detail on the orographic-precipitation toggle and per-cell channel
mechanics that informed L4's design.
[dwarffortresswiki.org/index.php/Advanced_world_generation](https://dwarffortresswiki.org/index.php/Advanced_world_generation)
