"""Public export endpoints: serialize a generated world and bundle a snapshot
plus per-layer rendered PNGs into a timestamped folder.

The serialized format is intentionally a *generic container*: every field on
``WorldSnapshot`` is an open-ended ``dict[str, Any]`` or ``list[dict[str, Any]]``
keyed by strings, with no fixed schema. Adding new per-hex fields, new
intermediate layers, or new metadata never requires changing ``WorldSnapshot``
itself — the new data slots in as another key.

Two public endpoints:

  - ``serialize_world(world) -> WorldSnapshot``
        Pure projection from ``GeneratedWorld`` into the generic container.
        No side effects, no timestamps, no I/O.

  - ``export_world(config, seed, output_root, stop_after=None) -> Path``
        Full bundle: generates the world (up to ``stop_after`` if set),
        serializes, writes JSON to disk, and renders one PNG per generation
        layer whose source data is available. World dimensions come from
        ``config.world``.

``save_snapshot`` / ``load_snapshot`` are the file I/O helpers; ``WorldSnapshot``
itself is format-agnostic (``to_dict`` / ``from_dict``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from worldgen._log import configure_logging
from worldgen.config_loader import load_worldgen_config
from worldgen.hex import Hex
from worldgen.pipeline import GeneratedWorld, PIPELINE_STEPS, generate
from worldgen.types import WorldgenConfig


@dataclass(frozen=True)
class WorldSnapshot:
    """Generic, serialization-friendly container for a generated world.

    Three open-ended bags:

      - ``metadata``: run parameters (world_width_km, world_height_km,
        hex_size_km, plus any extras layered on by the caller, e.g. seed
        and timestamp).
      - ``hexes``: list of per-hex records. Each record is a flat ``dict`` with
        ``q``, ``r``, and every ``HexData`` field as primitive JSON values.
      - ``layers``: layer-name → layer-level (non-per-hex) data dict.

    All three are ``dict[str, Any]`` / ``list[dict[str, Any]]`` so future
    additions (new per-hex fields, new intermediate layers, new metadata) do
    not require schema changes here.
    """

    metadata: dict[str, Any]
    hexes: list[dict[str, Any]]
    layers: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "hexes": self.hexes,
            "layers": self.layers,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorldSnapshot:
        return cls(
            metadata=d["metadata"],
            hexes=d["hexes"],
            layers=d["layers"],
        )


def _to_primitive(v: Any) -> Any:
    """Recursively convert a value into JSON-friendly primitives.

    ``Hex`` becomes ``{"q": q, "r": r}``. Dataclass instances are flattened
    field-by-field. Tuples become lists. Dict keys become strings.
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Hex):
        return {"q": v.q, "r": v.r}
    if is_dataclass(v) and not isinstance(v, type):
        return {f.name: _to_primitive(getattr(v, f.name)) for f in fields(v)}
    if isinstance(v, (list, tuple)):
        return [_to_primitive(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_primitive(val) for k, val in v.items()}
    return v  # leave as-is; json.dumps will raise if not encodable


def serialize_world(world: GeneratedWorld) -> WorldSnapshot:
    """Pure projection of a ``GeneratedWorld`` into a ``WorldSnapshot``.

    The result is deterministic: identical worlds serialize to identical
    snapshots. Runtime context like seed or timestamp is *not* set here —
    the caller (e.g. ``export_world``) layers that into ``metadata``.

    For partial pipelines (``world.stop_after`` < ``"biome"``), the per-hex
    record list is empty and only those ``layers`` whose source state is
    populated are emitted.
    """
    hex_records: list[dict[str, Any]] = []
    if world.hexes is not None:
        for h, data in world.hexes.items():
            rec: dict[str, Any] = {"q": h.q, "r": h.r}
            for f in fields(data):
                rec[f.name] = _to_primitive(getattr(data, f.name))
            hex_records.append(rec)

    layers: dict[str, dict[str, Any]] = {}
    if world.elevation is not None:
        layers["elevation"] = {"sea_level": world.elevation.sea_level}
    if world.plates is not None:
        layers["plates"] = {
            "count": len(world.plates.plates),
            "plates": [_to_primitive(p) for p in world.plates.plates],
        }

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "world_width_km": world.config.world.width_km,
        "world_height_km": world.config.world.height_km,
        "hex_count": len(world.hexes) if world.hexes is not None else 0,
        "hex_size_km": world.config.hex_size_km,
        "stop_after": world.stop_after,
        "tectonics": {
            "n_ticks": world.config.tectonics.n_ticks,
            "dt_myr": world.config.tectonics.dt_myr,
            "sea_level_km": world.config.tectonics.sea_level_km,
        },
    }

    return WorldSnapshot(metadata=metadata, hexes=hex_records, layers=layers)


def save_snapshot(snap: WorldSnapshot, path: Path) -> None:
    """Write a snapshot to ``path`` as JSON. Creates parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snap.to_dict(), indent=2, sort_keys=False),
        encoding="utf-8",
    )


def load_snapshot(path: Path) -> WorldSnapshot:
    """Load a snapshot from a JSON file written by ``save_snapshot``."""
    return WorldSnapshot.from_dict(json.loads(path.read_text(encoding="utf-8")))


# Standard renderable layers (in order). ``plates`` is appended automatically
# when ``world.plates is not None`` so non-plates worlds don't get a
# placeholder image.
DEFAULT_RENDER_LAYERS: tuple[str, ...] = (
    "elevation",
    "temperature",
    "precipitation",
    "flow",
    "biome",
    "composite",
    "currents",
    "wind",
    "continentality",
    "gyres",
    "ocean_depth",
    "plates_t0",
)


def _layers_available_at(stop_after: str, candidates: tuple[str, ...]) -> list[str]:
    """Filter render-layer names to those whose required pipeline step has run.

    ``LAYER_REQUIRES`` (in ``preview``) maps each layer to its minimum
    required step. A layer is kept when the index of its required step in
    ``PIPELINE_STEPS`` is ≤ the index of ``stop_after``.
    """
    from worldgen.preview import LAYER_REQUIRES  # lazy: avoids Pillow import here.

    stop_ix = PIPELINE_STEPS.index(stop_after)
    out: list[str] = []
    for layer in candidates:
        required = LAYER_REQUIRES.get(layer)
        if required is None or PIPELINE_STEPS.index(required) <= stop_ix:
            out.append(layer)
    return out


def _dump_run_config(
    out_dir: Path,
    *,
    config: WorldgenConfig,
    seed: int,
    cli_args: dict | None,
    timestamp: str,
    config_path: Path | None,
) -> None:
    """Write ``config.json`` (full reproducibility snapshot) + verbatim
    copy of the source TOML into ``out_dir``. Ported from the rigid-
    polygon prototype so worldgen exports gain the same "reproduce-from-
    output-dir-alone" guarantee.
    """
    import dataclasses
    import json
    import shutil

    def _to_jsonable(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {
                f.name: _to_jsonable(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
            }
        if isinstance(obj, (tuple, list)):
            return [_to_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, bool) or obj is None:
            return obj
        if isinstance(obj, (int, float, str)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        return repr(obj)

    payload = {
        "cli_args": _to_jsonable(cli_args or {}),
        "resolved": {
            "seed": seed,
            "timestamp": timestamp,
            "world_km": {
                "width": config.world.width_km,
                "height": config.world.height_km,
            },
        },
        "worldgen_config": _to_jsonable(config),
        "config_source_path": str(config_path) if config_path else None,
    }
    (out_dir / "config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
    )
    if config_path is not None:
        try:
            shutil.copy2(config_path, out_dir / "worldgen.source.toml")
        except OSError as exc:
            # Don't fail the export just because the source TOML
            # couldn't be copied (e.g. read-only filesystem on Windows).
            print(f"  (warning) could not copy source TOML: {exc}")


def _export_tectonic_sim_views(
    out_dir: Path,
    snapshot: dict,
) -> None:
    """Render the polygon-sim visualisations into ``out_dir``:

      - partition.png    (cell-mask + edges + plate-ID labels + hotspot markers)
      - crust.png        (continental tan / oceanic blue + age + thickness)
      - thickness.png    (km heatmap, navy→cyan→sandy→gold→red)
      - topography.png   (elevation colormap)
      - polygons.png     (alpha-complex outlines over full sim)
      - drift.gif        (partition + crust dual-panel animation)
      - thickness.gif    (thickness + crust dual-panel animation)
      - topography.gif   (topography + crust dual-panel animation)
      - hotspots.json    (positions + lifecycle for traceability)
    """
    import json
    out_dir.mkdir(parents=True, exist_ok=True)

    if snapshot.get("kind") != "polygon_sim":
        raise ValueError(
            f"unrecognised raw_snapshot kind {snapshot.get('kind')!r}; "
            "expected 'polygon_sim'."
        )

    _export_polygon_views(out_dir, snapshot)
    # Hotspot metadata (positions + lifecycle) as JSON.
    (out_dir / "hotspots.json").write_text(
        json.dumps(
            [
                {
                    "index": i,
                    "position_xy_km": list(h.position_xy_km),
                    "birth_tick": h.birth_tick,
                    "lifespan_ticks": h.lifespan_ticks,
                    "death_tick": h.birth_tick + h.lifespan_ticks,
                    "active_at_final_tick": h.is_active(
                        snapshot["sim_config"].n_ticks
                    ),
                }
                for i, h in enumerate(snapshot["hotspots"])
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def _export_polygon_views(out_dir: Path, snap: dict) -> None:
    """Render the polygon-sim's full visualization set. Uses
    tectonic_sim.polygon_sim renderers so output matches the sim's own
    perspective.

    All views show the FULL sim domain (the larger torus on which the
    simulation actually ran), including the buffer region outside the
    worldgen world rectangle. This makes plate drift, hotspots, and
    suture formation visible everywhere — not just inside the world
    crop. Worldgen's own per-hex layers (``layers/*.png``) remain
    world-sized because they're rendered from the hex grid directly.
    """
    from tectonic_sim.polygon_sim import (
        build_partition_image,
        build_crust_image,
        build_thickness_image,
        build_topography_image,
        overlay_hotspots,
        render_polygons_png,
        save_drift_gif,
    )

    plates = snap["plates"]
    owner = snap["owner"]
    crust = snap["crust"]
    age = snap["age"]
    thick = snap["thickness"]
    cell_km = snap["cell_km"]
    sim_domain = snap["sim_domain"]
    world_domain = snap["world_domain"]
    sim_cfg = snap["sim_config"]
    hotspots = snap["hotspots"]
    frames = snap["frames"]
    frames_thickness = snap["frames_thickness"]
    frames_topography = snap["frames_topography"]
    final_tick = sim_cfg.n_ticks

    gy, gx = owner.shape
    cap = (
        f"WORLDGEN POLYGON  sim {gx}×{gy}  "
        f"world {int(round(world_domain.width_km))}×"
        f"{int(round(world_domain.height_km))} km  "
        f"cell={cell_km:.1f}km  plates={sum(1 for p in plates if p.alive)}  "
        f"hotspots={sum(1 for h in hotspots if h.is_active(final_tick))}"
        f"/{len(hotspots)}"
    )
    upscale = 6

    def _save_with_hotspots(img, path):
        # Hotspots position in the sim's mantle frame — no crop offset
        # because the panels are the full sim grid.
        overlay_hotspots(
            img, hotspots, final_tick,
            cell_km=cell_km, gy=gy, gx=gx, upscale=upscale,
            x0_cells=0, y0_cells=0,
            only_active=False,
        )
        img.save(path)

    # 1. partition.png — plate ownership + edges + plate-ID labels.
    _save_with_hotspots(
        build_partition_image(owner, cap, upscale=upscale),
        out_dir / "partition.png",
    )
    # 2. crust.png — continental tan / oceanic blue.
    max_age = float(age.max() or 1)
    _save_with_hotspots(
        build_crust_image(
            owner, crust, age, thick, max_age, cap, upscale=upscale,
        ),
        out_dir / "crust.png",
    )
    # 3. thickness.png — heatmap.
    owned = owner != -1
    mean_thick = float(thick[owned].mean()) if owned.any() else 0.0
    thk_img = build_thickness_image(
        owner, thick,
        f"thickness  {gx}×{gy}  cell={cell_km:.1f}km  mean={mean_thick:.1f}km",
        upscale=upscale,
    )
    overlay_hotspots(
        thk_img, hotspots, final_tick,
        cell_km=cell_km, gy=gy, gx=gx, upscale=upscale,
        x0_cells=0, y0_cells=0, only_active=False,
    )
    thk_img.save(out_dir / "thickness.png")
    # 4. topography.png — elevation colormap.
    _save_with_hotspots(
        build_topography_image(
            owner, crust, age, thick, sim_cfg, cap, upscale=upscale,
        ),
        out_dir / "topography.png",
    )
    # 5. polygons.png — alpha-complex outlines (full sim).
    render_polygons_png(sim_domain, plates, out_dir / "polygons.png", cap)
    # 6. drift.gif / 7. thickness.gif / 8. topography.gif (full sim).
    if frames:
        save_drift_gif(frames, out_dir / "drift.gif", fps=10)
    if frames_thickness:
        save_drift_gif(frames_thickness, out_dir / "thickness.gif", fps=10)
    if frames_topography:
        save_drift_gif(frames_topography, out_dir / "topography.gif", fps=10)


def export_world(
    config: WorldgenConfig,
    seed: int,
    output_root: Path,
    render_layers: tuple[str, ...] = DEFAULT_RENDER_LAYERS,
    hex_px: float = 6.0,
    stop_after: str | None = None,
    config_path: Path | None = None,
    cli_args: dict | None = None,
) -> Path:
    """Generate a world, save its snapshot, render every available layer to PNG.

    Folder name is ``seed<seed>_<W>x<H>km_<YYYYMMDD-HHMMSS>`` under
    ``output_root``. World dimensions come from ``config.world``. Returns
    the path to the created folder.

    ``stop_after`` runs the pipeline up to (and including) the named step
    (see ``PIPELINE_STEPS``) and skips PNGs for layers whose source data
    isn't populated. Defaults to running the full pipeline.

    Layout::

        <folder>/
          snapshot.json
          layers/
            elevation.png
            temperature.png
            ...
          plates/
            plate_NN.png
          drift.gif

    Requires Pillow (the ``[preview]`` extra).
    """
    from worldgen import preview  # lazy: Pillow is only needed for rendering.

    world = generate(config=config, seed=seed, stop_after=stop_after)
    snap = serialize_world(world)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    snap.metadata["seed"] = seed
    snap.metadata["timestamp"] = timestamp

    w_km = int(round(config.world.width_km))
    h_km = int(round(config.world.height_km))
    folder = output_root / f"seed{seed}_{w_km}x{h_km}km_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)

    save_snapshot(snap, folder / "snapshot.json")

    layers_dir = folder / "layers"
    layers_dir.mkdir(exist_ok=True)

    # "plates" is auto-appended when the tectonics step ran.
    layers_to_render: list[str] = list(render_layers)
    if world.lithosphere is not None and "plates" not in layers_to_render:
        layers_to_render.append("plates")
    layers_to_render = _layers_available_at(world.stop_after, tuple(layers_to_render))

    for layer in layers_to_render:
        img = preview.render(world, layer, hex_px=hex_px)
        img.save(layers_dir / f"{layer}.png")

    # Per-plate final-state footprints (post-warp). One PNG per plate goes
    # into a ``plates/`` subfolder so the layers/ listing stays clean.
    # Iterate the *simulated* plate set (``lithosphere.plates``) rather
    # than the t=0 ``PlateField`` — under param_temperature randomization
    # the two can carry different ids.
    if world.lithosphere is not None:
        plates_dir = folder / "plates"
        plates_dir.mkdir(exist_ok=True)
        for plate in world.lithosphere.plates:
            img = preview.render_single_plate(world, plate.id, hex_px=hex_px)
            img.save(plates_dir / f"plate_{plate.id:02d}.png")

    # The drift / thickness / topography animations are emitted by the
    # polygon-sim view exporter below — they live inside
    # ``tectonic_sim_views/`` so the rendering stays close to the sim
    # that produced them. No top-level drift.gif.

    # --- Reproducibility snapshot (ported from prototype). ---
    # Writes config.json (CLI args + resolved WorldgenConfig + module
    # tunables touched by --temp + downstream seed) and copies the
    # source TOML verbatim — every run dir is fully self-describing.
    _dump_run_config(
        folder, config=config, seed=seed, cli_args=cli_args,
        timestamp=timestamp, config_path=config_path,
    )

    # --- tectonic_sim views (ported from prototype). ---
    # Render the underlying particle-cloud visualisations directly via
    # tectonic_sim/viz.py, into a sibling subdirectory. The raw Snapshot
    # is preserved on LithosphereState by tectonics_cast.py — no re-run
    # of the sim required.
    if (
        world.lithosphere is not None
        and world.lithosphere.raw_snapshot is not None
    ):
        _export_tectonic_sim_views(
            folder / "tectonic_sim_views",
            world.lithosphere.raw_snapshot,
        )

    return folder


def _find_py_spy() -> str | None:
    """Find py-spy executable in PATH or the active Python's script dirs.

    pip-installed py-spy is a Rust binary that lands in the active
    Python's ``Scripts`` directory (or user-install ``Scripts`` on
    Windows when site-packages isn't writeable). Not always on PATH.
    """
    import shutil
    import sys
    import sysconfig

    found = shutil.which("py-spy")
    if found:
        return found
    candidates: list[Path] = []
    for scheme in ("nt_user", "posix_user", "nt", "posix_prefix"):
        try:
            scripts = sysconfig.get_path("scripts", scheme)
        except KeyError:
            continue
        if scripts:
            candidates.append(Path(scripts))
    exe_name = "py-spy.exe" if sys.platform == "win32" else "py-spy"
    for d in candidates:
        cand = d / exe_name
        if cand.exists():
            return str(cand)
    return None


def _exec_under_pyspy(args: argparse.Namespace) -> None:
    """Re-exec the current ``python -m worldgen`` invocation under
    py-spy ``record`` mode, with ``--profile`` stripped so the child
    runs the normal worldgen path. Writes the flame-graph SVG to
    ``<out>/profiles/profile_<timestamp>_s<seed>.svg``.
    """
    import datetime as _dt
    import subprocess
    import sys
    import webbrowser

    py_spy = _find_py_spy()
    if py_spy is None:
        print(
            "ERROR: --profile requires py-spy, which is not installed.\n"
            "Install with: pip install -e .[dev]   (or: pip install py-spy)",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.out) / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_svg = out_dir / f"profile_{timestamp}_s{args.seed}.svg"

    # Re-build the worldgen invocation without the profile flags.
    target_cmd: list[str] = [sys.executable, "-m", "worldgen"]
    target_cmd += ["--seed", str(args.seed)]
    target_cmd += ["--config", str(args.config)]
    target_cmd += ["--out", str(args.out)]
    target_cmd += ["--hex-px", str(args.hex_px)]
    if args.stop_after is not None:
        target_cmd += ["--stop-after", args.stop_after]
    if args.quiet:
        target_cmd += ["-q"]

    pyspy_cmd = [
        py_spy, "record",
        "--rate", str(args.profile_sample_rate),
        "--output", str(out_svg),
        "--format", "flamegraph",
        "--subprocesses",
        "--",
        *target_cmd,
    ]
    print(f"Profiling worldgen under py-spy @ {args.profile_sample_rate} Hz")
    print(f"  command: {' '.join(pyspy_cmd)}")
    print(f"  flame graph → {out_svg}")
    result = subprocess.run(pyspy_cmd, check=False)
    if result.returncode != 0:
        print(
            f"\npy-spy exited with code {result.returncode}. "
            "On Windows / macOS you may need to run from an elevated "
            "shell (Admin / sudo) — py-spy needs OS-level access to "
            "read the target process's stack.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)
    if not out_svg.exists():
        print(f"ERROR: expected flame graph not written: {out_svg}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {out_svg} ({out_svg.stat().st_size / 1024:.0f} KB)")
    try:
        webbrowser.open(out_svg.as_uri())
    except Exception:
        pass


def main() -> None:
    """CLI entry point. Invoked by ``python -m worldgen``."""
    parser = argparse.ArgumentParser(
        description="Generate a world and export snapshot + per-layer PNGs.",
        prog="python -m worldgen",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("config/worldgen.toml"))
    parser.add_argument("--out", type=Path, default=Path("exports"))
    parser.add_argument("--hex-px", type=float, default=6.0)
    parser.add_argument(
        "--stop-after", choices=PIPELINE_STEPS, default=None,
        help=(
            "Stop the pipeline after this step (default: run all). "
            "Layers downstream of the stop point are skipped, and only PNGs "
            "whose source data is available are rendered."
        ),
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Silence all worldgen logs and progress bars.",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help=(
            "Re-exec the worldgen run under py-spy to record a wall-clock "
            "flame graph (SVG) into <out>/profiles/ alongside the export. "
            "Requires the [dev] extra (``pip install -e .[dev]``)."
        ),
    )
    parser.add_argument(
        "--profile-sample-rate", type=int, default=200,
        help="py-spy sample rate in Hz when --profile is set (default 200).",
    )
    args = parser.parse_args()

    # --profile mode: re-exec the current Python under py-spy. The
    # re-exec strips the --profile flag so the child runs normally,
    # while py-spy attaches a sampling profiler on top. We do this
    # before any other work so the profile captures the full run.
    if args.profile:
        _exec_under_pyspy(args)
        return

    if args.quiet:
        configure_logging(logging.WARNING)
    else:
        # Verbose (DEBUG) is the default — surfaces per-layer timing,
        # progress bars, and chatty per-hex details.
        configure_logging(logging.DEBUG)

    cfg = load_worldgen_config(args.config)
    folder = export_world(
        config=cfg,
        seed=args.seed,
        output_root=args.out,
        hex_px=args.hex_px,
        stop_after=args.stop_after,
        config_path=args.config,
        cli_args=vars(args),
    )
    print(f"Exported to {folder}")
