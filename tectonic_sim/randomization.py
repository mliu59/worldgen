"""Hyperparameter-driven randomization of physics configs.

One knob — ``param_temperature`` — controls the magnitude of random
exploration around a base config. At ``T = 0`` the returned config
equals the base byte-for-byte; at ``T > 0`` each numeric field is drawn
from ``Normal(base_value, T × hardcoded_std)``, clipped to a safe range
and rounded to int if applicable.

The design is two-layered so the same machinery can be reused for other
configs later (the obvious next case is ``WorldgenConfig``):

  - **Layer 1: a generic helper.** ``randomize_dataclass_fields(base,
    randomizers, param_temperature, rng)`` walks a tuple of
    ``FieldRandomizer`` specs and produces a new instance of the same
    dataclass with the listed fields perturbed.

  - **Layer 2: per-config public wrapper.** ``randomize_sim_config(
    base, param_temperature, seed)`` hides the generic helper and
    binds the ``SimConfig``-specific spec tuple. Worldgen would add
    its own ``randomize_worldgen_config`` later that delegates to the
    same Layer 1.

**Std + bounds are hardcoded.** The user only controls
``param_temperature``. The per-field std at ``T = 1`` is what the
designer of this module considers a "reasonable exploration unit" for
that field — usually ~10–25 % of the field's central value. Bounds
prevent extreme draws that would be physically nonsensical (negative
thickness, > 100 % continental fraction, ...).

**What's excluded.** World size is the user's explicit exclusion (it
sets the simulation domain, not a physics magnitude). Two other
categories are also left alone:

  - Temporal / output knobs: ``dt_myr``, ``n_ticks``,
    ``snapshot_period_ticks``.
  - Numerical / cadence knobs: ``contact_iterations``, ``erosion_period``.

These would all reshape the *run* rather than the *world* and are
better held fixed when sweeping ``param_temperature``.

**Independent draws — known limitation.** Fields are randomized
independently in v0. Some pairs have natural coupling (rift crust is
*usually* thinner than established continental crust; reference
thickness usually equals starting thickness). The clip bounds chosen
below prevent the most pathological cross-field combinations, but they
don't strictly enforce coupling. If you draw at ``T = 2`` you may get
configs that are physically possible but oddly tuned (e.g. rift crust
slightly thicker than continental). Future work: a constraint post-pass
or a derived-field option on ``FieldRandomizer``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from tectonic_sim.types import SimConfig


@dataclass(frozen=True)
class FieldRandomizer:
    """Specification for randomizing one numeric field on a dataclass.

    Attributes:
        field_name: the dataclass attribute to perturb.
        std: standard deviation at ``param_temperature = 1.0``. The
            effective std at temperature ``T`` is ``T × std``.
        minimum: clip lower bound. ``None`` means no lower clip.
        maximum: clip upper bound. ``None`` means no upper clip.
        is_integer: if True, the drawn value is rounded to the nearest
            integer (after clipping). Use for fields typed ``int``.
    """

    field_name: str
    std: float
    minimum: float | None
    maximum: float | None
    is_integer: bool = False


# ----------------------------------------------------------------------------
# Generic Layer 1
# ----------------------------------------------------------------------------

def randomize_dataclass_fields(
    base: Any,
    randomizers: tuple[FieldRandomizer, ...],
    param_temperature: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Any:
    """Return a new instance of ``base``'s dataclass with the listed
    fields perturbed by Normal noise scaled by ``param_temperature``.

    Fields not listed in ``randomizers`` pass through unchanged.
    Validates that every ``randomizer.field_name`` exists on the base
    dataclass — typos are bugs and should fail loud.

    Returns a new frozen-dataclass instance (uses ``dataclasses.replace``).
    """
    if param_temperature < 0:
        raise ValueError(
            f"param_temperature must be >= 0, got {param_temperature}",
        )
    if param_temperature == 0.0:
        # Identity path — return the input unchanged. Byte-identical
        # because no draws are made. The default value of 0 means the
        # safe / no-randomization path is what you get if you forget
        # to set the temperature.
        return base

    if rng is None:
        raise ValueError(
            "rng is required when param_temperature > 0",
        )

    known_fields = {f.name for f in fields(base)}
    overrides: dict[str, Any] = {}
    for spec in randomizers:
        if spec.field_name not in known_fields:
            raise ValueError(
                f"FieldRandomizer references unknown field "
                f"{spec.field_name!r} on {type(base).__name__}",
            )
        base_value = getattr(base, spec.field_name)
        sigma = spec.std * param_temperature
        drawn = float(rng.normal(loc=float(base_value), scale=sigma))
        if spec.minimum is not None:
            drawn = max(drawn, spec.minimum)
        if spec.maximum is not None:
            drawn = min(drawn, spec.maximum)
        if spec.is_integer:
            drawn_int = int(round(drawn))
            # Clip again after rounding to honour integer bounds exactly
            # (rounding can push a draw at ``minimum + 0.4`` below the
            # integer ``minimum`` if ``minimum`` itself was a float).
            if spec.minimum is not None:
                drawn_int = max(drawn_int, int(round(spec.minimum)))
            if spec.maximum is not None:
                drawn_int = min(drawn_int, int(round(spec.maximum)))
            overrides[spec.field_name] = drawn_int
        else:
            overrides[spec.field_name] = drawn

    return replace(base, **overrides)


# ----------------------------------------------------------------------------
# SimConfig-specific Layer 2
# ----------------------------------------------------------------------------

# Hardcoded per-field std and clip ranges for ``SimConfig``.
#
# Std picking heuristic: ~15–25 % of the typical central value, so a
# single ``T = 1`` draw shifts a parameter by a noticeable but not
# crazy amount. Some integer fields get tighter or looser stds based
# on whether the design space has steep cliffs near their bounds.
#
# Clip range picking: roughly ``±3σ`` from the typical central value,
# but pulled in further if a smaller value would break the sim
# (e.g. ``motion_speed_kmpy`` has a hard floor of ~1 km/Myr because
# the contact constraint can't engage at sub-overlap-radius drift).
_SIM_CONFIG_RANDOMIZERS: tuple[FieldRandomizer, ...] = (
    # --- Plate population ---
    FieldRandomizer("plate_count",
                    std=2.0, minimum=2, maximum=20, is_integer=True),
    FieldRandomizer("continental_fraction",
                    std=0.15, minimum=0.0, maximum=1.0),
    FieldRandomizer("motion_speed_kmpy",
                    std=20.0, minimum=1.0, maximum=200.0),
    FieldRandomizer("seed_radial_bias",
                    std=0.3, minimum=-1.0, maximum=1.0),

    # --- Initial particle layout ---
    # particle_spacing controls density; randomizing it makes runs hard
    # to compare visually so we use a conservative std and a tight
    # range — still varies, but doesn't 10× the particle count between
    # draws.
    FieldRandomizer("particle_spacing_km",
                    std=3.0, minimum=5.0, maximum=30.0),

    # --- Crust thicknesses ---
    # Continental and reference are highly correlated in nature; we
    # randomize independently but with overlapping ranges so a pair of
    # draws sits within a few km of each other at moderate T.
    FieldRandomizer("continental_thickness_km",
                    std=6.0, minimum=20.0, maximum=60.0),
    FieldRandomizer("oceanic_thickness_km",
                    std=1.5, minimum=4.0, maximum=12.0),
    FieldRandomizer("rift_thickness_km",
                    std=5.0, minimum=20.0, maximum=55.0),

    # --- Half-space cooling ---
    FieldRandomizer("ridge_depth_km",
                    std=0.5, minimum=1.0, maximum=4.0),
    FieldRandomizer("ridge_subsidence_rate",
                    std=0.1, minimum=0.15, maximum=0.6),
    FieldRandomizer("max_ocean_depth_km",
                    std=1.0, minimum=4.0, maximum=10.0),

    # --- Continental isostasy ---
    FieldRandomizer("continental_reference_thickness_km",
                    std=5.0, minimum=25.0, maximum=50.0),
    FieldRandomizer("continental_isostasy_factor",
                    std=0.04, minimum=0.08, maximum=0.25),
    # Sea level is *purely* a sampling threshold (the dynamics never read
    # it), so the only consequence of moving it is land/ocean mix.
    # A 1-km std covers Cretaceous-style high-stands and glacial lows.
    FieldRandomizer("sea_level_km",
                    std=1.0, minimum=-3.0, maximum=3.0),

    # --- Collision constants ---
    FieldRandomizer("orogeny_uplift_per_overlap_km",
                    std=0.03, minimum=0.01, maximum=0.30),
    # Range bumped to match the new default (0.5, reconciled from the
    # rigid-polygon prototype). Old range [0, 0.20] clipped every draw to
    # the upper bound. New std 0.15 keeps the ±30 % spread the old config
    # had relative to its base.
    FieldRandomizer("folding_ratio",
                    std=0.15, minimum=0.05, maximum=0.95),
    FieldRandomizer("folding_displacement_km",
                    std=0.6, minimum=0.0, maximum=4.0),
    FieldRandomizer("subduction_arc_uplift_km",
                    std=0.02, minimum=0.01, maximum=0.15),
    # Fold-belt geometry — width determines orogen breadth, decay sets
    # the dramatic-uplift core's radius. Std picked so a T=1 draw moves
    # the belt by ~30 % of the typical Himalayan scale.
    FieldRandomizer("folding_belt_depth_km",
                    std=30.0, minimum=0.0, maximum=400.0),
    FieldRandomizer("folding_belt_decay_km",
                    std=10.0, minimum=5.0, maximum=150.0),
    # Loser-side belt — tighter ranges. Ratio capped at 0.7 so it can't
    # alone exceed unity (would create mass); depth narrower than the
    # over-rider; decay correspondingly tighter.
    FieldRandomizer("folding_loser_side_ratio",
                    std=0.08, minimum=0.0, maximum=0.7),
    FieldRandomizer("folding_belt_loser_depth_km",
                    std=15.0, minimum=0.0, maximum=200.0),
    FieldRandomizer("folding_belt_loser_decay_km",
                    std=5.0, minimum=5.0, maximum=80.0),

    # --- Continental relief (Perlin "ancient basement topography") ---
    # Amplitude: ~25 % spread around the 6-km baseline at T=1; capped to
    # zero on the low side (negative amplitude is meaningless) and 15 km
    # on the high side (any larger and the noise dominates the 35 km
    # continental baseline, producing oceanic-floored continents).
    FieldRandomizer("continental_relief_amplitude_km",
                    std=2.0, minimum=0.0, maximum=15.0),
    # Wavelength: spans a 4× range at T=1 — 200 km (small islands /
    # archipelagos) to 3000 km (broad sub-continental basins).
    FieldRandomizer("continental_relief_wavelength_km",
                    std=400.0, minimum=200.0, maximum=3000.0),
    # Octaves: small integer field. Std 1 covers 3-5 typical settings.
    FieldRandomizer("continental_relief_octaves",
                    std=1.0, minimum=2, maximum=6, is_integer=True),
    # Persistence: tight std — extreme draws are visually noisy.
    FieldRandomizer("continental_relief_persistence",
                    std=0.1, minimum=0.2, maximum=0.7),

    # --- Velocity damping ---
    FieldRandomizer("velocity_damping_strength",
                    std=0.03, minimum=0.0, maximum=0.30),

    # --- Erosion ---
    # Period stays fixed; only the strength varies.
    FieldRandomizer("erosion_strength",
                    std=0.03, minimum=0.0, maximum=0.30),

    # Excluded from randomization, recorded here for documentation:
    #   n_ticks, dt_myr     — temporal knobs
    #   snapshot_period_ticks — output knob
    #   erosion_period      — numerical cadence
    # World size (WorldRect) is not a SimConfig field at all; the
    # caller passes it to the polygon sim separately.
)


def randomize_sim_config(
    base: SimConfig,
    param_temperature: float = 0.0,
    seed: int = 0,
) -> SimConfig:
    """Return a randomized ``SimConfig`` derived from ``base``.

    ``param_temperature`` controls the breadth of the exploration:

      - ``0.0`` (default) returns ``base`` unchanged (byte-identical,
        no draws). Default 0 makes the no-randomization path opt-in
        elsewhere: callers that don't set a temperature get the safe,
        deterministic config back.
      - ``1.0`` is the "natural" exploration breadth — each field is
        drawn from a Normal around ``base`` with hardcoded std.
      - ``> 1.0`` widens the draws (e.g. 2.0 doubles every std).

    ``seed`` controls determinism: same ``(base, temperature, seed)``
    yields the same output. Pass a different seed for a different draw
    at the same temperature. Defaults to 0; only consulted when
    ``param_temperature > 0`` (the identity path makes no draws).

    Fields not in ``_SIM_CONFIG_RANDOMIZERS`` (boundary mode, time step,
    output cadence, etc.) pass through unchanged — see this module's
    docstring for the exclusion rationale.

    Field bounds are clipped, not rejected, so the function always
    returns a usable config. Independent per-field draws mean some
    pairs may sit in unusual relative positions at high temperatures
    (e.g. rift crust slightly thicker than continental); future work
    can add a coupling layer if that becomes a problem.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    return randomize_dataclass_fields(
        base, _SIM_CONFIG_RANDOMIZERS, param_temperature, rng,
    )
