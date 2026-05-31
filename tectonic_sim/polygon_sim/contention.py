"""Per-cell ownership resolution + C-C fold-and-thrust belt deposition."""

from __future__ import annotations

import math

import numpy as np

from tectonic_sim.types import CRUST_CONTINENTAL

from tectonic_sim.polygon_sim.types import (
    PolygonPlate)


def _global_owner(plates: list[PolygonPlate], gy: int, gx: int) -> np.ndarray:
    """Snapshot the unique-owner-per-cell map (pid or -1)."""
    owner = np.full((gy, gx), -1, dtype=np.int64)
    for p in plates:
        if not p.alive:
            continue
        # If a cell is claimed by multiple plates at this snapshot (which
        # shouldn't happen at steady state), the later plate overwrites —
        # that's fine for "previous owner" purposes, this map only feeds
        # the trailing-edge gap fill.
        owner[p.cell_mask] = p.pid
    return owner



def _resolve_contention(
    plates: list[PolygonPlate], gy: int, gx: int, sim_config,
    cell_km: float,
) -> None:
    """At cells claimed by >1 plate after stamping, pick a per-cell
    winner and clear the loser masks (and apply C-C fold).

    Priority is per CELL, using the cell-level crust each plate stamped:
      * continental beats oceanic;
      * among continental, lower plate id wins (pyplatec uses smaller
        segment area; this is the simpler deterministic proxy);
      * among oceanic, younger cell age wins; ties by lower pid.

    C-C fold deposition (two-sided): when a continental winner
    overrides a continental loser at a contested cell, fold mass is
    redeposited on BOTH plates' near-suture interiors:

      * **Over-rider belt** — wide, broad. ``folding_ratio ·
        loser_thick`` distributed inland along the winner's −velocity
        across ``folding_belt_depth_km`` with e-folding
        ``folding_belt_decay_km``. Tibet-plateau analogue.

      * **Loser belt** — narrow, sharp. ``folding_loser_side_ratio ·
        loser_thick`` distributed along the LOSER's own −velocity
        across ``folding_belt_loser_depth_km`` with e-folding
        ``folding_belt_loser_decay_km``, starting *one cell into* the
        loser's interior (the suture itself just got cleared away from
        the loser). Himalayan-foothill analogue.

    Weights on each side are normalized to sum to 1 so the per-side
    total mass deposited equals the per-side ratio times the loser's
    column thickness. Belt cells that fall off the recipient plate or
    onto non-continental cells get dropped (mass lost — models belt
    running into ocean or off the continent's edge).
    """
    alive = [p for p in plates if p.alive]
    n = len(alive)
    if n == 0:
        return

    masks = np.stack([p.cell_mask for p in alive], axis=0)
    contend = masks.sum(axis=0)
    contested = contend > 1
    if not contested.any():
        return

    crusts = np.stack([p.crust for p in alive], axis=0)
    ages = np.stack([p.age for p in alive], axis=0)
    thicks = np.stack([p.thickness for p in alive], axis=0)
    pids = np.array([p.pid for p in alive], dtype=np.float64)

    # Per-plate-per-cell priority score, only where masked.
    # Mass-weighted priority: each plate's score at a contested cell is
    # its own total cell count, weighted by the cell's crust type at
    # that plate. Continental cells get a strong (sim_config.crust_continental_weight×)
    # boost so normal-sized continents dominate oceanic plates, but a
    # truly tiny continental island (mass << contesting plate / weight)
    # loses to a vastly larger oceanic neighbour — fixing the "small
    # island carves through a huge plate" artefact where a 5-cell
    # continental fragment used to win at every cell it stamped.
    cont_per_cell = crusts == CRUST_CONTINENTAL
    mass = masks.sum(axis=(1, 2)).astype(np.float64)
    crust_factor = np.where(cont_per_cell, sim_config.crust_continental_weight, 1.0)
    mass_score = mass[:, None, None] * crust_factor
    # Oceanic age tie-breaker (younger oceanic wins all else equal).
    # Continental cells get age_penalty=0 so age doesn't shift their
    # score; only their mass + crust weight count.
    age_penalty = np.where(cont_per_cell, 0.0, ages)
    score = mass_score - age_penalty - 1e-6 * pids[:, None, None]
    score = np.where(masks, score, -np.inf)

    winner = np.argmax(score, axis=0)        # (gy, gx)
    plate_idx = np.arange(n)[:, None, None]
    is_winner = plate_idx == winner[None, :, :]
    loser_mask = masks & ~is_winner & contested[None, :, :]

    # C-C fold: at contested cells where the winner is continental, sum
    # contributions from continental losers into a fold deposit.
    winner_crust = np.take_along_axis(crusts, winner[None], axis=0)[0]
    cc_loser_thick = np.where(
        loser_mask & (crusts == CRUST_CONTINENTAL), thicks, 0.0).sum(axis=0)
    fold_cell = contested & (winner_crust == CRUST_CONTINENTAL) & (cc_loser_thick > 0)

    # Clear loser masks (and paint at those cells).
    cleared_masks = masks & ~loser_mask
    cleared_thicks = np.where(loser_mask, 0.0, thicks)
    cleared_ages = np.where(loser_mask, 0.0, ages)
    cleared_crust = np.where(loser_mask, np.int8(0), crusts)

    # --- Inland fold-belt deposition --------------------------------------
    #
    # For each fold cell, distribute ``folding_ratio · cc_loser_thick``
    # across a belt of cells extending inland on the winner's plate.
    # "Inland" = opposite the winner plate's velocity unit vector — the
    # collision front sits at the over-rider's leading edge, so −velocity
    # points back into the continent's interior.
    #
    # The belt depth is ``folding_belt_depth_km`` cells; the weight at
    # offset k cells from the suture is ``exp(-k·cell_km / decay_km)``,
    # then renormalized so weights sum to 1 over the depth range. Since
    # ``cleared_thicks`` is the post-clearance thickness, the deposit
    # adds to the *winner's* row at each band-target cell, restricted to
    # cells that (a) belong to the winner's plate after clearance and
    # (b) are continental on the winner. Mass that would land outside
    # those constraints is dropped — that's the realistic "fold mass
    # bleeding into the ocean / off the plate" behaviour and keeps the
    # operation purely additive on continental crust.
    #
    # When the winner is stationary (or belt_depth <= cell_km), the
    # range collapses to k=0 and the deposit degenerates to the legacy
    # single-cell add.
    # Per-plate (idx in `alive`) inland unit vector. Plates with zero
    # velocity fall back to (0, 0): every k>0 step lands on the same
    # cell, so the band collapses to the suture — total mass still
    # conserved on the winner side because the weights still sum to 1
    # onto that one cell (legacy behaviour); on the loser side, the
    # collapse means the deposit fails the same-plate check (suture is
    # winner-owned now) so no mass lands. Both degrade gracefully.
    inland_dx = np.zeros(n, dtype=np.float64)
    inland_dy = np.zeros(n, dtype=np.float64)
    for k, p in enumerate(alive):
        vx, vy = float(p.velocity_kmpy[0]), float(p.velocity_kmpy[1])
        speed = math.hypot(vx, vy)
        if speed > 1e-6:
            inland_dx[k] = -vx / speed
            inland_dy[k] = -vy / speed

    # --- Over-rider belt deposition ---------------------------------------
    fold_add_total = sim_config.folding_ratio * cc_loser_thick * fold_cell
    if fold_add_total.any():
        belt_depth_cells = max(
            1, int(math.ceil(sim_config.folding_belt_depth_km / cell_km)),
        )
        decay_cells = max(
            1e-6, sim_config.folding_belt_decay_km / cell_km,
        )
        # Winner belt starts AT the suture (k=0) — the suture cell is
        # newly owned by the winner and is the peak of the deposit.
        weights = np.exp(
            -np.arange(belt_depth_cells, dtype=np.float64) / decay_cells,
        )
        weights /= weights.sum()

        # Build per-cell deposit additions to `cleared_thicks[winner]`.
        # We iterate over band offsets (small N — ~15) rather than fold
        # cells (could be thousands), so the inner work is vectorized.
        fy, fx = np.where(fold_cell)
        if fy.size > 0:
            cell_winners = winner[fy, fx]                      # (M,)
            cell_mass = fold_add_total[fy, fx]                 # (M,)
            ux = inland_dx[cell_winners]                       # (M,)
            uy = inland_dy[cell_winners]
            for step in range(belt_depth_cells):
                w = weights[step]
                if w <= 0.0:
                    continue
                ty = (fy + np.rint(uy * step).astype(np.int64)) % gy
                tx = (fx + np.rint(ux * step).astype(np.int64)) % gx
                # Only deposit where the target cell belongs to the
                # winner (post-clearance) AND is continental. Dropping
                # mass that violates either is by design: it models the
                # belt running off the edge of the continent.
                target_mask = cleared_masks[cell_winners, ty, tx] & (
                    cleared_crust[cell_winners, ty, tx] == CRUST_CONTINENTAL
                )
                if not target_mask.any():
                    continue
                # Scatter-add via np.add.at to handle multiple fold
                # cells projecting onto the same target.
                np.add.at(
                    cleared_thicks,
                    (cell_winners[target_mask], ty[target_mask], tx[target_mask]),
                    cell_mass[target_mask] * w,
                )

    # --- Loser belt deposition --------------------------------------------
    #
    # Mirror of the over-rider pass on the other side of the suture.
    # Each *loser* plate independently deposits a fraction of its own
    # cleared-cell thickness back into its near-suture interior, along
    # ITS OWN −velocity direction. Belt starts at step=1 (one cell into
    # the loser's territory; the suture cell now belongs to the winner
    # and would fail the same-plate check at step=0). Weights are
    # renormalized over the 1..N range so the per-loser per-cell total
    # deposited mass equals exactly ``folding_loser_side_ratio ·
    # loser_thick`` when the band fits inside the loser's territory.
    #
    # At a triple junction (>2 plates contesting one cell), each loser
    # contributes independently — Tibet-style outcome on the over-rider
    # plus two distinct Himalaya-style ridges on the two losers, each
    # propagating into its own continent.
    loser_ratio = sim_config.folding_loser_side_ratio
    if loser_ratio > 0.0:
        belt_depth_loser_cells = max(
            1, int(math.ceil(sim_config.folding_belt_loser_depth_km / cell_km)),
        )
        decay_loser_cells = max(
            1e-6, sim_config.folding_belt_loser_decay_km / cell_km,
        )
        # Weights start at step=1 (suture is over-rider-owned). The
        # step=1 cell is the peak — geologically the foothill ridge
        # sits right against the suture on the loser's side.
        loser_weights = np.exp(
            -np.arange(1, belt_depth_loser_cells + 1, dtype=np.float64)
            / decay_loser_cells,
        )
        loser_weights /= loser_weights.sum()

        # Per-loser-plate: find its contested continental cells, then
        # vectorize-deposit along its inland direction.
        for k_loser in range(n):
            loser_cells_k = (
                loser_mask[k_loser] & (crusts[k_loser] == CRUST_CONTINENTAL)
            )
            if not loser_cells_k.any():
                continue
            ux = float(inland_dx[k_loser])
            uy = float(inland_dy[k_loser])
            ly, lx = np.where(loser_cells_k)
            # Per-loser-cell mass to redeposit. Restricted to fold cells
            # whose winner is also continental (matches the over-rider
            # side's gate — at an O-C contention the underthrusting
            # oceanic crust doesn't get inland-deposit treatment, only
            # C-C does).
            ly_in_fold = fold_cell[ly, lx]
            if not ly_in_fold.any():
                continue
            ly = ly[ly_in_fold]
            lx = lx[ly_in_fold]
            cell_mass = loser_ratio * thicks[k_loser, ly, lx]
            for step_idx, w in enumerate(loser_weights):
                if w <= 0.0:
                    continue
                step = step_idx + 1
                ty = (ly + int(round(uy * step))) % gy
                tx = (lx + int(round(ux * step))) % gx
                # Target must be loser-owned (post-clearance) and
                # continental. Cells that wandered off the loser plate
                # or hit ocean drop their mass (belt runs off-plate).
                target_mask = cleared_masks[k_loser, ty, tx] & (
                    cleared_crust[k_loser, ty, tx] == CRUST_CONTINENTAL
                )
                if not target_mask.any():
                    continue
                np.add.at(
                    cleared_thicks[k_loser],
                    (ty[target_mask], tx[target_mask]),
                    cell_mass[target_mask] * w,
                )

    # Write back.
    for k, p in enumerate(alive):
        p.cell_mask = cleared_masks[k]
        p.crust = cleared_crust[k].astype(np.int8)
        p.age = cleared_ages[k]
        p.thickness = cleared_thicks[k]
