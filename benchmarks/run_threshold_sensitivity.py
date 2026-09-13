"""Sensitivity sweep over the adopted evidence-threshold defaults.

    py benchmarks/run_threshold_sensitivity.py

A plateau is evidence that a default is safe ONLY if the parameter could
have moved the answer. Two earlier versions of this sweep failed that test
in different ways (the 2026-09-01 bug sweep, then its own audit):

  v1  swept a_min and o_floor over a control where they provably could not
      act, and reported the flat line as "defaults confirmed".
  v2  reported verdicts that were artefacts: the try/except wrapped the
      split-half replicates as well as the main fit, so a replicate crash
      was recorded as the row's outcome; the a_min probe value (1.0) could
      not fail, because the test is support >= a_min and support was
      identically 1.0; and target_coverage re-targeted tau, absorbing the
      very penalty changes being swept.

This version is falsifiable by construction:

  * each parameter has a DRIVER, the quantity it is compared against inside
    the library. The driver is measured on the control and reported, and the
    swept values are chosen to STRADDLE it. A sweep whose range does not
    straddle its driver is reported as vacuous, not as a plateau.
  * the main fit and the replicate fits have separate error handling, so a
    replicate crash can never speak for the row.
  * no target_coverage: tau is fixed, so a penalty change is free to move
    the answer.
  * warnings are not silenced.

Driver per parameter, read off the code:

  min_valid_frac  low tail of the segment validity fractions
  rho_min         per-sample observed evidence fraction rho
  o_min           per-segment observation rate (per-resample assessability)
  o_floor         per-segment mean observation rate, via lam * max(rate, floor)
  a_min           per-segment resample support

The a_min driver deserves a note, because getting it wrong is why this file
was rewritten twice. Resample support is deterministic (identically 1.0)
whenever the observation patterns form detectable groups: _strata then
stratifies by class x observation group, and resample_indices draws exactly
len(idx) rows per stratum, so the observer count per resample has zero
variance BY CONSTRUCTION. Support varies, and a_min binds, only when the
patterns are too varied for observation_groups to group them. Both regimes
appear below.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import LOADERS, truth_segments  # noqa: E402

from robustsignalmaker import (  # noqa: E402
    BootstrapMaskSelector,
    InsufficientEvidenceError,
    SegmentGrid,
    make_signal_control,
    mask_dropout_stretches,
    mean_pairwise_jaccard,
    observation_groups,
    resample_indices,
)

RESULTS = Path(__file__).resolve().parent / "results"
DEFAULTS = {"min_valid_frac": 0.5, "rho_min": 0.1, "o_min": 0.1,
            "a_min": 0.5, "o_floor": 0.2}


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #
def raman_control():
    X, V, y, meta = LOADERS["raman"](seed=0)
    grid = SegmentGrid(X.shape[-1], segment=meta["segment"])
    truth = set(truth_segments(meta["truth_points"], grid).tolist())
    return dict(X=X, V=V, y=y, grid=grid, truth=truth, segment=meta["segment"],
                kw=dict(lam=0.08, n_iter=200, lr=0.1, n_bootstrap=10, tau=0.5))


def rare_band_control(rare=0.25, seed=90):
    """The only informative band is observed by a minority: the regime where
    the observation-scaled penalty decides whether it survives."""
    c = make_signal_control(n=400, n_points=256, task="binary", signal=2.5,
                            noise=1.0, region_cells=2, distractor_cells=2,
                            seed=seed)
    info = np.flatnonzero(c.informative)
    lo, hi = int(info.min()), int(info.max()) + 1
    rng = np.random.default_rng(seed)
    sees = rng.random(len(c.y)) < rare
    V = np.ones(c.X.shape, dtype=bool)
    V[~sees, :, lo:hi] = False
    grid = SegmentGrid(256, segment=8)
    return dict(X=np.where(V, c.X, 0.0), V=V, y=c.y, grid=grid,
                truth=set(c.truth_segments(grid).tolist()), segment=8,
                kw=dict(lam=0.3, n_iter=150, lr=0.1, n_bootstrap=20, tau=0.5))


def ungrouped_rare_control(share=0.12, seed=5):
    """The same rare-band idea, but with whole-segment dropouts at varied
    positions so the observation patterns are too many to group. Resample
    support is then genuinely intermediate and a_min can bind."""
    c = make_signal_control(n=300, n_points=256, task="binary", signal=2.5,
                            noise=1.0, region_cells=3, distractor_cells=2,
                            seed=seed)
    info = np.flatnonzero(c.informative)
    lo, hi = int(info.min()), int(info.max()) + 1
    V = mask_dropout_stretches(c.X.shape, n_stretches=2, min_len=16,
                               max_len=40, seed=seed + 1)
    rng = np.random.default_rng(seed)
    sees = rng.random(len(c.y)) < share
    V[~sees, :, lo:hi] = False
    grid = SegmentGrid(256, segment=8)
    return dict(X=np.where(V, c.X, 0.0), V=V, y=c.y, grid=grid,
                truth=set(c.truth_segments(grid).tolist()), segment=8,
                kw=dict(lam=0.08, n_iter=120, lr=0.1, n_bootstrap=10, tau=0.5))


# --------------------------------------------------------------------------- #
# drivers: the quantity each parameter is actually compared against
# --------------------------------------------------------------------------- #
def partial_validity_control(seed=11):
    """Dropout stretches SHORTER than a segment, so segment validity takes
    partial values: the only regime where min_valid_frac (an eligibility
    threshold on the validity fraction) can act. On the raman control the
    dropouts wipe whole segments, leaving validity at 1.0 or 0.0, which is
    why the swept range there is arithmetically inert."""
    c = make_signal_control(n=300, n_points=256, task="binary", signal=2.5,
                            noise=1.0, region_cells=3, distractor_cells=2,
                            seed=seed)
    V = mask_dropout_stretches(c.X.shape, n_stretches=6, min_len=2, max_len=5,
                               seed=seed + 1)
    grid = SegmentGrid(256, segment=8)
    return dict(X=np.where(V, c.X, 0.0), V=V, y=c.y, grid=grid,
                truth=set(c.truth_segments(grid).tolist()), segment=8,
                kw=dict(lam=0.08, n_iter=150, lr=0.1, n_bootstrap=10, tau=0.5))


def driver_min_valid_frac(ctl, sel):
    # the threshold acts on the LOW TAIL of the validity distribution: the
    # median is 1.0 whenever most segments are fully observed, which made an
    # earlier version of this driver mis-report a straddling sweep as vacuous
    _, VF = ctl["grid"].pool(ctl["X"], ctl["V"].astype(float))
    nz = VF[VF > 0]
    return (float(np.percentile(nz, 10)),
            "10th percentile of nonzero segment validity")


def driver_rho_min(ctl, sel):
    v_seg = ctl["grid"].segment_validity(ctl["V"].astype(float))
    return float(v_seg.mean(axis=1).min()), "min per-sample evidence fraction"


def driver_o_min(ctl, sel):
    return float(sel.obs_frac_.min()), "min per-segment observation rate"


def driver_o_floor(ctl, sel):
    truth = sorted(ctl["truth"])
    return (float(np.mean([sel.obs_frac_[s] for s in truth])),
            "mean observation rate of the truth segments")


def driver_a_min(ctl, sel):
    inter = sel.support_[(sel.support_ > 0) & (sel.support_ < 1)]
    if inter.size:
        return float(np.median(inter)), "median intermediate resample support"
    return 1.0, "resample support (deterministic: no intermediate values)"


SPECS = {
    "min_valid_frac": dict(control=partial_validity_control,
                           control_name="partial_validity",
                           values=[0.3, 0.6, 0.9], probe=0.999,
                           driver=driver_min_valid_frac),
    "rho_min": dict(control=raman_control, control_name="raman",
                    values=[0.1, 0.5, 0.9],
                    probe=0.999, driver=driver_rho_min),
    "o_min": dict(control=raman_control, control_name="raman",
                  values=[0.1, 0.5, 0.9],
                  probe=0.999, driver=driver_o_min),
    "o_floor": dict(control=lambda: rare_band_control(rare=0.25),
                    control_name="rare_band(rare=0.25)",
                    values=[0.1, 0.25, 0.5], probe=0.95,
                    driver=driver_o_floor),
    "a_min": dict(control=lambda: ungrouped_rare_control(share=0.12),
                  control_name="ungrouped_rare(share=0.12)",
                  values=[0.3, 0.6, 0.8], probe=1.0, driver=driver_a_min),
}


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for param, spec in SPECS.items():
        ctl = spec["control"]()
        X, V, y, grid = ctl["X"], ctl["V"], ctl["y"], ctl["grid"]
        truth, n = ctl["truth"], ctl["X"].shape[0]
        grouped = observation_groups(
            grid.segment_validity(V.astype(float))) is not None

        def fit(overrides, seed, Xs, ys, Vs):
            kwargs = dict(DEFAULTS)
            kwargs.update(overrides)
            return BootstrapMaskSelector(task="binary", segment=ctl["segment"],
                                         seed=seed, **ctl["kw"], **kwargs
                                         ).fit(Xs, ys, Vs)

        base = fit({}, 0, X, y, V)
        driver_value, driver_name = spec["driver"](ctl, base)
        values = list(spec["values"]) + [spec["probe"]]
        straddles = min(spec["values"]) <= driver_value <= max(spec["values"])
        print(f"\n=== {param} on the {spec['control_name']} control: "
              f"driver = {driver_name} = {driver_value:.3f}; "
              f"swept {spec['values']} "
              f"{'STRADDLES' if straddles else 'DOES NOT straddle'} it; "
              f"probe {spec['probe']}; observation groups "
              f"{'detected' if grouped else 'not detected'}", flush=True)

        selections = {}
        for val in values:
            role = ("probe" if val == spec["probe"] else
                    "default" if val == DEFAULTS[param] else "swept")
            row = {"param": param, "control": spec["control_name"],
                   "driver": driver_name,
                   "driver_value": round(driver_value, 3),
                   "groups_detected": grouped, "value": val, "role": role}
            try:  # the MAIN fit alone: a replicate crash must not speak for it
                sel = fit({param: val}, 0, X, y, V)
                chosen = tuple(sel.selected_segments_.tolist())
                s = set(chosen)
                row["n_selected"] = len(chosen)
                row["f1"] = (round(2 * len(s & truth) / (len(s) + len(truth)), 3)
                             if s else 0.0)
                row["main_fit"] = "ok"
            except InsufficientEvidenceError as exc:
                chosen = ("REFUSED",)
                row.update(n_selected="", f1="",
                           main_fit=f"refused: {type(exc).__name__}")
            selections[val] = chosen

            sets, failed = [], 0
            for r in range(3):  # replicates counted separately, never conflated
                rng = np.random.default_rng(1000 + r)
                idx = resample_indices(n, "half", rng, strat=y)
                try:
                    sets.append(fit({param: val}, r + 1,
                                    X[idx], y[idx], V[idx]).selected_segments_)
                except InsufficientEvidenceError:
                    failed += 1
            stab = (mean_pairwise_jaccard(sets, n_total=grid.n_segments)
                    if len(sets) > 1 else float("nan"))
            row["stability_adj"] = (round(float(stab), 3)
                                    if np.isfinite(stab) else "nan")
            row["replicates_failed"] = failed
            rows.append(row)
            print(f"  {role:8s} {param}={val}: selected={list(chosen)[:6]} "
                  f"f1={row['f1']} stab={row['stability_adj']} "
                  f"main={row['main_fit']} rep_failed={failed}", flush=True)

        swept_sets = {selections[v] for v in spec["values"]}
        probe_same = selections[spec["probe"]] in swept_sets
        if len(swept_sets) > 1:
            verdict = "SENSITIVE within the swept range"
        elif not straddles:
            verdict = ("VACUOUS: the swept range does not straddle the driver, "
                       "so the flat line is arithmetic, not evidence")
        elif not probe_same:
            verdict = "GENUINE PLATEAU: straddles the driver, and the probe moves it"
        else:
            verdict = ("INERT on this control: nothing moved it, probe included, "
                       "even though the range straddles the driver")
        rows.append({"param": param, "control": spec["control_name"],
                     "value": "VERDICT", "role": "summary",
                     "driver": driver_name,
                     "driver_value": round(driver_value, 3),
                     "groups_detected": grouped, "main_fit": verdict})
        print(f"  -> {verdict}", flush=True)

    out = RESULTS / "threshold_sensitivity.csv"
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
