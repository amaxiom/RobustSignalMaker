"""Sparsity-strength selection as a library feature (Milestone 4).

The RobustPixelMaker benchmarks showed one fixed ``lam`` under-pruning on one
dataset and over-pruning on another at the very same value, and that
project's findings state the consequence: per-dataset lam selection "must
become a library feature, not a benchmark script". This module is that
feature.

Two tools:

  * ``lam_frontier``: sweep lam and tabulate the score-relevant quantities
    per point (coverage, stability across resamples, ambiguity, support),
    so an operating point is chosen with the trade-offs visible. Stability
    is the chance-corrected mean pairwise Jaccard over the ASSESSABLE
    universe; raw Jaccard rewards keeping everything.
  * ``lam_for_coverage``: log-scale bisection to the lam whose coverage is
    nearest a target. It assumes coverage falls as lam rises, which holds in
    aggregate but is noisy per fit, so it tracks the best point seen and
    reports whether the target was actually reached; it never silently
    returns a miss as a hit.
"""
from __future__ import annotations

import numpy as np

from .metrics import mean_pairwise_jaccard
from .validity import InsufficientEvidenceError


def lam_frontier(make_selector, lams) -> list:
    """Fit ``make_selector(lam)`` per value and tabulate the frontier.

    ``make_selector`` returns a FITTED selector (a
    :class:`~robustsignalmaker.selection.BootstrapMaskSelector` or anything
    exposing ``coverage()``, ``ambiguous_fraction()``, ``resample_sets_``,
    ``assessable_universe()``, ``stability_report()``). Returns one dict per
    lam; no winner is chosen here, because the trade-off is the result.
    """
    rows = []
    for lam in lams:
        sel = make_selector(float(lam))
        stability = mean_pairwise_jaccard(
            sel.resample_sets_, min_size=1, n_total=sel.assessable_universe())
        report = sel.stability_report()
        rows.append({
            "lam": float(lam),
            "coverage": sel.coverage(),
            "stability_adjusted": stability,
            "ambiguous_fraction": report["ambiguous_fraction"],
            "unassessable_fraction": report["unassessable_fraction"],
            "n_selected": len(sel.selected_segments_),
            "n_failed_fits": report["n_failed_fits"],
            "degenerate": report["degenerate"],
        })
    return rows


def lam_for_coverage(make_selector, target: float, lo: float = 1e-3,
                     hi: float = 1.0, n_iter: int = 10, tol: float = 0.05) -> dict:
    """Log-scale bisection to the lam whose coverage is nearest ``target``.

    Returns ``{"lam", "coverage", "target_reached", "history"}``.
    ``target_reached`` is False when the bracket cannot produce the target
    (for example the weakest penalty still under-covers); the best point
    found is still returned, with its true coverage, so a miss is visible
    rather than silently renamed a hit.

    A lam strong enough to close every gate makes the selector refuse. That
    is recorded in ``history`` as ``coverage`` nan with ``refused`` True, and
    the search continues; it never wins best-point tracking and never raises.
    Any other exception from ``make_selector`` propagates unchanged.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError("target coverage must be in (0, 1]")
    if not 0.0 < lo < hi:
        raise ValueError("need 0 < lo < hi")

    history = []

    def evaluate(lam):
        # A strong penalty legitimately closes every gate, and the selector
        # then REFUSES rather than inventing a region. That is correct
        # behaviour, and for a function whose whole job is to sweep lam it is
        # a data point, not an error: the default bracket reaches lam = 1.0,
        # which refuses on most datasets, so propagating the exception made
        # lam_for_coverage unusable at its own defaults (found 2026-09-10
        # while testing the documented recipe). A refusal is recorded as nan
        # coverage, which _distance already scores as infinitely far, so it
        # can never win best-point tracking. Any other exception still
        # propagates; only the library's own "not enough evidence" is data.
        try:
            cov = make_selector(float(lam)).coverage()
            refused = False
        except InsufficientEvidenceError:
            cov, refused = float("nan"), True
        history.append({"lam": float(lam), "coverage": cov, "refused": refused})
        return cov, refused

    def _distance(row):
        # a nan coverage (degenerate selector) must never win best-point
        # tracking (2026-09-01 sweep: it displaced a within-tolerance hit)
        d = abs(row["coverage"] - target)
        return d if np.isfinite(d) else np.inf

    cov_lo, _ = evaluate(lo)         # weakest penalty: highest coverage
    cov_hi, hi_refused = evaluate(hi)  # strongest penalty: lowest coverage
    # A refusal at the strong end is not a failed bracket: it means the
    # penalty closed everything, so achievable coverage there is 0 and the
    # target is bracketed after all. Treating it as an unusable bracket made
    # the search give up and return the WEAKEST lam, reporting coverage 0.94
    # for a target of 0.25 that was reachable at lam 0.02.
    if hi_refused:
        cov_hi = 0.0
    best = min(history, key=_distance)

    def _result():
        reached = _distance(best) <= tol
        return {"lam": best["lam"], "coverage": best["coverage"],
                "target_reached": bool(reached), "history": history}

    if not (np.isfinite(cov_lo) and np.isfinite(cov_hi)
            and cov_hi <= target <= cov_lo):
        return _result()

    a, b = np.log(lo), np.log(hi)
    for _ in range(n_iter):
        mid = 0.5 * (a + b)
        cov, refused = evaluate(float(np.exp(mid)))
        best = min(history, key=_distance)
        if np.isfinite(cov) and abs(cov - target) <= tol:
            break
        if refused:
            # refusal means the penalty is TOO STRONG, so weaken it. The
            # original branch lumped this in with over-covering and
            # strengthened instead, walking the search further into the
            # region where every fit refuses.
            b = mid
        elif not np.isfinite(cov) or cov > target:
            a = mid  # degenerate or over-covering: strengthen the penalty
        else:
            b = mid
    return _result()
