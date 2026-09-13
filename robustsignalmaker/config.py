"""Configuration: user-facing knobs with robust defaults, and the fit-cost model.

Design constraints:
  * numpy-only: no torch, no device management. RPM's device detection is
    deliberately not ported; every fit is a hand-derived-gradient numpy loop.
  * User-defined where possible with robust defaults: every knob here has a
    sane default. The evidence thresholds are scientific choices adopted with
    the user, and were calibrated by the sensitivity sweep in
    Milestone 8 before being considered final.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class RSMConfig:
    """All user-facing knobs with robust defaults.

    Fields beyond the Milestone-1 scaffold are declared here so the config
    surface is stable as later milestones land (RPM convention).
    """

    # --- selection unit (sec.2) ---
    segment: int = 16  # points per segment; 1 gives exact point-wise mode
    channel_mode: str = "joint"  # joint | per_channel (per_channel is a later extension)

    # --- cross-validation ---
    k_outer: int = 5
    l_inner: int = 5
    repeated_outer_cv: int = 1
    n_iter: int = 20  # randomized hyperparameter configurations per inner CV

    # --- stability selection ---
    n_bootstrap: int = 20
    n_representations: int = 4
    tau: float = 0.7  # stability threshold (RMM default)
    lam: float = 0.03  # mask sparsity strength
    subsample: str = "bootstrap"  # bootstrap | half | complementary

    # --- masking / gates ---
    gate: str = "hardconcrete"  # hardconcrete | sigmoid
    tv: float = 0.0  # 1D total-variation prior strength (contiguous bands)

    # --- evidence thresholds (adopted defaults; M8 sensitivity sweep) ---
    min_valid_frac: float = 0.5  # segment observation fraction below which a
    #                              (sample, segment) feature is ineligible
    rho_min: float = 0.1  # per-sample observed-evidence floor; below it a
    #                       prediction is nan, never an amplified guess
    o_min: float = 0.1  # per-resample segment support floor for assessability
    a_min: float = 0.5  # fraction of resamples that must assess a segment
    #                     before pi is reported as a number rather than nan.
    #                     Measured (FINDINGS sec.10): this binds only when
    #                     the observation patterns are too varied to group.
    #                     With groups detected, resampling draws a fixed
    #                     count per stratum, support is deterministic, and
    #                     a_min cannot act.
    o_floor: float = 0.2  # floor for observation-scaled sparsity penalty
    n_min_hard: int = 8  # effective sample size below which a segment-channel
    #                      is unassessable (excluded, reported)
    n_min_warn: int = 30  # effective sample size below which it is flagged

    # --- missing y ---
    use_unlabelled_stats: bool = False  # opt-in: unlabelled TRAINING rows may
    #                                     contribute to X-only statistics

    # --- execution ---
    n_jobs: int = 1
    random_state: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def fit_count(
    k_outer: int,
    repeated_outer_cv: int,
    n_representations: int,
    n_bootstrap: int,
    n_iter: int,
    l_inner: int,
) -> int:
    """Total model-fit count (RPM's extension of RMM Eq.3, unchanged for 1D).

    F = K * R_out * (R_rep * B + n_iter * L + 1)  +  (R_rep * B + n_iter * L + 1)

    The trailing term is the final selection + final inner search + final
    refit on all training data. Use this to size runs before launching.
    """
    per = n_representations * n_bootstrap + n_iter * l_inner + 1
    return k_outer * repeated_outer_cv * per + per
