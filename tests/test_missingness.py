"""Milestone 7: systematic missingness regimes, the novelty experiment.

The claim lives or dies here, and it is pinned scenario by scenario, with
the measured boundary recorded in benchmarks/FINDINGS.md sec.2 rather than
smoothed over:

  A. NEVER-OBSERVED BAND: the masked pipeline refuses (verdicts
     unassessable), while interpolate-then-screen confidently mis-selects
     fabricated background (huge univariate F on segments nobody measured).
     Two positive side-findings are also pinned: RSM's own gated selector
     RESISTS interpolation fabrication (the frozen-head redundancy pruning),
     and the degenerate-scale guard catches mean-filled constants.
  B. GROUP BAND MISSINGNESS (ignorable): masked recovers the full truth
     band; interpolation is strictly worse on recovery and stability.
     Mean-impute TIES on recovery and is MORE split-half stable at this
     sample size; that is the honest bias-variance boundary (imputation is a
     legitimate variance reduction when missingness is ignorable), asserted
     as such, not hidden.
  C. NON-IGNORABLE (MNAR) MISSINGNESS: the masked pipeline raises the alarm
     and points at the right bands; imputation DESTROYS the alarm (after
     filling, the guard has no observation pattern left to test).
"""
from __future__ import annotations

import numpy as np
import pytest

from robustsignalmaker import (
    BootstrapMaskSelector,
    SegmentGrid,
    SegmentMean,
    drop_labels,
    make_signal_control,
    mask_band_by_group,
    mask_dropout_stretches,
    mask_saturation_censor,
    mean_pairwise_jaccard,
    missingness_association,
    resample_indices,
)

KW = dict(segment=8, lam=0.08, n_iter=200, lr=0.1, n_bootstrap=10, seed=0)


def interp_fill(X, V):
    """Linear interpolation across gaps: the typical quiet fabrication."""
    Xf = np.array(X, dtype=float)
    n, C, T = Xf.shape
    t = np.arange(T)
    for i in range(n):
        for c in range(C):
            obs = np.flatnonzero(V[i, c])
            Xf[i, c] = np.interp(t, obs, Xf[i, c, obs])
    return Xf


def mean_fill(X, V):
    """Per-point mean over observed samples written into the gaps."""
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    num = (X * V).sum(axis=0)
    den = V.sum(axis=0)
    mu = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return np.where(V > 0, X, mu[None])


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------

def test_dropout_stretches_are_contiguous_bounded_and_deterministic():
    V = mask_dropout_stretches((30, 2, 100), n_stretches=1, min_len=10,
                               max_len=20, seed=5)
    assert V.shape == (30, 2, 100) and V.dtype == bool
    assert np.array_equal(V, mask_dropout_stretches((30, 2, 100), 1, 10, 20, seed=5))
    for i in range(30):
        for c in range(2):
            gaps = np.flatnonzero(~V[i, c])
            assert 10 <= len(gaps) <= 20
            assert (np.diff(gaps) == 1).all()  # one contiguous run
    with pytest.raises(ValueError, match="min_len"):
        mask_dropout_stretches((3, 1, 10), min_len=0)


def test_band_by_group_is_exact():
    groups = np.array([0, 0, 1, 1, 2])
    V = mask_band_by_group((5, 2, 20), groups, {0: [(0, 5)], 1: [(5, 10), (15, 20)]})
    assert not V[0, :, 0:5].any() and V[0, :, 5:].all()
    assert V[2, :, 0:5].all() and not V[2, :, 5:10].any() and not V[2, :, 15:].any()
    assert V[4].all()  # group 2 unnamed: observes everything
    with pytest.raises(ValueError, match="does not fit"):
        mask_band_by_group((5, 2, 20), groups, {0: [(10, 25)]})
    with pytest.raises(ValueError, match="groups"):
        mask_band_by_group((5, 2, 20), groups[:3], {})


def test_saturation_censor_clips_and_marks_missing():
    X = np.array([[[0.0, 5.0, 10.0, -3.0]]])
    Xc, V = mask_saturation_censor(X, lower=-1.0, upper=8.0)
    assert Xc[0, 0].tolist() == [0.0, 5.0, 8.0, -1.0]
    assert V[0, 0].tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="at least one"):
        mask_saturation_censor(X)


def test_drop_labels_is_stratified_and_never_empties_a_class():
    y = np.array([0] * 50 + [1] * 6)
    yd = drop_labels(y, frac=0.5, seed=6)
    assert np.isnan(yd).sum() > 0
    for cls in (0, 1):
        assert np.sum(yd[y == cls] == cls) >= 1
    yr = drop_labels(np.random.default_rng(0).standard_normal(100), frac=0.2, seed=7)
    assert np.isnan(yr).sum() == 20


# ---------------------------------------------------------------------------
# scenario A: the never-observed band
# ---------------------------------------------------------------------------

def _scenario_a():
    """A wide bump (points 64..96, amplitude drives y) with points 88..120
    (segments 11..14) observed by NOBODY."""
    rng = np.random.default_rng(70)
    n, T = 240, 256
    a = np.abs(rng.standard_normal(n))
    t = np.arange(T)
    bump = np.zeros(T)
    sup = (t >= 64) & (t < 96)
    bump[sup] = 0.5 * (1 - np.cos(2 * np.pi * (t[sup] - 64 + 0.5) / 32))
    X = rng.normal(size=(n, 1, T)) + 2.0 * a[:, None, None] * bump[None, None, :]
    y = (a > np.median(a)).astype(int)
    V = mask_band_by_group(X.shape, np.zeros(n, dtype=int), {0: [(88, 120)]})
    return X, y, V


def test_make_or_break_A_refusal_vs_confident_fabrication():
    X, y, V = _scenario_a()
    never = [11, 12, 13, 14]

    masked = BootstrapMaskSelector(task="binary", **KW).fit(np.where(V, X, 0.0), y, V)
    for s in never:
        assert masked.verdicts_[s] == "unassessable"
        assert np.isnan(masked.pi_[s])
    assert set(masked.selected_segments_) <= {8, 9, 10}  # observed truth only

    # the standard pipeline: interpolate, then a univariate screen. It ranks
    # fabricated segments among its strongest findings, including segment 12,
    # which is PURE BACKGROUND nobody ever measured.
    from sklearn.feature_selection import f_classif

    grid = SegmentGrid(256, segment=8)
    F, _ = SegmentMean().transform(interp_fill(X, V), np.ones_like(X), grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        fstat, _ = f_classif(F[:, :, 0], y)
    top4 = set(np.argsort(fstat)[::-1][:4].tolist())
    assert top4 & {11, 12}, f"screen top4 {top4} should contain fabricated bands"
    assert 12 in np.argsort(fstat)[::-1][:6]  # the pure-background fabrication
    background = [s for s in range(16, 32)]
    # and it looks STRONG: several times any genuinely-background F statistic
    assert fstat[12] > 5 * np.nanmax(fstat[background])

    # positive finding: RSM's own gated selector resists the same fabrication
    # (a frozen head prunes segments that are redundant with the real band)
    interp_sel = BootstrapMaskSelector(task="binary", **KW).fit(interp_fill(X, V), y)
    assert not (set(interp_sel.selected_segments_) & {12, 13, 14})

    # and mean-filled constants trip the degenerate-scale guard: unassessable
    mean_sel = BootstrapMaskSelector(task="binary", **KW).fit(mean_fill(X, V), y)
    assert all(np.isnan(mean_sel.pi_[s]) for s in never)


# ---------------------------------------------------------------------------
# scenario B: ignorable group-band missingness
# ---------------------------------------------------------------------------

def _replicate_stability(X, y, V, n_rep=3):
    kw = dict(KW)
    kw.update(n_bootstrap=8, n_iter=150)
    sets = []
    for r in range(n_rep):
        rng = np.random.default_rng(200 + r)
        idx = resample_indices(len(y), "half", rng, strat=y)
        kw["seed"] = r
        sel = BootstrapMaskSelector(task="binary", **kw).fit(
            X[idx], y[idx], None if V is None else V[idx])
        sets.append(sel.selected_segments_)
    return mean_pairwise_jaccard(sets, n_total=32)


def _f1(selected, truth):
    s = set(np.asarray(selected).tolist())
    tp = len(s & truth)
    return 2 * tp / (len(s) + len(truth)) if s else 0.0


def test_make_or_break_B_truth_recovery_under_group_bands():
    c = make_signal_control(n=240, n_points=256, task="binary", signal=2.0,
                            noise=1.0, seed=71)
    groups = np.arange(240) % 2
    V = mask_band_by_group(c.X.shape, groups, {0: [(64, 80)]})  # half lack segs 8,9
    truth = {8, 9, 10}
    kw = dict(KW)
    kw.update(n_bootstrap=8, n_iter=150)

    masked = BootstrapMaskSelector(task="binary", **kw).fit(np.where(V, c.X, 0.0), c.y, V)
    interp = BootstrapMaskSelector(task="binary", **kw).fit(interp_fill(c.X, V), c.y)
    meanf = BootstrapMaskSelector(task="binary", **kw).fit(mean_fill(c.X, V), c.y)

    f1_masked, f1_interp, f1_mean = (_f1(s.selected_segments_, truth)
                                     for s in (masked, interp, meanf))
    assert f1_masked == 1.0  # full truth recovered from half the evidence
    assert f1_masked >= f1_mean
    assert f1_masked > f1_interp  # interpolation is strictly worse

    stab_masked = _replicate_stability(np.where(V, c.X, 0.0), c.y, V)
    stab_interp = _replicate_stability(interp_fill(c.X, V), c.y, None)
    assert stab_masked > stab_interp
    # deliberately NOT asserted: masked vs mean-impute stability. Measured:
    # mean-impute is MORE split-half stable here (0.40 vs 0.23 at n=120
    # halves), because when missingness is ignorable, imputation is a
    # legitimate variance reduction. The boundary is recorded in FINDINGS
    # sec.2; the masked method's case rests on scenarios A and C, where
    # imputation fabricates or launders.


# ---------------------------------------------------------------------------
# scenario C: non-ignorable missingness and the survival of the alarm
# ---------------------------------------------------------------------------

def test_make_or_break_C_imputation_destroys_the_mnar_alarm():
    rng = np.random.default_rng(80)
    X = rng.normal(size=(200, 1, 256))
    y = np.array([0, 1] * 100)
    V = mask_band_by_group(X.shape, y, {1: [(160, 192)]})  # class 1 lacks segs 20..23
    grid = SegmentGrid(256, segment=8)

    v_seg = grid.segment_validity(V.astype(float))
    with pytest.warns(Warning):
        masked_report = missingness_association(v_seg, y, "binary",
                                                n_permutations=50, seed=0)
    assert masked_report["verdict"] == "informative"
    assert set(masked_report["top_segments"].tolist()) & {20, 21, 22, 23}

    # after interpolation the validity pattern is gone: the guard cannot see
    # what it needed, and the pipeline proceeds with no warning at all
    Xi = interp_fill(X, V)
    imputed_report = missingness_association(
        grid.segment_validity(np.ones_like(Xi)), y, "binary",
        n_permutations=50, seed=0)
    assert imputed_report["verdict"] == "not_informative"
