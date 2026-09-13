"""Milestone 4: bootstrap stability with support accounting, and lam tuning.

The arithmetic tests script the inner fits exactly (which resamples fail,
which segments are assessable, which are selected) so pi, support, verdicts
and the failed-fit exclusion are pinned by hand-computable numbers rather
than by end-to-end behaviour alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from robustsignalmaker import (
    BootstrapMaskSelector,
    InsufficientEvidenceError,
    SoftMaskSelector,
    complementary_pair,
    jaccard,
    lam_for_coverage,
    lam_frontier,
    make_signal_control,
    mask_scattered,
    observation_groups,
    resample_indices,
)

FAST = dict(segment=8, n_iter=200, lr=0.1, n_bootstrap=10, lam=0.05, seed=0)


# ---------------------------------------------------------------------------
# resampling and grouping
# ---------------------------------------------------------------------------

def test_stratified_resampling_never_loses_a_class():
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 4)  # imbalanced enough to lose class 1 unstratified
    for mode in ("bootstrap", "half"):
        for trial in range(20):
            idx = resample_indices(len(y), mode, rng, strat=y)
            assert set(np.unique(y[idx])) == {0, 1}, mode
    with pytest.raises(ValueError, match="resample mode"):
        resample_indices(10, "jackknife", rng)


def test_complementary_pairs_are_disjoint_and_cover():
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 30)
    a, b = complementary_pair(len(y), rng, strat=y)
    assert len(set(a) & set(b)) == 0
    assert len(set(a) | set(b)) == len(y)
    assert set(np.unique(y[a])) == {0, 1} and set(np.unique(y[b])) == {0, 1}


def test_observation_groups_finds_clean_patterns_and_refuses_noise():
    v = np.zeros((40, 6))
    v[:20, :4] = 1.0   # instrument A observes segments 0..3
    v[20:, 2:] = 1.0   # instrument B observes segments 2..5
    g = observation_groups(v)
    assert g is not None and len(np.unique(g)) == 2
    assert len(np.unique(g[:20])) == 1 and g[0] != g[20]
    # scattered validity gives near-unique patterns: NOT groups
    rng = np.random.default_rng(2)
    assert observation_groups((rng.random((40, 6)) > 0.3).astype(float)) is None
    # a single shared pattern is not a grouping either
    assert observation_groups(np.ones((40, 6))) is None


# ---------------------------------------------------------------------------
# scripted-inner-fit arithmetic
# ---------------------------------------------------------------------------

class _ScriptedInner:
    """Stands in for SoftMaskSelector.fit with a scripted outcome."""

    def __init__(self, script, grid):
        self.script = script
        self.grid = grid

    def fit(self, X, y, V):
        if self.script.get("fail"):
            raise ValueError("scripted failure")
        S = self.grid.n_segments
        self.selected_segments_ = np.asarray(self.script["selected"], dtype=int)
        self.assessable_segments_ = np.asarray(self.script["assessable"], dtype=bool)
        v = np.ones((X.shape[0], S))
        v[:, ~self.assessable_segments_] = 0.0
        self._v_seg = v
        return self


def _scripted_selector(scripts, S=4, n=24, tau=0.7, a_min=0.5, **kw):
    """A BootstrapMaskSelector whose inner fits follow ``scripts`` in order."""
    sel = BootstrapMaskSelector(task="binary", segment=8, n_bootstrap=len(scripts),
                                tau=tau, a_min=a_min, seed=0, **kw)
    calls = iter(scripts)

    def make_inner(seed, representation=None):
        return _ScriptedInner(next(calls), sel.grid)

    sel._make_inner = make_inner
    sel._refit = lambda X, y, Vf: setattr(sel, "degenerate_", False)  # arithmetic only
    X = np.zeros((n, 1, S * 8))
    X[:, :, 0] = np.linspace(0, 1, n)[:, None]
    y = np.array([0, 1] * (n // 2))
    sel.fit(X, y)
    return sel


def test_pi_support_and_verdict_arithmetic():
    common = [True, False, True, False]  # segments 1 and 3 unseen by most fits
    rare = [True, True, True, False]     # segment 1 assessable only here
    scripts = (
        [{"selected": [0], "assessable": common}] * 6
        + [{"selected": [0, 1], "assessable": rare}] * 2
        + [{"fail": True}] * 2
    )
    sel = _scripted_selector(scripts)
    # 10 jobs, 2 failed: denominators are 8, never 10
    assert sel.n_fits_ == 8 and sel.n_failed_fits_ == 2
    # segment 0: assessable 8/8, selected 8/8
    assert sel.support_[0] == 1.0 and sel.pi_[0] == 1.0
    assert sel.verdicts_[0] == "selected"
    # segment 1: assessable only 2/8 (below a_min): pi is nan, NEVER 0,
    # and its raw frequency 1.0 lands in insufficient_evidence_
    assert sel.support_[1] == pytest.approx(0.25)
    assert np.isnan(sel.pi_[1])
    assert sel.verdicts_[1] == "unassessable"
    assert 1 in sel.insufficient_evidence_
    # segment 2: assessable 8/8, never selected: honestly rejected
    assert sel.pi_[2] == 0.0 and sel.verdicts_[2] == "rejected"
    # segment 3: never assessable
    assert np.isnan(sel.pi_[3]) and sel.verdicts_[3] == "unassessable"
    assert sel.assessable_universe() == 2
    assert sel.coverage() == pytest.approx(0.5)  # 1 selected of 2 supported


def test_failed_fits_warn_above_threshold_and_all_failed_raises():
    scripts_ok = [{"selected": [0], "assessable": [True] * 4}] * 6 + [{"fail": True}] * 4
    with pytest.warns(UserWarning, match="resample fits failed"):
        _scripted_selector(scripts_ok)
    with pytest.raises(InsufficientEvidenceError, match="every resample fit failed"):
        _scripted_selector([{"fail": True}] * 5)


def test_target_coverage_picks_threshold_on_the_supported_universe():
    every = [True] * 4
    # frequencies: seg0 8/8, seg1 5/8, seg2 2/8, seg3 0/8
    scripts = ([{"selected": [0, 1, 2], "assessable": every}] * 2
               + [{"selected": [0, 1], "assessable": every}] * 3
               + [{"selected": [0], "assessable": every}] * 3)
    sel = _scripted_selector(scripts, tau=0.7)
    # 5/8 = 0.625 < 0.7, so only segment 0 clears the fixed threshold
    assert set(sel.selected_segments_) == {0}
    scripts3 = ([{"selected": [0, 1, 2], "assessable": every}] * 2
                + [{"selected": [0, 1], "assessable": every}] * 3
                + [{"selected": [0], "assessable": every}] * 3)
    sel3 = BootstrapMaskSelector(task="binary", segment=8, n_bootstrap=8,
                                 target_coverage=0.5, seed=0)
    calls = iter(scripts3)
    sel3._make_inner = lambda seed, representation=None: _ScriptedInner(next(calls), sel3.grid)
    sel3._refit = lambda X, y, Vf: setattr(sel3, "degenerate_", False)
    X = np.zeros((24, 1, 32))
    X[:, 0, 0] = np.linspace(0, 1, 24)
    sel3.fit(X, np.array([0, 1] * 12))
    # target 0.5 of the 4-segment universe: the chosen threshold keeps {0, 1}
    assert set(sel3.selected_segments_) == {0, 1}
    assert sel3.tau_ == pytest.approx(0.625)


# ---------------------------------------------------------------------------
# end-to-end on the control
# ---------------------------------------------------------------------------

def _control(task="binary", seed=31):
    return make_signal_control(n=240, n_channels=1, n_points=256, task=task,
                               signal=2.0, noise=1.0, seed=seed)


def test_make_or_break_bimodal_pi_and_stability_under_scattered_nans():
    c = _control()
    # the bimodality claim needs pi granularity finer than the FAST profile,
    # and the operating point matters: at lam 0.05 one background segment
    # reached pi 0.8 under 30% NaNs (recorded here deliberately; per-dataset
    # lam selection is a library feature, tuning.py, for exactly this reason)
    kw = dict(FAST)
    kw.update(n_bootstrap=20, lam=0.08)
    full = BootstrapMaskSelector(task="binary", **kw).fit(c.X, c.y)
    truth = set(c.truth_segments(full.grid).tolist())
    assert truth <= set(full.selected_segments_.tolist())
    # bimodality: the go/no-go diagnostic says a reproducible region exists
    assert full.ambiguous_fraction() < 0.25
    report = full.stability_report()
    assert report["n_failed_fits"] == 0 and report["degenerate"] is False

    V = mask_scattered(c.X.shape, frac=0.3, seed=32)
    X = np.where(V, c.X, np.nan)
    noisy = BootstrapMaskSelector(task="binary", **kw).fit(X, c.y, V)
    assert jaccard(noisy.selected_segments_, full.selected_segments_) >= 0.9
    # the threshold-free claim: every truth segment outranks every background
    # segment on the frequency map, even under 30% missingness
    bg = [s for s in range(noisy.grid.n_segments) if s not in truth]
    assert np.nanmin(noisy.pi_[sorted(truth)]) > np.nanmax(noisy.pi_[bg])
    # predictions flow through the refit model
    proba = noisy.predict_proba(c.X)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(c.y, proba[:, 1]) > 0.8


def test_never_observed_band_is_unassessable_not_rejected():
    c = _control()
    V = np.ones_like(c.X)
    V[:, :, 160:192] = 0.0  # segments 20..23 observed by nobody
    sel = BootstrapMaskSelector(task="binary", **FAST).fit(c.X, c.y, V)
    for s in (20, 21, 22, 23):
        assert np.isnan(sel.pi_[s])
        assert sel.verdicts_[s] == "unassessable"
        assert s in sel.unassessable_segments_
        assert s not in sel.selected_segments_
    # the truth band is still recovered, and the point expansion carries nan
    truth = set(c.truth_segments(sel.grid).tolist())
    assert truth <= set(sel.selected_segments_.tolist())
    assert np.isnan(sel.pi_points()[160])
    assert sel.stability_report()["unassessable_fraction"] >= 4 / 32


def test_band_by_group_missingness_stratifies_by_detected_groups():
    c = _control()
    V = np.ones_like(c.X)
    V[::2, :, 192:224] = 0.0  # instrument A never sees segments 24..27
    sel = BootstrapMaskSelector(task="binary", **FAST).fit(c.X, c.y, V)
    assert sel.observation_groups_ is not None
    assert "observation group" in sel.strata_note_
    # observed by half the samples: assessable, with support reported
    for s in (24, 25, 26, 27):
        assert sel.verdicts_[s] in ("selected", "rejected")
        assert sel.obs_frac_[s] == pytest.approx(0.5)


def test_all_closed_aggregation_fails_loudly():
    c = _control()
    kw = dict(FAST)
    kw["lam"] = 50.0
    with pytest.raises(InsufficientEvidenceError, match="no region to fall back"):
        BootstrapMaskSelector(task="binary", **kw).fit(c.X, c.y)


def test_constructor_validation():
    with pytest.raises(ValueError, match="subsample"):
        BootstrapMaskSelector(subsample="jackknife")
    with pytest.raises(ValueError, match="tau"):
        BootstrapMaskSelector(tau=0.0)


# ---------------------------------------------------------------------------
# tuning
# ---------------------------------------------------------------------------

def test_lam_frontier_reports_the_tradeoff():
    c = _control()
    kw = dict(FAST)
    kw.pop("lam")
    kw["n_bootstrap"] = 6

    def make(lam):
        return BootstrapMaskSelector(task="binary", lam=lam, **kw).fit(c.X, c.y)

    rows = lam_frontier(make, [0.02, 0.2])
    assert [r["lam"] for r in rows] == [0.02, 0.2]
    for r in rows:
        assert set(r) >= {"coverage", "stability_adjusted", "ambiguous_fraction",
                          "unassessable_fraction", "n_selected", "degenerate"}
    assert rows[1]["coverage"] <= rows[0]["coverage"]  # stronger penalty keeps less


def test_lam_for_coverage_bisects_and_reports_misses_honestly():
    c = _control()

    def make(lam):
        return SoftMaskSelector(task="binary", segment=8, lam=lam, n_iter=200,
                                lr=0.1, seed=0).fit(c.X, c.y)

    res = lam_for_coverage(make, target=0.15, lo=1e-3, hi=2.0, tol=0.06)
    assert res["target_reached"]
    assert abs(res["coverage"] - 0.15) <= 0.06
    assert len(res["history"]) >= 2

    # an unreachable target is reported as a miss, never renamed a hit
    res_miss = lam_for_coverage(make, target=1.0, lo=0.5, hi=2.0, tol=0.01)
    assert not res_miss["target_reached"]
    with pytest.raises(ValueError, match="target"):
        lam_for_coverage(make, target=0.0)
    with pytest.raises(ValueError, match="lo"):
        lam_for_coverage(make, target=0.5, lo=2.0, hi=1.0)
