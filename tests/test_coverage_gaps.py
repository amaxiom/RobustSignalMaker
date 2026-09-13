"""Targeted tests for branches the milestone suites leave uncovered.

Coverage policy is 90% per module (whole-package measurement only); these
pin the guard/error/parallel branches that the behavioural suites do not
naturally reach, so no module sits on the enforcement line.
"""
from __future__ import annotations

import numpy as np
import pytest

from robustsignalmaker import (
    BootstrapMaskSelector,
    InsufficientEvidenceError,
    RepresentationEnsembleSelector,
    RSMResult,
    SegmentGrid,
    complementary_pair,
    fill_mean,
    lam_for_coverage,
    make_signal_control,
    masked_mean,
    masked_std,
    missingness_association,
    renormalised_convolve1d,
    renormalised_dot,
    resample_indices,
)

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# validity: guard and validation branches
# ---------------------------------------------------------------------------

def test_masked_reductions_over_all_axes():
    X = RNG.standard_normal((4, 5))
    W = np.ones_like(X)
    m, ok = masked_mean(X, W)  # axis=None scalar path
    assert bool(ok) and np.isclose(float(m), X.mean())
    s, ok2 = masked_std(X, W)
    assert bool(ok2) and np.isclose(float(s), X.std())


def test_masked_std_strict_raises_without_evidence():
    with pytest.raises(InsufficientEvidenceError):
        masked_std(np.ones((5, 2)), np.zeros((5, 2)), axis=0, strict=True)


def test_renormalised_convolve_validation():
    x = np.zeros(10)
    with pytest.raises(ValueError, match="does not match"):
        renormalised_convolve1d(x, np.ones(9), np.ones(3) / 3)
    with pytest.raises(ValueError, match="1D kernel"):
        renormalised_convolve1d(x, np.ones(10), np.ones((3, 3)))


def test_renormalised_dot_validation():
    U = np.ones((3, 4, 1))
    with pytest.raises(ValueError, match="must be"):
        renormalised_dot(U, np.ones(4), np.ones((3, 4)))
    with pytest.raises(ValueError, match="validity"):
        renormalised_dot(np.ones((3, 4)), np.ones(4), np.ones((3, 5)))
    with pytest.raises(ValueError, match="gate"):
        renormalised_dot(np.ones((3, 4)), np.ones(5), np.ones((3, 4)))


def test_missingness_association_regression_path_and_validation():
    rng = np.random.default_rng(8)
    y = rng.standard_normal(80)
    V = np.clip(rng.random((80, 6)), 0.5, 1.0)
    V[:, 2] = np.clip(0.5 + 0.4 * (y - y.min()) / np.ptp(y), 0, 1)  # tracks y
    report = missingness_association(V, y, "regression", n_permutations=40, seed=0)
    # tightened after the sweep: this fixture DOES produce an informative
    # verdict pointing at segment 2 (a disjunctive assert could never fail)
    assert report["verdict"] == "informative"
    assert 2 in report["top_segments"].tolist()
    with pytest.raises(ValueError, match="V_seg"):
        missingness_association(np.ones((5, 3)), np.ones(4), "regression")


def test_fill_mean_accepts_an_explicit_baseline():
    X = np.zeros((2, 1, 4))
    V = np.zeros_like(X)
    out = fill_mean(X, V, baseline=np.full((1, 4), 7.0))
    assert (out == 7.0).all()


# ---------------------------------------------------------------------------
# segments: validation branches
# ---------------------------------------------------------------------------

def test_segment_grid_shape_guards():
    with pytest.raises(ValueError, match="at least"):
        SegmentGrid.from_signal_shape(())
    g = SegmentGrid(16, segment=4)
    with pytest.raises(ValueError, match="does not match"):
        g.pool(np.zeros((2, 1, 16)), np.zeros((2, 1, 15)))
    with pytest.raises(ValueError, match="expected V"):
        g.segment_validity(np.zeros((2, 16)))
    with pytest.raises(ValueError):
        SegmentGrid(0)


# ---------------------------------------------------------------------------
# selection: unstratified resampling, parallel paths, degenerate branches
# ---------------------------------------------------------------------------

def test_unstratified_resampling_paths():
    rng = np.random.default_rng(9)
    half = resample_indices(20, "half", rng)
    assert len(half) == 10
    a, b = complementary_pair(21, rng)
    assert len(set(a) & set(b)) == 0 and len(a) + len(b) == 21


def _tiny_control():
    return make_signal_control(n=60, n_points=64, cell=4, region_cells=2,
                               distractor_cells=2, task="binary", seed=77)


def test_parallel_n_jobs_path_matches_serial():
    c = _tiny_control()
    kw = dict(task="binary", segment=8, lam=0.08, n_iter=60, lr=0.1,
              n_bootstrap=4, seed=0)
    serial = BootstrapMaskSelector(n_jobs=1, **kw).fit(c.X, c.y)
    parallel = BootstrapMaskSelector(n_jobs=2, **kw).fit(c.X, c.y)
    assert np.array_equal(serial.pi_, parallel.pi_, equal_nan=True)


def test_parallel_ensemble_path_runs():
    c = _tiny_control()
    ens = RepresentationEnsembleSelector(
        task="binary", segment=8, lam=0.08, n_iter=60, lr=0.1,
        n_bootstrap=2, n_representations=2, n_jobs=2, seed=0).fit(c.X, c.y)
    assert len(ens.pi_by_representation_) == 2


def test_choose_tau_falls_back_when_nothing_was_ever_selected():
    sel = BootstrapMaskSelector(tau=0.7, target_coverage=0.5)
    pi_raw = np.zeros(4)
    assert sel._choose_tau(pi_raw, np.ones(4, dtype=bool)) == 0.7


def test_strata_merges_up_when_a_cell_is_too_small():
    c = _tiny_control()
    groups = np.zeros(60, dtype=int)
    groups[0] = 1  # a 1-sample class x group cell
    sel = BootstrapMaskSelector(task="binary", segment=8, lam=0.08, n_iter=60,
                                lr=0.1, n_bootstrap=4, seed=0)
    sel.fit(c.X, c.y, obs_groups=groups)
    assert "class-only" in sel.strata_note_


def test_regression_with_observation_groups_stratifies_by_group_alone():
    c = make_signal_control(n=60, n_points=64, cell=4, region_cells=2,
                            distractor_cells=2, task="regression", seed=78)
    groups = np.arange(60) % 2
    sel = BootstrapMaskSelector(task="regression", segment=8, lam=0.08,
                                n_iter=60, lr=0.1, n_bootstrap=4, seed=0)
    sel.fit(c.X, c.y, obs_groups=groups)
    assert "observation group" in sel.strata_note_


def test_ensemble_records_a_lens_whose_fits_all_fail():
    class BrokenLens:
        name = "broken"

        def transform(self, X, V, grid):
            raise ValueError("scripted lens failure")

    from robustsignalmaker import SegmentMean

    c = _tiny_control()
    with pytest.warns(UserWarning, match="resample fits failed"):
        ens = RepresentationEnsembleSelector(
            task="binary", representations=[SegmentMean(), BrokenLens()],
            segment=8, lam=0.08, n_iter=60, lr=0.1, n_bootstrap=4,
            max_failed_frac=0.3, seed=0).fit(c.X, c.y)  # 4 of 8 fail: above 0.3
    assert np.isnan(ens.pi_by_representation_["broken"]).all()
    assert ens.n_failed_fits_ == 4


# ---------------------------------------------------------------------------
# tuning: the bisection loop walks both directions
# ---------------------------------------------------------------------------

def test_lam_for_coverage_bisects_both_directions():
    class Fake:
        def __init__(self, cov):
            self._c = cov

        def coverage(self):
            return self._c

    calls = []

    def make(lam):
        calls.append(lam)
        return Fake(max(0.0, 1.0 - lam))  # strictly decreasing in lam

    res = lam_for_coverage(make, target=0.5, lo=0.05, hi=1.0, n_iter=12, tol=0.01)
    assert res["target_reached"]
    assert abs(res["coverage"] - 0.5) <= 0.01
    assert len(calls) >= 4  # both bisection branches exercised


# ---------------------------------------------------------------------------
# results: empty tables, passthroughs
# ---------------------------------------------------------------------------

def test_results_with_empty_reports_round_trip(tmp_path):
    res = RSMResult(
        task="regression", random_state=0,
        per_fold_scores=np.array([-1.0, -1.1]),
        baseline_per_fold_scores=np.array([-1.2, -1.3]),
        verdict=None,
        oof_predictions=np.zeros(4), oof_count=np.ones(4, dtype=int),
        n_refused=0, n_unlabelled=0,
        selected_per_fold=[np.array([], dtype=int), np.array([], dtype=int)],
        selection_stability=float("nan"), selection_stability_adjusted=float("nan"),
        mnar_reports=[], fold_records=[
            {"repeat": 0, "fold": 0}, {"repeat": 0, "fold": 1}],
        final_estimator=None)
    tables = res.results_tables()
    assert tables["mnar_guard"] == [] and tables["selected_segments"] == []
    out = res.save(tmp_path, prefix="empty")
    assert (out / "empty_selected_segments.csv").read_text() == ""
    loaded = RSMResult.load(out / "empty_result.pkl")
    val, sd, name = loaded.display_score()
    assert name == "RMSE" and val == pytest.approx(1.05)
