"""Remaining guard, validation and alternative-path branches (98% policy).

test_coverage_gaps.py pinned the branches the milestone suites missed at the
90% policy; this file closes the rest so every module sits at or above 98%.
Three kinds of target, in order:

  * public surface no benchmark happens to call (``decision_function``,
    ``pi_points``, ``RSMResult.predict_proba``);
  * documented alternative paths that the default config never takes
    (``gate="sigmoid"``, ``subsample="complementary"``,
    ``use_unlabelled_stats=True``);
  * guards that need a deliberately hostile input, including the two
    third-party failure modes (a rank test refusing a sample, a CV splitter
    refusing a class count) that RSM must convert into "cannot assess"
    rather than a confident number.

The last group is where the family's bugs have historically hidden, so these
assert the SHAPE of the refusal (nan, "could_not_assess", a raise), never
merely that the call returned.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import robustsignalmaker.metrics as metrics_mod
from robustsignalmaker import (
    EnsembleMaskFoldEstimator,
    BootstrapMaskSelector,
    InsufficientEvidenceError,
    NestedCV,
    RandomConv1D,
    RSMResult,
    SavGolDerivative,
    SegmentGrid,
    SegmentMean,
    SoftMaskFoldEstimator,
    SoftMaskSelector,
    default_ensemble,
    drop_labels,
    loss_and_dz,
    make_signal_control,
    masked_standardise_fit,
    missingness_association,
    paired_comparison,
    resample_indices,
)
from robustsignalmaker.nested_cv import _score_with_refusals

FAST = dict(segment=8, lam=0.05, n_iter=150, lr=0.1, seed=0)
FAST_BOOT = dict(segment=8, lam=0.05, n_iter=120, lr=0.1, n_bootstrap=6, seed=0)
# the fold estimators are handed their seed by the engine, so theirs omit it
FAST_NOSEED = {k: v for k, v in FAST.items() if k != "seed"}
FAST_BOOT_NOSEED = {k: v for k, v in FAST_BOOT.items() if k != "seed"}


def _control(**kw):
    kw.setdefault("n", 60)
    kw.setdefault("n_points", 64)
    kw.setdefault("cell", 8)
    kw.setdefault("region_cells", 2)
    kw.setdefault("distractor_cells", 2)
    kw.setdefault("seed", 0)
    return make_signal_control(**kw)


# ---------------------------------------------------------------------------
# validity: validation and the two "cannot assess" escapes
# ---------------------------------------------------------------------------

def test_standardise_fit_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="W shape"):
        masked_standardise_fit(np.ones((5, 3)), np.ones((5, 2)))


def test_mnar_guard_refuses_when_the_observed_score_is_undefined(monkeypatch):
    """An undefined score on the real labels must not become a p-value."""
    V_seg = np.zeros((20, 3))
    V_seg[::2] = 1.0                      # varies between samples, so it is tested
    y = np.tile([0, 1], 10)
    monkeypatch.setattr(metrics_mod, "score_predictions",
                        lambda *a, **k: float("nan"))
    out = missingness_association(V_seg, y, task="binary", n_permutations=3)
    assert out["verdict"] == "could_not_assess"
    assert np.isnan(out["score"]) and np.isnan(out["p_value"])


def test_mnar_guard_refuses_when_every_permutation_is_undefined(monkeypatch):
    """Observed score fine, null undefined: still no p-value, by design."""
    V_seg = np.zeros((20, 3))
    V_seg[::2] = 1.0
    y = np.tile([0, 1], 10)
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        return 0.9 if calls["n"] == 1 else float("nan")

    monkeypatch.setattr(metrics_mod, "score_predictions", flaky)
    out = missingness_association(V_seg, y, task="binary", n_permutations=4)
    assert out["verdict"] == "could_not_assess"
    assert np.isnan(out["p_value"])


def test_mnar_guard_survives_a_splitter_that_refuses_the_class_counts():
    """A splitter that refuses the request (here n_splits=1) must come back as
    "could_not_assess", not as a propagated sklearn error."""
    rng = np.random.default_rng(0)
    V_seg = rng.random((14, 4))
    y = np.tile([0, 1], 7)
    out = missingness_association(V_seg, y, task="binary",
                                  n_permutations=3, n_splits=1)
    assert out["verdict"] == "could_not_assess"
    assert np.isnan(out["p_value"])


# ---------------------------------------------------------------------------
# masking: loss dispatch, lens contract, alternative gate, evidence floors
# ---------------------------------------------------------------------------

def test_loss_and_dz_rejects_an_unknown_task():
    Z = np.zeros((4, 1))
    with pytest.raises(ValueError, match="unknown task"):
        loss_and_dz("survival", Z, np.zeros((4, 1)), np.ones(4, dtype=bool))


def test_a_lens_that_changes_the_segment_count_is_rejected():
    """The lens invariant: a lens may change the features, never the
    coordinate system the answer is expressed in."""

    class WrongWidth:
        def transform(self, X, V, grid):
            F, FV = SegmentMean().transform(X, V, grid)
            return F[:, :-1], FV[:, :-1]

    c = _control(task="binary")
    with pytest.raises(ValueError, match="expected"):
        SoftMaskSelector(task="binary", representation=WrongWidth(),
                         **FAST).fit(c.X, c.y)


def test_sigmoid_gate_is_a_working_alternative_to_hard_concrete():
    c = _control(task="binary")
    sel = SoftMaskSelector(task="binary", gate="sigmoid", **FAST).fit(c.X, c.y)
    assert sel.mask_.shape == (sel.grid.n_segments,)
    assert np.all((sel.mask_ >= 0.0) & (sel.mask_ <= 1.0))
    assert np.isfinite(sel.final_loss_) and np.isfinite(sel.head_loss_)


def test_head_refuses_when_every_active_gate_is_closed():
    c = _control(task="binary")
    S = SegmentGrid.from_signal_shape(c.X.shape, segment=8).n_segments
    with pytest.raises(InsufficientEvidenceError, match="gates are closed"):
        SoftMaskSelector(task="binary", fixed_mask=np.zeros(S),
                         **FAST).fit(c.X, c.y)


def test_head_refuses_when_no_sample_clears_the_evidence_floor():
    """rho_min is a floor on observed evidence per SAMPLE; if no sample clears
    it the fit must refuse, not extrapolate from a sliver."""
    c = _control(task="binary")
    V = np.ones_like(c.X, dtype=bool)
    V[0::2, :, 32:] = False                # each sample sees half the axis, but
    V[1::2, :, :32] = False                # every segment is seen by half the rows
    with pytest.raises(InsufficientEvidenceError, match="rho_min"):
        SoftMaskSelector(task="binary", rho_min=0.95, **FAST).fit(c.X, c.y, V)


def test_decision_function_and_selected_points_are_available():
    c = _control(task="binary")
    sel = SoftMaskSelector(task="binary", **FAST).fit(c.X, c.y)
    Z = sel.decision_function(c.X)
    assert Z.shape == (len(c.y), 1)
    pts = sel.selected_points()
    assert pts.ndim == 1 and np.all(pts < c.X.shape[-1])
    assert np.array_equal(np.unique(pts), np.sort(pts))  # sorted, no repeats


# ---------------------------------------------------------------------------
# representations: input contract and ensemble sizing
# ---------------------------------------------------------------------------

def test_lens_input_validation():
    grid = SegmentGrid.from_signal_shape((6, 1, 32), segment=8)
    X = np.zeros((6, 1, 32))
    with pytest.raises(ValueError, match="does not match"):
        SegmentMean().transform(X, np.ones((6, 1, 16)), grid)


def test_savgol_and_randomconv_reject_impossible_geometry():
    with pytest.raises(ValueError, match="0 < deriv"):
        SavGolDerivative(window=7, polyorder=2, deriv=3)
    with pytest.raises(ValueError, match="kernel must be"):
        RandomConv1D(kernel=1)


def test_default_ensemble_pads_beyond_the_named_pool():
    big = default_ensemble(14, seed=0)
    assert len(big) == 14
    assert len({id(l) for l in big}) == 14


# ---------------------------------------------------------------------------
# synthetic: generator contracts
# ---------------------------------------------------------------------------

def test_control_rejects_cells_that_do_not_fit_the_axis():
    with pytest.raises(ValueError, match="do not fit"):
        make_signal_control(n=20, n_points=32, cell=8,
                            region_cells=3, distractor_cells=3)


def test_control_rejects_bad_class_counts_and_tasks():
    with pytest.raises(ValueError, match="n_classes"):
        _control(task="multiclass", n_classes=1)
    with pytest.raises(ValueError, match="task must be"):
        _control(task="ordinal")


def test_drop_labels_rejects_an_out_of_range_fraction():
    with pytest.raises(ValueError, match=r"frac must be in \[0, 1\)"):
        drop_labels(np.arange(10), frac=1.0)


# ---------------------------------------------------------------------------
# selection: resampling modes, strata notes, refusal, point-axis outputs
# ---------------------------------------------------------------------------

def test_unstratified_bootstrap_draws_n_of_n():
    rng = np.random.default_rng(0)
    idx = resample_indices(12, "bootstrap", rng, strat=None)
    assert idx.shape == (12,) and idx.min() >= 0 and idx.max() < 12


def test_complementary_pairs_subsampling_runs_and_covers_the_sample():
    c = _control(task="binary", n=48)
    sel = BootstrapMaskSelector(task="binary", subsample="complementary",
                                **FAST_BOOT).fit(c.X, c.y)
    assert len(sel.resample_sets_) == 6          # 3 pairs -> 6 halves
    assert sel.pi_.shape == (sel.grid.n_segments,)


def test_regression_without_observation_groups_is_unstratified_and_says_so():
    c = _control(task="regression")
    sel = BootstrapMaskSelector(task="regression", **FAST_BOOT).fit(c.X, c.y)
    assert "unstratified" in sel.strata_note_
    assert sel.observation_groups_ is None


def test_pi_points_expands_to_the_point_axis_keeping_nan():
    c = _control(task="binary")
    sel = BootstrapMaskSelector(task="binary", **FAST_BOOT).fit(c.X, c.y)
    pp = sel.pi_points()
    assert pp.shape == (c.X.shape[-1],)
    # unassessable segments stay nan on the point axis: absence of evidence is
    # never expanded into a zero frequency
    if sel.unassessable_segments_.size:
        bad = sel.grid.segments_to_points(sel.unassessable_segments_)
        assert np.all(np.isnan(pp[bad]))


def test_nothing_supported_means_refusal_not_a_fallback_region():
    """With the per-resample support floor set above 1 no segment can ever be
    assessed, so there is no region to fall back to and the fit must raise."""
    c = _control(task="binary")
    with pytest.raises(InsufficientEvidenceError):
        BootstrapMaskSelector(task="binary", o_min=1.01, **FAST_BOOT).fit(c.X, c.y)


def test_ambiguous_fraction_is_nan_when_nothing_is_supported():
    c = _control(task="binary")
    sel = BootstrapMaskSelector(task="binary", **FAST_BOOT).fit(c.X, c.y)
    sel._supported = np.zeros_like(sel._supported)
    assert np.isnan(sel.ambiguous_fraction())


# ---------------------------------------------------------------------------
# metrics: verdict serialisation and third-party test failures
# ---------------------------------------------------------------------------

def test_baseline_comparison_round_trips_to_a_dict():
    cmp = paired_comparison([0.8, 0.82, 0.79, 0.81, 0.80],
                            [0.5, 0.52, 0.49, 0.51, 0.50])
    d = cmp.to_dict()
    assert set(d) >= {"p_wilcoxon", "p_ttest", "mean_delta", "outcome"}
    assert d["outcome"] == cmp.outcome and d["mean_delta"] > 0


def test_a_rank_test_that_refuses_yields_nan_p_not_a_verdict(monkeypatch):
    """Both tests refusing must leave the p-values nan; the outcome then falls
    back to ``preserved`` on no evidence of difference, never to a claim."""
    import scipy.stats as st

    def refuse(*a, **k):
        raise ValueError("sample too small")

    monkeypatch.setattr(st, "wilcoxon", refuse)
    monkeypatch.setattr(st, "ttest_rel", refuse)
    cmp = paired_comparison([0.9, 0.8, 0.7], [0.1, 0.2, 0.3])
    assert np.isnan(cmp.p_wilcoxon) and np.isnan(cmp.p_ttest)
    assert cmp.outcome == "preserved" and cmp.mean_delta > 0


# ---------------------------------------------------------------------------
# nested_cv: refusal scoring, class alignment, adapters, unlabelled stats
# ---------------------------------------------------------------------------

def test_a_fold_that_refuses_every_sample_scores_nan_not_zero():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    s, n_ref = _score_with_refusals("regression", y, pred=np.full(4, np.nan))
    assert np.isnan(s) and n_ref == 4
    s2, n_ref2 = _score_with_refusals("binary", np.array([0, 1, 0, 1]),
                                      proba=np.full((4, 2), np.nan))
    assert np.isnan(s2) and n_ref2 == 4


def test_proba_alignment_without_a_class_list_refuses_the_fold():
    proba = np.array([[0.3, 0.7], [0.6, 0.4]])
    full = NestedCV._aligned_proba(proba, None, np.array([0, 1, 2]))
    assert full.shape == (2, 3) and np.all(np.isnan(full))


def test_fold_estimator_adapters_expose_predict_and_predict_proba():
    c = _control(task="binary")
    V = np.ones_like(c.X, dtype=bool)
    single = SoftMaskFoldEstimator(**FAST_NOSEED)
    single.fit(c.X, c.y, V=V, task="binary", seeds=None, fold_idx=0,
               repeat=0, inner_splitter=None)
    P = single.predict_proba(c.X, V)
    assert P.shape == (len(c.y), 2)

    ens = EnsembleMaskFoldEstimator(n_representations=2, **FAST_BOOT_NOSEED)
    ens.fit(c.X, c.y, V=V, task="binary", seeds=None, fold_idx=0,
            repeat=0, inner_splitter=None)
    pred = ens.predict(c.X, V)
    assert pred.shape == (len(c.y),)


def test_unlabelled_rows_may_feed_x_only_statistics():
    """use_unlabelled_stats routes unlabelled TRAINING rows into the fold fit;
    the labelled folds, and therefore the scores, must be unaffected in shape
    and the unlabelled rows must still be excluded from every partition."""
    c = _control(task="binary", n=64)
    y = drop_labels(c.y, frac=0.25, seed=1)
    n_unlab = int(np.isnan(y).sum())
    assert n_unlab > 0

    cv = NestedCV(estimator_factory=lambda: SoftMaskFoldEstimator(**FAST_NOSEED),
                  k_outer=3, l_inner=2, mnar_guard=False,
                  use_unlabelled_stats=True, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cv.run(c.X, y)
    assert res.n_unlabelled == n_unlab
    assert len(res.per_fold_scores) == 3
    assert res.oof_count.shape[0] == len(y)
    # an unlabelled row is never scored out of fold
    assert np.all(res.oof_count[np.isnan(y)] == 0)


# ---------------------------------------------------------------------------
# results: prediction passthrough, external baseline, RMSE display of it
# ---------------------------------------------------------------------------

def _minimal_result(task, per_fold, baseline=None, verdict=None):
    return RSMResult(
        task=task, random_state=0, per_fold_scores=np.asarray(per_fold),
        baseline_per_fold_scores=None if baseline is None else np.asarray(baseline),
        verdict=verdict, oof_predictions=np.zeros(4), oof_count=np.ones(4, dtype=int),
        n_refused=0, n_unlabelled=0, selected_per_fold=[np.array([0, 1])],
        selection_stability=1.0, selection_stability_adjusted=1.0,
        mnar_reports=[{"verdict": "not_informative"}], fold_records=[],
        final_estimator=None,
    )


def test_result_forwards_prediction_to_the_final_estimator():
    class Stub:
        def predict(self, X, V=None):
            return np.zeros(len(X))

        def predict_proba(self, X, V=None):
            return np.tile([0.4, 0.6], (len(X), 1))

    res = _minimal_result("binary", [0.7, 0.8])
    res.final_estimator = Stub()
    assert res.predict(np.zeros((3, 1, 8))).shape == (3,)
    assert res.predict_proba(np.zeros((3, 1, 8))).shape == (3, 2)


def test_result_compares_against_an_external_baseline():
    # eight folds: at five the two-sided Wilcoxon floor is 0.0625 and no real
    # advantage can reach alpha (the RMM caveat recorded in metrics.py)
    res = _minimal_result("binary", [0.90, 0.91, 0.89, 0.92, 0.90, 0.88, 0.93, 0.91])
    cmp = res.compare_to_baseline([0.60, 0.61, 0.59, 0.62, 0.60, 0.58, 0.63, 0.61])
    assert cmp.outcome == "sig.better" and cmp.mean_delta > 0


def test_regression_baseline_is_summarised_in_rmse_not_the_internal_sign():
    """score_mean is displayed as RMSE for regression, so the baseline beside
    it must be converted too; showing the internal higher-is-better number
    next to an RMSE would invert the comparison a reader makes by eye."""
    scores = [-2.0, -2.0, -2.0]            # internal convention: negative RMSE
    baseline = [-4.0, -4.0, -4.0]
    verdict = paired_comparison(scores, baseline)
    res = _minimal_result("regression", scores, baseline, verdict)
    s = res.summary()
    assert s["score_name"] == "RMSE"
    assert np.isclose(s["score_mean"], 2.0)
    assert np.isclose(s["baseline_score_mean"], 4.0)   # RMSE, not -4.0


def test_selected_points_expands_the_stable_region_to_the_point_axis():
    c = _control(task="binary")
    sel = BootstrapMaskSelector(task="binary", **FAST_BOOT).fit(c.X, c.y)
    pts = sel.selected_points()
    expected = len(sel.selected_segments_) * sel.grid.segment
    assert pts.shape == (expected,)
    assert np.array_equal(pts, np.sort(pts))


def test_an_unreachable_threshold_degrades_to_the_single_best_segment():
    """tau = 1.0 demands unanimity across resamples. When nothing is unanimous
    but something was selected sometimes, the fit falls back to the single
    highest-frequency segment AND flags itself degenerate, so the caller can
    see that the threshold, not the data, chose the answer."""
    # a weak signal makes the resamples disagree, so nothing is unanimous
    c = _control(task="binary", n=48, signal=0.1, seed=5)
    sel = BootstrapMaskSelector(task="binary", tau=1.0, **FAST_BOOT).fit(c.X, c.y)
    assert sel.degenerate_ is True
    assert len(sel.selected_segments_) == 1
    best = int(sel.selected_segments_[0])
    assert sel.pi_[best] == np.nanmax(sel.pi_)
    assert sel.pi_[best] < 1.0            # genuinely below the threshold asked for
    assert sel.stability_report()["degenerate"] is True


# ---------------------------------------------------------------------------
# the contrast-branch denominator is a MEASURED choice; pin it (FINDINGS sec.12)
# ---------------------------------------------------------------------------

def test_contrast_branch_normalises_by_l1_mass_not_l2_or_count():
    """RSM normalises contrast filters by observed ABSOLUTE (L1) filter mass.

    RobustPixelMaker uses L2 for the same branch, and porting that rule here
    was measured and REJECTED: on truth recovery of the selection it lost to
    L1 21 to 9 across 10 seeds (FINDINGS sec.12). Because both rules are
    defensible in the abstract and only measurement separates them, a silent
    swap is exactly the kind of change that would not show up as a test
    failure anywhere else, while quietly moving selections on 8 of 12
    configurations. So the denominator itself is asserted here.
    """
    from scipy.ndimage import convolve1d

    from robustsignalmaker import renormalised_convolve1d

    rng = np.random.default_rng(11)
    X = rng.normal(size=(4, 1, 64)) * 3.0
    Vm = (rng.random(X.shape) > 0.25).astype(float)
    filt = rng.normal(size=9)
    filt -= filt.mean()                      # zero-sum: the contrast path

    out, v_out = renormalised_convolve1d(X, Vm, filt)

    ones = np.ones_like(filt)
    Xz = np.where(Vm > 0, X, 0.0)
    num = convolve1d(Xz * Vm, filt, axis=-1, mode="reflect")
    evidence = convolve1d(Vm, ones, axis=-1, mode="reflect")
    inv = np.where(evidence > 0, 1.0 / np.where(evidence > 0, evidence, 1.0), 0.0)
    w = convolve1d(Vm, filt, axis=-1, mode="reflect")
    num = num - convolve1d(Xz * Vm, ones, axis=-1, mode="reflect") * inv * w

    l1 = convolve1d(Vm, np.abs(filt), axis=-1, mode="reflect")
    l2 = np.sqrt(np.maximum(convolve1d(Vm, filt ** 2, axis=-1, mode="reflect"), 0.0))
    ref_l1 = num * np.abs(filt).sum() / np.where(l1 > 0, l1, 1.0)
    ref_l2 = num * np.sqrt((filt ** 2).sum()) / np.where(l2 > 0, l2, 1.0)

    live = v_out > 0
    assert live.any()
    np.testing.assert_allclose(out[live], ref_l1[live], rtol=1e-10)
    # and the two rules genuinely differ here, so the assertion above has teeth
    assert not np.allclose(ref_l1[live], ref_l2[live], rtol=1e-3)


def test_level_branch_normalises_by_observed_filter_weight():
    """The level branch IS an identity: a constant returns exactly, any kernel.

    Unlike the contrast rule this is not a calibration choice, so it is
    asserted to 1e-12 on a deliberately non-uniform kernel.
    """
    from robustsignalmaker import renormalised_convolve1d

    a = np.arange(11) - 5.0
    g = np.exp(-(a ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()                          # non-uniform, sums to 1: level

    X = np.full((3, 1, 64), 7.0)
    Vm = np.ones_like(X)
    Vm[:, :, :20] = 0.0                      # a straight gap edge

    out, v_out = renormalised_convolve1d(X, Vm, g)
    live = v_out > 0
    assert live.any()
    np.testing.assert_allclose(out[live], 7.0, atol=1e-12)


# ---------------------------------------------------------------------------
# lam_for_coverage must survive a lam strong enough to make the selector refuse
# ---------------------------------------------------------------------------

class _FakeSelector:
    def __init__(self, coverage):
        self._c = coverage

    def coverage(self):
        return self._c


def _refusing_maker(refuse_above):
    """coverage falls smoothly with lam, then the selector refuses entirely."""
    def make(lam):
        if lam >= refuse_above:
            raise InsufficientEvidenceError("all gates closed at this penalty")
        return _FakeSelector(max(0.0, min(1.0, 0.05 / lam)))
    return make


def test_lam_for_coverage_does_not_crash_when_a_penalty_refuses():
    """A refusal is a data point for a lam sweep, not an error to propagate.

    The default bracket reaches lam = 1.0, which closes every gate on most
    datasets, so propagating the exception made this function unusable at
    its own defaults.
    """
    from robustsignalmaker import lam_for_coverage

    out = lam_for_coverage(_refusing_maker(0.5), target=0.25)
    assert any(row["refused"] for row in out["history"])
    assert np.isnan([r["coverage"] for r in out["history"] if r["refused"]][0])


def test_a_refusal_at_the_strong_end_still_brackets_the_target():
    """Refusal at hi means achievable coverage there is 0, so the target IS
    bracketed; treating it as an unusable bracket made the search give up and
    return the weakest lam."""
    from robustsignalmaker import lam_for_coverage

    out = lam_for_coverage(_refusing_maker(0.5), target=0.25, tol=0.02)
    assert out["target_reached"] is True
    assert abs(out["coverage"] - 0.25) <= 0.02
    assert out["lam"] > 1e-3            # it searched rather than returning lo


def test_a_refusal_mid_search_weakens_the_penalty():
    """Refusal means TOO STRONG. The original branch lumped it in with
    over-covering and strengthened, walking further into the refusing region."""
    from robustsignalmaker import lam_for_coverage

    # refuses over most of the bracket, so the search must walk back down
    out = lam_for_coverage(_refusing_maker(0.02), target=0.5, tol=0.05)
    refused = [r["lam"] for r in out["history"] if r["refused"]]
    assert refused, "the probe should have hit the refusing region"
    assert out["lam"] < min(refused), "search should end below the refusals"


def test_a_non_evidence_exception_still_propagates():
    """Only the library's own "not enough evidence" is data; a genuine bug
    in the user's factory must not be swallowed."""
    from robustsignalmaker import lam_for_coverage

    def broken(lam):
        raise KeyError("a real mistake in the caller's factory")

    with pytest.raises(KeyError):
        lam_for_coverage(broken, target=0.25)
