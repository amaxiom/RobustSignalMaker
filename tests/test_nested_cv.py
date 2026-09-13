"""Milestone 5: the nested-CV wrap, verdict pipeline, and result object."""
from __future__ import annotations

import numpy as np
import pytest

from robustsignalmaker import (
    MissingnessInformativeWarning,
    NestedCV,
    RSMResult,
    SoftMaskFoldEstimator,
    infer_task,
    make_signal_control,
    make_outer_splitter,
)

FAST_SEL = dict(lam=0.08, n_iter=150, lr=0.1, n_bootstrap=8)  # segment comes from NestedCV


def _control(task="binary", n=240, seed=41):
    return make_signal_control(n=n, n_channels=1, n_points=256, task=task,
                               signal=2.0, noise=1.0, seed=seed)


def test_infer_task():
    assert infer_task(np.array([0, 1, 1, 0])) == "binary"
    assert infer_task(np.array([0, 1, 2, 0, 1, 2])) == "multiclass"
    assert infer_task(np.random.default_rng(0).standard_normal(100)) == "regression"
    # float-encoded class labels are classification, and nan y is ignored
    y = np.array([0.0, 1.0, np.nan, 1.0, 0.0] * 10)
    assert infer_task(y) == "binary"


def test_grouped_splitters_are_group_safe_types():
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    assert isinstance(make_outer_splitter("binary", np.zeros(10), 3, 0),
                      StratifiedGroupKFold)
    assert isinstance(make_outer_splitter("regression", np.zeros(10), 3, 0),
                      GroupKFold)


def test_make_or_break_preserved_verdict_at_low_coverage(tmp_path):
    c = _control("binary")
    eng = NestedCV(k_outer=4, segment=8, mnar_permutations=30,
                   random_state=0, **FAST_SEL)
    res = eng.run(c.X, c.y)

    # the claim: performance preserved (or better) at meaningfully reduced coverage
    assert res.verdict is not None
    assert res.verdict.outcome in ("preserved", "sig.better")
    mean_cov = np.mean([len(s) for s in res.selected_per_fold]) / 32
    assert mean_cov < 0.3
    assert res.mean_score > 0.85  # AUC on the control
    # per-fold selections are stable, and the truth band recurs
    assert res.selection_stability >= 0.7
    truth = set(c.truth_segments(res.final_estimator.selector_.grid).tolist())
    assert truth <= set(res.final_estimator.selected_.tolist())
    # every labelled sample got an out-of-fold prediction
    assert (res.oof_count >= 1).all()
    # MNAR guard ran and found nothing (missingness is not informative here)
    assert not res.mnar_flag
    assert all(r["mnar_verdict"] == "not_informative" for r in res.fold_records)

    # export round-trip: one CSV per table, JSON, pickle
    out = res.save(tmp_path, prefix="ctl")
    for name in ("ctl_summary.json", "ctl_overview.csv", "ctl_folds.csv",
                 "ctl_selected_segments.csv", "ctl_mnar_guard.csv", "ctl_result.pkl"):
        assert (out / name).exists(), name
    loaded = RSMResult.load(out / "ctl_result.pkl")
    assert np.array_equal(loaded.per_fold_scores, res.per_fold_scores)
    assert loaded.summary()["verdict"] == res.verdict.outcome
    # the final estimator predicts through the result object
    proba = loaded.predict_proba(c.X)
    assert proba.shape == (len(c.y), 2)


def test_regression_with_single_mask_estimator_and_missing_y():
    c = _control("regression", seed=42)
    y = c.y.astype(float).copy()
    y[:30] = np.nan  # thirty samples have signals but no target
    eng = NestedCV(estimator_factory=lambda: SoftMaskFoldEstimator(
                       segment=8, lam=0.05, n_iter=200, lr=0.1),
                   k_outer=4, segment=8, mnar_guard=False, random_state=1)
    res = eng.run(c.X, y)
    assert res.task == "regression"
    assert res.n_unlabelled == 30
    unlabelled = set(range(30))
    for rec in res.fold_records:
        assert not (set(rec["test_idx"]) & unlabelled)  # never in a test partition
    assert (res.oof_count[:30] == 0).all()   # and never given an OOF prediction
    assert (res.oof_count[30:] >= 1).all()
    val, sd, name = res.display_score()
    assert name == "RMSE" and np.isfinite(val)
    assert res.verdict is not None


def test_mnar_guard_stamps_informative_missingness():
    c = _control("binary", seed=43)
    V = np.ones_like(c.X)
    V[c.y == 1, :, 224:256] = 0.0  # class 1 signals lack the last four segments
    eng = NestedCV(k_outer=3, segment=8, mnar_permutations=40,
                   random_state=2, **FAST_SEL)
    with pytest.warns(MissingnessInformativeWarning):
        res = eng.run(c.X, c.y, V)
    assert res.mnar_flag
    assert res.summary()["missingness_informative"] is True
    informative = [r for r in res.mnar_reports if r["verdict"] == "informative"]
    assert informative
    flagged = {s for r in informative for s in r["top_segments"]}
    assert flagged & {28, 29, 30, 31}  # the guard points at the right bands


def test_input_validation_fails_loudly():
    c = _control("binary")
    eng = NestedCV(k_outer=5, **FAST_SEL)
    with pytest.raises(ValueError, match="rows"):
        eng.run(c.X, c.y[:-3])
    y = c.y.astype(float).copy()
    y[8:] = np.nan
    with pytest.raises(ValueError, match="labelled"):
        eng.run(c.X, y)
