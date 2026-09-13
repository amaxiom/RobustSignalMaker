"""Milestone 5: the leakage contract, PROVED by canaries rather than asserted.

The canary estimator hashes every training row it is fitted on; scoring a row
it has seen raises. A clean nested-CV run therefore certifies structurally
that no test row ever reached a fit. A second canary certifies the missing-y
rule (an unlabelled row appears in no fit and no test partition), and a third
that a group never spans train and test.
"""
from __future__ import annotations

import numpy as np
import pytest

from robustsignalmaker import NestedCV, make_signal_control


class CanaryFoldEstimator:
    """Hashes training rows in fit; refuses to score any row it trained on."""

    def __init__(self, registry=None):
        self.registry = registry  # shared across folds when provided
        self.selected_ = np.array([], dtype=int)

    def fit(self, X, y, *, V, task, seeds, fold_idx, repeat, inner_splitter,
            groups=None, X_unlabelled=None, V_unlabelled=None):
        self.task = task
        X = np.ascontiguousarray(np.asarray(X, dtype=float))
        self.train_hashes = {row.tobytes() for row in X.reshape(X.shape[0], -1)}
        self.n_classes = 1 if task == "regression" else len(np.unique(y))
        if self.registry is not None:
            self.registry["fitted_rows"] |= self.train_hashes
            if groups is not None:
                self.registry["fit_groups"].append(set(np.asarray(groups).tolist()))
        return self

    def _check(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=float))
        for row in X.reshape(X.shape[0], -1):
            if row.tobytes() in self.train_hashes:
                raise RuntimeError("LEAK: predict saw a row it was trained on")
        return X.shape[0]

    def predict(self, X, V=None):
        return np.zeros(self._check(X))

    def predict_proba(self, X, V=None):
        n = self._check(X)
        return np.full((n, self.n_classes), 1.0 / self.n_classes)


def _control(seed=51):
    return make_signal_control(n=120, n_channels=1, n_points=128, task="binary",
                               signal=2.0, noise=1.0, seed=seed)


def test_canary_certifies_no_test_row_reaches_fit():
    c = _control()
    eng = NestedCV(estimator_factory=CanaryFoldEstimator, baseline=False,
                   mnar_guard=False, k_outer=4, segment=8, random_state=0)
    res = eng.run(c.X, c.y)  # a leak would raise inside scoring
    assert len(res.per_fold_scores) == 4


def test_canary_trips_when_leakage_is_deliberately_injected():
    c = _control()
    canary = CanaryFoldEstimator()
    canary.fit(c.X, c.y, V=np.ones_like(c.X), task="binary", seeds=None,
               fold_idx=0, repeat=0, inner_splitter=None)
    with pytest.raises(RuntimeError, match="LEAK"):
        canary.predict(c.X[:5])


def test_missing_y_rows_reach_neither_fit_nor_test_partitions():
    c = _control(seed=52)
    y = c.y.astype(float).copy()
    y[:20] = np.nan
    registry = {"fitted_rows": set(), "fit_groups": []}
    eng = NestedCV(estimator_factory=lambda: CanaryFoldEstimator(registry),
                   baseline=False, mnar_guard=False, k_outer=4, segment=8,
                   random_state=1)
    res = eng.run(c.X, y)
    unlabelled_rows = {np.ascontiguousarray(c.X[i].ravel()).tobytes()
                      for i in range(20)}
    assert not (registry["fitted_rows"] & unlabelled_rows)
    for rec in res.fold_records:
        assert not (set(rec["test_idx"]) & set(range(20)))


def test_groups_never_span_train_and_test():
    c = _control(seed=53)
    groups = np.repeat(np.arange(12), 10)  # 12 subjects, 10 signals each
    registry = {"fitted_rows": set(), "fit_groups": []}
    eng = NestedCV(estimator_factory=lambda: CanaryFoldEstimator(registry),
                   baseline=False, mnar_guard=False, k_outer=4, segment=8,
                   random_state=2)
    res = eng.run(c.X, c.y, groups=groups)
    assert res.config["grouped"] is True
    for rec, fit_groups in zip(res.fold_records, registry["fit_groups"]):
        test_groups = set(groups[rec["test_idx"]].tolist())
        assert not (fit_groups & test_groups), "a group leaked across the split"
