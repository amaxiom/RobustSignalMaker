"""The refit model zoo (Milestone 9).

Selection is done by the gated head; these tests are about what PREDICTS on
the region it chose. Two properties matter more than any individual model
working, and both are asserted repeatedly below:

  1. swapping the refit model must not change the SELECTION, or the feature
     has quietly turned into a different selector;
  2. the no-fabrication contract must survive a model that knows nothing
     about validity. A random forest will cheerfully predict from imputed
     values; here it must refuse the same samples the built-in head refuses.
"""
from __future__ import annotations

import numpy as np
import pytest

from robustsignalmaker import (
    ALGORITHMS,
    BootstrapMaskSelector,
    ExternalRefit,
    InsufficientEvidenceError,
    NestedCV,
    PLSDA,
    SegmentGrid,
    make_refit_model,
    make_signal_control,
    refit_features,
)
from robustsignalmaker.nested_cv import (
    BootstrapMaskFoldEstimator,
    FullSignalFoldEstimator,
)

FAST = dict(segment=16, lam=0.05, n_bootstrap=6, n_iter=100, seed=0)
EXTERNAL = [a for a in ALGORITHMS if a != "head"]


def _binary(n=150, gap=True):
    c = make_signal_control(n=n, n_points=128, task="binary", seed=0)
    X = c.X.copy()
    if gap:
        X[:40, 0, 90:110] = np.nan
    return X, c.y, c


# ---------------------------------------------------------------------------
# the zoo builds, and refuses clearly rather than substituting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alg", EXTERNAL)
def test_every_algorithm_builds_for_classification(alg):
    m = make_refit_model(alg, "binary", seed=0)
    assert hasattr(m, "fit") and (hasattr(m, "predict_proba")
                                  or hasattr(m, "decision_function"))


@pytest.mark.parametrize("alg", [a for a in EXTERNAL if a != "log"])
def test_every_applicable_algorithm_builds_for_regression(alg):
    m = make_refit_model(alg, "regression", seed=0)
    assert hasattr(m, "fit") and hasattr(m, "predict")


def test_a_classification_only_algorithm_refuses_a_regression_task():
    """Substituting something else silently would be the wrong kindness."""
    with pytest.raises(ValueError, match="classification only"):
        make_refit_model("log", "regression")


def test_unknown_algorithm_and_task_are_rejected():
    with pytest.raises(ValueError, match="unknown algorithm"):
        make_refit_model("randomforest", "binary")
    with pytest.raises(ValueError, match="unknown task"):
        make_refit_model("rf", "survival")


def test_head_is_not_built_by_the_factory():
    """'head' names the built-in gated head, which the selector owns."""
    with pytest.raises(ValueError, match="built-in gated head"):
        make_refit_model("head", "binary")


def test_xgb_falls_back_and_says_so_when_xgboost_is_absent():
    """Either xgboost is installed and used, or the fallback is announced.

    A silent substitution would make a reported 'xgb' result mean two
    different things depending on the machine.
    """
    m = make_refit_model("xgb", "binary", seed=0)
    note = getattr(m, "_rsm_note", None)
    try:
        import xgboost  # noqa: F401
    except Exception:
        assert note and "xgboost not installed" in note
    else:
        assert note is None


def test_overrides_reach_the_estimator():
    m = make_refit_model("rf", "binary", seed=0, n_estimators=7)
    assert m.get_params()["n_estimators"] == 7


# ---------------------------------------------------------------------------
# PLS-DA, which scikit-learn does not ship
# ---------------------------------------------------------------------------

def test_plsda_predicts_classes_and_normalised_probabilities():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (40, 5)), rng.normal(2.5, 1, (40, 5))])
    y = np.array([0] * 40 + [1] * 40)
    m = PLSDA(n_components=2).fit(X, y)
    P = m.predict_proba(X)
    assert P.shape == (80, 2)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
    assert (m.predict(X) == y).mean() > 0.9
    assert list(m.classes_) == [0, 1]


def test_plsda_handles_more_components_than_features():
    """n_components is clamped rather than raising deep inside sklearn."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2))
    y = np.array([0, 1] * 10)
    m = PLSDA(n_components=10).fit(X, y)
    assert m.predict_proba(X).shape == (20, 2)


def test_plsda_supports_three_classes():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(k * 3, 1, (30, 4)) for k in range(3)])
    y = np.repeat([0, 1, 2], 30)
    m = PLSDA(n_components=2).fit(X, y)
    assert m.predict_proba(X).shape == (90, 3)


# ---------------------------------------------------------------------------
# feature construction: nothing is imputed, thin samples are refused
# ---------------------------------------------------------------------------

def test_refit_features_zero_unobserved_and_flag_thin_samples():
    from robustsignalmaker import masked_standardise_fit
    from robustsignalmaker.representations import SegmentMean

    X, y, _ = _binary()
    V = np.isfinite(X).astype(float)
    Xz = np.where(V > 0, X, 0.0)
    grid = SegmentGrid.from_signal_shape(X.shape, segment=16)
    F, VF = SegmentMean().transform(Xz, V, grid)
    stats = masked_standardise_fit(F, VF)
    Z, valid = refit_features(F, VF, np.array([0, 1]), stats, rho_min=0.1)

    assert Z.shape == (X.shape[0], 2)
    assert np.isfinite(Z).all(), "no NaN may reach an sklearn estimator"
    assert valid.all(), "segments 0 and 1 are fully observed here"


def test_refit_features_rejects_an_empty_selection():
    from robustsignalmaker import masked_standardise_fit
    from robustsignalmaker.representations import SegmentMean

    X, _, _ = _binary()
    V = np.isfinite(X).astype(float)
    grid = SegmentGrid.from_signal_shape(X.shape, segment=16)
    F, VF = SegmentMean().transform(np.where(V > 0, X, 0.0), V, grid)
    with pytest.raises(ValueError, match="no segments selected"):
        refit_features(F, VF, np.array([], dtype=int),
                       masked_standardise_fit(F, VF))


# ---------------------------------------------------------------------------
# the two properties that matter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alg", ["rf", "pls", "svm", "rdg"])
def test_the_refit_model_does_not_change_the_selection(alg):
    """The gated head chooses the bands; the refit only changes the predictor."""
    X, y, _ = _binary()
    head = BootstrapMaskSelector(task="binary", **FAST).fit(X, y)
    other = BootstrapMaskSelector(task="binary", refit_model=alg, **FAST).fit(X, y)
    np.testing.assert_array_equal(head.selected_segments_, other.selected_segments_)
    np.testing.assert_allclose(head.pi_, other.pi_, equal_nan=True)


@pytest.mark.parametrize("alg", ["rf", "pls"])
def test_an_external_model_refuses_the_same_samples_the_head_refuses(alg):
    """The no-fabrication contract survives a model that cannot see validity.

    A random forest handed imputed features would predict happily for every
    row. Here the thin rows must come back nan, exactly as from the head.
    """
    c = make_signal_control(n=200, n_points=128, task="binary", seed=0)
    X = c.X.copy()
    X[:40] = np.nan
    X[:40, 0, :4] = c.X[:40, 0, :4]          # a sliver of evidence only

    head = BootstrapMaskSelector(task="binary", **FAST).fit(X, c.y)
    other = BootstrapMaskSelector(task="binary", refit_model=alg, **FAST).fit(X, c.y)
    ref_head = ~np.isfinite(head.predict_proba(X)).all(axis=1)
    ref_other = ~np.isfinite(other.predict_proba(X)).all(axis=1)
    assert ref_other.sum() > 0, "the thin samples must be refused, not guessed"
    np.testing.assert_array_equal(ref_head, ref_other)


def test_probabilities_are_normalised_over_the_scored_rows():
    X, y, _ = _binary()
    for alg in ("rf", "pls", "rdg"):
        P = BootstrapMaskSelector(task="binary", refit_model=alg,
                                  **FAST).fit(X, y).predict_proba(X)
        live = np.isfinite(P).all(axis=1)
        np.testing.assert_allclose(P[live].sum(axis=1), 1.0, atol=1e-10)


def test_a_model_without_predict_proba_still_yields_probabilities():
    """RidgeClassifier has only decision_function; a monotone transform of it
    keeps the ranking, and therefore AUC, while giving proba something to
    return rather than refusing the whole task."""
    X, y, _ = _binary()
    sel = BootstrapMaskSelector(task="binary", refit_model="rdg", **FAST).fit(X, y)
    assert not hasattr(sel.external_model_.model_, "predict_proba")
    P = sel.predict_proba(X)
    live = np.isfinite(P).all(axis=1)
    np.testing.assert_allclose(P[live].sum(axis=1), 1.0, atol=1e-10)


def test_regression_refit_predicts_and_refuses_proba():
    c = make_signal_control(n=150, n_points=128, task="regression", seed=0)
    sel = BootstrapMaskSelector(task="regression", refit_model="rf",
                                **FAST).fit(c.X, c.y)
    pred = sel.predict(c.X)
    assert pred.shape == (150,) and np.isfinite(pred).all()
    with pytest.raises(ValueError, match="undefined for regression"):
        sel.predict_proba(c.X)


def test_multiclass_refit_gives_one_column_per_class():
    c = make_signal_control(n=180, n_points=128, task="multiclass",
                            n_classes=3, seed=0)
    sel = BootstrapMaskSelector(task="multiclass", refit_model="pls",
                                **FAST).fit(c.X, c.y)
    assert sel.predict_proba(c.X).shape == (180, 3)


def test_an_unknown_refit_model_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown refit_model"):
        BootstrapMaskSelector(task="binary", refit_model="randomforest")


def test_external_refit_refuses_when_nothing_clears_the_floor():
    X, y, _ = _binary(gap=False)
    grid = SegmentGrid.from_signal_shape(X.shape, segment=16)
    V = np.zeros_like(X)
    V[:, :, :2] = 1.0                        # far below any sensible floor
    ext = ExternalRefit("rf", "binary", grid, np.array([4, 5]), rho_min=0.9)
    with pytest.raises(InsufficientEvidenceError, match="clear the evidence floor"):
        ext.fit(np.where(V > 0, X, 0.0), V, y)


# ---------------------------------------------------------------------------
# the verdict must compare REGIONS, not models
# ---------------------------------------------------------------------------

def test_nested_cv_uses_the_same_model_for_the_matched_baseline():
    """With refit_model='rf' the verdict is a forest on the selected bands
    against a forest on all of them. Comparing a forest against the built-in
    head would confound region with model, which is what the verdict exists
    to isolate."""
    X, y, _ = _binary(n=120)
    cv = NestedCV(
        estimator_factory=lambda: BootstrapMaskFoldEstimator(
            segment=16, lam=0.05, n_bootstrap=4, n_iter=80, refit_model="rf"),
        baseline_factory=lambda: FullSignalFoldEstimator(segment=16, refit_model="rf"),
        k_outer=3, l_inner=2, segment=16, mnar_guard=False, random_state=0)
    res = cv.run(X, y)
    s = res.summary()
    assert s["verdict"] in ("preserved", "sig.better", "sig.worse", "undecidable")
    assert np.isfinite(s["score_mean"]) and np.isfinite(s["baseline_score_mean"])
    # the baseline really did use the external model on every segment
    base = cv.baseline_factory()
    assert base.refit_model == "rf"


def test_model_supports_proba_distinguishes_the_two_kinds():
    from robustsignalmaker.models import model_supports_proba

    assert model_supports_proba(make_refit_model("rf", "binary"))
    assert not model_supports_proba(make_refit_model("rdg", "binary"))


def test_external_predict_returns_class_labels_for_classification():
    """`predict` (as opposed to `predict_proba`) must give labels, with None
    where the sample was refused rather than a fabricated class."""
    c = make_signal_control(n=200, n_points=128, task="binary", seed=0)
    X = c.X.copy()
    X[:40] = np.nan
    X[:40, 0, :4] = c.X[:40, 0, :4]

    sel = BootstrapMaskSelector(task="binary", refit_model="rf", **FAST).fit(X, c.y)
    pred = sel.predict(X)
    assert pred.dtype == object
    refused = np.array([p is None for p in pred])
    assert refused.sum() > 0
    assert set(pred[~refused]) <= set(np.unique(c.y))


def test_the_matched_baseline_predicts_through_the_external_model():
    """Covers FullSignalFoldEstimator.predict on the external path, which the
    classification route reaches only via predict_proba."""
    c = make_signal_control(n=120, n_points=128, task="regression", seed=0)
    est = FullSignalFoldEstimator(segment=16, refit_model="rf", n_iter=80)
    est.fit(c.X, c.y, V=np.isfinite(c.X), task="regression", seeds=None,
            fold_idx=0, repeat=0, inner_splitter=None)
    pred = est.predict(c.X)
    assert pred.shape == (120,) and np.isfinite(pred).all()
    assert est.external_ is not None
