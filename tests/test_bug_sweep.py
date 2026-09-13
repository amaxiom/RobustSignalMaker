"""Regression tests from the 2026-09-01 deep bug sweep.

Each test pins a defect that was found and fixed (or a scope decision that
was measured and documented); the docstrings say which.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from robustsignalmaker import (
    NestedCV,
    SoftMaskSelector,
    make_signal_control,
    paired_comparison,
)


def test_string_class_labels_survive_predict_and_refusal():
    """predict() crashed on non-numeric labels (classes_[idx].astype(float));
    refusals now read None for object labels, nan for numeric ones."""
    c = make_signal_control(n=90, n_points=128, task="multiclass", n_classes=3,
                            seed=1)
    y = np.array(["alpha", "beta", "gamma"])[c.y]
    sel = SoftMaskSelector(task="multiclass", segment=8, lam=0.05, n_iter=80,
                           lr=0.1, seed=0).fit(c.X, y)
    pred = sel.predict(c.X[:5])
    assert set(pred) <= {"alpha", "beta", "gamma"}
    # a refused sample is None, not a guessed class
    V = np.ones_like(c.X[:2], dtype=bool)
    V[1] = False
    V[1, :, :2] = True
    pred2 = sel.predict(c.X[:2], V)
    assert pred2[1] is None and pred2[0] in {"alpha", "beta", "gamma"}
    # binary with string labels too
    cb = make_signal_control(n=80, n_points=128, task="binary", seed=2)
    yb = np.array(["neg", "pos"])[cb.y]
    selb = SoftMaskSelector(task="binary", segment=8, lam=0.05, n_iter=80,
                            lr=0.1, seed=0).fit(cb.X, yb)
    assert set(selb.predict(cb.X[:5])) <= {"neg", "pos"}


def test_exported_csvs_are_well_formed(tmp_path):
    """test_idx lists (and any comma-bearing cell) corrupted every folds.csv
    row; exports now go through the csv module with proper quoting."""
    c = make_signal_control(n=80, n_points=128, task="binary", seed=2)
    res = NestedCV(k_outer=3, segment=8, lam=0.08, n_bootstrap=4, n_iter=60,
                   lr=0.1, mnar_guard=False, random_state=0).run(c.X, c.y)
    out = res.save(tmp_path, prefix="sweep")
    for name in ("sweep_folds.csv", "sweep_overview.csv",
                 "sweep_selected_segments.csv"):
        with open(out / name, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            continue
        width = len(rows[0])
        assert all(len(r) == width for r in rows), f"{name} is malformed"
    # the list-valued cell survives a round trip
    with open(out / "sweep_folds.csv", newline="", encoding="utf-8") as fh:
        folds = list(csv.DictReader(fh))
    assert all("test_idx" in r and r["test_idx"].startswith("[") for r in folds)


def test_paired_comparison_with_nan_folds_is_never_confident():
    """nan fold scores produced outcome 'preserved' with p nan (failing toward
    confidence); pairs with nan are now excluded, and fewer than two
    comparable folds is 'undecidable'."""
    r = paired_comparison([0.9, np.nan, 0.8, 0.7, 0.75],
                          [0.85, 0.8, np.nan, 0.72, 0.7])
    assert np.isfinite(r.mean_delta)  # computed over the three comparable pairs
    r2 = paired_comparison([np.nan, 0.9], [0.8, np.nan])
    assert r2.outcome == "undecidable"
    assert np.isnan(r2.mean_delta)


def test_level_renormalisation_is_exact_for_nonuniform_kernels():
    """The count-based rescale fabricated +39% level shifts on a FLAT signal
    near a gap for Gaussian kernels; the filter-weighted normalisation is
    exact for any kernel (D1, the sweep's most serious finding)."""
    from robustsignalmaker import renormalised_convolve1d

    x = np.arange(-8, 9, dtype=float)
    k = np.exp(-0.5 * (x / 2.0) ** 2)
    k /= k.sum()
    sig = np.full(101, 3.0)
    V = np.ones(101)
    V[40:70] = 0.0
    out, v_out = renormalised_convolve1d(sig, V, k)
    assert np.allclose(out[v_out > 0], 3.0, rtol=1e-12), (
        f"fabricated level shift: {np.abs(out[v_out > 0] - 3.0).max()}")
    # end to end: the Gaussian lens on a flat gapped signal stays flat
    from robustsignalmaker import GaussianScale1D, SegmentGrid

    grid = SegmentGrid(101, segment=10)
    F, FV = GaussianScale1D(sigmas=(2.0,)).transform(
        sig[None, None, :], V[None, None, :], grid)
    assert np.allclose(F[FV > 0], 3.0, rtol=1e-12)


def test_nan_and_inf_payloads_never_leak_through_public_primitives():
    """masked_mean/std/standardise_fit, renormalised_convolve1d and
    renormalised_dot returned nan WITH valid=True for nan payloads (D2 to
    D4); payloads of any value are now bitwise-irrelevant."""
    from robustsignalmaker import (
        masked_mean,
        masked_standardise_fit,
        masked_std,
        renormalised_convolve1d,
        renormalised_dot,
    )

    W = np.array([[0.0, 1.0, 1.0]])
    for payload in (np.nan, np.inf, -np.inf, 1e300):
        X = np.array([[payload, 1.0, 2.0]])
        m, ok = masked_mean(X, W, axis=1)
        assert ok[0] and m[0] == 1.5
        s, ok2 = masked_std(X, W, axis=1, min_count=2)
        assert ok2[0] and np.isfinite(s[0])
    rng = np.random.default_rng(11)
    F = rng.standard_normal((20, 3))
    Wf = np.ones_like(F)
    Wf[:10, 0] = 0.0
    Fp = F.copy()
    Fp[:10, 0] = np.nan
    stats_clean = masked_standardise_fit(np.where(Wf > 0, F, 0.0), Wf)
    stats_payload = masked_standardise_fit(Fp, Wf)
    assert np.array_equal(stats_clean.mu, stats_payload.mu)
    assert stats_payload.assessable.all()

    x = rng.standard_normal(40)
    v = (rng.random(40) > 0.4).astype(float)
    base, vb = renormalised_convolve1d(np.where(v > 0, x, 0.0), v,
                                       np.array([-1.0, 0.0, 1.0]) / 2)
    xn = np.where(v > 0, x, np.nan)
    out, vo = renormalised_convolve1d(xn, v, np.array([-1.0, 0.0, 1.0]) / 2)
    assert np.array_equal(out, base) and np.array_equal(vo, vb)

    U = np.ones((2, 4))
    Un = U.copy()
    validity = np.ones((2, 4))
    validity[0, 2] = 0.0
    Un[0, 2] = np.nan
    z1, ok1, _ = renormalised_dot(U * (validity > 0), np.ones(4), validity)
    z2, ok2, _ = renormalised_dot(Un, np.ones(4), validity)
    assert np.array_equal(z1, z2) and np.array_equal(ok1, ok2)


def test_zero_evidence_sample_is_invalid_even_at_rho_min_zero():
    """renormalised_dot returned a confident z=0.0 for a sample that observed
    NOTHING when rho_min=0 (D4)."""
    from robustsignalmaker import renormalised_dot

    validity = np.array([[1.0, 1.0], [0.0, 0.0]])
    z, valid, _ = renormalised_dot(np.ones((2, 2)), np.ones(2), validity,
                                   rho_min=0.0)
    assert valid[0] and not valid[1]
    assert np.isnan(z[1])


def test_subnormal_evidence_is_no_evidence_not_inf():
    """Evidence around 1e-320 overflowed the reciprocal into trusted inf/nan
    outputs (D5); validity mass below float-tiny now reads unobserved."""
    from robustsignalmaker import renormalised_convolve1d

    x = np.full(20, 2.0)
    v = np.zeros(20)
    v[10] = 1e-320
    out, v_out = renormalised_convolve1d(x, v, np.ones(3) / 3)
    assert np.isfinite(out).all()
    assert (v_out == 0.0).all()


def test_multiclass_single_class_raises():
    """Multiclass fit silently accepted one class (D8)."""
    c = make_signal_control(n=60, n_points=128, task="binary", seed=3)
    with pytest.raises(ValueError, match="at least 2 classes"):
        SoftMaskSelector(task="multiclass", segment=8, n_iter=40,
                         seed=0).fit(c.X, np.zeros(60))


def test_missingness_association_rejects_task_typos():
    """A typo'd task fell into the classification branch and came back
    'could_not_assess' instead of raising (D10)."""
    from robustsignalmaker import missingness_association

    with pytest.raises(ValueError, match="task must be"):
        missingness_association(np.ones((30, 4)), np.arange(30.0), "regresion")


def test_fixed_mask_path_exposes_final_loss_and_tv_is_reported():
    """final_loss_ existed only on the learned path (D11), and the TV energy
    was optimised but missing from the reported loss (D6)."""
    c = make_signal_control(n=80, n_points=128, task="binary", seed=4)
    probe = SoftMaskSelector(task="binary", segment=8, lam=0.05, n_iter=60,
                             lr=0.1, seed=0).fit(c.X, c.y)
    fixed = np.zeros(probe.grid.n_segments)
    fixed[probe.selected_segments_ if len(probe.selected_segments_) else [0]] = 1.0
    sel = SoftMaskSelector(task="binary", segment=8, fixed_mask=fixed,
                           n_iter=60, lr=0.1, seed=0).fit(c.X, c.y)
    assert np.isfinite(sel.final_loss_)
    tv_sel = SoftMaskSelector(task="binary", segment=8, lam=0.05, tv=0.5,
                              n_iter=80, lr=0.1, seed=0).fit(c.X, c.y)
    assert np.isfinite(tv_sel.final_loss_)


def test_grouped_multiclass_survives_a_group_confined_class():
    """With StratifiedGroupKFold a class can leave training entirely, so the
    fold model knows fewer classes than the engine; the out-of-fold
    accumulator crashed on the shape mismatch (sweep finding B1)."""
    c = make_signal_control(n=150, n_points=128, task="multiclass",
                            n_classes=3, seed=5)
    groups = np.arange(150) % 10
    y = c.y.copy()
    # confine class 2 to a single group: some folds train without it
    y[y == 2] = 0
    y[groups == 7] = 2
    res = NestedCV(k_outer=4, segment=8, lam=0.08, n_bootstrap=4, n_iter=60,
                   lr=0.1, mnar_guard=False, random_state=0).run(
                       c.X, y, groups=groups)
    assert len(res.per_fold_scores) == 4
    assert res.oof_predictions.shape == (150, 3)


def test_grouped_regression_repeats_are_refused_not_duplicated():
    """GroupKFold is deterministic, so repeats would duplicate folds and
    manufacture Wilcoxon significance from pseudo-replication (B2)."""
    c = make_signal_control(n=120, n_points=128, task="regression", seed=6)
    groups = np.arange(120) % 8
    with pytest.warns(UserWarning, match="repeated_outer_cv forced to 1"):
        res = NestedCV(k_outer=4, repeated_outer_cv=3, segment=8, lam=0.08,
                       n_bootstrap=4, n_iter=60, lr=0.1, mnar_guard=False,
                       random_state=0).run(c.X, c.y, groups=groups)
    assert res.config["repeated_outer_cv"] == 1
    assert len(res.per_fold_scores) == 4


def test_object_dtype_missing_labels_are_excluded_and_counted():
    """A pandas round trip delivers object arrays with None/nan holes; these
    crashed with an opaque TypeError instead of counting as missing (B5)."""
    c = make_signal_control(n=100, n_points=128, task="binary", seed=7)
    y = np.array(["neg", "pos"], dtype=object)[c.y]
    y[:12] = None
    y[12:20] = np.nan
    res = NestedCV(k_outer=3, segment=8, lam=0.08, n_bootstrap=4, n_iter=60,
                   lr=0.1, mnar_guard=False, random_state=0).run(c.X, y)
    assert res.n_unlabelled == 20
    for rec in res.fold_records:
        assert not (set(rec["test_idx"]) & set(range(20)))


def test_nan_like_string_labels_are_flagged_not_silently_classed():
    """'nan' as a STRING is a common CSV artifact; it stays a class (we
    cannot know better) but the user is told (B12)."""
    c = make_signal_control(n=90, n_points=128, task="binary", seed=8)
    y = np.array(["healthy", "disease"], dtype=object)[c.y]
    y[:15] = "nan"
    with pytest.warns(UserWarning, match="missing-value"):
        NestedCV(k_outer=3, segment=8, lam=0.08, n_bootstrap=4, n_iter=60,
                 lr=0.1, mnar_guard=False, random_state=0).run(c.X, y)


def test_constant_y_raises_instead_of_burning_a_run():
    """A degenerate target inferred 'multiclass' and produced an all-nan run
    with no warning (B11)."""
    c = make_signal_control(n=80, n_points=128, task="binary", seed=9)
    with pytest.raises(ValueError, match="constant"):
        NestedCV(k_outer=3, segment=8, mnar_guard=False,
                 random_state=0).run(c.X, np.full(80, 3.0))


def test_fold_adapters_forward_groups_to_resample_stratification():
    """The adapters dropped caller groups, voiding the documented
    instrument-stratification guarantee (B3)."""
    from robustsignalmaker import BootstrapMaskFoldEstimator, Seeds

    c = make_signal_control(n=100, n_points=128, task="binary", seed=10)
    groups = np.arange(100) % 2
    est = BootstrapMaskFoldEstimator(segment=8, lam=0.08, n_bootstrap=4,
                                     n_iter=60, lr=0.1)
    est.fit(c.X, c.y, V=np.ones_like(c.X, dtype=bool), task="binary",
            seeds=Seeds(base=0), fold_idx=0, repeat=0, inner_splitter=None,
            groups=groups)
    assert "observation group" in est.selector_.strata_note_
    # and the stability universe is the assessable one, not the raw count
    assert est.n_candidates_ == est.selector_.assessable_universe()


def test_selected_set_is_row_order_invariant_for_single_fits():
    """Measured scope decision: permuting rows leaves the mask and selection
    unchanged (fitted coefficients may differ in the last fp bits; the
    reproducibility contract is same seed AND same row order)."""
    c = make_signal_control(n=80, n_points=128, task="binary", seed=2)
    perm = np.random.default_rng(0).permutation(80)
    a = SoftMaskSelector(task="binary", segment=8, lam=0.05, n_iter=120,
                         lr=0.1, seed=0).fit(c.X, c.y)
    b = SoftMaskSelector(task="binary", segment=8, lam=0.05, n_iter=120,
                         lr=0.1, seed=0).fit(c.X[perm], c.y[perm])
    assert set(a.selected_segments_) == set(b.selected_segments_)
    assert np.array_equal(a.mask_, b.mask_)
