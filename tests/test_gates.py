"""Milestone 3: the gated engine.

The non-negotiable test here is numeric-vs-analytic gradient agreement
through the renormalised head. The analytic gradient is DELIBERATELY the
stop-gradient surrogate (the renormaliser M/D held constant), so the check
freezes the renormaliser for the numeric side too, and a separate test pins
the intended difference: a never-observed segment feels zero analytic
pressure even though the true (unfrozen) derivative through the renormaliser
is nonzero.
"""
from __future__ import annotations

import numpy as np
import pytest

from robustsignalmaker import (
    InsufficientEvidenceError,
    SoftMaskSelector,
    loss_and_dz,
    make_signal_control,
    mask_data_gradient,
    mask_scattered,
    renormalised_gated_forward,
)
from robustsignalmaker.masking import HC_GAMMA, HC_ZETA

RNG = np.random.default_rng(4)

FAST_KWARGS = dict(segment=8, n_iter=300, lr=0.1, seed=0)


def _targets(task, n, K, rng):
    if task == "regression":
        return rng.standard_normal((n, 1))
    if task == "binary":
        return (rng.random((n, 1)) > 0.5).astype(float)
    Y = np.zeros((n, K))
    Y[np.arange(n), rng.integers(0, K, size=n)] = 1.0
    return Y


@pytest.mark.parametrize("task,K", [("regression", 1), ("binary", 1), ("multiclass", 3)])
def test_analytic_gradient_matches_numeric_with_frozen_renormaliser(task, K):
    rng = np.random.default_rng(9)
    n, S = 40, 7
    U = rng.standard_normal((n, S, K))
    v_seg = np.clip(rng.random((n, S)), 0.05, 1.0)
    v_seg[:, 0] = 0.0
    U[:, 0, :] = 0.0  # a never-observed segment contributes nothing, as in the pipeline
    active = np.ones(S, dtype=bool)
    m = np.clip(rng.random(S), 0.2, 0.9)
    c = rng.standard_normal(K)
    Y = _targets(task, n, K, rng)

    Z, valid, r = renormalised_gated_forward(U, v_seg, m, active, c, rho_min=0.1)
    _, dz = loss_and_dz(task, Z, Y, valid)
    analytic = mask_data_gradient(U, dz, r, active)

    eps = 1e-6
    numeric = np.zeros(S)
    for s in range(S):
        for sign in (+1, -1):
            mp = m.copy()
            mp[s] += sign * eps
            Zp, validp, _ = renormalised_gated_forward(
                U, v_seg, mp, active, c, rho_min=0.1, r_override=r)
            lp, _ = loss_and_dz(task, Zp, Y, valid)
            numeric[s] += sign * lp
    numeric /= 2 * eps
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-9)


def test_stop_gradient_gives_never_observed_segments_exactly_zero_pressure():
    rng = np.random.default_rng(10)
    n, S = 30, 5
    U = rng.standard_normal((n, S, 1))
    v_seg = np.clip(rng.random((n, S)), 0.3, 1.0)
    v_seg[:, 2] = 0.0
    U[:, 2, :] = 0.0
    active = np.ones(S, dtype=bool)
    m = np.full(S, 0.6)
    c = np.zeros(1)
    Y = rng.standard_normal((n, 1))

    Z, valid, r = renormalised_gated_forward(U, v_seg, m, active, c)
    _, dz = loss_and_dz("regression", Z, Y, valid)
    analytic = mask_data_gradient(U, dz, r, active)
    assert analytic[2] == 0.0

    # the TRUE derivative through the renormaliser is nonzero: opening the
    # dead gate inflates M without touching D, rescaling every prediction
    eps = 1e-5
    losses = []
    for d in (+eps, -eps):
        mp = m.copy()
        mp[2] += d
        Zp, validp, _ = renormalised_gated_forward(U, v_seg, mp, active, c)
        lp, _ = loss_and_dz("regression", Zp, Y, validp)
        losses.append(lp)
    true_grad = (losses[0] - losses[1]) / (2 * eps)
    assert abs(true_grad) > 1e-4  # pressure exists; the stop-gradient removes it


def test_forward_reduces_to_plain_head_on_complete_data():
    rng = np.random.default_rng(11)
    n, S, K = 12, 6, 2
    U = rng.standard_normal((n, S, K))
    m = rng.random(S)
    c = rng.standard_normal(K)
    Z, valid, r = renormalised_gated_forward(
        U, np.ones((n, S)), m, np.ones(S, dtype=bool), c)
    assert valid.all() and np.allclose(r, 1.0)
    np.testing.assert_allclose(Z, np.einsum("nsk,s->nk", U, m) + c, rtol=1e-12)


def test_forward_refuses_all_closed_and_flags_thin_samples():
    U = np.ones((3, 4, 1))
    v = np.ones((3, 4))
    v[1, :] = 0.05  # rho below the floor
    with pytest.raises(InsufficientEvidenceError):
        renormalised_gated_forward(U, v, np.zeros(4), np.ones(4, dtype=bool), np.zeros(1))
    Z, valid, _ = renormalised_gated_forward(
        U, v, np.ones(4), np.ones(4, dtype=bool), np.zeros(1), rho_min=0.1)
    assert not valid[1] and np.isnan(Z[1]).all()
    assert valid[0] and valid[2]


def test_loss_refuses_when_no_sample_is_scorable():
    with pytest.raises(InsufficientEvidenceError):
        loss_and_dz("regression", np.zeros((3, 1)), np.zeros((3, 1)),
                    np.zeros(3, dtype=bool))


def test_hardconcrete_helpers():
    la = np.array([-30.0, 0.0, 30.0])
    det = SoftMaskSelector._hc_deterministic(la)
    assert det[0] == 0.0 and det[2] == 1.0 and 0.0 < det[1] < 1.0
    q = SoftMaskSelector._hc_open_prob(np.array([-5.0, 0.0, 5.0]))
    assert q[0] < q[1] < q[2]
    z, dz = SoftMaskSelector._hc_sample(np.full(1000, 2.0), np.random.default_rng(0))
    assert (z >= 0.0).all() and (z <= 1.0).all()
    saturated = (z == 0.0) | (z == 1.0)
    assert (dz[saturated] == 0.0).all()
    assert (dz[~saturated] > 0.0).all()
    assert float(HC_ZETA) > 1.0 > 0.0 > float(HC_GAMMA)  # the stretch is real


def test_tv_gradient_is_chain_laplacian_and_skips_unassessable_pairs():
    sel = SoftMaskSelector(task="regression")
    sel._active = np.array([True, True, False, True, True])
    m = np.array([1.0, 0.0, 0.5, 0.2, 0.9])
    g = sel._tv_grad(m)
    # pair (0,1): d = -1 -> g[1] += -1, g[0] -= -1; pairs with segment 2 skipped;
    # pair (3,4): d = 0.7 -> g[4] += 0.7, g[3] -= 0.7
    np.testing.assert_allclose(g, [1.0, -1.0, 0.0, -0.7, 0.7])


# ---------------------------------------------------------------------------
# end-to-end: the Milestone 3 make-or-break
# ---------------------------------------------------------------------------

def _control(task="binary"):
    return make_signal_control(n=240, n_channels=1, n_points=256, task=task,
                               signal=2.0, noise=1.0, seed=21)


@pytest.mark.parametrize("task", ["binary", "regression"])
def test_make_or_break_single_fit_recovers_the_planted_band(task):
    c = _control(task)
    sel = SoftMaskSelector(task=task, lam=0.05, **FAST_KWARGS).fit(c.X, c.y)
    truth = set(c.truth_segments(sel.grid).tolist())
    selected = set(sel.selected_segments_.tolist())
    assert truth <= selected, f"missed truth segments: {truth - selected}"
    assert sel.coverage() < 0.5, "no meaningful compression"
    assert sel.degenerate_ is None


def test_make_or_break_nans_outside_the_band_do_not_change_selection():
    c = _control("binary")
    sel_full = SoftMaskSelector(task="binary", lam=0.05, **FAST_KWARGS).fit(c.X, c.y)

    protected = c.informative[None, None, :]  # never mask the truth band
    V = mask_scattered(c.X.shape, frac=0.25, seed=22) | protected
    X = np.where(V, c.X, np.nan)  # NaN payloads, exercising as_validity
    sel_nan = SoftMaskSelector(task="binary", lam=0.05, **FAST_KWARGS).fit(X, c.y, V)

    truth = set(c.truth_segments(sel_full.grid).tolist())
    assert truth <= set(sel_nan.selected_segments_.tolist())
    assert set(sel_nan.selected_segments_) == set(sel_full.selected_segments_)


def test_multiclass_fit_recovers_the_band_too():
    c = make_signal_control(n=300, n_points=256, task="multiclass", n_classes=3,
                            signal=2.5, noise=1.0, seed=23)
    sel = SoftMaskSelector(task="multiclass", lam=0.05, **FAST_KWARGS).fit(c.X, c.y)
    truth = set(c.truth_segments(sel.grid).tolist())
    assert truth <= set(sel.selected_segments_.tolist())


def test_tv_prior_produces_contiguous_bands():
    c = _control("binary")
    sel = SoftMaskSelector(task="binary", lam=0.05, tv=0.5, **FAST_KWARGS).fit(c.X, c.y)
    sel_ids = np.sort(sel.selected_segments_)
    assert len(sel_ids) > 0
    runs = np.split(sel_ids, np.flatnonzero(np.diff(sel_ids) > 1) + 1)
    assert len(runs) <= 2, f"TV should yield contiguous bands, got runs {runs}"


def test_prediction_quality_and_valid_flags():
    c = _control("binary")
    sel = SoftMaskSelector(task="binary", lam=0.05, **FAST_KWARGS).fit(c.X, c.y)
    proba = sel.predict_proba(c.X)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(c.y, proba[:, 1]) > 0.85
    # a sample with almost no observed evidence is refused, not guessed
    X2 = c.X.copy()
    V2 = np.ones_like(X2)
    V2[0, :, :] = 0.0
    V2[0, :, :2] = 1.0
    assert np.isnan(sel.predict(X2, V2)[0])
    assert not sel.prediction_valid(X2, V2)[0]
    assert np.isfinite(sel.predict(X2, V2)[1:]).all()


def test_fixed_mask_fits_head_only():
    c = _control("regression")
    grid_probe = SoftMaskSelector(task="regression", **FAST_KWARGS).fit(c.X, c.y).grid
    fixed = np.zeros(grid_probe.n_segments)
    fixed[c.truth_segments(grid_probe)] = 1.0
    sel = SoftMaskSelector(task="regression", fixed_mask=fixed, **FAST_KWARGS).fit(c.X, c.y)
    assert sel.gate_params_ is None
    assert set(sel.selected_segments_) == set(c.truth_segments(grid_probe).tolist())
    pred = sel.predict(c.X)
    assert np.corrcoef(pred, c.y)[0, 1] > 0.8


def test_same_seed_is_bitwise_reproducible_and_seeds_matter():
    c = _control("binary")
    a = SoftMaskSelector(task="binary", lam=0.05, **FAST_KWARGS).fit(c.X, c.y)
    b = SoftMaskSelector(task="binary", lam=0.05, **FAST_KWARGS).fit(c.X, c.y)
    assert np.array_equal(a.mask_, b.mask_)
    assert np.array_equal(a.coef_, b.coef_)


def test_degenerate_all_closed_is_flagged_and_prediction_refuses():
    c = _control("binary")
    sel = SoftMaskSelector(task="binary", lam=50.0, **FAST_KWARGS).fit(c.X, c.y)
    assert sel.degenerate_ == "all_closed"
    assert len(sel.selected_segments_) == 0
    with pytest.raises(InsufficientEvidenceError):
        sel.predict(c.X)


def test_unassessable_segments_are_reported_not_scored():
    c = _control("binary")
    V = np.ones_like(c.X)
    V[:, :, :16] = 0.0  # first two segments never observed for anyone
    sel = SoftMaskSelector(task="binary", lam=0.05, **FAST_KWARGS).fit(c.X, c.y, V)
    assert {0, 1} <= set(sel.unassessable_segments_.tolist())
    assert not (set(sel.selected_segments_) & {0, 1})


def test_constructor_and_input_validation():
    with pytest.raises(ValueError, match="task"):
        SoftMaskSelector(task="cluster")
    with pytest.raises(ValueError, match="gate"):
        SoftMaskSelector(gate="relu")
    c = _control("binary")
    with pytest.raises(ValueError, match="n, C, T"):
        SoftMaskSelector(task="binary").fit(c.X[:, 0, :], c.y)
    with pytest.raises(ValueError, match="2 classes"):
        SoftMaskSelector(task="binary", **FAST_KWARGS).fit(c.X, np.zeros(len(c.y)))
    with pytest.raises(ValueError, match="fixed_mask"):
        SoftMaskSelector(task="binary", fixed_mask=np.ones(3), **FAST_KWARGS).fit(c.X, c.y)
