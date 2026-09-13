"""Milestone 1: masked-computation primitives.

The densest test file in the package, deliberately: the family's worst bugs
hide in shared helpers, and validity.py is the shared helper everything else
will stand on. The named invariants:

  * masked == unmasked on fully observed data, exactly;
  * the payload under a masked entry is never read (bitwise invariance);
  * scale invariance at 1e-200 and 1e300;
  * guards fail loudly (valid=False, nan, or an exception), never a
    confident-looking number.
"""
from __future__ import annotations

import numpy as np
import pytest

from robustsignalmaker import (
    InsufficientEvidenceError,
    MissingnessInformativeWarning,
    as_validity,
    fabricated_edge,
    fill_zero,
    is_contrast_filter,
    masked_mean,
    masked_standardise,
    masked_standardise_fit,
    masked_std,
    missingness_association,
    observation_support,
    renormalised_convolve1d,
    renormalised_dot,
)

RNG = np.random.default_rng(0)

DERIV = np.array([-1.0, 0.0, 1.0]) / 2.0  # contrast (zero-sum)
BOX = np.ones(5) / 5.0  # level


# ---------------------------------------------------------------------------
# as_validity
# ---------------------------------------------------------------------------

def test_as_validity_derives_mask_from_nan_and_zeroes_payload():
    X = np.array([[1.0, np.nan, 3.0]])
    Xc, V = as_validity(X)
    assert V.tolist() == [[True, False, True]]
    assert Xc[0, 1] == 0.0 and Xc[0, 0] == 1.0


def test_as_validity_rejects_nan_claimed_observed():
    X = np.array([[1.0, np.nan]])
    V = np.array([[True, True]])
    with pytest.raises(ValueError, match="observed"):
        as_validity(X, V)


def test_as_validity_rejects_inf_at_observed_positions():
    with pytest.raises(ValueError, match="finite"):
        as_validity(np.array([[np.inf, 1.0]]))


def test_as_validity_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        as_validity(np.zeros((2, 3)), np.ones((3, 2), dtype=bool))


# ---------------------------------------------------------------------------
# masked moments
# ---------------------------------------------------------------------------

def test_masked_mean_equals_plain_mean_on_full_validity():
    X = RNG.standard_normal((7, 11))
    W = np.ones_like(X)
    m, ok = masked_mean(X, W, axis=0)
    assert ok.all()
    np.testing.assert_allclose(m, X.mean(axis=0), rtol=1e-13)


def test_masked_mean_ignores_masked_entries():
    X = np.array([[1.0, 100.0], [3.0, 100.0]])
    W = np.array([[1.0, 0.0], [1.0, 0.0]])
    m, ok = masked_mean(X, W, axis=0)
    assert m[0] == 2.0
    assert not ok[1] and m[1] == 0.0  # placeholder, flagged invalid


def test_masked_mean_zero_evidence_fails_loudly_in_strict_mode():
    X = np.ones((3, 2))
    W = np.zeros((3, 2))
    with pytest.raises(InsufficientEvidenceError):
        masked_mean(X, W, axis=0, strict=True)


def test_masked_std_matches_plain_std_and_enforces_ess_floor():
    X = RNG.standard_normal((20, 4))
    sd, ok = masked_std(X, np.ones_like(X), axis=0)
    assert ok.all()
    np.testing.assert_allclose(sd, X.std(axis=0), rtol=1e-12)
    # a single observed row cannot support a std
    W = np.zeros_like(X)
    W[0, :] = 1.0
    sd1, ok1 = masked_std(X, W, axis=0, min_count=2)
    assert not ok1.any()
    assert (sd1 == 0.0).all()


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e300])
def test_masked_moments_are_scale_invariant(scale):
    X = RNG.standard_normal((15, 6))
    W = (RNG.random((15, 6)) > 0.3).astype(float)
    W[:5, :] = 1.0  # keep every column assessable
    m1, _ = masked_mean(X, W, axis=0)
    m2, _ = masked_mean(X * scale, W, axis=0)
    np.testing.assert_allclose(m2, m1 * scale, rtol=1e-12)
    s1, _ = masked_std(X, W, axis=0)
    s2, _ = masked_std(X * scale, W, axis=0)
    np.testing.assert_allclose(s2, s1 * scale, rtol=1e-12)


# ---------------------------------------------------------------------------
# masked standardisation
# ---------------------------------------------------------------------------

def test_standardise_equals_plain_zscore_on_full_validity():
    F = RNG.standard_normal((40, 5))
    W = np.ones_like(F)
    stats = masked_standardise_fit(F, W)
    assert stats.assessable.all()
    Z = masked_standardise(F, W, stats)
    plain = (F - F.mean(axis=0)) / F.std(axis=0)
    np.testing.assert_allclose(Z, plain, rtol=1e-11)


@pytest.mark.parametrize("scale", [1e-200, 1e300])
def test_standardise_is_scale_invariant_at_extremes(scale):
    F = RNG.standard_normal((40, 5))
    W = (RNG.random((40, 5)) > 0.2).astype(float)
    W[:10, :] = 1.0
    Z1 = masked_standardise(F, W, masked_standardise_fit(F, W))
    Z2 = masked_standardise(F * scale, W, masked_standardise_fit(F * scale, W))
    np.testing.assert_allclose(Z2, Z1, rtol=1e-10)
    assert np.all(np.isfinite(Z2))


def test_standardise_removes_degenerate_and_unassessable_features():
    F = RNG.standard_normal((40, 3))
    F[:, 1] = 5.0  # constant: scale-degenerate
    W = np.ones_like(F)
    W[4:, 2] = 0.0  # only 4 observations: below the hard ESS floor of 8
    stats = masked_standardise_fit(F, W, n_min_hard=8)
    assert stats.degenerate_scale[1] and not stats.assessable[1]
    assert not stats.assessable[2]
    assert stats.assessable[0]
    Z = masked_standardise(F, W, stats)
    assert (Z[:, 1] == 0.0).all() and (Z[:, 2] == 0.0).all()
    assert Z[:, 0].std() > 0


def test_standardise_low_support_is_flagged_not_hidden():
    F = RNG.standard_normal((20, 1))
    W = np.zeros_like(F)
    W[:10, 0] = 1.0  # ESS 10: above hard floor 8, below warn floor 30
    stats = masked_standardise_fit(F, W)
    assert stats.assessable[0] and stats.low_support[0]


def test_standardise_strict_raises_when_nothing_is_assessable():
    F = np.ones((5, 2))  # constant everywhere
    with pytest.raises(InsufficientEvidenceError, match="assessable"):
        masked_standardise_fit(F, np.ones_like(F), strict=True)


# ---------------------------------------------------------------------------
# renormalised 1D convolution
# ---------------------------------------------------------------------------

def test_filter_classification():
    assert is_contrast_filter(DERIV)
    assert not is_contrast_filter(BOX)


@pytest.mark.parametrize("filt", [DERIV, BOX])
def test_renorm_convolve_reduces_to_plain_on_full_validity(filt):
    from scipy.ndimage import convolve1d

    X = RNG.standard_normal((3, 2, 50))
    V = np.ones_like(X)
    out, v_out = renormalised_convolve1d(X, V, filt)
    plain = convolve1d(X, filt, axis=-1, mode="reflect")
    np.testing.assert_allclose(out, plain, rtol=1e-12, atol=1e-14)
    assert (v_out == 1.0).all()


def test_flat_signal_with_gap_fabricates_no_edge():
    """The Milestone 1 make-or-break property, as a unit test."""
    x = np.full(200, 3.7)
    v = np.ones(200)
    v[80:120] = 0.0  # a dropped stretch
    out, _ = renormalised_convolve1d(x, v, DERIV, min_frac=0.0)
    assert float(np.abs(out).max()) < 1e-12
    # whereas zero-fill invents an edge of the order of the signal level
    from scipy.ndimage import convolve1d

    fabricated = convolve1d(fill_zero(x, v), DERIV, axis=-1, mode="reflect")
    assert float(np.abs(fabricated).max()) > 1.0


def test_level_filter_reports_average_of_surviving_evidence():
    x = np.full(60, 2.5)
    v = np.ones(60)
    v[10:30] = 0.0
    out, v_out = renormalised_convolve1d(x, v, BOX)
    # wherever any evidence exists, the local average of a constant is itself
    np.testing.assert_allclose(out[v_out > 0], 2.5, rtol=1e-12)


def test_contrast_validity_threshold_and_override():
    x = RNG.standard_normal(40)
    v = np.ones(40)
    v[10:14] = 0.0
    filt = np.array([-1.0, 0.0, 0.0, 0.0, 1.0]) / 2.0  # width 5, contrast
    _, v_out = renormalised_convolve1d(x, v, filt)
    # a window with under half its evidence is untrusted by default
    assert (v_out[(v_out > 0)] >= 0.5).all()
    _, v_all = renormalised_convolve1d(x, v, filt, min_frac=0.0)
    assert (v_all[11:13] > 0).any() or (v_all > 0).sum() > (v_out > 0).sum()


def test_renorm_convolve_masked_payload_is_never_read():
    x = RNG.standard_normal(64)
    v = (RNG.random(64) > 0.3).astype(float)
    base, vb = renormalised_convolve1d(np.where(v > 0, x, 0.0), v, DERIV)
    for payload in (123.0, -1.0, 1e300):
        xp = np.where(v > 0, x, payload)
        out, vo = renormalised_convolve1d(xp, v, DERIV)
        assert np.array_equal(out, base)
        assert np.array_equal(vo, vb)


# ---------------------------------------------------------------------------
# renormalised head inner product
# ---------------------------------------------------------------------------

def test_renormalised_dot_reduces_to_plain_gated_sum_when_complete():
    U = RNG.standard_normal((6, 9))
    gate = RNG.random(9)
    z, ok, rho = renormalised_dot(U, gate, np.ones_like(U))
    assert ok.all() and np.allclose(rho, 1.0)
    np.testing.assert_allclose(z, U @ gate, rtol=1e-12)


def test_renormalised_dot_does_not_attenuate_under_uniform_missingness():
    """The property that pins the design choice: an available-case sum would
    return half the complete-data value here; the renormalised form returns
    it exactly."""
    U = np.ones((4, 10))
    gate = np.ones(10)
    z_full, _, _ = renormalised_dot(U, gate, np.ones_like(U))
    z_half, ok, rho = renormalised_dot(U, gate, np.full_like(U, 0.5))
    assert ok.all() and np.allclose(rho, 0.5)
    np.testing.assert_allclose(z_half, z_full, rtol=1e-12)


def test_renormalised_dot_refuses_slivers_of_evidence():
    U = np.ones((2, 10))
    gate = np.ones(10)
    validity = np.zeros_like(U)
    validity[0, :] = 1.0
    validity[1, 0] = 0.5  # rho = 0.05, below the 0.1 floor
    z, ok, rho = renormalised_dot(U, gate, validity, rho_min=0.1)
    assert ok[0] and not ok[1]
    assert np.isnan(z[1]) and np.isfinite(z[0])


def test_renormalised_dot_all_gates_closed_raises():
    with pytest.raises(InsufficientEvidenceError, match="gates"):
        renormalised_dot(np.ones((3, 4)), np.zeros(4), np.ones((3, 4)))


# ---------------------------------------------------------------------------
# observation support and the MNAR honesty guard
# ---------------------------------------------------------------------------

def test_observation_support_counts_samples_meeting_the_floor():
    V_seg = np.array([[1.0, 0.4, 0.0],
                      [1.0, 0.6, 0.0],
                      [0.2, 0.9, 0.0],
                      [1.0, 0.1, 0.0]])
    o = observation_support(V_seg, min_valid_frac=0.5)
    np.testing.assert_allclose(o, [0.75, 0.5, 0.0])


def test_missingness_association_fires_on_a_constructed_shortcut():
    rng = np.random.default_rng(3)
    n, S = 80, 12
    y = np.array([0, 1] * (n // 2))
    V_seg = np.clip(rng.random((n, S)), 0.6, 1.0)
    V_seg[:, 4] = np.where(y == 1, 0.0, 1.0)  # segment 4 observed only for class 0
    with pytest.warns(MissingnessInformativeWarning):
        report = missingness_association(V_seg, y, task="binary",
                                         n_permutations=60, seed=0)
    assert report["verdict"] == "informative"
    assert 4 in report["top_segments"].tolist()


def test_missingness_association_is_quiet_on_unrelated_missingness():
    rng = np.random.default_rng(5)
    n, S = 80, 12
    y = np.array([0, 1] * (n // 2))
    V_seg = (rng.random((n, S)) > 0.2).astype(float)  # independent of y
    report = missingness_association(V_seg, y, task="binary",
                                     n_permutations=60, seed=0)
    assert report["verdict"] == "not_informative"


def test_missingness_association_with_no_variation_cannot_implicate():
    V_seg = np.ones((30, 6))
    y = np.arange(30, dtype=float)
    report = missingness_association(V_seg, y, task="regression", n_permutations=20)
    assert report["verdict"] == "not_informative"
    assert report["p_value"] == 1.0


# ---------------------------------------------------------------------------
# fabricated-edge diagnostic
# ---------------------------------------------------------------------------

def test_fabricated_edge_diagnostic_separates_renorm_from_fills():
    x = np.full(300, 5.0)
    v = np.ones(300)
    v[100:180] = 0.0
    report = fabricated_edge(x, v, [DERIV])
    assert report["renorm"] < 1e-12
    assert report["zero"] > 0.1  # invents an edge of the order of the level
    assert report["mean"] <= report["zero"]


def test_fabricated_edge_rejects_level_filters():
    with pytest.raises(ValueError, match="contrast"):
        fabricated_edge(np.ones(50), np.ones(50), [BOX])
