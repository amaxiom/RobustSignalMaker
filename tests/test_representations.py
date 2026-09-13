"""Milestone 2: lenses. The lens invariant, validity reduction, payload hygiene."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import convolve1d

from robustsignalmaker import (
    GaussianScale1D,
    RandomConv1D,
    SavGolDerivative,
    SegmentGrid,
    SegmentMean,
    SegmentStats,
    default_ensemble,
)

RNG = np.random.default_rng(2)
N, C, T, SEG = 5, 2, 64, 8
GRID = SegmentGrid(T, segment=SEG)
X = RNG.standard_normal((N, C, T))
ONES = np.ones_like(X)

ALL_LENSES = [SegmentMean(), SegmentStats(), SavGolDerivative(),
              GaussianScale1D(), RandomConv1D(seed=7)]


def plain_pool(A):
    """Unmasked per-segment means, (n, C, T) -> (n, S, C)."""
    return A.reshape(N, C, T // SEG, SEG).mean(axis=-1).swapaxes(1, 2)


def unmasked_reference(lens):
    """Each lens's plain (validity-free) form, built independently here."""
    if isinstance(lens, SegmentMean):
        return plain_pool(X)
    if isinstance(lens, SegmentStats):
        mean = plain_pool(X)
        std = X.reshape(N, C, T // SEG, SEG).std(axis=-1).swapaxes(1, 2)
        return np.concatenate([mean, std], axis=2)
    if isinstance(lens, SavGolDerivative):
        out = convolve1d(X, lens._kernel(), axis=-1, mode="reflect")
        return plain_pool(np.abs(out))
    if isinstance(lens, GaussianScale1D):
        blocks = [plain_pool(convolve1d(X, lens._kernel(s), axis=-1, mode="reflect"))
                  for s in lens.sigmas]
        return np.concatenate(blocks, axis=2)
    if isinstance(lens, RandomConv1D):
        blocks = [plain_pool(np.maximum(convolve1d(X, f, axis=-1, mode="reflect"), 0.0))
                  for f in lens._filters()]
        return np.concatenate(blocks, axis=2)
    raise AssertionError(f"no reference for {lens!r}")


@pytest.mark.parametrize("lens", ALL_LENSES, ids=lambda l: l.name)
def test_lens_reduces_exactly_to_unmasked_form_on_complete_data(lens):
    F, FV = lens.transform(X, ONES, GRID)
    assert (FV == 1.0).all()
    np.testing.assert_allclose(F, unmasked_reference(lens), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("lens", ALL_LENSES, ids=lambda l: l.name)
def test_lens_obeys_the_coordinate_invariant(lens):
    F, FV = lens.transform(X, ONES, GRID)
    # features are indexed by the grid's segments, whatever the channel count
    assert F.shape[0] == N and F.shape[1] == GRID.n_segments
    assert F.shape == FV.shape


@pytest.mark.parametrize("lens", ALL_LENSES, ids=lambda l: l.name)
def test_lens_never_reads_masked_payloads(lens):
    V = (RNG.random(X.shape) > 0.3).astype(float)
    base_F, base_FV = lens.transform(np.where(V > 0, X, 0.0), V, GRID)
    for payload in (-1.0, 1e300, np.nan):
        Xp = np.where(V > 0, X, payload)
        F, FV = lens.transform(Xp, V, GRID)
        assert np.array_equal(F, base_F), f"{lens.name} read a masked payload"
        assert np.array_equal(FV, base_FV)


def test_segment_stats_std_needs_two_observed_points():
    V = np.zeros_like(X)
    V[:, :, 0] = 1.0  # one observed point in the first segment, none elsewhere
    F, FV = SegmentStats().transform(X, V, GRID)
    S = GRID.n_segments
    std_block_FV = FV[:, :, C:]
    assert (std_block_FV == 0.0).all()  # a std from one point is not evidence
    mean_block_FV = FV[:, :, :C]
    assert (mean_block_FV[:, 0, :] > 0).all()
    assert (mean_block_FV[:, 1:, :] == 0.0).all()
    assert F.shape == (N, S, 2 * C)


def test_segment_stats_is_scale_safe_at_extremes():
    for scale in (1e-200, 1e300):
        F1, _ = SegmentStats().transform(X, ONES, GRID)
        F2, _ = SegmentStats().transform(X * scale, ONES, GRID)
        np.testing.assert_allclose(F2, F1 * scale, rtol=1e-10)
        assert np.all(np.isfinite(F2))


def test_channel_permutation_equivariance():
    perm = np.array([1, 0])
    Xp = X[:, perm, :]
    F, _ = SegmentMean().transform(X, ONES, GRID)
    Fp, _ = SegmentMean().transform(Xp, ONES, GRID)
    np.testing.assert_array_equal(Fp, F[:, :, perm])
    # block-per-filter layout: each filter block permutes within itself
    lens = GaussianScale1D()
    G, _ = lens.transform(X, ONES, GRID)
    Gp, _ = lens.transform(Xp, ONES, GRID)
    for b in range(len(lens.sigmas)):
        np.testing.assert_array_equal(
            Gp[:, :, b * C:(b + 1) * C], G[:, :, b * C:(b + 1) * C][:, :, perm]
        )


def test_gap_in_flat_signal_gives_zero_derivative_energy():
    flat = np.full((1, 1, T), 4.2)
    V = np.ones_like(flat)
    V[0, 0, 24:40] = 0.0
    F, FV = SavGolDerivative().transform(flat, V, GRID)
    assert float(np.abs(F).max()) < 1e-12  # no fabricated band at the gap


def test_default_ensemble_is_diverse_and_deterministic():
    ens = default_ensemble(4, seed=0)
    assert [l.name for l in ens] == ["mean", "savgol_d1w11", "gauss2_8", "randconv1000"]
    assert len({l.name for l in default_ensemble(9, seed=0)}) == 9
    a = default_ensemble(4, seed=0)[3]._filters()
    b = default_ensemble(4, seed=0)[3]._filters()
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        default_ensemble(0)


def test_lens_shape_validation_fails_loudly():
    with pytest.raises(ValueError, match="n, C, T"):
        SegmentMean().transform(np.zeros((3, T)), np.zeros((3, T)), GRID)
    with pytest.raises(ValueError, match="does not match"):
        SegmentMean().transform(np.zeros((2, 1, T + 1)), np.zeros((2, 1, T + 1)), GRID)
    with pytest.raises(ValueError):
        SavGolDerivative(window=4)
    with pytest.raises(ValueError):
        GaussianScale1D(sigmas=())
