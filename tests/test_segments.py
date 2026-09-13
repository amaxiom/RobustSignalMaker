"""Milestone 2: SegmentGrid pooling algebra and the point-wise path."""
from __future__ import annotations

import numpy as np
import pytest

from robustsignalmaker import SegmentGrid

RNG = np.random.default_rng(1)


def test_partition_including_ragged_trailing_segment():
    g = SegmentGrid(10, segment=4)
    assert g.n_segments == 3
    assert g.point_segment.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert g.segment_lengths.tolist() == [4.0, 4.0, 2.0]


def test_pool_equals_plain_segment_means_on_full_validity():
    g = SegmentGrid(20, segment=5)
    X = RNG.standard_normal((6, 3, 20))
    V = np.ones_like(X)
    F, VF = g.pool(X, V)
    assert F.shape == (6, 4, 3) and VF.shape == (6, 4, 3)
    assert (VF == 1.0).all()
    ref = X.reshape(6, 3, 4, 5).mean(axis=-1).swapaxes(1, 2)
    np.testing.assert_allclose(F, ref, rtol=1e-13)


def test_pool_ignores_masked_entries_and_reports_fractions():
    g = SegmentGrid(8, segment=4)
    X = np.zeros((1, 1, 8))
    X[0, 0, :4] = [1.0, 3.0, 999.0, 999.0]
    V = np.ones_like(X)
    V[0, 0, 2:4] = 0.0  # the 999 payloads are unobserved
    V[0, 0, 4:] = 0.0   # second segment entirely unobserved
    F, VF = g.pool(X, V)
    assert F[0, 0, 0] == 2.0 and VF[0, 0, 0] == 0.5
    assert F[0, 1, 0] == 0.0 and VF[0, 1, 0] == 0.0  # placeholder, flagged


def test_pool_ragged_segment_validity_uses_true_length():
    g = SegmentGrid(10, segment=4)  # last segment holds 2 points
    X = np.ones((1, 1, 10))
    V = np.ones_like(X)
    V[0, 0, 9] = 0.0
    _, VF = g.pool(X, V)
    assert VF[0, 2, 0] == 0.5  # 1 of 2 points, not 1 of 4


def test_segment_one_is_exact_pointwise_mode():
    g = SegmentGrid(12, segment=1)
    assert g.n_segments == 12
    X = RNG.standard_normal((4, 2, 12))
    V = (RNG.random((4, 2, 12)) > 0.3).astype(float)
    F, VF = g.pool(X, V)
    np.testing.assert_array_equal(F, np.where(V > 0, X, 0.0).swapaxes(1, 2))
    np.testing.assert_array_equal(VF, V.swapaxes(1, 2))


def test_segment_validity_averages_channels():
    g = SegmentGrid(8, segment=4)
    V = np.ones((1, 2, 8))
    V[0, 1, :] = 0.0  # channel 1 never observed
    sv = g.segment_validity(V)
    np.testing.assert_allclose(sv, [[0.5, 0.5]])


def test_expand_and_segments_to_points_roundtrip():
    g = SegmentGrid(10, segment=4)
    seg = np.array([1.0, 2.0, 3.0])
    pt = g.expand(seg)
    assert pt.tolist() == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
    assert g.segments_to_points([2]).tolist() == [8, 9]
    # expand broadcasts over leading axes
    stacked = g.expand(np.stack([seg, seg + 10]))
    assert stacked.shape == (2, 10)


@pytest.mark.parametrize("scale", [1e-200, 1e300])
def test_pool_is_scale_invariant(scale):
    g = SegmentGrid(32, segment=8)
    X = RNG.standard_normal((5, 2, 32))
    V = (RNG.random((5, 2, 32)) > 0.4).astype(float)
    F1, VF1 = g.pool(X, V)
    F2, VF2 = g.pool(X * scale, V)
    np.testing.assert_allclose(F2, F1 * scale, rtol=1e-12)
    np.testing.assert_array_equal(VF1, VF2)


def test_shape_validation_fails_loudly():
    g = SegmentGrid(10, segment=4)
    with pytest.raises(ValueError, match="n, C, T"):
        g.pool(np.zeros((3, 10)), np.zeros((3, 10)))
    with pytest.raises(ValueError, match="does not match"):
        g.pool(np.zeros((2, 1, 12)), np.zeros((2, 1, 12)))
    with pytest.raises(ValueError, match="n_segments"):
        g.expand(np.zeros(5))
    with pytest.raises(ValueError):
        SegmentGrid(10, segment=0)
