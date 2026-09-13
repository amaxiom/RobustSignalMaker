"""Milestone 2: synthetic control, and the milestone's make-or-break experiment.

Make-or-break: every lens separates the planted informative band from the
background AND the equal-variance distractor at feature level, with and
without scattered NaNs. Separation is judged by ranking segments on their
strongest evidence-weighted association with the target.
"""
from __future__ import annotations

import numpy as np
import pytest

from robustsignalmaker import (
    SegmentGrid,
    default_ensemble,
    make_signal_control,
    mask_scattered,
)

SEG = 8


def weighted_corr_scores(F, FV, y):
    """Per-segment max over feature channels of |weighted corr with y|.

    Weights are the feature validities, so unobserved evidence neither helps
    nor hurts a segment's score.
    """
    y = np.asarray(y, dtype=float)
    n, S, K = F.shape
    scores = np.zeros(S)
    for s in range(S):
        best = 0.0
        for k in range(K):
            w = FV[:, s, k]
            if w.sum() <= 2:
                continue
            f = F[:, s, k]
            wm_f = np.average(f, weights=w)
            wm_y = np.average(y, weights=w)
            cov = np.average((f - wm_f) * (y - wm_y), weights=w)
            vf = np.average((f - wm_f) ** 2, weights=w)
            vy = np.average((y - wm_y) ** 2, weights=w)
            if vf > 0 and vy > 0:
                best = max(best, abs(cov) / np.sqrt(vf * vy))
        scores[s] = best
    return scores


def test_control_shapes_balance_and_determinism():
    c1 = make_signal_control(n=60, n_channels=2, n_points=128, task="binary", seed=3)
    c2 = make_signal_control(n=60, n_channels=2, n_points=128, task="binary", seed=3)
    assert c1.X.shape == (60, 2, 128) and c1.y.shape == (60,)
    assert np.array_equal(c1.X, c2.X) and np.array_equal(c1.y, c2.y)
    assert c1.informative.sum() == 3 * 8
    assert not (c1.informative & c1.distractor).any()
    # binary labels are reasonably balanced (sign of a latent sum)
    assert 0.3 <= c1.y.mean() <= 0.7


def test_multiclass_targets_are_balanced_by_construction():
    c = make_signal_control(n=90, task="multiclass", n_classes=3, seed=1)
    counts = np.bincount(c.y)
    assert len(counts) == 3 and counts.min() >= 25


def test_truth_and_distractor_segments():
    c = make_signal_control(n=10, n_points=128, seed=0)
    grid = SegmentGrid(128, segment=SEG)
    truth = c.truth_segments(grid)
    dist = c.distractor_segments(grid)
    assert len(truth) >= 3 and len(dist) >= 3
    assert not set(truth) & set(dist)
    assert c.informative[grid.segments_to_points(truth)].mean() >= 0.5


def test_mask_scattered_rate_and_determinism():
    V = mask_scattered((50, 2, 100), frac=0.25, seed=4)
    assert V.dtype == bool
    assert abs((~V).mean() - 0.25) < 0.02
    assert np.array_equal(V, mask_scattered((50, 2, 100), frac=0.25, seed=4))
    with pytest.raises(ValueError):
        mask_scattered((3, 3), frac=1.0)


@pytest.mark.parametrize("lens", default_ensemble(5, seed=0), ids=lambda l: l.name)
@pytest.mark.parametrize("nan_frac", [0.0, 0.2], ids=["complete", "scattered20"])
def test_make_or_break_every_lens_separates_the_planted_band(lens, nan_frac):
    c = make_signal_control(n=240, n_channels=1, n_points=256, task="regression",
                            signal=2.0, noise=1.0, seed=11)
    grid = SegmentGrid(256, segment=SEG)
    V = np.ones_like(c.X)
    if nan_frac > 0:
        V = mask_scattered(c.X.shape, frac=nan_frac, seed=12).astype(float)
    X = np.where(V > 0, c.X, 0.0)
    F, FV = lens.transform(X, V, grid)
    scores = weighted_corr_scores(F, FV, c.y)
    truth = set(c.truth_segments(grid).tolist())
    # the top-ranked segment must lie in the informative band, and the band
    # must outscore the background (which includes the distractor) on average
    assert int(np.argmax(scores)) in truth, f"{lens.name} ranked a background segment first"
    truth_idx = sorted(truth)
    bg_idx = [s for s in range(grid.n_segments) if s not in truth]
    assert scores[truth_idx].mean() > 2.0 * scores[bg_idx].mean(), (
        f"{lens.name} does not separate the planted band"
    )
