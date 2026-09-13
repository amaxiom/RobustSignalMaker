"""Milestone 1: config surface and metrics (the RPM-style unit sweep)."""
from __future__ import annotations

import numpy as np
import pytest

from robustsignalmaker import (
    RSMConfig,
    adjusted_jaccard,
    expected_jaccard,
    fit_count,
    jaccard,
    mean_pairwise_jaccard,
    paired_comparison,
    rmse_from_score,
    score_predictions,
)


def test_config_defaults_match_the_adopted_contract():
    cfg = RSMConfig()
    assert cfg.tau == 0.7  # RMM/RPM convention
    assert cfg.segment == 16 and cfg.channel_mode == "joint"
    assert cfg.min_valid_frac == 0.5
    assert cfg.rho_min == 0.1 and cfg.o_min == 0.1
    assert cfg.a_min == 0.5 and cfg.o_floor == 0.2
    assert cfg.n_min_hard == 8 and cfg.n_min_warn == 30
    assert cfg.use_unlabelled_stats is False
    d = cfg.to_dict()
    assert d["gate"] == "hardconcrete"


def test_fit_count_formula():
    # F = K * R_out * per + per, per = R_rep * B + n_iter * L + 1
    per = 4 * 20 + 20 * 5 + 1
    assert fit_count(5, 1, 4, 20, 20, 5) == 5 * per + per


def test_score_conventions():
    y = np.array([0.0, 1.0, 2.0])
    s = score_predictions("regression", y, y_pred=y + 1.0)
    assert s == pytest.approx(-1.0)
    assert rmse_from_score(s) == pytest.approx(1.0)
    auc = score_predictions("binary", np.array([0, 1, 0, 1]),
                            y_proba=np.array([0.1, 0.9, 0.2, 0.8]))
    assert auc == 1.0


def test_score_undefined_fold_returns_nan_not_a_crash():
    assert np.isnan(score_predictions("binary", np.array([1, 1, 1]),
                                      y_proba=np.array([0.5, 0.5, 0.5])))


def test_multiclass_score_and_unknown_task():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.full((6, 3), 1e-3)
    proba[np.arange(6), y] = 0.998
    assert score_predictions("multiclass", y, y_proba=proba) == 1.0
    # an undefined multiclass fold (a class absent) is nan, not a crash
    assert np.isnan(score_predictions("multiclass", np.zeros(4, dtype=int),
                                      y_proba=np.full((4, 3), 1 / 3)))
    with pytest.raises(ValueError, match="unknown task"):
        score_predictions("ranking", y, y_pred=y)


def test_adjusted_jaccard_saturated_chance_baseline():
    # two empty sets: expected Jaccard is 1, and the corrected index is 1
    assert adjusted_jaccard([], [], 10) == 1.0
    # adjusted path through mean_pairwise_jaccard
    val = mean_pairwise_jaccard([[1, 2], [2, 3]], n_total=10)
    assert -1.0 <= val <= 1.0


def test_paired_comparison_degenerate_inputs_are_undecidable_not_confident():
    # a single fold pair supports no verdict; since the 2026-09-01 sweep the
    # outcome is "undecidable", never a confident-looking "preserved"
    res = paired_comparison([0.5], [0.9])
    assert res.outcome == "undecidable"
    assert np.isnan(res.p_ttest) and np.isnan(res.mean_delta)


def test_jaccard_family():
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)
    assert jaccard([], []) == 1.0
    assert expected_jaccard(0, 0, 10) == 1.0
    # identical sets are perfectly stable even after chance correction
    assert adjusted_jaccard([1, 2, 3], [1, 2, 3], 100) == pytest.approx(1.0)
    # chance-level overlap scores about zero after correction
    a, b = list(range(0, 50)), list(range(25, 75))
    assert abs(adjusted_jaccard(a, b, 100)) < 0.02


def test_empty_selection_never_scores_perfect_stability():
    # the RPM hazard: J(empty, empty) = 1 would crown a collapsed selector
    assert np.isnan(mean_pairwise_jaccard([[], []], min_size=1))
    assert mean_pairwise_jaccard([[1, 2], [1, 2]]) == 1.0
    assert np.isnan(mean_pairwise_jaccard([[1, 2]]))  # fewer than two sets


def test_paired_comparison_outcomes():
    base = np.array([0.70, 0.72, 0.71, 0.69, 0.73, 0.70, 0.71, 0.72])
    same = paired_comparison(base, base)
    assert same.outcome == "preserved" and same.mean_delta == 0.0
    better = paired_comparison(base + 0.1, base)
    assert better.outcome == "sig.better" and better.mean_delta == pytest.approx(0.1)
    worse = paired_comparison(base - 0.1, base)
    assert worse.outcome == "sig.worse"
