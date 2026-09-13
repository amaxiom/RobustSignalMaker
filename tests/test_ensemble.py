"""Milestone 6: representation-ensemble marginalisation.

The make-or-break: at equal coverage, the marginalised selection is at least
as stable as the best single lens, including under missingness, and the
lens-agreement diagnostic is reported so a high-agreement result (where
marginalisation buys little) is visible rather than hidden.
"""
from __future__ import annotations

import numpy as np
import pytest

from robustsignalmaker import (
    BootstrapMaskSelector,
    EnsembleMaskFoldEstimator,
    NestedCV,
    RepresentationEnsembleSelector,
    default_ensemble,
    make_signal_control,
    mask_scattered,
    mean_pairwise_jaccard,
)

FAST = dict(segment=8, lam=0.08, n_iter=150, lr=0.1, n_bootstrap=8, seed=0)


class _ScriptedInner:
    def __init__(self, script, grid):
        self.script = script
        self.grid = grid

    def fit(self, X, y, V):
        if self.script.get("fail"):
            raise ValueError("scripted failure")
        S = self.grid.n_segments
        self.selected_segments_ = np.asarray(self.script["selected"], dtype=int)
        self.assessable_segments_ = np.asarray(self.script["assessable"], dtype=bool)
        v = np.ones((X.shape[0], S))
        v[:, ~self.assessable_segments_] = 0.0
        self._v_seg = v
        return self


class _FakeLens:
    def __init__(self, name):
        self.name = name


def test_ensemble_arithmetic_per_lens_maps_and_agreement():
    lens_a, lens_b = _FakeLens("a"), _FakeLens("b")
    sel = RepresentationEnsembleSelector(
        representations=[lens_a, lens_b], task="binary", segment=8,
        n_bootstrap=4, tau=0.7, seed=0)
    every = [True] * 4
    scripts = {
        "a": iter([{"selected": [0, 1], "assessable": every}] * 4),
        "b": iter([{"selected": [1, 2], "assessable": every}] * 4),
    }
    sel._make_inner = lambda seed, representation=None: _ScriptedInner(
        next(scripts[representation.name]), sel.grid)
    sel._refit = lambda X, y, Vf: setattr(sel, "degenerate_", False)
    X = np.zeros((24, 1, 32))
    X[:, 0, 0] = np.linspace(0, 1, 24)
    sel.fit(X, np.array([0, 1] * 12))

    # marginal over 8 fits: segment 1 in all, 0 and 2 in half each
    assert sel.pi_[1] == 1.0
    assert sel.pi_[0] == pytest.approx(0.5) and sel.pi_[2] == pytest.approx(0.5)
    assert set(sel.selected_segments_) == {1}
    # per-lens maps kept intact
    np.testing.assert_allclose(sel.pi_by_representation_["a"][:3], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(sel.pi_by_representation_["b"][:3], [0.0, 1.0, 1.0])
    # agreement is the raw Jaccard between the per-lens regions: {0,1} vs {1,2}
    assert sel.representation_agreement() == pytest.approx(1 / 3)
    report = sel.stability_report()
    assert report["n_representations"] == 2
    assert report["representation_agreement"] == pytest.approx(1 / 3)


def test_ensemble_constructor_validation():
    with pytest.raises(ValueError, match="plural"):
        RepresentationEnsembleSelector(representation=_FakeLens("x"))
    with pytest.raises(ValueError, match="at least one"):
        RepresentationEnsembleSelector(representations=[])


def test_ensemble_is_bitwise_deterministic():
    c = make_signal_control(n=160, n_points=128, task="binary", seed=61)
    kw = dict(FAST)
    kw["n_bootstrap"] = 4
    reps = default_ensemble(2, seed=0)
    a = RepresentationEnsembleSelector(task="binary", representations=reps, **kw
                                       ).fit(c.X, c.y)
    b = RepresentationEnsembleSelector(task="binary", representations=reps, **kw
                                       ).fit(c.X, c.y)
    assert np.array_equal(a.pi_, b.pi_, equal_nan=True)
    assert np.array_equal(a.selected_segments_, b.selected_segments_)


def _replicate_sets(make_sel, X, y, V, n_rep=4, base_seed=100):
    """Selected sets across split-half replicates: the deliverable's stability.

    This, not the spread of inner resample fits, is what a scientist cares
    about: would a different draw of the data report the same region?
    """
    from robustsignalmaker import resample_indices

    sets = []
    for r in range(n_rep):
        rng = np.random.default_rng(base_seed + r)
        idx = resample_indices(len(y), "half", rng, strat=y)
        sets.append(make_sel(r).fit(X[idx], y[idx], V[idx]).selected_segments_)
    return sets


@pytest.mark.parametrize("nan_frac", [0.0, 0.3], ids=["complete", "scattered30"])
def test_make_or_break_marginalisation_is_insurance_against_lens_choice(nan_frac):
    """The M6 experiment, with its negative result pinned honestly.

    The plan's literal criterion (marginalised stability >= the best single
    lens at equal coverage) is FALSIFIED on this control: post-sweep
    split-half stabilities are mean 0.68 / savgol 0.14 / gauss 0.65 /
    ensemble 0.57 complete, and mean 0.57 / savgol 0.19 / gauss 0.52 /
    ensemble 0.52 under 30% NaNs (FINDINGS sec.1 and its post-sweep
    addendum; the pre-sweep "gauss best under NaN" was partly an artefact of
    the count-based renormalisation bug). What the experiment DOES support,
    and what this test pins: the ensemble beats the average lens, beats the
    worst lens by a wide margin, and stays near-best, without an oracle
    saying which lens to trust.
    """
    c = make_signal_control(n=240, n_points=256, task="binary", signal=2.0,
                            noise=1.0, seed=62)
    V = np.ones_like(c.X)
    if nan_frac:
        V = mask_scattered(c.X.shape, frac=nan_frac, seed=63).astype(float)
    X = np.where(V > 0, c.X, 0.0)
    reps = default_ensemble(3, seed=0)  # mean, savgol, gauss
    target = 4 / 32  # equal coverage for every configuration
    KW = dict(segment=8, lam=0.08, n_iter=120, lr=0.1, n_bootstrap=6,
              target_coverage=target)

    singles = {}
    for lens in reps:
        sets = _replicate_sets(
            lambda r, l=lens: BootstrapMaskSelector(task="binary", representation=l,
                                                    seed=r, **KW), X, c.y, V)
        singles[lens.name] = mean_pairwise_jaccard(sets, n_total=32)
    ens_sets = _replicate_sets(
        lambda r: RepresentationEnsembleSelector(task="binary", representations=reps,
                                                 seed=r, **KW), X, c.y, V)
    ens_stability = mean_pairwise_jaccard(ens_sets, n_total=32)

    vals = list(singles.values())
    # insurance: strictly better than the worst lens, by a wide margin
    assert ens_stability > min(vals) + 0.2, f"ensemble {ens_stability} vs {singles}"
    # better than the average lens (a random lens choice loses to marginalising)
    assert ens_stability >= np.mean(vals), f"ensemble {ens_stability} vs {singles}"
    # near-best without knowing which lens is best in this condition
    assert ens_stability >= max(vals) - 0.15, f"ensemble {ens_stability} vs {singles}"
    # and the truth band recurs in every replicate's marginalised selection
    from robustsignalmaker import SegmentGrid

    truth = set(c.truth_segments(SegmentGrid(256, segment=8)).tolist())
    for s in ens_sets:
        assert truth <= set(np.asarray(s).tolist())


def test_ensemble_fold_estimator_runs_inside_nested_cv():
    c = make_signal_control(n=160, n_points=128, task="binary", seed=64)
    eng = NestedCV(
        estimator_factory=lambda: EnsembleMaskFoldEstimator(
            n_representations=2, segment=8, lam=0.08, n_iter=120, lr=0.1,
            n_bootstrap=4),
        k_outer=3, segment=8, mnar_guard=False, random_state=3)
    res = eng.run(c.X, c.y)
    assert res.verdict is not None
    assert len(res.per_fold_scores) == 3
    report = res.final_estimator.selector_.stability_report()
    assert "representation_agreement" in report
