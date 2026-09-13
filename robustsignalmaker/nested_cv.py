"""Leakage-safe nested cross-validation with validity and missing-y (Milestone 5).

The leakage-safety contract is STRUCTURAL, not conventional (per RPM): the
engine slices train/test indices and only ever passes the TRAIN partition to
a fold estimator's ``fit``; the held-out partition is touched only by
``predict`` at scoring time. The guarantee is proved by canary tests (a probe
estimator that hashes its training rows and raises if ``predict`` ever sees
one), not asserted by convention.

RSM extensions over the RPM engine:

  * ``run(X, y, V=None, groups=None)``: the validity mask travels with the
    split automatically (it is per-sample data), and every fitted statistic
    lives inside the fold estimator, so missing data adds no new leakage
    channel.
  * Missing y: rows with nan targets are excluded from splits, fits, scoring
    and resampling (a test-fold row without y contributes nothing anywhere;
    covariate leakage is still leakage). They are counted in the result.
    ``use_unlabelled_stats=True`` additionally hands the unlabelled
    TRAINING-partition rows to the fold estimator as ``X_unlabelled`` /
    ``V_unlabelled`` for X-only statistics; estimators that cannot use them
    ignore them.
  * Refusal-aware scoring: a fold estimator may return nan predictions for
    samples below its evidence floor. Folds are scored over the samples the
    model was willing to score; refusals are counted per fold and in total,
    never silently imputed.
  * The MNAR honesty guard runs on every training fold
    (:func:`~robustsignalmaker.validity.missingness_association`) and its
    three-valued verdict is stamped into the fold record and the result.
  * The full-signal baseline is built in: a matched fold estimator with
    every segment retained runs on the same partitions, and the result
    carries the paired preserved / sig.better / sig.worse verdict (RMM
    sec.6.1), because "preserved performance at reduced coverage" is the
    claim the library exists to test.
"""
from __future__ import annotations

import time
import warnings
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .masking import SoftMaskSelector
from .metrics import mean_pairwise_jaccard, score_predictions
from .reproducibility import Seeds
from .results import RSMResult
from .segments import SegmentGrid
from .selection import BootstrapMaskSelector
from .validity import as_validity, missingness_association


# --------------------------------------------------------------------------- #
# Task inference & splitters (RMM-style, ported from RPM)
# --------------------------------------------------------------------------- #
def infer_task(y) -> str:
    """Infer 'binary' | 'multiclass' | 'regression' from the LABELLED targets."""
    y = np.asarray(y)
    if y.dtype.kind == "f":
        y = y[~np.isnan(y)]
        if np.all(np.equal(np.mod(y, 1), 0)) and len(np.unique(y)) <= max(20, int(0.05 * len(y))):
            return "binary" if len(np.unique(y)) == 2 else "multiclass"
        return "regression"
    return "binary" if len(np.unique(y)) == 2 else "multiclass"


def make_outer_splitter(task: str, groups, n_splits: int, seed: int):
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedGroupKFold,
        StratifiedKFold,
    )

    if groups is not None:
        # instrument/session/subject grouping: no group may span train and test
        if task in ("binary", "multiclass"):
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return GroupKFold(n_splits=n_splits)  # deterministic by group order
    if task in ("binary", "multiclass"):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_inner_splitter(task: str, grouped: bool, n_splits: int, seed: int):
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedGroupKFold,
        StratifiedKFold,
    )

    if grouped:
        if task in ("binary", "multiclass"):
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return GroupKFold(n_splits=n_splits)
    if task in ("binary", "multiclass"):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


# --------------------------------------------------------------------------- #
# Fold-estimator contract and adapters
# --------------------------------------------------------------------------- #
@runtime_checkable
class FoldEstimator(Protocol):
    """Contract for the per-fold model. ``fit`` sees TRAIN data only.

    ``V`` is the validity mask for the training partition. ``X_unlabelled`` /
    ``V_unlabelled`` are optional unlabelled TRAINING rows for X-only
    statistics; an estimator may ignore them. ``predict`` / ``predict_proba``
    may return nan rows for samples below the estimator's evidence floor.
    """

    def fit(self, X, y, *, V, task: str, seeds: Seeds, fold_idx: int, repeat: int,
            inner_splitter, groups=None, X_unlabelled=None,
            V_unlabelled=None) -> "FoldEstimator": ...

    def predict(self, X, V=None): ...

    def predict_proba(self, X, V=None): ...

    selected_: np.ndarray


class SoftMaskFoldEstimator:
    """Single learned-mask fold estimator (everything fitted inside ``fit``)."""

    def __init__(self, **selector_kwargs):
        self.kw = selector_kwargs
        self.selected_ = np.array([], dtype=int)

    def fit(self, X, y, *, V, task, seeds, fold_idx, repeat, inner_splitter,
            groups=None, X_unlabelled=None, V_unlabelled=None):
        seed = 0 if seeds is None else seeds.outer(fold_idx, repeat)
        self.selector_ = SoftMaskSelector(task=task, seed=int(seed), **self.kw).fit(X, y, V)
        self.selected_ = self.selector_.selected_segments_
        # the stability universe is the ASSESSABLE segments (metrics contract)
        self.n_candidates_ = int(self.selector_.assessable_segments_.sum())
        self.classes_ = getattr(self.selector_, "classes_", None)
        return self

    def predict(self, X, V=None):
        return self.selector_.predict(X, V)

    def predict_proba(self, X, V=None):
        return self.selector_.predict_proba(X, V)


class BootstrapMaskFoldEstimator:
    """Stability-selection fold estimator: the aggregation, threshold choice
    and final refit all happen inside ``fit`` on the training partition."""

    def __init__(self, **selector_kwargs):
        self.kw = selector_kwargs
        self.selected_ = np.array([], dtype=int)

    def fit(self, X, y, *, V, task, seeds, fold_idx, repeat, inner_splitter,
            groups=None, X_unlabelled=None, V_unlabelled=None):
        seed = 0 if seeds is None else seeds.outer(fold_idx, repeat)
        # caller-supplied groups reach the selector's resample stratification
        # (2026-09-01 sweep: the adapter silently dropped them)
        self.selector_ = BootstrapMaskSelector(task=task, seed=int(seed), **self.kw
                                               ).fit(X, y, V, obs_groups=groups)
        self.selected_ = self.selector_.selected_segments_
        self.pi_ = self.selector_.pi_
        self.support_ = self.selector_.support_
        self.n_candidates_ = self.selector_.assessable_universe()
        self.classes_ = getattr(self.selector_.model_, "classes_", None)
        return self

    def predict(self, X, V=None):
        return self.selector_.predict(X, V)

    def predict_proba(self, X, V=None):
        return self.selector_.predict_proba(X, V)


class EnsembleMaskFoldEstimator:
    """Representation-ensemble fold estimator (Milestone 6)."""

    def __init__(self, n_representations: int = 4, representations=None,
                 **selector_kwargs):
        self.n_representations = n_representations
        self.representations = representations
        self.kw = selector_kwargs
        self.selected_ = np.array([], dtype=int)

    def fit(self, X, y, *, V, task, seeds, fold_idx, repeat, inner_splitter,
            groups=None, X_unlabelled=None, V_unlabelled=None):
        from .selection import RepresentationEnsembleSelector

        seed = 0 if seeds is None else seeds.outer(fold_idx, repeat)
        self.selector_ = RepresentationEnsembleSelector(
            task=task, seed=int(seed), representations=self.representations,
            n_representations=self.n_representations, **self.kw
        ).fit(X, y, V, obs_groups=groups)
        self.selected_ = self.selector_.selected_segments_
        self.pi_ = self.selector_.pi_
        self.support_ = self.selector_.support_
        self.n_candidates_ = self.selector_.assessable_universe()
        self.classes_ = getattr(self.selector_.model_, "classes_", None)
        return self

    def predict(self, X, V=None):
        return self.selector_.predict(X, V)

    def predict_proba(self, X, V=None):
        return self.selector_.predict_proba(X, V)


class FullSignalFoldEstimator:
    """The matched baseline: the same head with EVERY segment retained.

    This is what the verdict compares against; "preserved" means the selected
    region lost nothing the full signal had.
    """

    def __init__(self, segment: int = 16, refit_model: str = "head",
                 **selector_kwargs):
        self.segment = segment
        self.refit_model = refit_model
        self.kw = selector_kwargs
        self.selected_ = np.array([], dtype=int)
        self.external_ = None

    def fit(self, X, y, *, V, task, seeds, fold_idx, repeat, inner_splitter,
            groups=None, X_unlabelled=None, V_unlabelled=None):
        seed = 0 if seeds is None else seeds.outer(fold_idx, repeat)
        grid = SegmentGrid.from_signal_shape(np.asarray(X).shape, segment=self.segment)
        S = grid.n_segments
        self.selector_ = SoftMaskSelector(task=task, seed=int(seed), segment=self.segment,
                                          fixed_mask=np.ones(S), **self.kw).fit(X, y, V)
        self.selected_ = self.selector_.selected_segments_
        self.n_candidates_ = int(self.selector_.assessable_segments_.sum())
        self.classes_ = getattr(self.selector_, "classes_", None)
        if self.refit_model != "head":
            # the baseline must use the SAME model as the selection arm, or
            # the verdict conflates the choice of region with the choice of
            # model, which is the one thing it exists to isolate
            from .models import ExternalRefit
            from .validity import as_validity

            Xa, Vb = as_validity(X, V)
            self.external_ = ExternalRefit(
                self.refit_model, task, grid, np.arange(S), seed=int(seed),
                rho_min=self.kw.get("rho_min", 0.1),
                n_min_hard=self.kw.get("n_min_hard", 8),
                n_min_warn=self.kw.get("n_min_warn", 30),
            ).fit(Xa, Vb.astype(float), y)
            self.classes_ = self.external_.classes_
        return self

    def predict(self, X, V=None):
        if self.external_ is not None:
            return self.external_.predict(X, V)
        return self.selector_.predict(X, V)

    def predict_proba(self, X, V=None):
        if self.external_ is not None:
            return self.external_.predict_proba(X, V)
        return self.selector_.predict_proba(X, V)


# --------------------------------------------------------------------------- #
# Refusal-aware scoring
# --------------------------------------------------------------------------- #
def _score_with_refusals(task, y_true, pred=None, proba=None):
    """(score, n_refused): score over the samples the model would score."""
    if task == "regression":
        ok = np.isfinite(np.asarray(pred))
        n_ref = int((~ok).sum())
        if not ok.any():
            return float("nan"), n_ref
        return score_predictions(task, y_true[ok], y_pred=np.asarray(pred)[ok]), n_ref
    proba = np.asarray(proba)
    ok = np.isfinite(proba).all(axis=1)
    n_ref = int((~ok).sum())
    if not ok.any():
        return float("nan"), n_ref
    return score_predictions(task, y_true[ok], y_proba=proba[ok]), n_ref


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class NestedCV:
    """Leakage-safe nested CV over a :class:`FoldEstimator` factory.

    ``estimator_factory`` defaults to the bootstrap stability selector;
    ``baseline_factory`` defaults to the matched full-signal estimator (pass
    ``baseline=False`` to skip the verdict). ``segment`` is used for the
    MNAR guard's validity features and the default estimators.
    """

    def __init__(self, estimator_factory=None, *, baseline: bool = True,
                 baseline_factory=None, k_outer: int = 5, l_inner: int = 5,
                 repeated_outer_cv: int = 1, segment: int = 16,
                 use_unlabelled_stats: bool = False, mnar_guard: bool = True,
                 mnar_permutations: int = 100, random_state: int = 0,
                 **default_estimator_kwargs):
        self.estimator_factory = estimator_factory or (
            lambda: BootstrapMaskFoldEstimator(segment=segment,
                                               **default_estimator_kwargs))
        self.baseline = baseline
        self.baseline_factory = baseline_factory or (
            lambda: FullSignalFoldEstimator(segment=segment))
        self.k_outer = k_outer
        self.l_inner = l_inner
        self.repeated_outer_cv = repeated_outer_cv
        self.segment = segment
        self.use_unlabelled_stats = use_unlabelled_stats
        self.mnar_guard = mnar_guard
        self.mnar_permutations = mnar_permutations
        self.random_state = random_state

    def _labelled_mask(self, y):
        y = np.asarray(y)
        if y.dtype.kind == "f":
            return ~np.isnan(y)
        if y.dtype == object:
            # a pandas round trip commonly delivers object arrays with real
            # nan/None holes; these are missing labels, not a class
            labelled = np.array([not (v is None or (isinstance(v, float)
                                                    and np.isnan(v)))
                                 for v in y])
            suspicious = [v for v in y[labelled]
                          if isinstance(v, str) and v.strip().lower() in ("nan", "")]
            if suspicious:
                warnings.warn(
                    f"y contains string labels that look like missing-value "
                    f"artifacts ({sorted(set(suspicious))!r}); they are being "
                    f"treated as CLASSES. Convert them to real nan if they "
                    f"mean missing.", stacklevel=2)
            return labelled
        return np.ones(len(y), dtype=bool)

    @staticmethod
    def _aligned_proba(proba, fold_classes, classes):
        """Map a fold estimator's proba columns onto the engine's class list.

        With grouped CV a class can be absent from a training fold, so the
        fold model legitimately knows fewer classes (2026-09-01 sweep: this
        crashed the out-of-fold accumulator). Unknown classes get probability
        zero, which is what the fold model asserts; without a class list the
        fold is treated as refused rather than mis-broadcast.
        """
        proba = np.asarray(proba)
        if proba.shape[1] == len(classes):
            return proba
        full = np.full((proba.shape[0], len(classes)), np.nan)
        if fold_classes is None:
            return full  # cannot align: refuse the fold loudly downstream
        full[:] = 0.0
        refused = ~np.isfinite(proba).all(axis=1)
        for j, cls in enumerate(fold_classes):
            pos = np.flatnonzero(classes == cls)
            if pos.size:
                full[:, pos[0]] = proba[:, j]
        full[refused] = np.nan
        return full

    def run(self, X, y, V=None, groups=None) -> RSMResult:
        X, Vb = as_validity(X, V)
        y = np.asarray(y)
        groups = None if groups is None else np.asarray(groups)
        n_all = X.shape[0]
        if y.shape[0] != n_all:
            raise ValueError(f"y has {y.shape[0]} rows, X has {n_all}")

        labelled = self._labelled_mask(y)
        lab_idx = np.flatnonzero(labelled)
        unlab_idx = np.flatnonzero(~labelled)
        if lab_idx.size < 2 * self.k_outer:
            raise ValueError(
                f"only {lab_idx.size} labelled samples for k_outer={self.k_outer}")
        Xl, yl, Vl = X[lab_idx], y[lab_idx], Vb[lab_idx]
        gl = None if groups is None else groups[lab_idx]

        task = infer_task(yl)
        uniq = np.unique(yl.astype(str)) if yl.dtype == object else np.unique(yl)
        if len(uniq) < 2:
            raise ValueError(
                "y is constant over the labelled samples; nothing to model")
        seeds = Seeds(base=self.random_state)
        classes = np.unique(yl) if task != "regression" else None
        grid = SegmentGrid.from_signal_shape(X.shape, segment=self.segment)

        repeats = self.repeated_outer_cv
        if repeats > 1 and gl is not None and task == "regression":
            # GroupKFold has no shuffle: every repeat would duplicate the same
            # folds and pseudo-replication would manufacture significance
            # (2026-09-01 sweep). One honest pass instead, said out loud.
            warnings.warn(
                "repeated_outer_cv forced to 1: grouped regression splits are "
                "deterministic and repeats would duplicate folds", stacklevel=2)
            repeats = 1

        if task == "regression":
            oof_sum = np.zeros(n_all)
        else:
            oof_sum = np.zeros((n_all, len(classes)))
        oof_count = np.zeros(n_all, dtype=int)

        fold_records, mnar_reports = [], []
        scores, base_scores, selected_per_fold = [], [], []
        n_candidates = 0
        total_refused = 0

        for repeat in range(repeats):
            outer = make_outer_splitter(task, gl, self.k_outer, seeds.outer(0, repeat))
            for fold_idx, (tr, te) in enumerate(outer.split(Xl, yl, gl)):
                t0 = time.perf_counter()
                inner = make_inner_splitter(task, gl is not None, self.l_inner,
                                            seeds.inner(fold_idx, repeat))
                fit_kwargs = dict(
                    V=Vl[tr], task=task, seeds=seeds, fold_idx=fold_idx,
                    repeat=repeat, inner_splitter=inner,
                    groups=None if gl is None else gl[tr])
                if self.use_unlabelled_stats and unlab_idx.size:
                    fit_kwargs["X_unlabelled"] = X[unlab_idx]
                    fit_kwargs["V_unlabelled"] = Vb[unlab_idx]

                mnar = {"verdict": "not_run"}
                if self.mnar_guard:
                    v_seg_tr = grid.segment_validity(Vl[tr].astype(float))
                    mnar = missingness_association(
                        v_seg_tr, yl[tr], task,
                        n_permutations=self.mnar_permutations,
                        seed=seeds.inner(fold_idx, repeat) + 1)
                    mnar["top_segments"] = [int(s) for s in
                                            np.asarray(mnar.get("top_segments", [])).ravel()]
                mnar_reports.append(mnar)

                est = self.estimator_factory()
                est.fit(Xl[tr], yl[tr], **fit_kwargs)
                orig_te = lab_idx[te]
                if task == "regression":
                    pred = est.predict(Xl[te], Vl[te])
                    score, n_ref = _score_with_refusals(task, yl[te], pred=pred)
                    ok = np.isfinite(np.asarray(pred))
                    oof_sum[orig_te[ok]] += np.asarray(pred)[ok]
                else:
                    proba = self._aligned_proba(est.predict_proba(Xl[te], Vl[te]),
                                                getattr(est, "classes_", None),
                                                classes)
                    score, n_ref = _score_with_refusals(task, yl[te], proba=proba)
                    ok = np.isfinite(np.asarray(proba)).all(axis=1)
                    oof_sum[orig_te[ok]] += np.asarray(proba)[ok]
                oof_count[orig_te[ok]] += 1
                total_refused += n_ref

                base_score = float("nan")
                if self.baseline:
                    base = self.baseline_factory()
                    base.fit(Xl[tr], yl[tr], **fit_kwargs)
                    if task == "regression":
                        base_score, _ = _score_with_refusals(
                            task, yl[te], pred=base.predict(Xl[te], Vl[te]))
                    else:
                        base_proba = self._aligned_proba(
                            base.predict_proba(Xl[te], Vl[te]),
                            getattr(base, "classes_", None), classes)
                        base_score, _ = _score_with_refusals(
                            task, yl[te], proba=base_proba)

                scores.append(score)
                base_scores.append(base_score)
                selected_per_fold.append(np.asarray(est.selected_, dtype=int))
                n_candidates = max(n_candidates, int(getattr(est, "n_candidates_", 0)))
                fold_records.append({
                    "repeat": repeat, "fold": fold_idx,
                    "n_train": int(len(tr)), "n_test": int(len(te)),
                    "test_idx": [int(i) for i in orig_te],
                    "n_selected": int(len(est.selected_)),
                    "score": float(score), "baseline_score": float(base_score),
                    "n_refused": int(n_ref),
                    "mnar_verdict": mnar.get("verdict"),
                    "seed_outer": seeds.outer(fold_idx, repeat),
                    "seed_inner": seeds.inner(fold_idx, repeat),
                    "seconds": time.perf_counter() - t0,
                })

        safe = np.maximum(oof_count, 1)
        oof = oof_sum / (safe if task == "regression" else safe[:, None])

        final = self.estimator_factory()
        final.fit(Xl, yl, V=Vl, task=task, seeds=seeds, fold_idx=0, repeat=0,
                  inner_splitter=make_inner_splitter(task, gl is not None,
                                                     self.l_inner, seeds.inner(0, 0)),
                  groups=gl)

        scores_arr = np.asarray(scores, dtype=float)
        base_arr = np.asarray(base_scores, dtype=float) if self.baseline else None
        verdict = None
        if self.baseline and np.isfinite(scores_arr).any() and np.isfinite(base_arr).any():
            from .metrics import paired_comparison

            verdict = paired_comparison(scores_arr, base_arr)

        return RSMResult(
            task=task,
            random_state=self.random_state,
            per_fold_scores=scores_arr,
            baseline_per_fold_scores=base_arr,
            verdict=verdict,
            oof_predictions=oof,
            oof_count=oof_count,
            n_refused=total_refused,
            n_unlabelled=int(unlab_idx.size),
            selected_per_fold=selected_per_fold,
            selection_stability=mean_pairwise_jaccard(selected_per_fold),
            selection_stability_adjusted=(
                mean_pairwise_jaccard(selected_per_fold, n_total=n_candidates)
                if n_candidates else float("nan")),
            mnar_reports=mnar_reports,
            fold_records=fold_records,
            final_estimator=final,
            classes=None if classes is None else np.asarray(classes),
            config={
                "k_outer": self.k_outer, "l_inner": self.l_inner,
                "repeated_outer_cv": repeats,
                "segment": self.segment, "grouped": groups is not None,
                "baseline": self.baseline, "mnar_guard": self.mnar_guard,
                "use_unlabelled_stats": self.use_unlabelled_stats,
            },
        )
