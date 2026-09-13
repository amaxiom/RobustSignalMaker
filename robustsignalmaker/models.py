"""Refit model zoo: what predicts on the selected region (Milestone 9).

RSM's selection comes from gates trained by gradient descent THROUGH the
head, so the head that does the selecting must be differentiable and is
hand-differentiated. That is why there is exactly one of it, and why a
random forest cannot go there the way it can in RobustModelMaker, whose
selection is importance-based rather than gate-based.

The FINAL refit is a different matter. Once the bands are chosen, nothing
needs a gradient, so the predictor on the selected region can be any model
at all. This module is that zoo, with the same ``alg`` vocabulary as RMM so
a study can move between the two libraries, plus the two PLS variants that
every chemometrician will look for first and that RMM has no reason to
carry.

    lin  ordinary least squares / logistic without penalty
    rdg  ridge
    las  lasso
    eln  elastic net
    log  logistic regression (classification only; alias of lin there)
    svm  support vector machine, RBF kernel
    rf   random forest
    xgb  gradient boosting (xgboost if installed, else sklearn's)
    mlp  multi-layer perceptron
    pls  partial least squares; PLS-DA for classification
    head the built-in renormalised gated head (the default, and the only
         one that consumes validity directly rather than through pooling)

**The missing-data contract is preserved, and this is the load-bearing
part.** An sklearn estimator cannot see a validity mask, so the selected
region is pooled to validity-weighted segment means and standardised with
RSM's masked standardisation, which sends unobserved entries to exactly 0
in standardised space: the same convention the built-in head uses, so an
unobserved band contributes nothing rather than contributing a fabricated
value. Samples whose observed evidence over the selected region falls below
``rho_min`` are REFUSED, exactly as the head refuses them, and come back as
nan rather than as a confident guess from a model that would happily have
produced one. No value is imputed anywhere in this module.
"""
from __future__ import annotations

import numpy as np

ALGORITHMS = ("head", "lin", "rdg", "las", "eln", "log", "svm", "rf", "xgb",
              "mlp", "pls")

#: algorithms that are classifiers, regressors, or both
_CLASSIFICATION_ONLY = frozenset({"log"})


class PLSDA:
    """PLS discriminant analysis: PLS regression onto one-hot targets.

    scikit-learn ships no PLS-DA, and it is the standard classifier in
    chemometrics, so it is built here rather than left out. Class scores are
    the PLS predictions for each one-hot column; probabilities are their
    softmax, which is monotone in the scores and so leaves the ranking (and
    therefore AUC) untouched while giving ``predict_proba`` something
    defensible to return.
    """

    def __init__(self, n_components: int = 2, scale: bool = False):
        self.n_components = int(n_components)
        self.scale = bool(scale)

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        n_comp = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
        self.model_ = PLSRegression(n_components=n_comp, scale=self.scale)
        self.model_.fit(X, Y)
        return self

    def decision_function(self, X):
        return np.asarray(self.model_.predict(np.asarray(X, dtype=float)))

    def predict_proba(self, X):
        Z = self.decision_function(X)
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except Exception:
        return False
    return True  # pragma: no cover - depends on the environment


def make_refit_model(alg: str, task: str, seed: int = 0, **overrides):
    """Build the estimator named by ``alg`` for ``task``.

    Raises ``ValueError`` for an unknown algorithm or an algorithm that does
    not apply to the task, rather than silently substituting something else.
    ``xgb`` falls back to scikit-learn's gradient boosting when xgboost is
    not installed, and says so in the estimator's ``_rsm_note`` attribute.
    """
    if alg not in ALGORITHMS:
        raise ValueError(
            f"unknown algorithm {alg!r}; choose from {', '.join(ALGORITHMS)}")
    if alg == "head":
        raise ValueError(
            "alg='head' is the built-in gated head and is not built here; "
            "the selector handles it directly")
    if task not in ("binary", "multiclass", "regression"):
        raise ValueError(f"unknown task {task!r}")
    clf = task in ("binary", "multiclass")
    if alg in _CLASSIFICATION_ONLY and not clf:
        raise ValueError(f"alg={alg!r} is classification only; task is {task!r}")

    note = None
    if alg in ("lin", "log"):
        if clf:
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=2000, random_state=seed)
        else:
            from sklearn.linear_model import LinearRegression
            m = LinearRegression()
    elif alg == "rdg":
        from sklearn.linear_model import Ridge, RidgeClassifier
        m = RidgeClassifier(random_state=seed) if clf else Ridge(alpha=1.0)
    elif alg == "las":
        if clf:
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(penalty="l1", solver="liblinear",
                                   max_iter=2000, random_state=seed)
        else:
            from sklearn.linear_model import Lasso
            m = Lasso(alpha=0.01, max_iter=5000, random_state=seed)
    elif alg == "eln":
        if clf:
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(penalty="elasticnet", solver="saga",
                                   l1_ratio=0.5, max_iter=5000, random_state=seed)
        else:
            from sklearn.linear_model import ElasticNet
            m = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000,
                           random_state=seed)
    elif alg == "svm":
        from sklearn.svm import SVC, SVR
        m = (SVC(probability=True, random_state=seed) if clf
             else SVR())
    elif alg == "rf":
        from sklearn.ensemble import (RandomForestClassifier,
                                      RandomForestRegressor)
        cls = RandomForestClassifier if clf else RandomForestRegressor
        m = cls(n_estimators=200, random_state=seed, n_jobs=1)
    elif alg == "xgb":
        if _xgboost_available():  # pragma: no cover - xgboost not installed here
            import xgboost as xgb
            cls = xgb.XGBClassifier if clf else xgb.XGBRegressor
            m = cls(n_estimators=200, max_depth=4, learning_rate=0.05,
                    random_state=seed, verbosity=0, n_jobs=1)
        else:
            from sklearn.ensemble import (GradientBoostingClassifier,
                                          GradientBoostingRegressor)
            cls = GradientBoostingClassifier if clf else GradientBoostingRegressor
            m = cls(n_estimators=200, max_depth=3, learning_rate=0.05,
                    random_state=seed)
            note = ("xgboost not installed; used scikit-learn gradient "
                    "boosting instead")
    elif alg == "mlp":
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        cls = MLPClassifier if clf else MLPRegressor
        m = cls(hidden_layer_sizes=(64,), max_iter=1000, random_state=seed)
    elif alg == "pls":
        if clf:
            m = PLSDA(n_components=2)
        else:
            from sklearn.cross_decomposition import PLSRegression
            m = PLSRegression(n_components=2, scale=False)
    else:  # pragma: no cover - ALGORITHMS is exhaustive above
        raise ValueError(f"unhandled algorithm {alg!r}")

    if overrides:
        m.set_params(**overrides)
    m._rsm_note = note
    return m


def model_supports_proba(model) -> bool:
    """Whether the estimator can give calibrated-ish class probabilities."""
    return hasattr(model, "predict_proba")


def refit_features(F, VF, selected, stats, rho_min=0.1):
    """Features for an external model, plus which samples may be scored.

    ``F``/``VF`` are the pooled ``(n, S, C)`` segment means and validity from
    ``SegmentGrid.pool``; ``stats`` are frozen standardisation statistics
    fitted on the training fold. Returns ``(Z, valid)`` where ``Z`` is
    ``(n, len(selected) * C)`` with unobserved entries at exactly 0 in
    standardised space, and ``valid`` marks samples whose observed fraction
    over the selected region reaches ``rho_min``.

    Nothing is imputed. A sample below the floor is refused rather than
    filled, which is why ``valid`` is returned instead of a complete matrix.
    """
    from .validity import masked_standardise

    F = np.asarray(F, dtype=float)
    VF = np.asarray(VF, dtype=float)
    sel = np.asarray(selected, dtype=int)
    if sel.size == 0:
        raise ValueError("no segments selected; nothing to refit on")

    Fs = masked_standardise(F, VF, stats)          # unobserved -> exactly 0.0
    Z = Fs[:, sel, :].reshape(F.shape[0], -1)
    rho = VF[:, sel, :].mean(axis=(1, 2))
    valid = rho >= rho_min
    return Z, valid


class ExternalRefit:
    """Fit an external model on a chosen region, preserving the refusal rule.

    Shared by the selector's refit and by the matched full-signal baseline,
    so a verdict computed with ``refit_model="rf"`` compares a random forest
    on the SELECTED bands against a random forest on ALL bands. Comparing a
    forest on the selection against the built-in head on everything would
    confound the choice of region with the choice of model, which is the one
    thing the verdict is supposed to isolate.
    """

    def __init__(self, alg: str, task: str, grid, selected, seed: int = 0,
                 rho_min: float = 0.1, n_min_hard: int = 8, n_min_warn: int = 30):
        self.alg = alg
        self.task = task
        self.grid = grid
        self.selected = np.asarray(selected, dtype=int)
        self.seed = int(seed)
        self.rho_min = float(rho_min)
        self.n_min_hard = int(n_min_hard)
        self.n_min_warn = int(n_min_warn)

    def fit(self, X, V, y):
        from .representations import SegmentMean
        from .validity import InsufficientEvidenceError, masked_standardise_fit

        F, VF = SegmentMean().transform(X, V, self.grid)
        self.stats_ = masked_standardise_fit(
            F, VF, n_min_hard=self.n_min_hard, n_min_warn=self.n_min_warn)
        Z, valid = refit_features(F, VF, self.selected, self.stats_,
                                  rho_min=self.rho_min)
        y = np.asarray(y)
        if y.dtype.kind in "fc":
            labelled = valid & np.isfinite(y)
        else:
            labelled = valid & np.array([v is not None for v in y])
        if int(labelled.sum()) < 2:
            raise InsufficientEvidenceError(
                f"refit_model={self.alg!r}: fewer than two samples clear the "
                "evidence floor on the chosen region")
        self.model_ = make_refit_model(self.alg, self.task, seed=self.seed)
        self.model_.fit(Z[labelled], y[labelled])
        self.note_ = getattr(self.model_, "_rsm_note", None)
        self.n_excluded_ = int((~labelled).sum())
        self.classes_ = getattr(self.model_, "classes_", None)
        return self

    def _features(self, X, V):
        from .representations import SegmentMean
        from .validity import as_validity

        Xa, Vb = as_validity(X, V)
        F, VF = SegmentMean().transform(Xa, Vb.astype(float), self.grid)
        return refit_features(F, VF, self.selected, self.stats_,
                              rho_min=self.rho_min)

    def predict(self, X, V=None):
        Z, valid = self._features(X, V)
        if self.task == "regression":
            out = np.full(Z.shape[0], np.nan)
            if valid.any():
                out[valid] = np.asarray(
                    self.model_.predict(Z[valid]), dtype=float).ravel()
            return out
        out = np.full(Z.shape[0], None, dtype=object)
        if valid.any():
            out[valid] = np.asarray(self.model_.predict(Z[valid])).ravel()
        return out

    def predict_proba(self, X, V=None):
        if self.task == "regression":
            raise ValueError("predict_proba is undefined for regression")
        Z, valid = self._features(X, V)
        classes = np.asarray(self.classes_)
        out = np.full((Z.shape[0], len(classes)), np.nan)
        if valid.any():
            m = self.model_
            if hasattr(m, "predict_proba"):
                out[valid] = m.predict_proba(Z[valid])
            else:
                d = np.asarray(m.decision_function(Z[valid]), dtype=float)
                if d.ndim == 1:
                    d = np.column_stack([-d, d])
                d = d - d.max(axis=1, keepdims=True)
                e = np.exp(d)
                out[valid] = e / e.sum(axis=1, keepdims=True)
        return out
