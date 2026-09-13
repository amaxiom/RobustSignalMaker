"""Benchmark competitors: selection methods behind one explicit registry.

Every selector has the signature

    select(X, V, y, grid, task, seed, k, **cfg) -> np.ndarray of segment ids

where ``k`` is the target selection size (filter methods take exactly k;
RSM variants hit it via coverage targeting so comparisons are at equal
coverage) and ``cfg`` is the dataset's declared selector configuration
(``meta["selector_config"]``: for example the evidence scale of sparse
spike data). Selectors that have no use for a key ignore it. Names are explicit and self-describing, including the fill
strategy (`*_mean_fill`, `*_interp_fill`): the RPM harness lessons all
reduced to implicit defaults and shared names silently collapsing two
variants into one, so here every variant names itself and ``METHOD_ORDER``
derives from the registry.

The impute-then-select family exists to be beaten where it should lose and
to win honestly where it does not (see FINDINGS sec.2): imputation deletes
the validity mask, so those selectors receive filled data and V of ones.
"""
from __future__ import annotations

import numpy as np

from robustsignalmaker import BootstrapMaskSelector, SegmentMean

RSM_KW = dict(lam=0.08, n_iter=200, lr=0.1, n_bootstrap=10)


# --------------------------------------------------------------------------- #
# fills (the fabrication step of the impute-then-select pipelines)
# --------------------------------------------------------------------------- #
def mean_fill(X, V):
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    num = (X * V).sum(axis=0)
    den = V.sum(axis=0)
    mu = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return np.where(V > 0, X, mu[None])


def interp_fill(X, V):
    X = np.asarray(X, dtype=float)
    Xf = X.copy()
    n, C, T = X.shape
    t = np.arange(T)
    for i in range(n):
        for c in range(C):
            obs = np.flatnonzero(V[i, c])
            if obs.size:
                Xf[i, c] = np.interp(t, obs, X[i, c, obs])
    return Xf


def _segment_means(X, V, grid):
    """Complete-data segment-mean features (n, S), channel-averaged."""
    F, _ = SegmentMean().transform(X, np.ones_like(np.asarray(X, dtype=float)), grid)
    return F.mean(axis=2)


def _top_k(importance, k):
    imp = np.nan_to_num(np.asarray(importance, dtype=float), nan=-np.inf)
    return np.sort(np.argsort(imp)[::-1][:k]).astype(int)


# --------------------------------------------------------------------------- #
# base selectors (each receives ALREADY-FILLED data when used via a fill)
# --------------------------------------------------------------------------- #
def _anova(X, V, y, grid, task, seed, k, **cfg):
    from sklearn.feature_selection import f_classif, f_regression

    F = _segment_means(X, V, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat, _ = (f_regression(F, y) if task == "regression" else f_classif(F, y))
    return _top_k(stat, k)


def _lasso(X, V, y, grid, task, seed, k, **cfg):
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.preprocessing import StandardScaler

    F = StandardScaler().fit_transform(_segment_means(X, V, grid))
    if task == "regression":
        coef = Lasso(alpha=0.05, random_state=seed).fit(F, y).coef_
    else:
        # bounded, seeded saga per the RPM lesson (uncapped liblinear wedged runs)
        coef = LogisticRegression(penalty="l1", solver="saga", C=1.0,
                                  max_iter=2000, random_state=seed).fit(F, y).coef_
        coef = np.abs(coef).max(axis=0)
    return _top_k(np.abs(coef), k)


def _rf(X, V, y, grid, task, seed, k, **cfg):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    F = _segment_means(X, V, grid)
    model = (RandomForestRegressor if task == "regression"
             else RandomForestClassifier)(n_estimators=200, random_state=seed, n_jobs=1)
    return _top_k(model.fit(F, y).feature_importances_, k)


def _ipls(X, V, y, grid, task, seed, k, **cfg):
    """Interval PLS: rank each segment by its own small PLS model's CV error."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold

    X = np.asarray(X, dtype=float)
    yv = np.asarray(y, dtype=float)
    scores = np.full(grid.n_segments, np.nan)
    cv = KFold(n_splits=3, shuffle=True, random_state=seed)
    for s in range(grid.n_segments):
        pts = grid.segments_to_points([s])
        Fs = X[:, :, pts].reshape(X.shape[0], -1)
        if np.std(Fs) == 0:
            continue
        errs = []
        try:
            for tr, te in cv.split(Fs):
                with np.errstate(all="ignore"):
                    pls = PLSRegression(n_components=min(2, Fs.shape[1]))
                    pls.fit(Fs[tr], yv[tr])
                    pred = pls.predict(Fs[te]).ravel()
                errs.append(np.mean((pred - yv[te]) ** 2))
        except Exception:
            continue  # a numerically degenerate interval scores nan (skipped)
        if errs and np.isfinite(errs).all():
            scores[s] = -float(np.mean(errs))  # higher is better
    return _top_k(scores, k)


def _mcuve(X, V, y, grid, task, seed, k, **cfg):
    """Monte-Carlo uninformative variable elimination on segment means."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    F = StandardScaler().fit_transform(_segment_means(X, V, grid))
    yv = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    B, n = 30, len(yv)
    coefs = np.zeros((B, F.shape[1]))
    for b in range(B):
        idx = rng.choice(n, size=max(2, int(0.7 * n)), replace=False)
        coefs[b] = Ridge(alpha=1.0).fit(F[idx], yv[idx]).coef_
    sd = coefs.std(axis=0)
    reliability = np.divide(coefs.mean(axis=0), sd,
                            out=np.zeros(F.shape[1]), where=sd > 0)
    return _top_k(np.abs(reliability), k)


def _rsm(X, V, y, grid, task, seed, k, **cfg):
    sel = BootstrapMaskSelector(task=task, segment=grid.segment, seed=seed,
                                target_coverage=k / grid.n_segments,
                                **RSM_KW, **cfg)
    sel.fit(X, y, V)
    return np.asarray(sel.selected_segments_, dtype=int)


# --------------------------------------------------------------------------- #
# registry: every variant names itself, fills included
# --------------------------------------------------------------------------- #
def _with_fill(base, fill):
    def run(X, V, y, grid, task, seed, k, **cfg):
        Xf = fill(X, V)
        # imputation deletes the validity mask, so the dataset's declared
        # evidence scale is irrelevant downstream of a fill
        return base(Xf, np.ones_like(Xf, dtype=bool), y, grid, task, seed, k)

    return run


def select_full_signal(X, V, y, grid, task, seed, k, **cfg):
    return np.arange(grid.n_segments, dtype=int)


SELECTORS = {
    "full_signal": select_full_signal,
    "rsm_masked": _rsm,
    "rsm_mean_fill": _with_fill(_rsm, mean_fill),
    "rsm_interp_fill": _with_fill(_rsm, interp_fill),
    "anova_mean_fill": _with_fill(_anova, mean_fill),
    "anova_interp_fill": _with_fill(_anova, interp_fill),
    "lasso_mean_fill": _with_fill(_lasso, mean_fill),
    "rf_mean_fill": _with_fill(_rf, mean_fill),
    "ipls_mean_fill": _with_fill(_ipls, mean_fill),
    "mcuve_mean_fill": _with_fill(_mcuve, mean_fill),
}

METHOD_ORDER = list(SELECTORS)  # derived, never hardcoded elsewhere
