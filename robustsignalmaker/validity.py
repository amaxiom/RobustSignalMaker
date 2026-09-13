"""Masked computation: missing data is absent, never fabricated (Milestone 1).

This module is the novelty axis of RobustSignalMaker. Neither sibling treats
input missingness as a first-class citizen: RobustModelMaker median-imputes
per training fold, and RobustPixelMaker has no input-missingness handling at
all (its maskfill module concerns the LEARNED selection mask). Here the two
ideas are unified: a learned gate and a data validity mask share the same
algebra (both mean "this evidence is absent") and compose multiplicatively.

Contract, enforced module-wide:
  * No value is ever fabricated. Every statistic is computed over the evidence
    that actually exists, renormalised by how much evidence that was.
  * A guard never substitutes a value. When something cannot be computed it is
    removed from play and reported (a ``valid`` flag, a nan, or an
    :class:`InsufficientEvidenceError`), never a confident-looking number.
  * Values at invalid positions are never read. The payload under a masked
    entry (0, -1, 1e300, anything) must not change any output bit; this is a
    named test, and it forbids relying on nan-propagation anywhere.
  * Scale safety: renormalisations use evidence-count reciprocals and moments
    are computed in max-abs-factored space, so inputs at 1e-200 and 1e300
    behave identically up to floating point. No epsilon ever meets a data
    magnitude.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


class InsufficientEvidenceError(ValueError):
    """Raised when a requested quantity has no evidence to be computed from."""


class MissingnessInformativeWarning(UserWarning):
    """The observation pattern alone predicts the target.

    Selection on the observed data is then confounded with the observation
    process (missingness is informative, MNAR): the selected region may encode
    instrument identity rather than signal content. RSM warns and reports; it
    does not pretend to fix MNAR.
    """


# ---------------------------------------------------------------------------
# canonicalisation
# ---------------------------------------------------------------------------

def as_validity(X, V=None):
    """Canonicalise a (values, validity) pair.

    ``V`` defaults to ``~np.isnan(X)``. When ``V`` is supplied, ``X`` must be
    finite everywhere ``V`` is True (a NaN or inf at a claimed-observed
    position is a contradiction and raises). Values at invalid positions are
    overwritten with 0.0 so that no downstream code can accidentally read a
    masked payload; the contract is that they are never read at all.

    Returns ``(X_clean, V_bool)`` with ``X_clean`` float64 and ``V_bool`` bool,
    both of the input shape.
    """
    X = np.asarray(X, dtype=float)
    if V is None:
        V = ~np.isnan(X)
    else:
        V = np.asarray(V).astype(bool)
        if V.shape != X.shape:
            raise ValueError(f"V shape {V.shape} does not match X shape {X.shape}")
        if not np.all(np.isfinite(X[V])):
            raise ValueError(
                "X contains non-finite values at positions V marks as observed; "
                "an observed value must be finite"
            )
    if not np.all(np.isfinite(X[V])):  # V derived from isnan can still admit inf
        raise ValueError("X contains inf at observed positions; observed values must be finite")
    X_clean = np.where(V, X, 0.0)
    return X_clean, V


# ---------------------------------------------------------------------------
# masked moments (scale-safe)
# ---------------------------------------------------------------------------

def _scale_factor(X, W, axis):
    """Max-abs of the observed values along ``axis`` (keepdims), 0 where none.

    Factoring this out before any sum keeps every intermediate O(n), so
    magnitudes like 1e300 cannot overflow a reduction and magnitudes like
    1e-200 cannot underflow one.
    """
    absx = np.where(W > 0, np.abs(X), 0.0)
    return np.max(absx, axis=axis, keepdims=True)


def masked_mean(X, W, axis=None, strict=False):
    """Validity-weighted mean over observed entries, scale-safely.

    ``W`` is a validity weight in [0, 1] (bool accepted). Returns
    ``(mean, valid)``; where the total weight along ``axis`` is zero the mean
    is reported as 0.0 with ``valid=False`` (or raises with ``strict=True``).
    The 0.0 is a placeholder that must never be read where ``valid`` is False.
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    X = np.where(W > 0, X, 0.0)  # payloads never reach arithmetic (nan*0 is nan)
    s = _scale_factor(X, W, axis)
    g = np.divide(X, s, out=np.zeros_like(X), where=s > 0)
    wsum = np.sum(W, axis=axis)
    num = np.sum(W * g, axis=axis)
    valid = wsum > 0
    mean_scaled = np.divide(num, wsum, out=np.zeros_like(num), where=valid)
    mean = mean_scaled * np.squeeze(s, axis=axis) if axis is not None else mean_scaled * float(s)
    if strict and not np.all(valid):
        raise InsufficientEvidenceError("masked_mean: no observed evidence along the reduced axis")
    return mean, valid


def masked_std(X, W, axis=None, min_count=2, strict=False):
    """Validity-weighted standard deviation over observed entries, scale-safely.

    Two-pass shifted computation in max-abs-factored space. ``valid`` requires
    the effective sample size (Kish, ``(sum W)^2 / sum W^2``) to reach
    ``min_count``; below it the std is untrustworthy and is reported as 0.0
    with ``valid=False`` (or raises with ``strict=True``).
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    X = np.where(W > 0, X, 0.0)  # payloads never reach arithmetic
    s = _scale_factor(X, W, axis)
    g = np.divide(X, s, out=np.zeros_like(X), where=s > 0)
    wsum = np.sum(W, axis=axis)
    w2sum = np.sum(W * W, axis=axis)
    ess = np.divide(wsum * wsum, w2sum, out=np.zeros_like(wsum), where=w2sum > 0)
    has_any = wsum > 0
    mu = np.divide(np.sum(W * g, axis=axis), wsum,
                   out=np.zeros_like(wsum), where=has_any)
    mu_e = np.expand_dims(mu, axis) if axis is not None else mu
    var = np.divide(np.sum(W * (g - mu_e) ** 2, axis=axis), wsum,
                    out=np.zeros_like(wsum), where=has_any)
    std_scaled = np.sqrt(np.maximum(var, 0.0))
    std = std_scaled * np.squeeze(s, axis=axis) if axis is not None else std_scaled * float(s)
    valid = has_any & (ess >= min_count)
    std = np.where(valid, std, 0.0)
    if strict and not np.all(valid):
        raise InsufficientEvidenceError(
            f"masked_std: effective sample size below min_count={min_count}"
        )
    return std, valid


# ---------------------------------------------------------------------------
# masked standardisation (fit on a training fold, applied frozen)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StandardisationStats:
    """Frozen per-feature standardisation statistics, in max-abs-factored space.

    ``scale`` is the max-abs factor s; ``mu`` and ``sigma`` are moments of
    ``x / s``. The standardised value is computed as ``(x / s - mu) / sigma``
    and raw-unit moments (``s * mu``, ``s * sigma``) are never formed, which is
    what makes the transform exact at scales like 1e-200 and 1e300.

    ``assessable`` is False where the effective sample size is below the hard
    floor or the feature is scale-degenerate (constant on the training data);
    such features are removed from play (standardised to 0 and flagged), never
    silently rescaled. ``low_support`` flags assessable features whose
    effective sample size is below the warn floor.
    """

    scale: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    ess: np.ndarray
    assessable: np.ndarray
    low_support: np.ndarray
    degenerate_scale: np.ndarray

    @property
    def n_assessable(self) -> int:
        return int(np.sum(self.assessable))


def masked_standardise_fit(F, W, n_min_hard=8, n_min_warn=30, sigma_tol=1e-12,
                           strict=False) -> StandardisationStats:
    """Fit per-feature masked standardisation over rows (axis 0).

    ``F`` is ``(n_rows, ...)`` features with validity weights ``W`` of the
    same shape, values in [0, 1]. Fit this on a training fold (or resample)
    only, and apply it frozen everywhere else; fitting it on anything that
    touches a test partition is leakage.

    With ``strict=True`` an entirely unassessable feature set raises
    :class:`InsufficientEvidenceError`; otherwise the stats simply record what
    could not be assessed.
    """
    F = np.asarray(F, dtype=float)
    W = np.asarray(W, dtype=float)
    if F.shape != W.shape:
        raise ValueError(f"W shape {W.shape} does not match F shape {F.shape}")
    F = np.where(W > 0, F, 0.0)  # payloads never reach arithmetic
    s = _scale_factor(F, W, axis=0)  # (1, ...) keepdims
    g = np.divide(F, s, out=np.zeros_like(F), where=s > 0)
    wsum = np.sum(W, axis=0)
    w2sum = np.sum(W * W, axis=0)
    ess = np.divide(wsum * wsum, w2sum, out=np.zeros_like(wsum), where=w2sum > 0)
    has_any = wsum > 0
    mu = np.divide(np.sum(W * g, axis=0), wsum, out=np.zeros_like(wsum), where=has_any)
    var = np.divide(np.sum(W * (g - mu[None, ...]) ** 2, axis=0), wsum,
                    out=np.zeros_like(wsum), where=has_any)
    sigma = np.sqrt(np.maximum(var, 0.0))

    degenerate = has_any & (sigma <= sigma_tol)
    assessable = (ess >= n_min_hard) & ~degenerate
    low_support = assessable & (ess < n_min_warn)

    stats = StandardisationStats(
        scale=np.squeeze(s, axis=0),
        mu=mu,
        sigma=sigma,
        ess=ess,
        assessable=assessable,
        low_support=low_support,
        degenerate_scale=degenerate,
    )
    if strict and stats.n_assessable == 0:
        raise InsufficientEvidenceError(
            "masked_standardise_fit: no feature is assessable "
            f"(effective sample size floor {n_min_hard}, sigma tolerance {sigma_tol}); "
            "the training data cannot support standardisation"
        )
    return stats


def masked_standardise(F, W, stats: StandardisationStats):
    """Apply frozen standardisation; unassessable features are removed from play.

    Observed values on assessable features become ``(F / s - mu) / sigma``.
    Everything else (unobserved entries, degenerate or unassessable features)
    becomes exactly 0.0 and contributes nothing downstream. Removal, not
    clipping: a feature that was constant on the training data cannot be
    honestly scaled at predict time, so it is excluded and flagged rather than
    given an invented spread.
    """
    F = np.asarray(F, dtype=float)
    W = np.asarray(W, dtype=float)
    ok = stats.assessable[None, ...] & (W > 0)
    g = np.divide(F, stats.scale[None, ...], out=np.zeros_like(F),
                  where=ok & (stats.scale[None, ...] > 0))
    z = np.divide(g - stats.mu[None, ...], stats.sigma[None, ...],
                  out=np.zeros_like(F), where=ok & (stats.sigma[None, ...] > 0))
    return np.where(ok, z, 0.0)


# ---------------------------------------------------------------------------
# renormalised 1D convolution (the lens primitive)
# ---------------------------------------------------------------------------

CONTRAST_MIN_FRAC = 0.5  # a slope estimated from under half a gapped window is
#                          high-variance evidence; below this the output is invalid


def is_contrast_filter(filt) -> bool:
    """A zero-sum filter measures contrast; a nonzero-sum filter measures level."""
    filt = np.asarray(filt, dtype=float)
    return abs(float(filt.sum())) < 1e-8 * max(1.0, float(np.abs(filt).sum()))


def renormalised_convolve1d(X, V, filt, min_frac=None, mode="reflect"):
    """Partial 1D convolution measured against the surviving evidence only.

    1D port of RobustPixelMaker's ``maskfill.renormalised_convolve``, with
    fractional validity in and out. ``X`` is ``(..., T)``, ``V`` the same
    shape with values in [0, 1] (bool accepted), ``filt`` a 1D kernel applied
    along the last axis.

    A LEVEL filter (weights summing to something nonzero, such as a local
    average) is renormalised by its OBSERVED FILTER WEIGHT,

        out = sum_obs f_j x_j * ( sum f / sum_obs f_j ),

    so a partially covered window reports the filter-weighted average of the
    evidence that exists, exactly, for any kernel shape. A zero-mean CONTRAST
    filter needs a different correction: over a partial support its weights
    no longer sum to zero, so a perfectly flat region would produce a
    response at a gap boundary, a fabricated edge of exactly the kind this
    construction exists to prevent. It is therefore re-centred over the
    surviving support and normalised by the observed absolute filter mass,

        out = [ sum_obs f_j x_j - xbar_obs * sum_obs f_j ] * ( sum|f| / sum_obs |f_j| ),

    with ``xbar_obs`` the evidence-weighted mean of the window, identically
    zero on a flat region. Both cases reduce exactly to a plain convolution
    on a fully observed window. Denominators are validity masses, which are
    dimensionless, so the only floor is the float-subnormal boundary and no
    epsilon ever meets a data magnitude. (The first release rescaled both
    branches by the observed COUNT, which is exact only for uniform kernels
    and fabricated level shifts near gaps for Gaussian kernels; found and
    fixed in the 2026-09-01 bug sweep, and a deliberate divergence from the
    RPM maskfill construction it was ported from.)

    Output validity is the renormalised evidence fraction of the window,
    thresholded by filter class: a contrast response needs at least
    ``CONTRAST_MIN_FRAC`` of its window (default 0.5), a level response counts
    any evidence. ``min_frac`` overrides the threshold for both classes.
    Positions failing the threshold return 0.0 with validity 0.0 and must not
    be read.

    Returns ``(out, v_out)``, both shaped like ``X``; ``v_out`` is fractional
    in [0, 1] and flows into pooling as a weight, so evidence near a gap is
    partially trusted rather than binarised.
    """
    from scipy.ndimage import convolve1d

    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.shape != X.shape:
        raise ValueError(f"V shape {V.shape} does not match X shape {X.shape}")
    filt = np.asarray(filt, dtype=float)
    if filt.ndim != 1 or filt.size == 0:
        raise ValueError("filt must be a non-empty 1D kernel")
    width = float(filt.size)
    ones = np.ones_like(filt)
    # payload hygiene: a masked value (nan, inf, anything) must never reach
    # arithmetic; nan * 0 is nan, so multiply-by-V alone is not enough
    X = np.where(V > 0, X, 0.0)

    num = convolve1d(X * V, filt, axis=-1, mode=mode)
    evidence = convolve1d(V, ones, axis=-1, mode=mode)
    # validity mass is dimensionless, so a subnormal-floor here never touches
    # a data magnitude: evidence below float-tiny IS no evidence
    tiny = np.finfo(float).tiny
    has_any = evidence > tiny

    contrast = is_contrast_filter(filt)
    if contrast:
        # re-centre over the surviving support, then normalise the response
        # by the OBSERVED ABSOLUTE (L1) FILTER MASS. The count-based rescale
        # of the first release was exact only for uniform kernels and
        # inflated responses near gaps.
        #
        # L1 here is a measured choice, not an oversight, and the measurement
        # is worth keeping because the sibling library reached a DIFFERENT
        # answer. RobustPixelMaker normalises its contrast branch by the L2
        # norm, because for exchangeable residuals a re-centred response has
        # magnitude ~sqrt(sum_obs f^2) and L2 is what preserves it; on RPM's
        # criterion (recovering the full-data convolution response with
        # random zero-mean 2D kernels) L2 beat count 79.9% against 93.2% RMS
        # error. Ported here and scored on the criterion that actually
        # governs RSM, truth recovery of the SELECTION, L2 lost: over 10
        # seeds x 2 dropout regimes x {SavGol, RandomConv1D}, paired per
        # configuration, L1 beat L2 21 to 9 with 10 ties, mean F1 +0.019.
        # SavGol coefficients are smooth and antisymmetric rather than
        # iid-like, so the exchangeable-residual argument behind L2 does not
        # hold for them, and the >=0.5 window-validity threshold plus segment
        # pooling damp what is left. Count and L1 tie on this criterion
        # (16 to 14 with 10 ties), so L1 is kept as the principled
        # generalisation of count to weighted kernels. See FINDINGS sec.12;
        # do not port the RPM rule back without re-running that experiment.
        inv = np.where(has_any, 1.0 / np.where(has_any, evidence, 1.0), 0.0)
        weight_obs = convolve1d(V, filt, axis=-1, mode=mode)
        xbar = convolve1d(X * V, ones, axis=-1, mode=mode) * inv
        num = num - xbar * weight_obs
        mass_obs = convolve1d(V, np.abs(filt), axis=-1, mode=mode)
        mass_full = float(np.abs(filt).sum())
    else:
        # a LEVEL filter renormalises by the observed FILTER WEIGHT so a
        # partially covered window reports the filter-weighted average of
        # the evidence that exists (count-based rescaling fabricated level
        # shifts for non-uniform kernels; found in the 2026-09-01 sweep)
        mass_obs = convolve1d(V, filt, axis=-1, mode=mode)
        mass_full = float(filt.sum())
    ok_mass = has_any & (mass_obs > tiny)
    out = np.where(ok_mass,
                   num * np.divide(mass_full, mass_obs,
                                   out=np.ones_like(mass_obs), where=ok_mass),
                   0.0)
    has_any = ok_mass
    frac = evidence / width

    if min_frac is None:
        q = CONTRAST_MIN_FRAC if contrast else 0.0
    else:
        q = float(min_frac)
    trusted = has_any & (frac >= q) if q > 0 else has_any
    v_out = np.where(trusted, frac, 0.0)
    out = np.where(trusted, out, 0.0)
    return out, v_out


# ---------------------------------------------------------------------------
# renormalised head inner product
# ---------------------------------------------------------------------------

def renormalised_dot(U, gate, validity, rho_min=0.1):
    """Per-sample renormalised gated sum: the head's analogue of partial convolution.

    ``U`` is ``(n, S)`` per-segment contributions (already channel-summed and
    standardised), ``gate`` is ``(S,)`` in [0, 1], ``validity`` is ``(n, S)``
    in [0, 1]. With M = sum_k gate_k and D_i = sum_k gate_k * validity_ik:

        z_i = (M / D_i) * sum_k gate_k * validity_ik * U_ik

    which reduces exactly to the plain gated sum when every gated segment is
    observed, and under MCAR is a ratio-form Horvitz-Thompson estimate of the
    full-evidence sum: first-order unbiased, variance inflated by 1/rho_i with
    rho_i = D_i / M the sample's observed fraction of the gated evidence. The
    alternative (an available-case sum, equivalent to mean-filling in
    standardised space) attenuates the estimate by rho_i and lets "unobserved"
    masquerade as "unimportant"; renormalisation instead converts missingness
    into visible variance, which stability selection converts into low
    selection frequency. That trade is the point.

    Samples with rho_i below ``rho_min`` return nan with ``valid=False``: a
    prediction amplified from a sliver of evidence is not a prediction worth
    reporting. All gates closed raises :class:`InsufficientEvidenceError`.

    Returns ``(z, valid, rho)``. Callers add any intercept themselves. Note
    for gradient code (Milestone 3): the renormaliser M / D_i is treated as a
    constant (stop-gradient) when differentiating with respect to the gate, so
    a never-observed segment receives exactly zero gate gradient.
    """
    U = np.asarray(U, dtype=float)
    gate = np.asarray(gate, dtype=float)
    validity = np.asarray(validity, dtype=float)
    if U.ndim != 2:
        raise ValueError(f"U must be (n, S); got shape {U.shape}")
    if validity.shape != U.shape:
        raise ValueError(f"validity shape {validity.shape} does not match U shape {U.shape}")
    if gate.shape != (U.shape[1],):
        raise ValueError(f"gate shape {gate.shape} does not match S={U.shape[1]}")

    U = np.where(validity > 0, U, 0.0)  # payloads never reach arithmetic
    M = float(gate.sum())
    if M <= 0.0:
        raise InsufficientEvidenceError("renormalised_dot: all gates are closed")
    D = validity @ gate
    rho = D / M
    valid = (D > 0) & (rho >= rho_min)  # zero evidence is never a valid 0.0
    raw = (U * validity) @ gate
    z = np.where(valid, np.divide(raw * M, D, out=np.zeros_like(D), where=D > 0), np.nan)
    return z, valid, rho


# ---------------------------------------------------------------------------
# observation support and the MNAR honesty guard
# ---------------------------------------------------------------------------

def observation_support(V_seg, min_valid_frac=0.5):
    """Fraction of samples observing each segment at ``min_valid_frac`` or better.

    ``V_seg`` is ``(n, S)`` per-sample segment validity fractions. This is the
    map that must always be reported beside selection frequencies: a segment
    nobody observed can be "cannot assess", never "stably rejected".
    """
    V_seg = np.asarray(V_seg, dtype=float)
    return np.mean(V_seg >= min_valid_frac, axis=0)


def missingness_association(V_seg, y, task, n_permutations=200, n_splits=5,
                            alpha=0.05, seed=0, top_k=5):
    """Score how well the missingness pattern ALONE predicts the target.

    Fits a small regularised linear model from the per-sample segment validity
    fractions ``V_seg`` (n, S) to ``y`` under k-fold cross-validation, and
    calibrates the score against ``n_permutations`` label permutations. Run
    this inside a training fold only.

    Returns a dict with a three-valued ``verdict``:
      "informative"      the pattern predicts y (permutation p < alpha); a
                         :class:`MissingnessInformativeWarning` is emitted,
      "not_informative"  it does not; this includes the trivial case where
                         the validity pattern does not vary between samples
                         (nothing to exploit, reported with p = 1),
      "could_not_assess" the test could not be run (degenerate labels or a
                         failed fit); never silently reported as safe.
    plus ``score`` (task score of the validity-only model), ``p_value``, and
    ``top_segments`` (indices whose validity is most associated with y).
    """
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

    from .metrics import score_predictions

    if task not in ("regression", "binary", "multiclass"):
        raise ValueError(  # a typo must never fall into the other branch
            f"task must be 'regression', 'binary' or 'multiclass'; got {task!r}")
    V_seg = np.asarray(V_seg, dtype=float)
    y = np.asarray(y)
    if V_seg.ndim != 2 or V_seg.shape[0] != y.shape[0]:
        raise ValueError(f"V_seg must be (n, S) matching y; got {V_seg.shape} vs {y.shape}")

    out = {"verdict": "could_not_assess", "score": float("nan"),
           "p_value": float("nan"), "top_segments": np.array([], dtype=int),
           "task": task, "n_permutations": int(n_permutations)}

    varying = np.std(V_seg, axis=0) > 0
    if not np.any(varying):
        # No missingness variation between samples: there is nothing to test,
        # which is a benign reason to be unable to assess.
        out["verdict"] = "not_informative"
        out["p_value"] = 1.0
        return out

    Vv = V_seg[:, varying]
    rng = np.random.default_rng(seed)

    def _score(labels):
        if task == "regression":
            from sklearn.linear_model import Ridge

            model = Ridge(alpha=1.0)
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            pred = cross_val_predict(model, Vv, labels, cv=splitter)
            return score_predictions(task, labels, y_pred=pred)
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1.0, max_iter=500)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        proba = cross_val_predict(model, Vv, labels, cv=splitter, method="predict_proba")
        return score_predictions(task, labels, y_proba=proba)

    try:
        obs = _score(y)
        if np.isnan(obs):
            return out
        perm = np.empty(n_permutations)
        for p in range(n_permutations):
            perm[p] = _score(rng.permutation(y))
        valid_perm = perm[~np.isnan(perm)]
        if valid_perm.size == 0:
            return out
        p_value = float((1 + np.sum(valid_perm >= obs)) / (1 + valid_perm.size))
    except ValueError:
        return out  # verdict stays "could_not_assess", loudly not silently safe

    if task == "regression":
        from sklearn.feature_selection import f_regression

        f_stat, _ = f_regression(Vv, y.astype(float))
    else:
        from sklearn.feature_selection import f_classif

        f_stat, _ = f_classif(Vv, y)
    f_stat = np.nan_to_num(f_stat, nan=0.0)
    order = np.argsort(f_stat)[::-1][:top_k]
    top = np.flatnonzero(varying)[order]

    out.update(score=float(obs), p_value=p_value, top_segments=top)
    if p_value < alpha:
        out["verdict"] = "informative"
        warnings.warn(
            f"missingness pattern alone predicts the target "
            f"(score={obs:.3f}, permutation p={p_value:.4f}; "
            f"most associated segments: {top.tolist()}); observed-data selection "
            f"is confounded with the observation process",
            MissingnessInformativeWarning,
            stacklevel=2,
        )
    else:
        out["verdict"] = "not_informative"
    return out


# ---------------------------------------------------------------------------
# fill baselines and the fabricated-edge diagnostic (things to beat)
# ---------------------------------------------------------------------------

def fill_zero(X, V):
    """Write zeros into the unobserved region (the naive, fabricating choice)."""
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    return X * V


def fill_mean(X, V, baseline=None):
    """Write a per-point baseline (masked mean over samples) into the gaps."""
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    if baseline is None:
        baseline, ok = masked_mean(X, V, axis=0)
        baseline = np.where(ok, baseline, 0.0)
    else:
        baseline = np.asarray(baseline, dtype=float)
    return X * V + baseline[None, ...] * (1.0 - V)


def fabricated_edge(x_flat, V, filters, eps=1e-12):
    """How much fabricated response each strategy invents at a gap boundary.

    The measurement needs a case where the right answer is known. On a FLAT
    signal a zero-mean contrast filter must report zero everywhere, so any
    response is an edge that the fill strategy invented rather than something
    in the data. Reported as the largest absolute response anywhere, relative
    to the signal level; zero is perfect. Level filters are rejected because
    the diagnostic is undefined for them.

    (The RobustPixelMaker version of this diagnostic originally compared masked
    features against the full image, which rewarded strategies for letting
    dropped data leak in; the flat-signal known-answer form is the corrected
    one and is what this ports.)
    """
    from scipy.ndimage import convolve1d

    x = np.asarray(x_flat, dtype=float)
    V = np.asarray(V, dtype=float)
    for f in filters:
        if not is_contrast_filter(f):
            raise ValueError("fabricated_edge is defined for zero-mean contrast filters only")
    level = float(np.abs(x).mean()) + eps

    def _plain(sig):
        return max(
            float(np.abs(convolve1d(sig, np.asarray(f, dtype=float), axis=-1, mode="reflect")).max())
            for f in filters
        )

    out = {
        "zero": _plain(fill_zero(x, V)),
        "mean": _plain(fill_mean(x[None, ...], V[None, ...])[0]),
    }
    out["renorm"] = max(
        float(np.abs(renormalised_convolve1d(x, V, f, min_frac=0.0)[0]).max())
        for f in filters
    )
    return {k: v / level for k, v in out.items()}
