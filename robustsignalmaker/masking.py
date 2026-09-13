"""Soft-mask selection: the 1D gated engine (Milestone 3).

A single GLOBAL soft mask over a segment grid: one gate vector shared across
all signals. The gates weight standardised per-segment features and a linear,
logistic or softmax head predicts the target. Training is two-phase,
VTF-style, ported from RobustPixelMaker's masking module:

  Phase 1  fit the head on the FULL feature set (mask = 1), then FREEZE it.
  Phase 2  train only the mask against the frozen head, with a sparsity
           penalty.

Freezing matters (the RPM lesson, kept verbatim): trained jointly, the mask
and head are scale-degenerate: the head can grow its weights to cancel a
shrinking mask, so the sparsity penalty drives the mask toward zero without
changing predictions. With the head fixed, shrinking a gate genuinely changes
predictions, so only segments whose removal does not hurt the frozen head are
pruned.

Gate parameterisations (RPM defaults, evidence-backed there):

  ``gate="hardconcrete"``  DEFAULT. Hard-Concrete / L0 gates (Louizos,
                           Welling and Kingma, 2018): stochastic and
                           differentiable in training, deterministic at test
                           time (exactly 0 or 1 once training saturates a
                           gate; a mid-range gate stays fractional, and
                           ``selected_segments_`` binarises at
                           ``mask_threshold``); the penalty is on the
                           expected COUNT of open gates.
  ``gate="sigmoid"``       m = sigmoid(theta) with an L1 penalty on sum(m).

The 1D total-variation prior couples neighbouring gates so contiguous bands
are kept or dropped together; only pairs of ASSESSABLE segments are coupled
(coupling an assessable gate to a data-free neighbour would drag it toward an
arbitrary state).

What is new relative to RPM: validity is threaded end to end, and three
decisions carry the honesty contract.

  * The RENORMALISED HEAD. With per-(sample, segment, channel) feature
    weights ``omega`` (validity fractions, zeroed below ``min_valid_frac``),
    per-segment contributions are ``U[i,s,k] = sum_c omega[i,s,c] *
    fstd[i,s,c] * W[s,c,k]``, and the decision is

        z_i = (M / D_i) * sum_s m_s U[i,s,:] + b,
        M = sum_s m_s (assessable s),  D_i = sum_s m_s * vseg[i,s]

    which reduces exactly to the plain gated head on complete data and is a
    ratio-form Horvitz-Thompson estimate under MCAR: missingness becomes
    visible variance, never quiet attenuation ("unobserved" must not
    masquerade as "unimportant"). Note validity enters ONCE, inside the
    channel sum; the plan-file shorthand ``m_k v_ik u_ik`` with v also inside
    u would square it for a single channel and re-introduce attenuation.
  * The STOP-GRADIENT RENORMALISER. M / D_i is treated as a constant when
    differentiating with respect to the gates, so a never-observed segment
    receives exactly zero gate gradient (through the true derivative it
    would feel pressure via the renormaliser while changing no prediction).
  * The OBSERVATION-SCALED PENALTY. The data gradient of gate s scales with
    its observation rate while the sparsity pressure does not, so a rarely
    observed segment would drift closed. The penalty is scaled per segment,
    ``lam_s = lam * max(obs_rate_s, o_floor)``, which restores the ratio
    deterministically instead of amplifying an already-noisy gradient.

Guards fail loudly: no assessable segment, or no sample with enough observed
evidence, raises InsufficientEvidenceError; samples below ``rho_min``
observed gated evidence get nan predictions and are excluded (and counted),
never amplified.

Deliberately numpy-only with hand-derived analytic gradients (checked against
numeric differentiation in the tests, which the masked head makes
non-negotiable).
"""
from __future__ import annotations

import numpy as np

from .segments import SegmentGrid
from .validity import (
    InsufficientEvidenceError,
    as_validity,
    masked_standardise,
    masked_standardise_fit,
)

# Hard-Concrete stretch constants (Louizos et al. 2018, sec.4; RPM values)
HC_BETA = 2.0 / 3.0
HC_GAMMA = -0.1
HC_ZETA = 1.1


def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))),
                    np.exp(np.clip(z, -50, 50)) / (1.0 + np.exp(np.clip(z, -50, 50))))


def renormalised_gated_forward(U, v_seg, m, active, intercept, rho_min=0.1,
                               r_override=None):
    """The renormalised gated head, as a pure function (see module docstring).

    ``U`` is (n, S, K) per-segment head contributions, ``v_seg`` (n, S)
    per-sample segment validity, ``m`` (S,) gate values, ``active`` (S,) bool
    assessability. Returns ``(z, valid, r)``: decisions (n, K) with nan rows
    where a sample's observed gated evidence fraction is below ``rho_min``,
    the validity flags, and the renormaliser used. ``r_override`` freezes the
    renormaliser (the stop-gradient surrogate; used by the gradient tests).
    Raises when every active gate is closed.
    """
    U = np.asarray(U, dtype=float)
    m_act = np.asarray(m, dtype=float) * np.asarray(active, dtype=float)
    M = float(m_act.sum())
    if M <= 0.0:
        raise InsufficientEvidenceError("gated forward: all active gates are closed")
    D = np.asarray(v_seg, dtype=float) @ m_act
    rho = D / M
    valid = (D > 0) & (rho >= rho_min)
    if r_override is None:
        r = np.where(valid, np.divide(M, D, out=np.zeros_like(D), where=D > 0), 0.0)
    else:
        r = np.asarray(r_override, dtype=float)
    z = r[:, None] * np.einsum("nsk,s->nk", U, m_act) + intercept[None, :]
    z = np.where(valid[:, None], z, np.nan)
    return z, valid, r


def mask_data_gradient(U, dz, r, active):
    """Analytic stop-gradient dLoss/dm: exactly zero for never-observed segments.

    ``dz`` is dLoss/dz with excluded samples' rows already zeroed; ``r`` the
    renormaliser at the current gates, treated as constant (stop-gradient).
    """
    dm = np.einsum("nk,nsk->s", np.asarray(dz) * np.asarray(r)[:, None], np.asarray(U))
    return dm * np.asarray(active, dtype=float)


def loss_and_dz(task, Z, Y, valid):
    """Task loss and dLoss/dZ over the valid samples only.

    Excluded samples (``valid`` False) contribute nothing: their dz rows are
    zero and the mean runs over the valid count. Nan rows in Z are never read.
    """
    n_ok = int(np.sum(valid))
    if n_ok == 0:
        raise InsufficientEvidenceError("loss: no sample has enough observed evidence")
    Zv = np.where(valid[:, None], np.nan_to_num(Z, nan=0.0), 0.0)
    if task == "multiclass":
        # exact cross-entropy via log-sum-exp: no probability epsilon, so the
        # analytic gradient (softmax - Y) is the exact derivative of the loss
        # reported (the gradient check depends on this)
        zmax = Zv.max(axis=1, keepdims=True)
        expZ = np.exp(Zv - zmax)
        sumexp = expZ.sum(axis=1, keepdims=True)
        lse = (zmax + np.log(sumexp))[:, 0]
        loss = np.sum(valid * (lse - np.sum(Y * Zv, axis=1))) / n_ok
        Pm = expZ / sumexp
        dz = np.where(valid[:, None], (Pm - Y) / n_ok, 0.0)
        return loss, dz
    if task == "binary":
        # exact BCE via stable softplus: loss = y softplus(-z) + (1-y) softplus(z),
        # whose exact derivative is sigmoid(z) - y
        z = Zv[:, 0]
        softplus = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
        per = Y[:, 0] * (softplus - z) + (1 - Y[:, 0]) * softplus
        loss = np.sum(valid * per) / n_ok
        dz = np.where(valid[:, None], (sigmoid(Zv) - Y) / n_ok, 0.0)
        return loss, dz
    if task == "regression":
        loss = 0.5 * np.sum(valid[:, None] * (Zv - Y) ** 2) / n_ok
        dz = np.where(valid[:, None], (Zv - Y) / n_ok, 0.0)
        return loss, dz
    raise ValueError(f"unknown task: {task!r}")


class SoftMaskSelector:
    """Global segment-level soft mask plus a linear/logistic/softmax head.

    ``task`` in {"binary", "multiclass", "regression"}. Head weights are
    (S, C_feat, K) with K = 1 for binary/regression and K = n_classes for
    multiclass, so one set of gradient expressions covers every task.

    ``fixed_mask``: when supplied the mask is not learned; the head is fitted
    against it and phase 2 is skipped (used by the bootstrap aggregator to
    fit the final model on the stability-selected region).

    ``representation``: a lens with ``transform(X, V, grid) -> (F, FV)``;
    None means the plain validity-weighted segment mean. A lens changes which
    segments rank highly, never the coordinate system of the answer.
    """

    def __init__(self, task: str = "binary", segment: int = 16, lam: float = 0.03,
                 l2: float = 1e-3, lr: float = 0.08, n_iter: int = 800,
                 warmup_frac: float = 0.3, mask_threshold: float = 0.5,
                 gate: str = "hardconcrete", tv: float = 0.0, seed: int = 0,
                 fixed_mask=None, representation=None, min_valid_frac: float = 0.5,
                 rho_min: float = 0.1, o_floor: float = 0.2,
                 n_min_hard: int = 8, n_min_warn: int = 30):
        if task not in ("binary", "multiclass", "regression"):
            raise ValueError("task must be one of {'binary','multiclass','regression'}")
        if gate not in ("sigmoid", "hardconcrete"):
            raise ValueError("gate must be one of {'sigmoid','hardconcrete'}")
        self.task = task
        self.segment = segment
        self.lam = lam
        self.l2 = l2
        self.lr = lr
        self.n_iter = n_iter
        self.warmup_frac = warmup_frac
        self.mask_threshold = mask_threshold
        self.gate = gate
        self.tv = tv
        self.seed = seed
        self.fixed_mask = None if fixed_mask is None else np.asarray(fixed_mask, dtype=float)
        self.representation = representation
        self.min_valid_frac = min_valid_frac
        self.rho_min = rho_min
        self.o_floor = o_floor
        self.n_min_hard = n_min_hard
        self.n_min_warn = n_min_warn

    # -- feature pipeline (fit on train only) ------------------------------- #
    def _features(self, X, V):
        """Lens features and eligibility weights: (F, omega), both (n, S, C)."""
        if self.representation is None:
            from .representations import SegmentMean

            F, FV = SegmentMean().transform(X, V, self.grid)
        else:
            F, FV = self.representation.transform(X, V, self.grid)
            if F.shape[1] != self.grid.n_segments:
                raise ValueError(
                    f"representation returned {F.shape[1]} segments, "
                    f"expected {self.grid.n_segments}")
        omega = np.where(FV >= self.min_valid_frac, FV, 0.0)
        return F, omega

    def _prepare_targets(self, y):
        """(n, K) target matrix; regression targets standardised scale-safely.

        RPM standardised y with a raw np.std, which overflows at 1e300; here
        the max-abs factor comes out first (the MissLearn scale lesson).
        """
        y = np.asarray(y)
        if self.task == "multiclass":
            self.classes_ = np.unique(y)
            if len(self.classes_) < 2:
                raise ValueError(
                    f"multiclass task needs at least 2 classes; got {self.classes_}")
            idx = np.searchsorted(self.classes_, y)
            Y = np.zeros((len(y), len(self.classes_)))
            Y[np.arange(len(y)), idx] = 1.0
            return Y
        if self.task == "binary":
            self.classes_ = np.unique(y)
            if len(self.classes_) != 2:
                raise ValueError(f"binary task needs exactly 2 classes; got {self.classes_}")
            pos = self.classes_[-1]
            return (y == pos).astype(float).reshape(-1, 1)
        y = y.astype(float)
        self._y_scale = float(np.max(np.abs(y))) or 1.0
        g = y / self._y_scale
        self._y_mu = float(np.mean(g))
        self._y_sd = float(np.std(g)) or 1.0
        return ((g - self._y_mu) / self._y_sd).reshape(-1, 1)

    def _tv_grad(self, m):
        """Gradient of (1/2) sum of squared adjacent-gate differences (chain
        Laplacian), coupling only pairs where both segments are assessable."""
        g = np.zeros_like(m)
        both = self._active[1:] & self._active[:-1]
        d = np.where(both, m[1:] - m[:-1], 0.0)
        g[1:] += d
        g[:-1] -= d
        return g

    # -- Hard-Concrete gate helpers (RPM, verbatim) -------------------------- #
    @staticmethod
    def _hc_sample(log_alpha, rng):
        """Stochastic gate value and d(gate)/d(log_alpha) for the training pass."""
        u = rng.uniform(1e-6, 1 - 1e-6, size=log_alpha.shape)
        s = sigmoid((np.log(u) - np.log1p(-u) + log_alpha) / HC_BETA)
        s_bar = s * (HC_ZETA - HC_GAMMA) + HC_GAMMA
        z = np.clip(s_bar, 0.0, 1.0)
        active = (s_bar > 0.0) & (s_bar < 1.0)
        dz = np.where(active, (HC_ZETA - HC_GAMMA) * s * (1 - s) / HC_BETA, 0.0)
        return z, dz

    @staticmethod
    def _hc_deterministic(log_alpha):
        """Test-time gate: no noise; exactly 0 or 1 outside the stretch
        interval, fractional inside it (selection binarises separately)."""
        s = sigmoid(log_alpha)
        return np.clip(s * (HC_ZETA - HC_GAMMA) + HC_GAMMA, 0.0, 1.0)

    @staticmethod
    def _hc_open_prob(log_alpha):
        """P(gate > 0), the differentiable stand-in for the L0 count penalty."""
        return sigmoid(log_alpha - HC_BETA * np.log(-HC_GAMMA / HC_ZETA))

    # -- fitting ------------------------------------------------------------ #
    def fit(self, X, y, V=None):
        X, Vb = as_validity(X, V)
        if X.ndim != 3:
            raise ValueError(f"expected X shaped (n, C, T); got {X.shape}")
        self.grid = SegmentGrid.from_signal_shape(X.shape, segment=self.segment)
        Vf = Vb.astype(float)
        F, omega = self._features(X, Vf)
        n, S, C = F.shape

        self.stats_ = masked_standardise_fit(
            F, omega, n_min_hard=self.n_min_hard, n_min_warn=self.n_min_warn, strict=True)
        Fstd = masked_standardise(F, omega, self.stats_)
        G = Fstd * omega  # validity enters the head exactly once, here

        # segment-level validity and assessability
        self._v_seg = omega.mean(axis=2)                          # (n, S)
        self._active = self.stats_.assessable.any(axis=1)         # (S,)
        self.assessable_segments_ = self._active.copy()
        self.obs_rate_ = self._v_seg.mean(axis=0)                 # (S,)
        if not self._active.any():  # pragma: no cover - unreachable invariant
            # masked_standardise_fit(strict=True) above already raises when no
            # (segment, channel) is assessable, and _active is that same array
            # reduced over channels, so this can only fire if the two ever
            # disagree. Kept as an assertion of the invariant, not dead weight.
            raise InsufficientEvidenceError(
                "no segment is assessable; the data cannot support selection")

        Y = self._prepare_targets(y)
        K = Y.shape[1]
        rng = np.random.default_rng(self.seed)
        b1, b2, aeps = 0.9, 0.999, 1e-8

        if self.fixed_mask is not None:
            if self.fixed_mask.shape != (S,):
                raise ValueError(
                    f"fixed_mask has shape {self.fixed_mask.shape}, expected ({S},)")
            self._fit_head(G, Y, self.fixed_mask, n_iter=self.n_iter)
            self.gate_params_ = None
            self.mask_ = self.fixed_mask
            self.final_loss_ = self.head_loss_  # both paths expose final_loss_
            self._finalise_selection()
            return self

        # -- Phase 1: head on the full feature set (mask = 1), then frozen --- #
        ones = np.ones(S)
        self._fit_head(G, Y, ones, n_iter=max(1, int(self.warmup_frac * self.n_iter)))
        W3 = self.coef_
        c = self.intercept_

        # per-segment head contributions with the head frozen
        U = np.einsum("nsc,sck->nsk", G, W3)
        lam_s = self.lam * np.maximum(self.obs_rate_, self.o_floor)

        # -- Phase 2: freeze (W, c); train the gates -------------------------- #
        params = np.zeros(S) if self.gate == "sigmoid" else np.full(S, 2.0)
        mt = np.zeros(S)
        vt = np.zeros(S)
        mask_steps = max(1, self.n_iter - max(1, int(self.warmup_frac * self.n_iter)))
        loss = None
        for t in range(1, mask_steps + 1):
            if self.gate == "sigmoid":
                m = sigmoid(params)
                dgate = m * (1 - m)
                penalty_grad = lam_s * dgate
                penalty_val = float((lam_s * m)[self._active].sum())
            else:
                m, dgate = self._hc_sample(params, rng)
                q = self._hc_open_prob(params)
                penalty_grad = lam_s * q * (1 - q)
                penalty_val = float((lam_s * q)[self._active].sum())

            if float((m * self._active).sum()) > 0.0:
                Z, valid, r = renormalised_gated_forward(
                    U, self._v_seg, m, self._active, c, rho_min=self.rho_min)
                data_loss, dz = loss_and_dz(self.task, Z, Y, valid)
                dm = mask_data_gradient(U, dz, r, self._active)
            else:
                # a stochastic draw closed every gate: no data term this step;
                # the sparsity penalty alone moves the parameters, and if the
                # FINAL mask is all-closed that is flagged degenerate (and
                # prediction refuses loudly) rather than papered over
                data_loss, dm = 0.0, np.zeros_like(m)
            if self.tv:
                dm = dm + self.tv * self._tv_grad(m)
            grad = (dm * dgate + penalty_grad) * self._active

            mt = b1 * mt + (1 - b1) * grad
            vt = b2 * vt + (1 - b2) * grad ** 2
            params = params - self.lr * (mt / (1 - b1 ** t)) / (np.sqrt(vt / (1 - b2 ** t)) + aeps)
            # the reported loss carries EVERY term the gradient optimises
            # (2026-09-01 sweep: TV was optimised but unreported)
            tv_val = 0.0
            if self.tv:
                both = self._active[1:] & self._active[:-1]
                tv_val = 0.5 * self.tv * float(
                    (np.where(both, m[1:] - m[:-1], 0.0) ** 2).sum())
            loss = data_loss + penalty_val + tv_val

        self.gate_params_ = params
        self.mask_ = sigmoid(params) if self.gate == "sigmoid" else self._hc_deterministic(params)
        self.final_loss_ = loss
        self._finalise_selection()
        return self

    def _fit_head(self, G, Y, m, n_iter):
        """Adam fit of (W, c) against gates ``m`` (phase 1 and fixed-mask refits)."""
        n, S, C = G.shape
        K = Y.shape[1]
        m_act = m * self._active
        Gm = (G * m_act[None, :, None]).reshape(n, S * C)
        M = float(m_act.sum())
        if M <= 0.0:
            raise InsufficientEvidenceError("head fit: all active gates are closed")
        D = self._v_seg @ m_act
        rho = D / M
        valid = (D > 0) & (rho >= self.rho_min)
        if not valid.any():
            raise InsufficientEvidenceError(
                "head fit: no sample reaches the rho_min observed-evidence floor")
        r = np.where(valid, np.divide(M, D, out=np.zeros_like(D), where=D > 0), 0.0)
        self.n_excluded_samples_ = int((~valid).sum())

        W = np.zeros((S * C, K))
        c = np.zeros(K)
        b1, b2, aeps = 0.9, 0.999, 1e-8
        mt = np.zeros(S * C * K + K)
        vt = np.zeros(S * C * K + K)
        Gr = Gm * r[:, None]
        loss = None
        for t in range(1, n_iter + 1):
            Z = Gr @ W + c
            loss, dz = loss_and_dz(self.task, Z, Y, valid)
            grad = np.concatenate([(Gr.T @ dz + self.l2 * W).ravel(), dz.sum(axis=0)])
            mt = b1 * mt + (1 - b1) * grad
            vt = b2 * vt + (1 - b2) * grad ** 2
            step = (mt / (1 - b1 ** t)) / (np.sqrt(vt / (1 - b2 ** t)) + aeps)
            W = W - self.lr * step[:S * C * K].reshape(S * C, K)
            c = c - self.lr * step[S * C * K:]
        self.coef_ = W.reshape(S, C, K)
        self.intercept_ = c
        # reported with the L2 term the gradient optimises
        self.head_loss_ = (loss + 0.5 * self.l2 * float((W ** 2).sum())
                           if loss is not None else None)

    def _finalise_selection(self):
        open_gates = (self.mask_ >= self.mask_threshold) & self._active
        self.selected_segments_ = np.flatnonzero(open_gates).astype(int)
        self.unassessable_segments_ = np.flatnonzero(~self._active).astype(int)
        n_active = int(self._active.sum())
        n_open = len(self.selected_segments_)
        if n_open == 0:
            self.degenerate_ = "all_closed"
        elif n_open == n_active:
            self.degenerate_ = "all_open"
        else:
            self.degenerate_ = None

    # -- prediction --------------------------------------------------------- #
    def _decision(self, X, V=None):
        X, Vb = as_validity(X, V)
        F, omega = self._features(X, Vb.astype(float))
        Fstd = masked_standardise(F, omega, self.stats_)
        G = Fstd * omega
        U = np.einsum("nsc,sck->nsk", G, self.coef_)
        v_seg = omega.mean(axis=2)
        return renormalised_gated_forward(
            U, v_seg, self.mask_, self._active, self.intercept_, rho_min=self.rho_min)

    def decision_function(self, X, V=None):
        """Raw decisions (n, K); nan rows where evidence is below ``rho_min``."""
        Z, _, _ = self._decision(X, V)
        return Z

    def prediction_valid(self, X, V=None):
        """Which samples have enough observed gated evidence to be predicted."""
        _, valid, _ = self._decision(X, V)
        return valid

    def _labels_with_refusals(self, labels, valid):
        """Refused samples read as nan (numeric labels) or None (other dtypes),
        never as a guessed class."""
        if np.issubdtype(np.asarray(self.classes_).dtype, np.number):
            return np.where(valid, np.asarray(labels, dtype=float), np.nan)
        out = np.asarray(labels, dtype=object)
        out[~valid] = None
        return out

    def predict(self, X, V=None):
        """Predictions with nan/None where a sample cannot honestly be scored."""
        Z, valid, _ = self._decision(X, V)
        if self.task == "regression":
            out = (Z[:, 0] * self._y_sd + self._y_mu) * self._y_scale
            return np.where(valid, out, np.nan)
        if self.task == "multiclass":
            idx = np.nan_to_num(Z, nan=-np.inf).argmax(axis=1)
            return self._labels_with_refusals(self.classes_[idx], valid)
        p = sigmoid(np.nan_to_num(Z[:, 0], nan=0.0))
        labels = np.where(p >= 0.5, self.classes_[-1], self.classes_[0])
        return self._labels_with_refusals(labels, valid)

    def predict_proba(self, X, V=None):
        Z, valid, _ = self._decision(X, V)
        Zv = np.nan_to_num(Z, nan=0.0)
        if self.task == "multiclass":
            Zs = Zv - Zv.max(axis=1, keepdims=True)
            expZ = np.exp(Zs)
            P = expZ / expZ.sum(axis=1, keepdims=True)
        else:
            p = sigmoid(Zv[:, 0])
            P = np.column_stack([1 - p, p])
        return np.where(valid[:, None], P, np.nan)

    # -- selection outputs -------------------------------------------------- #
    def selected_points(self) -> np.ndarray:
        return self.grid.segments_to_points(self.selected_segments_)

    def coverage(self) -> float:
        """Fraction of ASSESSABLE segments retained (never-observed segments
        are not candidates, so counting them would flatter the compression)."""
        n_active = int(self._active.sum())
        return len(self.selected_segments_) / n_active if n_active else float("nan")
