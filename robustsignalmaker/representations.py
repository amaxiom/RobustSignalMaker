"""Signal lenses, and the ensemble over which selection is later marginalised.

Ported contract from RobustPixelMaker's representations module: the objection
to any band-selection method is that the answer depends on how the signal was
represented; choose a different lens and different bands look important.
Rather than picking one lens and defending it, the representation is treated
as a nuisance variable and the selection is marginalised over an ensemble of
them (Milestone 6), so what survives is what is selected regardless of lens.

The crucial invariant, verbatim from RPM: a lens changes *which segments rank
highly*, never *the coordinate system of the answer*. Every lens returns
features indexed ``(n_samples, n_segments, n_feature_channels)``, the mask
gates whole segments across all their feature channels, and the deliverable
stays a region of the physical sampling axis.

RSM adds a second contract clause the siblings do not have: **every lens
consumes and emits validity.** ``transform(X, V, grid) -> (F, FV)`` with
``F`` and ``FV`` both ``(n, S, K)``; convolutional lenses route through
``validity.renormalised_convolve1d`` so a gap in the input is absent, never a
fabricated edge, and the output validity is the (fractional) surviving
evidence, which flows into pooling as a weight. Two properties are named
tests: a lens must reduce exactly to its unmasked form when V is all ones,
and a masked payload must never change any output bit.

Feature-channel layout for multi-filter lenses is block-per-filter: feature
channel ``f * C + c`` is input channel ``c`` seen through filter ``f``.
"""
from __future__ import annotations

import numpy as np

from .validity import renormalised_convolve1d


def _canonical(X, V, grid):
    """Common lens entry: shape checks, float casts, and payload hygiene.

    Values at ``V == 0`` are overwritten with 0.0 before any arithmetic, so a
    masked payload (including NaN or 1e300) cannot reach a sum, a square, or a
    max. This is the line that makes the bitwise payload-invariance test hold
    for every lens rather than for the careful ones.
    """
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"expected X shaped (n, C, T); got {X.shape}")
    if V.shape != X.shape:
        raise ValueError(f"V shape {V.shape} does not match X shape {X.shape}")
    if X.shape[-1] != grid.n_points:
        raise ValueError(
            f"signal length {X.shape[-1]} does not match grid n_points {grid.n_points}"
        )
    return np.where(V > 0, X, 0.0), V


class SegmentMean:
    """Validity-weighted per-segment mean: the plainest lens, and the default.

    Also the lens the final refit predictor uses (the ensemble decides WHICH
    region to keep; the deliverable should not inherit one member's feature
    space, per RPM selection.py).
    """

    name = "mean"

    def transform(self, X, V, grid):
        X, V = _canonical(X, V, grid)
        return grid.pool(X, V)


class SegmentStats:
    """Per-segment mean and dispersion (a texture-aware lens).

    The within-segment standard deviation is computed scale-safely: the
    per-signal max-abs is factored out before anything is squared, so a
    spectrum at 1e300 cannot overflow the second moment (the MissLearn
    lesson: scale-free quantities must not be computed scale-dependently).
    A std pooled from fewer than ``min_count`` observed points is not
    evidence; its feature validity is 0, never a confident value.
    """

    def __init__(self, min_count: int = 2):
        self.min_count = int(min_count)
        self.name = "stats"

    def transform(self, X, V, grid):
        X, V = _canonical(X, V, grid)
        # factor out per-(sample, channel) max-abs of the observed values
        s = np.max(np.where(V > 0, np.abs(X), 0.0), axis=-1, keepdims=True)
        g = np.divide(X, s, out=np.zeros_like(X), where=s > 0)
        mean_g, VF = grid.pool(g, V)
        mean_pt = grid.expand(mean_g.swapaxes(1, 2))  # (n, C, T)
        var_g, _ = grid.pool((g - mean_pt) ** 2, V)
        counts = VF * grid.segment_lengths[None, :, None]
        s_seg = s[:, :, 0][:, None, :]  # (n, 1, C) broadcast over segments
        mean = mean_g * s_seg
        std = np.sqrt(np.maximum(var_g, 0.0)) * s_seg
        enough = counts >= self.min_count
        std = np.where(enough, std, 0.0)
        F = np.concatenate([mean, std], axis=2)
        FV = np.concatenate([VF, np.where(enough, VF, 0.0)], axis=2)
        return F, FV


class SavGolDerivative:
    """Savitzky-Golay derivative energy per segment, made NaN-safe.

    The classic chemometrics move (derivatives suppress smooth baselines and
    expose band structure), computed through the renormalised convolution so
    a gap contributes no fabricated slope: the SG derivative kernel is
    zero-sum, which exercises the contrast-filter re-centring path. Pooled as
    rectified energy (absolute response), which is linear in the data and so
    stays scale-safe.
    """

    def __init__(self, window: int = 11, polyorder: int = 2, deriv: int = 1):
        if window % 2 != 1 or window < 3:
            raise ValueError("window must be odd and >= 3")
        if not (0 < deriv <= polyorder < window):
            raise ValueError("need 0 < deriv <= polyorder < window")
        self.window = int(window)
        self.polyorder = int(polyorder)
        self.deriv = int(deriv)
        self.name = f"savgol_d{deriv}w{window}"

    def _kernel(self):
        from scipy.signal import savgol_coeffs

        return savgol_coeffs(self.window, self.polyorder, deriv=self.deriv)

    def transform(self, X, V, grid):
        X, V = _canonical(X, V, grid)
        out, v_out = renormalised_convolve1d(X, V, self._kernel())
        return grid.pool(np.abs(out), v_out)


class GaussianScale1D:
    """Segment means of Gaussian-smoothed signals: the same lens at coarser scales.

    A normalised Gaussian is a level filter, so the renormalised convolution
    reports the average of the evidence that survived a gap rather than a
    dimmed value. Multiple sigmas separate sharp-peak information from
    broad-envelope information (Raman lines vs UV-Vis envelopes).
    """

    def __init__(self, sigmas=(2.0, 8.0), truncate: float = 4.0):
        self.sigmas = tuple(float(s) for s in sigmas)
        if not self.sigmas or any(s <= 0 for s in self.sigmas):
            raise ValueError("sigmas must be positive")
        self.truncate = float(truncate)
        self.name = "gauss" + "_".join(f"{s:g}" for s in self.sigmas)

    def _kernel(self, sigma):
        radius = max(1, int(self.truncate * sigma + 0.5))
        x = np.arange(-radius, radius + 1, dtype=float)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        return k / k.sum()

    def transform(self, X, V, grid):
        X, V = _canonical(X, V, grid)
        fs, fvs = [], []
        for sigma in self.sigmas:
            out, v_out = renormalised_convolve1d(X, V, self._kernel(sigma))
            F, FV = grid.pool(out, v_out)
            fs.append(F)
            fvs.append(FV)
        return np.concatenate(fs, axis=2), np.concatenate(fvs, axis=2)


class RandomConv1D:
    """Random zero-mean 1D filters, rectified and pooled per segment (ROCKET-lite).

    Each seed yields a genuinely different lens; zero-meaning makes a channel
    respond to local structure rather than overall level, which would
    duplicate the plain mean lens. Zero-sum filters take the contrast path of
    the renormalised convolution, so gaps fabricate no response.
    """

    def __init__(self, n_filters: int = 4, kernel: int = 9, seed: int = 0):
        if kernel < 2:
            raise ValueError("kernel must be >= 2")
        self.n_filters = int(n_filters)
        self.kernel = int(kernel)
        self.seed = int(seed)
        self.name = f"randconv{seed}"

    def _filters(self):
        rng = np.random.default_rng(self.seed)
        f = rng.normal(size=(self.n_filters, self.kernel))
        f -= f.mean(axis=1, keepdims=True)
        norm = np.sqrt((f ** 2).sum(axis=1, keepdims=True))
        return f / np.maximum(norm, 1e-8)

    def transform(self, X, V, grid):
        X, V = _canonical(X, V, grid)
        fs, fvs = [], []
        for filt in self._filters():
            out, v_out = renormalised_convolve1d(X, V, filt)
            F, FV = grid.pool(np.maximum(out, 0.0), v_out)
            fs.append(F)
            fvs.append(FV)
        return np.concatenate(fs, axis=2), np.concatenate(fvs, axis=2)


def default_ensemble(n_representations: int = 4, seed: int = 0) -> list:
    """A deliberately diverse default ensemble (RPM convention).

    Diversity is the point: members that agree by construction would make the
    marginalisation vacuous. The set spans the raw mean lens, a derivative
    lens, a coarser smoothed scale, and random convolutional lenses; further
    members are extra random lenses at distinct seeds.
    """
    pool = [
        SegmentMean(),
        SavGolDerivative(),
        GaussianScale1D(),
        RandomConv1D(seed=seed + 1000),
        SegmentStats(),
    ]
    pool += [RandomConv1D(seed=seed + 1001 + i) for i in range(7)]
    if n_representations > len(pool):
        pool += [RandomConv1D(seed=seed + 2000 + i)
                 for i in range(n_representations - len(pool))]
    if n_representations < 1:
        raise ValueError("n_representations must be >= 1")
    return pool[:n_representations]
