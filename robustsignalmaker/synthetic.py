"""Synthetic control: signals with a KNOWN informative band.

Milestone-2 validation fixture and the substrate for the synthetic benchmark
porting the discriminating design of RobustPixelMaker's
synthetic module to 1D. The informative band is a run of ``region_cells``
independent cells of ``cell`` points, each carrying its own latent, and the
target depends on the SUM of those latents, so every informative cell is
jointly necessary (no single cell suffices). A distractor band of equal-size,
equal-variance cells carries latents that do NOT enter the target, and the
rest of the axis is noise. A correct selector must recover ALL informative
cells while rejecting both the noise and the distractor.

Two deliberate 1D departures from the RPM design, both needed to make the
control fair to the whole lens ensemble rather than to level lenses only:

  * Cell latents are NON-NEGATIVE amplitudes (half-normal), like peak heights
    driven by concentrations, not signed Gaussians. A rectified-energy lens
    (derivative energy, rectified random convolutions) produces even
    functions of the latent, and an even function of a sign-symmetric latent
    is uncorrelated with it by symmetry (E[|z| z] = 0): with signed latents
    the control would be unlearnable for energy lenses by construction, a
    property of the fixture rather than of the lenses.
  * Each cell carries a smooth raised-cosine bump rather than a flat block.
    A flat block responds to contrast filters only at its edges, which sit
    exactly on segment boundaries and smear the evidence into neighbouring
    segments; a bump keeps the derivative structure inside the cell, which is
    also what real spectral peaks look like.

Multi-channel: every channel carries the same planted structure with
independent noise, matching the joint-gating deliverable (the answer is a
band of the shared axis). Multiclass targets are quantile bins of the latent
sum, so classes are balanced by construction.

Missingness generators (Milestone 7): the four first-class patterns from
the four regimes, each as an explicit, seeded generator returning a bool
validity mask (True = observed):

  * ``mask_scattered``        isolated point NaNs (tabular-style MCAR)
  * ``mask_dropout_stretches``contiguous runs within signals (detector
                              dropout, censored regions)
  * ``mask_band_by_group``    systematically missing bands per sample
                              subgroup (instruments with different ranges)
  * ``mask_saturation_censor``censored-at-limit points marked missing
                              (the v1 semantics: absent, never fabricated)
  * ``drop_labels``           missing y

No generator ever writes a value; they only say what was observed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SignalControl:
    X: np.ndarray            # (n, C, T)
    y: np.ndarray            # (n,)
    informative: np.ndarray  # (T,) bool, ground-truth predictive band
    distractor: np.ndarray   # (T,) bool, non-predictive equal-variance band
    task: str

    def truth_segments(self, grid, min_overlap: float = 0.5) -> np.ndarray:
        """Segment ids whose overlap with the informative band reaches ``min_overlap``."""
        frac = np.add.reduceat(self.informative.astype(float), grid._starts)
        frac = frac / grid.segment_lengths
        return np.flatnonzero(frac >= min_overlap)

    def distractor_segments(self, grid, min_overlap: float = 0.5) -> np.ndarray:
        frac = np.add.reduceat(self.distractor.astype(float), grid._starts)
        frac = frac / grid.segment_lengths
        return np.flatnonzero(frac >= min_overlap)


def make_signal_control(
    n: int = 200,
    n_channels: int = 1,
    n_points: int = 256,
    task: str = "binary",
    signal: float = 2.0,
    noise: float = 1.0,
    cell: int = 8,
    region_cells: int = 3,
    distractor_cells: int = 3,
    n_classes: int = 3,
    seed: int = 0,
) -> SignalControl:
    """Generate a shared-grid synthetic control (see module docstring).

    The informative band starts at T/4; the distractor band follows after a
    one-cell gap. ``task`` is one of ``binary`` (latent sum above its median),
    ``multiclass`` (``n_classes`` quantile bins of the latent sum, balanced by
    construction), or ``regression`` (latent sum plus small noise). Raises if
    the requested cells do not fit on the axis.
    """
    rng = np.random.default_rng(seed)
    T, C = int(n_points), int(n_channels)
    info_start = T // 4
    dist_start = info_start + region_cells * cell + cell  # one-cell gap
    info_cells = [info_start + k * cell for k in range(region_cells)]
    dist_cells = [dist_start + k * cell for k in range(distractor_cells)]
    for start in info_cells + dist_cells:
        if start + cell > T:
            raise ValueError(
                f"cells do not fit on a {T}-point axis (cell={cell}, "
                f"region_cells={region_cells}, distractor_cells={distractor_cells}); "
                f"use more points or fewer/smaller cells"
            )

    # non-negative amplitude latents and a smooth in-cell profile: see module
    # docstring for why both are load-bearing for lens fairness
    S = np.abs(rng.normal(size=(n, len(info_cells))))
    D = np.abs(rng.normal(size=(n, len(dist_cells))))
    bump = 0.5 * (1.0 - np.cos(2.0 * np.pi * (np.arange(cell) + 0.5) / cell))
    X = rng.normal(scale=noise, size=(n, C, T))
    informative = np.zeros(T, dtype=bool)
    distractor = np.zeros(T, dtype=bool)
    for k, start in enumerate(info_cells):
        X[:, :, start:start + cell] += S[:, k, None, None] * signal * bump[None, None, :]
        informative[start:start + cell] = True
    for k, start in enumerate(dist_cells):
        X[:, :, start:start + cell] += D[:, k, None, None] * signal * bump[None, None, :]
        distractor[start:start + cell] = True

    agg = S.sum(axis=1) if info_cells else np.zeros(n)
    if task == "regression":
        y = agg + rng.normal(scale=0.3, size=n)
    elif task == "binary":
        y = ((agg > np.median(agg)).astype(int) if info_cells
             else (rng.normal(size=n) > 0).astype(int))
    elif task == "multiclass":
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        edges = np.quantile(agg, np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.searchsorted(edges, agg).astype(int)
    else:
        raise ValueError("task must be one of {'binary', 'multiclass', 'regression'}")

    return SignalControl(X=X, y=y, informative=informative, distractor=distractor, task=task)


def mask_scattered(shape, frac: float = 0.2, seed: int = 0) -> np.ndarray:
    """Scattered point missingness: each entry independently unobserved with ``frac``.

    Returns a bool validity array of the given shape (True = observed). The
    tabular-style MCAR pattern; the structured generators (dropout stretches,
    band-by-group, saturation censoring) arrive with Milestone 7.
    """
    if not (0.0 <= frac < 1.0):
        raise ValueError("frac must be in [0, 1)")
    rng = np.random.default_rng(seed)
    return rng.random(shape) >= frac


def mask_dropout_stretches(shape, n_stretches: int = 1, min_len: int = 8,
                           max_len: int = 32, seed: int = 0) -> np.ndarray:
    """Contiguous unobserved runs within each signal (detector dropout).

    ``shape`` is (n, C, T). Each (sample, channel) trace independently loses
    ``n_stretches`` runs of uniform random length in [min_len, max_len] at
    uniform random positions (runs may overlap each other).
    """
    n, C, T = shape
    if not 1 <= min_len <= max_len <= T:
        raise ValueError("need 1 <= min_len <= max_len <= T")
    rng = np.random.default_rng(seed)
    V = np.ones(shape, dtype=bool)
    for i in range(n):
        for c in range(C):
            for _ in range(n_stretches):
                length = int(rng.integers(min_len, max_len + 1))
                start = int(rng.integers(0, T - length + 1))
                V[i, c, start:start + length] = False
    return V


def mask_band_by_group(shape, groups, bands) -> np.ndarray:
    """Systematically missing bands per sample subgroup (instrument ranges).

    ``groups`` is (n,) labels; ``bands`` maps a label to a list of point
    ranges ``(lo, hi)`` (half-open) that samples of that group never observe,
    on every channel. Labels absent from ``bands`` observe everything.
    Deterministic by construction (no randomness to seed).
    """
    n, C, T = shape
    groups = np.asarray(groups)
    if groups.shape != (n,):
        raise ValueError(f"groups must be shaped ({n},); got {groups.shape}")
    V = np.ones(shape, dtype=bool)
    for label, ranges in bands.items():
        rows = np.flatnonzero(groups == label)
        for lo, hi in ranges:
            if not 0 <= lo < hi <= T:
                raise ValueError(f"band ({lo}, {hi}) does not fit on [0, {T})")
            V[rows[:, None], :, np.arange(lo, hi)[None, :]] = False
    return V


def mask_saturation_censor(X, lower=None, upper=None):
    """Censored-at-limit points marked missing (v1 semantics: absent).

    Returns ``(X_censored, V)``: values beyond a limit are CLIPPED to it (as
    a real detector would record them) and marked unobserved, so the honest
    pipeline never reads the fabricated limit value. At least one limit must
    be given; informative-censoring models are a documented later extension.
    """
    if lower is None and upper is None:
        raise ValueError("give at least one of lower, upper")
    X = np.asarray(X, dtype=float)
    V = np.ones(X.shape, dtype=bool)
    Xc = X.copy()
    if upper is not None:
        hit = X >= upper
        Xc[hit] = upper
        V &= ~hit
    if lower is not None:
        hit = X <= lower
        Xc[hit] = lower
        V &= ~hit
    return Xc, V


def drop_labels(y, frac: float = 0.2, seed: int = 0) -> np.ndarray:
    """Missing y: a random fraction of targets replaced with nan (float out).

    Classification labels are dropped per class (stratified) so no class can
    lose all its labelled members; at least one labelled sample per class is
    always kept.
    """
    if not 0.0 <= frac < 1.0:
        raise ValueError("frac must be in [0, 1)")
    y = np.asarray(y, dtype=float).copy()
    rng = np.random.default_rng(seed)
    values = np.unique(y)
    if len(values) <= max(20, int(0.05 * len(y))) and np.all(np.mod(y, 1) == 0):
        for v in values:  # stratified: drop within each class
            idx = np.flatnonzero(y == v)
            k = min(int(round(frac * len(idx))), len(idx) - 1)
            y[rng.choice(idx, size=k, replace=False)] = np.nan
    else:
        k = int(round(frac * len(y)))
        y[rng.choice(len(y), size=k, replace=False)] = np.nan
    return y
