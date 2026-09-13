"""Selection units: the segment partition of a shared sampling axis.

The 1D analogue of RobustPixelMaker's PatchGrid, and the same argument
carries over from its regimes module: neighbouring points of a signal are
near-equivalent under autocorrelation, so a point-level mask rarely
stabilises, whereas a segment/band mask does. ``segment=1`` IS the point-wise
mode, through the same code path with no special cases; the docs and the
stability report, not the code, carry the warning about using it on smooth
spectra. Peak-aware or change-point segmentation are later backends behind
the same interface (the selection layers see only ``n_segments``, ``pool``,
``expand``).

One deliberate deviation from PatchGrid: RPM pre-normalises its pooling
matrix by fixed pixel counts. RSM cannot, because the denominator is the
per-sample OBSERVED evidence in each segment, which changes with the validity
mask. Pooling therefore normalises at pool time, and returns the segment
validity fractions alongside the pooled values so nothing downstream ever
mistakes "pooled from little evidence" for "pooled from all of it".
"""
from __future__ import annotations

import numpy as np


class SegmentGrid:
    """A regular partition of a shared 1D sampling axis into fixed windows.

    Provides validity-weighted pooling (signal -> per-segment mean feature
    plus validity fraction) and expansion (per-segment vector -> point map).
    The trailing segment may be shorter when ``segment`` does not divide
    ``n_points``; its validity fractions are computed against its true length.
    """

    def __init__(self, n_points: int, segment: int = 16):
        if segment < 1:
            raise ValueError("segment must be >= 1")
        if n_points < 1:
            raise ValueError("n_points must be >= 1")
        self.n_points = int(n_points)
        self.segment = int(segment)
        self.n_segments = (self.n_points + self.segment - 1) // self.segment

        # point -> segment id map (T,) and reduceat boundaries (S,)
        self.point_segment = (np.arange(self.n_points) // self.segment).astype(int)
        self._starts = np.arange(0, self.n_points, self.segment)
        self.segment_lengths = np.bincount(
            self.point_segment, minlength=self.n_segments
        ).astype(float)

    @classmethod
    def from_signal_shape(cls, shape, segment: int = 16) -> "SegmentGrid":
        """Build from an array shape; the sampling axis is the last axis."""
        if len(shape) < 1:
            raise ValueError("need at least a (T,) shape")
        return cls(shape[-1], segment=segment)

    def _check(self, X, V):
        X = np.asarray(X, dtype=float)
        V = np.asarray(V, dtype=float)
        if X.ndim != 3:
            raise ValueError(f"expected X shaped (n, C, T); got {X.shape}")
        if V.shape != X.shape:
            raise ValueError(f"V shape {V.shape} does not match X shape {X.shape}")
        if X.shape[-1] != self.n_points:
            raise ValueError(
                f"signal length {X.shape[-1]} does not match grid n_points {self.n_points}"
            )
        return X, V

    def pool(self, X, V):
        """(n, C, T), validity (n, C, T) -> (F, VF), both (n, S, C).

        ``F[i, s, c]`` is the validity-weighted mean of the observed points of
        segment ``s`` (0.0 with ``VF = 0`` when nothing was observed; a
        placeholder that must never be read). ``VF[i, s, c]`` is the fraction
        of the segment's evidence that existed, in [0, 1]. ``V`` may be bool
        or fractional (a lens's inherited validity). Values at ``V == 0`` are
        multiplied out before any sum, so masked payloads are never read.
        Fully observed input reduces exactly to plain segment means with
        ``VF == 1``.
        """
        X, V = self._check(X, V)
        num = np.add.reduceat(X * V, self._starts, axis=-1)  # (n, C, S)
        den = np.add.reduceat(V, self._starts, axis=-1)
        has = den > 0
        F = np.divide(num, den, out=np.zeros_like(num), where=has)
        VF = den / self.segment_lengths[None, None, :]
        return F.swapaxes(1, 2), VF.swapaxes(1, 2)

    def segment_validity(self, V):
        """(n, C, T) validity -> (n, S) per-sample segment validity for gating.

        The mean over channels of the segment validity fractions: a segment
        fully observed in one of four channels has validity 0.25. The
        per-channel fractions stay fully granular in ``pool``'s VF; this is
        the collapsed view the shared gate sees.
        """
        V = np.asarray(V, dtype=float)
        if V.ndim != 3 or V.shape[-1] != self.n_points:
            raise ValueError(f"expected V shaped (n, C, T={self.n_points}); got {V.shape}")
        den = np.add.reduceat(V, self._starts, axis=-1)  # (n, C, S)
        return (den / self.segment_lengths[None, None, :]).mean(axis=1)

    def expand(self, seg_arr):
        """(..., S) -> (..., T) point map by segment assignment."""
        seg_arr = np.asarray(seg_arr)
        if seg_arr.shape[-1] != self.n_segments:
            raise ValueError(
                f"last axis {seg_arr.shape[-1]} does not match n_segments {self.n_segments}"
            )
        return np.take(seg_arr, self.point_segment, axis=-1)

    def segments_to_points(self, segment_ids):
        """Point indices covered by the given segment ids."""
        ids = list({int(s) for s in segment_ids})
        return np.flatnonzero(np.isin(self.point_segment, ids))
