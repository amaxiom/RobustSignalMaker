"""Bootstrap stability selection with observation-support accounting (Milestone 4).

The Meinshausen-Buehlmann construction, ported from RobustPixelMaker: a single
mask fit is a function of the particular training sample, so selection is
aggregated over resamples,

    pi_k = fraction of resamples on which segment k was selected,
    S_hat = { k : pi_k >= tau },

and noise, which does not reproduce, is thresholded away.

What RSM adds is the support-conditioned semantics that make pi honest under
missing data:

  * A segment is ASSESSABLE on a resample only if the resample observed it
    (at least ``o_min`` of the resample's samples have any evidence in it)
    and the inner fit could standardise it (the ESS floors). ``pi_k`` counts
    hits over assessable resamples ONLY.
  * A segment assessed by fewer than ``a_min`` of the successful resamples
    gets ``pi_k = nan``: the verdict is three-valued, selected / rejected /
    unassessable, and a band nobody measured can never read as "stably
    rejected". Segments whose raw frequency clears tau but whose support does
    not are reported in ``insufficient_evidence_``, not silently selected.
  * A FAILED resample fit is excluded from every denominator and counted in
    ``n_failed_fits_``. (RPM counted a failed fit as an all-zero vote, which
    biases pi toward 0; that defect is deliberately not ported.)
  * Resampling is stratified by class x observation group, so a resample
    cannot lose an instrument subgroup and silently change which segments are
    assessable. Groups come from the caller when the instrument/session id is
    known; the only automatic fallback is exact validity-pattern grouping
    (``observation_groups``), never clustering guesswork.

Every stability figure this module emits travels with its support: reports
carry coverage, resample support, observation rate and the ambiguous
fraction together, because each one alone can look excellent for a bad
reason (a lesson from the RobustPixelMaker benchmarks, not from this
package's own FINDINGS.md).
"""
from __future__ import annotations

import warnings

import numpy as np

from .masking import SoftMaskSelector
from .reproducibility import Seeds
from .segments import SegmentGrid
from .validity import InsufficientEvidenceError, as_validity


def resample_indices(n, mode, rng, strat=None):
    """Indices for one resample; stratified when ``strat`` labels are supplied."""
    if mode == "bootstrap":
        if strat is None:
            return rng.integers(0, n, size=n)
        out = []
        for lab in np.unique(strat):
            idx = np.where(strat == lab)[0]
            out.append(rng.choice(idx, size=len(idx), replace=True))
        return np.concatenate(out)
    if mode == "half":
        if strat is None:
            return rng.permutation(n)[: max(1, n // 2)]
        out = []
        for lab in np.unique(strat):
            idx = rng.permutation(np.where(strat == lab)[0])
            out.append(idx[: max(1, len(idx) // 2)])
        return np.concatenate(out)
    raise ValueError(f"unknown resample mode {mode!r}")


def complementary_pair(n, rng, strat=None):
    """Two disjoint halves covering the sample (Shah and Samworth pairs)."""
    if strat is None:
        perm = rng.permutation(n)
        cut = n // 2
        return perm[:cut], perm[cut:]
    a, b = [], []
    for lab in np.unique(strat):
        idx = rng.permutation(np.where(strat == lab)[0])
        cut = len(idx) // 2
        a.append(idx[:cut])
        b.append(idx[cut:])
    return np.concatenate(a), np.concatenate(b)


def observation_groups(v_seg, max_groups: int = 8, min_size: int = 4):
    """Exact validity-pattern groups, or None when patterns do not form groups.

    ``v_seg`` is (n, S) per-sample segment validity. Samples sharing exactly
    the same observed-segment pattern (instrument ranges produce these) form
    a group. Scattered missingness gives near-unique patterns, which are not
    groups: when there are more than ``max_groups`` patterns or any pattern
    holds fewer than ``min_size`` samples, this returns None rather than
    inventing structure. Clustering fallbacks are deliberately not offered.
    """
    patterns = np.asarray(v_seg) > 0
    _, inverse, counts = np.unique(patterns, axis=0, return_inverse=True,
                                   return_counts=True)
    if len(counts) < 2 or len(counts) > max_groups or counts.min() < min_size:
        return None
    return inverse


def _frequency_map(results, S, a_min):
    """Support-aware frequency map over (hit, assessable) results.

    Returns ``(pi, pi_raw, support, supported)``: hits counted over
    assessable results only, nan where support falls below ``a_min``.
    """
    hits = np.stack([h for h, _ in results])
    assess = np.stack([a for _, a in results]).astype(float)
    A = assess.sum(axis=0)
    H = (hits * assess).sum(axis=0)
    with np.errstate(invalid="ignore"):
        pi_raw = np.divide(H, A, out=np.full(S, np.nan), where=A > 0)
    support = A / len(results)
    # a segment assessed by ZERO resamples is never supported, whatever
    # a_min says (2026-09-01 sweep: a_min=0 turned nan pi into "rejected")
    supported = (support >= a_min) & np.isfinite(pi_raw)
    return np.where(supported, pi_raw, np.nan), pi_raw, support, supported


class BootstrapMaskSelector:
    """Stability selection over resampled mask fits, with support accounting.

    Parameters mirror :class:`~robustsignalmaker.masking.SoftMaskSelector`
    for the inner fits, plus the aggregation controls. ``target_coverage``
    picks the threshold whose retained fraction (of the SUPPORTED universe)
    is closest to the target, on the already-computed frequency map, for
    free; None uses the fixed ``tau``.
    """

    def __init__(self, task: str = "binary", segment: int = 16, lam: float = 0.03,
                 tau: float = 0.7, n_bootstrap: int = 20, subsample: str = "bootstrap",
                 target_coverage=None, gate: str = "hardconcrete", tv: float = 0.0,
                 l2: float = 1e-3, lr: float = 0.08, n_iter: int = 500,
                 warmup_frac: float = 0.3, mask_threshold: float = 0.5,
                 min_valid_frac: float = 0.5, rho_min: float = 0.1,
                 o_floor: float = 0.2, o_min: float = 0.1, a_min: float = 0.5,
                 n_min_hard: int = 8, n_min_warn: int = 30,
                 max_failed_frac: float = 0.2, n_jobs: int = 1, seed: int = 0,
                 representation=None, refit_model: str = "head"):
        from .models import ALGORITHMS

        if refit_model not in ALGORITHMS:
            raise ValueError(
                f"unknown refit_model {refit_model!r}; choose from "
                f"{', '.join(ALGORITHMS)}")
        if subsample not in ("bootstrap", "half", "complementary"):
            raise ValueError("subsample must be one of {'bootstrap','half','complementary'}")
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        self.refit_model = refit_model
        self.task = task
        self.segment = segment
        self.lam = lam
        self.tau = tau
        self.n_bootstrap = n_bootstrap
        self.subsample = subsample
        self.target_coverage = target_coverage
        self.gate = gate
        self.tv = tv
        self.l2 = l2
        self.lr = lr
        self.n_iter = n_iter
        self.warmup_frac = warmup_frac
        self.mask_threshold = mask_threshold
        self.min_valid_frac = min_valid_frac
        self.rho_min = rho_min
        self.o_floor = o_floor
        self.o_min = o_min
        self.a_min = a_min
        self.n_min_hard = n_min_hard
        self.n_min_warn = n_min_warn
        self.max_failed_frac = max_failed_frac
        self.n_jobs = n_jobs
        self.seed = seed
        # the lens the INNER fits look through; the final refit always uses
        # the plain SegmentMean (the deliverable does not inherit a lens)
        self.representation = representation

    def _inner_kwargs(self):
        return dict(task=self.task, segment=self.segment, lam=self.lam, l2=self.l2,
                    lr=self.lr, n_iter=self.n_iter, warmup_frac=self.warmup_frac,
                    mask_threshold=self.mask_threshold, gate=self.gate, tv=self.tv,
                    min_valid_frac=self.min_valid_frac, rho_min=self.rho_min,
                    o_floor=self.o_floor, n_min_hard=self.n_min_hard,
                    n_min_warn=self.n_min_warn)

    def _make_inner(self, seed, representation=None):
        return SoftMaskSelector(seed=int(seed), representation=representation,
                                **self._inner_kwargs())

    def _one_fit(self, X, y, V, idx, seed, representation=None):
        """(hit, assessable) for one resample, or an error STRING for a
        failed fit.

        A failed fit (a class vanished, nothing assessable, no scorable
        sample) is EXCLUDED from every denominator, never an all-zero vote.
        The error travels in the return value so parallel workers cannot
        lose it (2026-09-01 sweep: appending to self dropped every message
        under n_jobs > 1).
        """
        try:
            sel = self._make_inner(seed, representation).fit(X[idx], y[idx], V[idx])
        except Exception as exc:  # noqa: BLE001 - counted and surfaced, not hidden
            return f"{type(exc).__name__}: {exc}"
        S = sel.grid.n_segments
        hit = np.zeros(S)
        hit[sel.selected_segments_] = 1.0
        o_b = (sel._v_seg > 0).mean(axis=0)
        assessable = sel.assessable_segments_ & (o_b >= self.o_min)
        return hit, assessable

    def _resamples(self, n, strat, seeds):
        """(indices, seed) pairs, deterministic and independent of execution order."""
        jobs = []
        if self.subsample == "complementary":
            for p in range(max(1, self.n_bootstrap // 2)):
                rng = np.random.default_rng(seeds.bootstrap(p))
                a, b = complementary_pair(n, rng, strat)
                jobs.append((a, seeds.bootstrap(2 * p)))
                jobs.append((b, seeds.bootstrap(2 * p + 1)))
        else:
            for b in range(self.n_bootstrap):
                rng = np.random.default_rng(seeds.bootstrap(b))
                jobs.append((resample_indices(n, self.subsample, rng, strat),
                             seeds.bootstrap(b)))
        return jobs

    def _strata(self, y, v_seg, obs_groups):
        """Class x observation-group labels for stratified resampling."""
        parts = []
        if self.task != "regression":
            parts.append(np.asarray(y).astype(str))
        groups = obs_groups if obs_groups is not None else observation_groups(v_seg)
        self.observation_groups_ = None if groups is None else np.asarray(groups)
        if groups is not None:
            parts.append(np.asarray(groups).astype(str))
        if not parts:
            self.strata_note_ = "unstratified (regression, no observation groups)"
            return None
        strat = parts[0]
        for p in parts[1:]:
            strat = np.char.add(np.char.add(strat, "|"), p)
        _, counts = np.unique(strat, return_counts=True)
        if counts.min() < 2 and len(parts) > 1:
            # a class x group cell too small to resample: merge upward (drop
            # the group axis) and say so, rather than resampling a singleton
            self.strata_note_ = ("class-only stratification: a class x group "
                                 "cell had fewer than 2 samples")
            return parts[0]
        self.strata_note_ = "stratified by " + (
            "class x observation group" if len(parts) == 2
            else ("class" if self.task != "regression" else "observation group"))
        return strat

    def _choose_tau(self, pi_raw, supported) -> float:
        """Fixed tau, or the threshold nearest the target coverage, over the
        SUPPORTED universe only (unassessable segments are not candidates)."""
        if self.target_coverage is None:
            return self.tau
        vals = pi_raw[supported]
        vals = vals[np.isfinite(vals)]
        candidates = np.unique(vals[vals > 0])
        if candidates.size == 0:
            return self.tau
        cov = np.array([(vals >= t).mean() for t in candidates])
        return float(candidates[int(np.argmin(np.abs(cov - self.target_coverage)))])

    def fit(self, X, y, V=None, obs_groups=None):
        X, Vb = as_validity(X, V)
        y = np.asarray(y)
        Vf = Vb.astype(float)
        self.grid = SegmentGrid.from_signal_shape(X.shape, segment=self.segment)
        S = self.grid.n_segments
        n = X.shape[0]
        seeds = Seeds(base=self.seed, n_bootstrap=max(2 * self.n_bootstrap, 1))
        v_seg = self.grid.segment_validity(Vf)
        self.obs_frac_ = (v_seg > 0).mean(axis=0)
        strat = self._strata(y, v_seg, obs_groups)

        jobs = self._resamples(n, strat, seeds)
        if self.n_jobs == 1:
            raw = [self._one_fit(X, y, Vf, idx, s, self.representation)
                   for idx, s in jobs]
        else:
            from joblib import Parallel, delayed

            raw = Parallel(n_jobs=self.n_jobs)(
                delayed(self._one_fit)(X, y, Vf, idx, s, self.representation)
                for idx, s in jobs)
        self._aggregate(raw, len(jobs), S)
        self._refit(X, y, Vf)
        return self

    def _aggregate(self, raw, n_jobs_total, S):
        results = [r for r in raw if isinstance(r, tuple)]
        errors = [r for r in raw if isinstance(r, str)]
        self.n_fits_ = len(results)
        self.n_failed_fits_ = n_jobs_total - len(results)
        self.fit_errors_ = errors[:5]
        if not results:
            raise InsufficientEvidenceError(
                "every resample fit failed; last errors: "
                + "; ".join(self.fit_errors_[-3:]))
        if self.n_failed_fits_ > self.max_failed_frac * n_jobs_total:
            warnings.warn(
                f"{self.n_failed_fits_}/{n_jobs_total} resample fits failed; "
                f"stability frequencies rest on thinner evidence than requested "
                f"(first errors: {'; '.join(self.fit_errors_[:2])})",
                stacklevel=2)

        pi, pi_raw, support, supported = _frequency_map(results, S, self.a_min)
        self.support_ = support
        self._supported = supported
        self.pi_ = pi
        self._pi_raw = pi_raw
        self.resample_sets_ = [np.flatnonzero(h * a) for h, a in results]

        self.tau_ = self._choose_tau(pi_raw, self._supported)
        sel_mask = self._supported & (np.nan_to_num(pi_raw, nan=-1.0) >= self.tau_)
        self.selected_segments_ = np.flatnonzero(sel_mask).astype(int)
        self.rejected_segments_ = np.flatnonzero(
            self._supported & ~sel_mask).astype(int)
        self.unassessable_segments_ = np.flatnonzero(~self._supported).astype(int)
        self.insufficient_evidence_ = np.flatnonzero(
            ~self._supported & (np.nan_to_num(pi_raw, nan=-1.0) >= self.tau_)
        ).astype(int)

        verdicts = np.full(S, "rejected", dtype=object)
        verdicts[~self._supported] = "unassessable"
        verdicts[sel_mask] = "selected"
        self.verdicts_ = verdicts

    def _refit(self, X, y, Vf):
        """Final predictor on the stable region (plain SegmentMean lens; the
        aggregation decided WHICH region, the deliverable does not inherit a
        lens's feature space)."""
        S = self.grid.n_segments
        if self.selected_segments_.size == 0:
            finite = np.isfinite(self.pi_) & (self.pi_ > 0)
            if not finite.any():
                raise InsufficientEvidenceError(
                    "aggregation selected nothing and no supported segment was "
                    "ever selected; there is no region to fall back to")
            self.selected_segments_ = np.array([int(np.nanargmax(self.pi_))])
            self.degenerate_ = True
        else:
            self.degenerate_ = False
        binary = np.zeros(S)
        binary[self.selected_segments_] = 1.0
        self.mask_ = binary
        self.model_ = SoftMaskSelector(seed=int(self.seed), fixed_mask=binary,
                                       **self._inner_kwargs()).fit(X, y, Vf)
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        self.refit_note_ = None
        self.external_model_ = None
        if self.refit_model != "head":
            self._fit_external(X, y, Vf)

    def _fit_external(self, X, y, Vf):
        """Refit an sklearn-style model on the selected region.

        Selection is unchanged: the gated head chose the bands, and this
        only swaps what predicts on them. Unobserved entries reach the model
        as exactly 0 in standardised space (never a fabricated value) and
        samples below ``rho_min`` are dropped from the fit and refused at
        predict time, so the no-fabrication contract survives a model that
        knows nothing about validity.
        """
        from .models import ExternalRefit

        ext = ExternalRefit(self.refit_model, self.task, self.grid,
                            self.selected_segments_, seed=int(self.seed),
                            rho_min=self.rho_min, n_min_hard=self.n_min_hard,
                            n_min_warn=self.n_min_warn).fit(X, Vf, y)
        self.external_model_ = ext
        self.refit_stats_ = ext.stats_
        self.refit_note_ = ext.note_
        self.n_refit_excluded_ = ext.n_excluded_
        self.classes_ = ext.classes_

    # -- prediction --------------------------------------------------------- #
    def predict(self, X, V=None):
        if self.external_model_ is None:
            return self.model_.predict(X, V)
        return self.external_model_.predict(X, V)

    def predict_proba(self, X, V=None):
        if self.external_model_ is None:
            return self.model_.predict_proba(X, V)
        return self.external_model_.predict_proba(X, V)

    # -- selection outputs -------------------------------------------------- #
    def pi_points(self) -> np.ndarray:
        """Selection frequency expanded to the point axis (nan = unassessable)."""
        return self.grid.expand(self.pi_)

    def selected_points(self) -> np.ndarray:
        return self.grid.segments_to_points(self.selected_segments_)

    def assessable_universe(self) -> int:
        """Number of supported segments: the universe for coverage and for
        chance-corrected Jaccard (never the raw segment count)."""
        return int(self._supported.sum())

    def coverage(self) -> float:
        u = self.assessable_universe()
        return len(self.selected_segments_) / u if u else float("nan")

    def ambiguous_fraction(self, lo: float = 0.2, hi: float = 0.8) -> float:
        """Fraction of SUPPORTED segments with pi in the undecided band: the
        go/no-go diagnostic for whether a reproducible region exists at all
        (bimodal pi is trustworthy; diffuse pi means no threshold can help).
        nan when nothing is supported."""
        if not self._supported.any():
            return float("nan")
        vals = self.pi_[self._supported]
        return float(np.mean((vals >= lo) & (vals < hi)))

    def _report_extras(self) -> dict:
        return {}

    def stability_report(self) -> dict:
        """The frequency map summary; support travels with every figure."""
        sup = self.pi_[self._supported]
        return {
            "n_fits": self.n_fits_,
            "n_failed_fits": self.n_failed_fits_,
            "tau": self.tau_,
            "coverage": self.coverage(),
            "assessable_universe": self.assessable_universe(),
            "n_segments": self.grid.n_segments,
            "unassessable_fraction": float(np.mean(~self._supported)),
            "insufficient_evidence": [int(s) for s in self.insufficient_evidence_],
            "pi_mean": float(sup.mean()) if sup.size else float("nan"),
            "pi_max": float(sup.max()) if sup.size else float("nan"),
            "frac_confident_in": float(np.mean(sup >= 0.8)) if sup.size else float("nan"),
            "frac_confident_out": float(np.mean(sup < 0.2)) if sup.size else float("nan"),
            "ambiguous_fraction": self.ambiguous_fraction(),
            "mean_obs_frac": float(self.obs_frac_.mean()),
            "strata": self.strata_note_,
            "degenerate": self.degenerate_,
            **self._report_extras(),
        }


class RepresentationEnsembleSelector(BootstrapMaskSelector):
    """Double marginalisation: over data resamples AND representations.

    The RPM central claim, ported with support accounting. The selection
    frequency aggregates over the product of resamples and lenses,

        pi_k = hits / assessable fits, over all (lens, resample) fits,

    so a segment survives only if it is chosen regardless of which lens is
    looking; the representation is a nuisance variable, integrated out rather
    than fixed and defended.

    The per-lens frequency maps are kept, because their disagreement is the
    quantity the claim rests on: ``representation_agreement()`` low means a
    single-lens result would have been lens-specific and the marginalised one
    is the defensible finding; high (RPM used > 0.8 as the line) means one
    lens would have sufficed, and the honest report says so. The final
    predictor uses the plain SegmentMean lens: the ensemble decided WHICH
    region to keep, and the deliverable does not inherit one member's
    feature space.
    """

    def __init__(self, representations=None, n_representations: int = 4, **kwargs):
        if kwargs.pop("representation", None) is not None:
            raise ValueError(
                "pass `representations` (plural); the ensemble supplies the lenses")
        super().__init__(**kwargs)
        if representations is None:
            from .representations import default_ensemble

            representations = default_ensemble(n_representations, seed=self.seed)
        self.representations = list(representations)
        if not self.representations:
            raise ValueError("need at least one representation")

    def fit(self, X, y, V=None, obs_groups=None):
        X, Vb = as_validity(X, V)
        y = np.asarray(y)
        Vf = Vb.astype(float)
        self.grid = SegmentGrid.from_signal_shape(X.shape, segment=self.segment)
        S = self.grid.n_segments
        n = X.shape[0]
        # bootstrap-stream capacity covers the resample draws (indices up to
        # 2B) plus one fit seed per (lens, resample); the Seeds constructor
        # asserts loudly if this ever outgrows the stream spacing
        n_jobs_est = max(2 * self.n_bootstrap, 1)
        seeds = Seeds(base=self.seed,
                      n_bootstrap=(len(self.representations) + 1) * n_jobs_est,
                      n_representations=len(self.representations))
        v_seg = self.grid.segment_validity(Vf)
        self.obs_frac_ = (v_seg > 0).mean(axis=0)
        strat = self._strata(y, v_seg, obs_groups)

        jobs = self._resamples(n, strat, seeds)
        # one distinct fit seed per (lens, resample), inside the bootstrap
        # stream and clear of the resample-draw indices (the additive
        # s + representation(r) scheme collided whenever b + r matched, and
        # landed in the inner-splitter stream; 2026-09-01 sweep)
        tasks = [(r_idx, rep, idx, seeds.bootstrap((r_idx + 1) * len(jobs) + j))
                 for r_idx, rep in enumerate(self.representations)
                 for j, (idx, s) in enumerate(jobs)]
        if self.n_jobs == 1:
            raw = [self._one_fit(X, y, Vf, idx, s, rep) for _, rep, idx, s in tasks]
        else:
            from joblib import Parallel, delayed

            raw = Parallel(n_jobs=self.n_jobs)(
                delayed(self._one_fit)(X, y, Vf, idx, s, rep)
                for _, rep, idx, s in tasks)

        self._aggregate(raw, len(tasks), S)

        # per-lens frequency maps and regions, with the same support rules
        # and thresholded at the SAME operating point as the reported answer
        # (2026-09-01 sweep: the fixed tau here diverged from tau_ whenever
        # target_coverage chose a different threshold)
        self.pi_by_representation_ = {}
        self._per_rep_sets = []
        for r_idx, rep in enumerate(self.representations):
            rows = [r for (ri, _, _, _), r in zip(tasks, raw)
                    if ri == r_idx and isinstance(r, tuple)]
            name = getattr(rep, "name", f"rep{r_idx}")
            if rows:
                pi_r, raw_r, _, supported_r = _frequency_map(rows, S, self.a_min)
                self.pi_by_representation_[name] = pi_r
                self._per_rep_sets.append(np.flatnonzero(
                    supported_r & (np.nan_to_num(raw_r, nan=-1.0) >= self.tau_)))
            else:
                self.pi_by_representation_[name] = np.full(S, np.nan)
                self._per_rep_sets.append(np.array([], dtype=int))

        self._refit(X, y, Vf)
        return self

    def representation_agreement(self) -> float:
        """Mean pairwise Jaccard between the regions chosen by each lens alone.

        Raw Jaccard, deliberately (comparable to the RPM benchmark numbers,
        0.36 to 0.45 on real images); nan when any lens selected nothing.
        """
        from .metrics import mean_pairwise_jaccard

        return mean_pairwise_jaccard(self._per_rep_sets)

    def _report_extras(self) -> dict:
        return {
            "n_representations": len(self.representations),
            "representation_agreement": self.representation_agreement(),
        }
