# RobustSignalMaker Implementation Guide

How the pipeline works inside, for readers extending it or checking it. The
module docstrings carry the derivations; API_REFERENCE.md lists every public
symbol with its signature; RELATED_WORK.md defends the design against prior
art; this guide is the map.

## Contents

1. Pipeline overview
2. The validity algebra (validity.py)
3. Selection units (segments.py)
4. Lenses (representations.py)
5. The gated engine (masking.py)
6. Stability aggregation (selection.py)
7. The nested-CV engine (nested_cv.py)
8. Reproducibility contract
9. Guards and failure semantics
10. Speed and scaling
11. Extending RSM

## 1. Pipeline overview

```
X (n, C, T) + V ──lens──> F, FV (n, S, K)     renormalised conv + validity-weighted pooling
                 eligibility: FV >= min_valid_frac
                 masked standardisation      scale-safe, TRAIN statistics only
                 two-phase soft-mask fit     head on all segments -> freeze -> prune gates
                 B resamples ──> pi, support, three-valued verdicts
                 refit head on the selected region (SegmentMean lens)
   all wrapped in nested CV: split first, everything above per training fold
```

## 2. The validity algebra (validity.py)

One module holds every masked-computation primitive, because the learned
gate and the data validity mask share the same algebra (both mean "this
evidence is absent") and compose multiplicatively.

- Masked moments and standardisation factor out the max-abs of the observed
  values before any sum or square, so 1e-200 and 1e300 inputs behave
  identically; raw-unit moments are never formed. Effective-sample-size
  floors decide assessability.
- `renormalised_convolve1d` is the 1D partial convolution with the
  level-vs-contrast correction (a zero-sum filter is re-centred over the
  surviving support, otherwise a flat signal fabricates an edge at a gap
  boundary). The reciprocal is exact where evidence exists; no epsilon ever
  meets a data magnitude. Output validity is fractional and flows into
  pooling as a weight.
- `renormalised_dot` is the head's analogue: the gated sum rescaled by
  M / D_i (gated mass over observed gated mass), a ratio-form
  Horvitz-Thompson estimate. Validity enters the head ONCE, inside the
  channel sum.
- `missingness_association` is the MNAR guard: a small CV'd linear model
  from the validity pattern to y, calibrated by permutation, three-valued
  verdict.

## 3. Selection units (segments.py)

`SegmentGrid` partitions the axis into fixed windows; `segment=1` is exact
point-wise mode through the same code path. Pooling normalises at pool time
(the denominator is per-sample observed evidence, so no precomputed
normalised matrix as in RPM) and returns validity fractions beside values.
The selection layers touch only `n_segments`, `pool`, `expand`, so a future
peak-aware grid slots in without touching them.

## 4. Lenses (representations.py)

Contract: `transform(X, V, grid) -> (F, FV)`, both `(n, S, K)`. Invariant
(from RPM, verbatim): a lens changes which segments rank highly, never the
coordinate system of the answer. RSM adds: every lens consumes and emits
validity, reduces exactly to its unmasked form on complete data, and never
reads a masked payload (the shared `_canonical` entry zeroes payloads before
any arithmetic; this makes the bitwise payload-invariance test hold for
every lens rather than the careful ones).

## 5. The gated engine (masking.py)

Two-phase VTF fit (RPM): fit the head on all segments, FREEZE it, then train
only the gates. Frozen, because jointly the mask and head are
scale-degenerate and the sparsity penalty collapses the mask without
changing predictions.

The three RSM-specific mechanisms, each with its reason:

- Renormalised head with STOP-GRADIENT renormaliser: a never-observed
  segment receives exactly zero gate gradient (through the true derivative
  it would feel pressure via M while changing no prediction).
- Observation-scaled penalty `lam_s = lam * max(obs_rate_s, o_floor)`: the
  data gradient scales with observation rate and the penalty does not, so
  without this, rarely observed drifts to closed ("unobserved" masquerading
  as "unimportant"). Scaling the penalty adds no gradient variance;
  amplifying the data gradient would.
- Losses are exact (stable softplus / log-sum-exp, no probability epsilon),
  so the analytic gradient is the exact derivative of the reported loss and
  the numeric gradient checks in test_gates.py pass at 1e-5.

Gates: Hard-Concrete L0 (default; penalty on the expected open count) or
sigmoid L1. TV-1D couples adjacent gates, but only pairs where both
segments are assessable.

## 6. Stability aggregation (selection.py)

Per resample, a segment is ASSESSABLE iff observed by at least `o_min` of
the resample and standardisable under the ESS floors. `pi_k` counts hits
over assessable resamples only; below `a_min` resample support the verdict
is "unassessable" and pi is nan. Failed resample fits are excluded from
every denominator, counted, and warned about above `max_failed_frac` (RPM
counted them as zero votes, a bias toward 0 deliberately not ported).
Resampling stratifies by class x observation group; the only automatic
grouping is exact validity patterns. All-closed aggregation raises rather
than falling back to an arbitrary argmax (deliberate divergence from RPM).

The ensemble selector aggregates over lenses x resamples with the same
rules, keeps per-lens maps, and reports lens agreement.

## 7. The nested-CV engine (nested_cv.py)

The leakage contract is structural: the engine slices indices and hands
ONLY the training partition to `fit`; the test partition is touched only by
`predict`. It is proved by canary tests (hashed training rows; predict
raises on a seen row), not asserted. Missing y: labelled rows only, split
and score; refusal-aware scoring counts nan predictions instead of imputing
them; the MNAR guard stamps every fold; the built-in baseline is the same
head with every segment retained, feeding the paired verdict.

## 8. Reproducibility contract

One `random_state`; `Seeds` derives every stream with documented additive
offsets (bootstrap +10000, representation +20000, inner +30000, repeat
stride 100000) and asserts non-collision at construction. All derivations
are pure functions of (base, index, repeat): results are independent of
execution order and n_jobs. Same seed, bitwise-identical masks and
frequencies (tested).

## 9. Guards and failure semantics

A guard never substitutes a value. The ladder: a statistic without evidence
returns a flagged placeholder (`valid=False`); a feature that cannot be
standardised is removed from play and flagged; a sample below `rho_min` gets
nan predictions, counted; a segment without support gets nan pi and the
"unassessable" verdict; a fit with nothing assessable raises
`InsufficientEvidenceError`; an aggregation that selected nothing anywhere
raises. Masked payloads (0, -1, 1e300, NaN) never change any output bit.

## 10. Speed and scaling

Cost model: `fit_count` in config.py. The heavy loop is B resample fits per
selection, each an Adam loop over (S x K) parameters; n_jobs parallelises
resamples via joblib. Tests use FAST profiles (small B, n_iter around 150);
production defaults are in the parameter reference. Wall-clock figures on
shared machines are untrustworthy; confirm the machine is quiet before
quoting seconds.

## 11. Extending RSM

- New lens: implement `transform(X, V, grid) -> (F, FV)` obeying section 4;
  add to `default_ensemble` if it earns a place; the payload-invariance and
  reduction tests in test_representations.py are the acceptance bar.
- New selection unit: mirror `SegmentGrid`'s interface.
- New fold estimator: satisfy the `FoldEstimator` protocol (fit sees train
  only, may refuse via nan predictions, exposes `selected_`).
- New benchmark dataset or competitor: register a loader returning
  `(X, V, y, meta)` in benchmarks/datasets.py, or a selector
  `select(X, V, y, grid, task, seed, k, **cfg)` in
  benchmarks/competitors.py. Names are explicit, including the fill
  strategy; tables derive from the registries.
