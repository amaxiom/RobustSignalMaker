# RobustSignalMaker User Guide

How to use RSM on your own signals. The companion guides:
API_REFERENCE.md (every public symbol, with signatures),
INTERPRETATION_GUIDE.md (how to read what it gives you) and
IMPLEMENTATION_GUIDE.md (how it works inside).

Every code block in this guide has been executed; they run in sequence from
section 3.

## Contents

1. Install and environment
2. Data model: X, V, y
3. Quick start
4. The one-call pipeline: NestedCV
5. Reading the result
6. The selectors, from single fit to full pipeline
7. Missing data: what RSM does and never does
8. Missing labels
9. Grouped data (instruments, sessions, subjects)
10. Choosing segment size and sparsity strength
11. The lens ensemble
11b. Choosing the predictor: the refit model zoo
12. Saving and loading
13. Parameter reference

## 1. Install and environment

```
py -m pip install -e .
py -m pytest tests -q
```

Dependencies: numpy, scipy, scikit-learn (Python 3.9 or newer). On this
project's reference machine the scientific stack lives in the `py`
interpreter; run everything with `py`, never bare `python`.

## 2. Data model: X, V, y

- `X`: float array shaped `(n_samples, n_channels, n_points)` on a SHARED
  sampling grid (same wavelengths / m/z bins / time stamps for every
  sample). Single-channel data is `n_channels = 1`.
- `V`: bool validity mask, same shape, True where a value was actually
  measured. If you omit it, `V = ~isnan(X)`. An inf, or a NaN where V says
  observed, is rejected loudly.
- `y`: one target per sample. Regression, binary, or multiclass; the task is
  inferred (or pass `task=` explicitly to the selectors). `y` may contain
  NaN for unlabelled samples (see section 8).

What counts as missing is YOUR modelling decision, made explicitly:
detector dropout and gaps are NaN; saturated/clipped values should be marked
missing (`mask_saturation_censor` does both steps); mass-spec zeros are
missing only if you say so (`zeros_are_missing` in your loader). RSM never
guesses.

## 3. Quick start

Every code block in this guide runs, and they run in sequence: this one
makes the data the later sections use. Substitute your own `X` and `y` and
the rest follows unchanged.

```python
import numpy as np
from robustsignalmaker import BootstrapMaskSelector, make_signal_control

# a control with a known informative band, so the answer is checkable.
# Replace these two lines with your own (n, C, T) array and targets.
c = make_signal_control(n=200, n_points=256, task="binary", seed=0)
X, y = c.X.copy(), c.y
X[:80, 0, 180:210] = np.nan        # a detector dropout on 80 samples
X_new = X[:20]                     # stand-in for unseen spectra

sel = BootstrapMaskSelector(task="binary", segment=16, lam=0.05,
                            n_bootstrap=50, seed=0)
sel.fit(X, y)                      # V defaults to ~isnan(X)

print(sel.selected_segments_)      # the retained bands (segment ids)
print(sel.selected_points()[:8])   # the same, as point indices on your axis
print(sel.stability_report())      # pi, support, coverage, diagnostics
proba = sel.predict_proba(X_new)   # nan rows = "not enough evidence to say"

truth = sorted(set(np.flatnonzero(c.informative) // 16))
print("planted band was:", truth)  # compare: the control's known answer
```

## 4. The one-call pipeline: NestedCV

For an honest generalisation estimate and the headline verdict, wrap the
selector in nested cross-validation:

```python
from robustsignalmaker import NestedCV

eng = NestedCV(k_outer=5, segment=16, lam=0.05, n_bootstrap=50,
               random_state=0)
result = eng.run(X, y)             # V and groups optional
print(result.summary())
result.save("my_rsm_results")
```

Everything (standardisation, selection, thresholds, the MNAR guard) is
fitted per training fold and applied frozen; the built-in baseline is the
same head with every segment retained, and `result.verdict.outcome` says
whether the selected region `preserved` the full signal's performance (the
success criterion), did `sig.better`, or `sig.worse`.

## 5. Reading the result

See INTERPRETATION_GUIDE.md for the full treatment. The essentials:

- `result.summary()`: score, verdict, stability, refusals, MNAR flag.
- Per segment, the selector's verdict is THREE-VALUED: selected, rejected,
  or unassessable. Unassessable (`pi` is nan) means the data could not
  support a judgement; it is never evidence of absence.
- Every stability figure travels with coverage and observation support.
  Read them together, always.

## 6. The selectors, from single fit to full pipeline

- `SoftMaskSelector`: one learned gate vector (Hard-Concrete L0 by default),
  one fit. Fast, but a single fit's selection moves under resampling.
- `BootstrapMaskSelector`: the stability layer; aggregates the mask over
  resamples into selection frequencies `pi_` with support accounting. This
  is the recommended default.
- `RepresentationEnsembleSelector`: additionally marginalises over a lens
  ensemble (section 11).
- `NestedCV` + fold estimators: the leakage-safe wrapper producing
  `RSMResult` with the verdict.

## 7. Missing data: what RSM does and never does

RSM computes every quantity over the evidence that exists, renormalised by
how much evidence that was. It never fabricates a value: no imputation, no
zero-fill, no interpolation, anywhere in the method.

- A segment a sample did not observe contributes nothing for that sample.
- A sample with too little observed evidence overall (below `rho_min`) gets
  a nan prediction: a refusal, counted and reported, never a guess.
- A segment observed by too few samples or resamples is UNASSESSABLE.
- If the missingness pattern alone predicts y, every training fold's MNAR
  guard warns (`MissingnessInformativeWarning`) and stamps the result:
  selection on observed data is then confounded with the observation
  process. RSM warns; it does not pretend to fix MNAR.

When is plain imputation fine? When you know the missingness is ignorable
(independent of y and of the signal's informative structure), mean
imputation is a legitimate variance reduction and can be MORE stable
(benchmarks/FINDINGS.md sec.2). The danger zones, where you want RSM, are
never-observed regions and non-ignorable missingness, where imputation
fabricates findings or destroys the warning you needed.

## 8. Missing labels

`y` may contain NaN. Labelled rows alone enter splits, fits, resampling and
scoring; unlabelled rows are counted in `result.n_unlabelled`. Opt-in
(`use_unlabelled_stats=True` on NestedCV), unlabelled TRAINING rows are
offered to the fold estimator for X-only statistics.

## 9. Grouped data (instruments, sessions, subjects)

Pass `groups=` to `NestedCV.run` and no group will span a train/test split
(StratifiedGroupKFold / GroupKFold). Pass `obs_groups=` to the selectors to
stratify resampling by instrument; without it, the only automatic fallback
is exact validity-pattern grouping (`observation_groups`), never clustering.

## 10. Choosing segment size and sparsity strength

- `segment`: the selection unit, in points. Choose it near the width of the
  narrowest physically meaningful band; `segment=1` is point-wise mode and
  rarely stabilises on smooth spectra (the report will show it).
- `lam`: fixed sparsity strength; or set `target_coverage` to pick the
  stability threshold that retains a target fraction; or use
  `lam_frontier` / `lam_for_coverage` (tuning module) to choose per dataset
  with the trade-offs visible. A weak `lam` makes the fixed threshold
  fragile even when the frequency map's ordering is robust (the M4
  note).

## 11. The lens ensemble

`default_ensemble(n)` supplies diverse lenses (segment means, Savitzky-Golay
derivative energy, multi-scale Gaussian, random convolutions, segment
stats). `RepresentationEnsembleSelector` marginalises selection over them
and reports `representation_agreement()`. Expectation management, from the
benchmarks: marginalisation is INSURANCE against picking the wrong lens
(near-best whichever regime you are in), not a guaranteed stability boost;
when lenses agree strongly, a single lens would have sufficed and the report
will show that.

## 11b. Choosing the predictor: the refit model zoo

The gated head does the selecting, because the gates are trained through it.
What PREDICTS on the chosen bands is your choice, and the vocabulary matches
RobustModelMaker: `lin`, `rdg`, `las`, `eln`, `log`, `svm`, `rf`, `xgb`,
`mlp`, plus `pls` (PLS regression, PLS-DA for classification) which
chemometrics will want first.

```python
sel = BootstrapMaskSelector(task="binary", segment=16, lam=0.05,
                            n_bootstrap=50, seed=0,
                            refit_model="pls").fit(X, y)
print(sel.selected_segments_)      # identical to the default: selection is unchanged
print(sel.predict_proba(X_new)[:3])
```

Two things to hold on to.

**Selection does not depend on the choice.** `pi_`, `support_`, `verdicts_`
and the selected set are identical whichever model you pick. You are choosing
a predictor, not a selector.

**The refusal rule still applies.** A random forest cannot see a validity
mask and would happily predict from imputed values. RSM does not impute:
the selected region is pooled and standardised so unobserved entries are
exactly 0 in standardised space, and any sample below `rho_min` is refused
and comes back NaN, exactly as the built-in head refuses it.

Under nested CV, give the same model to the baseline, or the verdict compares
a forest against the built-in head and confounds the region with the model:

```python
from robustsignalmaker import NestedCV
from robustsignalmaker.nested_cv import (BootstrapMaskFoldEstimator,
                                         FullSignalFoldEstimator)

cv = NestedCV(
    estimator_factory=lambda: BootstrapMaskFoldEstimator(
        segment=16, lam=0.05, n_bootstrap=20, refit_model="rf"),
    baseline_factory=lambda: FullSignalFoldEstimator(segment=16, refit_model="rf"),
    k_outer=5, segment=16, random_state=0)
print(cv.run(X, y).summary()["verdict"])
```

## 12. Saving and loading

```python
from robustsignalmaker import RSMResult

result.save("out_dir", prefix="mystudy")   # CSV per table + JSON + pickle
loaded = RSMResult.load("out_dir/mystudy_result.pkl")
print(loaded.summary()["verdict"])
```

`save` writes one CSV per table from `results_tables()`, a JSON summary, and
a pickle of the whole result including the refit estimator. The CSVs are for
reading and for a paper; the pickle is what `load` reads back.

## 13. Parameter reference (defaults)

Selection: `segment=16`, `lam=0.03`, `tau=0.7`, `n_bootstrap=20`,
`subsample="bootstrap"` (or `"half"`, `"complementary"`),
`gate="hardconcrete"` (or `"sigmoid"`), `tv=0.0` (raise for contiguous
bands), `target_coverage=None`.

Evidence thresholds: `min_valid_frac=0.5`, `rho_min=0.1`, `o_min=0.1`,
`a_min=0.5`, `o_floor=0.2`, `n_min_hard=8`, `n_min_warn=30`. All five were
sensitivity-swept against the quantity each is actually compared with, with
probes to prove the sweep could fail (FINDINGS sec.10): `o_floor` sits on a
genuine plateau; `rho_min` and `o_min` are live and the defaults are
deliberately loose, since tightening them toward the data's own evidence
level discards real signal (`o_min` at 0.9 dropped truth segments and
halved recovery); `min_valid_frac` did not move the selection anywhere in
0.3 to 0.999 on a control whose validity low tail it straddles, but it IS
the one to set per dataset when the evidence scale differs, as sparse mass
spectra do; and `a_min` binds only when your missingness patterns are too
varied for `observation_groups` to group them (with groups detected,
resample support is deterministic and a_min cannot act).

Engine: `l2=1e-3`, `lr=0.08`, `n_iter=800` (single fit) / `500` (per
resample), `warmup_frac=0.3`, `mask_threshold=0.5`.

CV: `k_outer=5`, `l_inner=5`, `repeated_outer_cv=1`, `baseline=True`,
`mnar_guard=True`, `mnar_permutations=100`, `random_state=0`.
