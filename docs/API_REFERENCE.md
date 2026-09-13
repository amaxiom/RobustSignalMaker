# RobustSignalMaker API Reference

Every name exported from `robustsignalmaker`, grouped by the module it lives
in. Signatures are taken from the code by introspection, not transcribed, so
they match the installed version.

Read this alongside the other three guides: USER_GUIDE.md for how to run a
study, INTERPRETATION_GUIDE.md for what the outputs mean, and
IMPLEMENTATION_GUIDE.md for why the internals are built as they are.

**What is public.** Everything listed here is in `robustsignalmaker.__all__`
and is intended for use. Anything not listed, and anything whose name begins
with an underscore, is internal and may change without notice.

**Naming conventions.** Trailing-underscore attributes (`pi_`, `support_`)
exist only after `fit`. `X` is always `(n_samples, n_channels, n_points)`,
`V` is a validity array of the same shape with True or a weight in [0, 1]
meaning observed, and `y` may contain NaN for unlabelled samples.

## Contents

1. [Selectors](#1-selectors) the things you call
2. [The pipeline: NestedCV and RSMResult](#2-the-pipeline-nestedcv-and-rsmresult)
3. [Fitted attributes](#3-fitted-attributes)
4. [Lenses (representations)](#4-lenses-representations)
5. [Fold estimators and splitters](#5-fold-estimators-and-splitters)
6. [Tuning](#6-tuning)
7. [Validity primitives](#7-validity-primitives)
8. [Segment grid](#8-segment-grid)
9. [Metrics](#9-metrics)
10. [Synthetic data and missingness generators](#10-synthetic-data-and-missingness-generators)
11. [Reproducibility](#11-reproducibility)
12. [Configuration](#12-configuration)
13. [Errors and warnings](#13-errors-and-warnings)
14. [Refit model zoo](#14-refit-model-zoo)

## 1. Selectors

Three selectors, in increasing order of what they average over. For most
work use `BootstrapMaskSelector`, or wrap it in `NestedCV` (section 2) to
learn what the selection is worth.

### `BootstrapMaskSelector`

```
BootstrapMaskSelector(task="binary", segment=16, lam=0.03, tau=0.7,
                      n_bootstrap=20, subsample="bootstrap",
                      target_coverage=None, gate="hardconcrete", tv=0.0,
                      l2=1e-3, lr=0.08, n_iter=500, warmup_frac=0.3,
                      mask_threshold=0.5, min_valid_frac=0.5, rho_min=0.1,
                      o_floor=0.2, o_min=0.1, a_min=0.5, n_min_hard=8,
                      n_min_warn=30, max_failed_frac=0.2, n_jobs=1, seed=0,
                      representation=None)
```

Stability selection over resampled mask fits, with support accounting. This
is the main entry point.

| parameter | meaning |
|---|---|
| `task` | `"binary"`, `"multiclass"` or `"regression"` |
| `segment` | window width in points; `segment=1` gives point-wise selection |
| `lam` | sparsity strength; see `lam_frontier` rather than guessing |
| `tau` | selection threshold on `pi_`; 0.7 is the family convention |
| `n_bootstrap` | number of resample fits |
| `subsample` | `"bootstrap"`, `"half"`, or `"complementary"` for Shah-Samworth pairs |
| `target_coverage` | if set, `tau` is chosen to hit this coverage instead of being fixed |
| `gate` | `"hardconcrete"` (default) or `"sigmoid"` |
| `tv` | total-variation strength on adjacent gates, encouraging contiguous bands |
| `min_valid_frac` | a segment counts as observed for a sample at this validity fraction |
| `rho_min` | per-sample floor on observed gated evidence; below it a prediction is refused |
| `o_min` | per-resample floor on segment observation for the segment to be assessable |
| `a_min` | fraction of resamples that must assess a segment before `pi_` is trusted |
| `o_floor` | floor on the observation-scaled sparsity penalty |
| `n_min_hard` / `n_min_warn` | effective-sample-size floors: unassessable, and flagged low support |
| `max_failed_frac` | fraction of resample fits allowed to fail before the whole fit raises |
| `representation` | a lens (section 4); `None` means `SegmentMean` |
| `refit_model` | what predicts on the selected region (section 14); `"head"` is the built-in gated head |

Methods:

- `.fit(X, y, V=None, obs_groups=None)` fit; `obs_groups` stratifies
  resampling by observation pattern as well as class.
- `.predict(X, V=None)`, `.predict_proba(X, V=None)` predictions from the
  refit-on-selected-region model, with NaN where evidence is too thin.
- `.selected_points()` the selection as point indices on your axis.
- `.pi_points()` `pi_` expanded to the point axis, NaN where unassessable.
- `.coverage()` fraction of the assessable universe retained.
- `.assessable_universe()` number of supported segments, the denominator for
  coverage and for chance-corrected Jaccard.
- `.ambiguous_fraction(lo=0.2, hi=0.8)` fraction of supported segments whose
  `pi_` sits in the undecided band. The go/no-go diagnostic.
- `.stability_report()` one dict of run-level diagnostics.

### `RepresentationEnsembleSelector`

```
RepresentationEnsembleSelector(representations=None, n_representations=4,
                               **BootstrapMaskSelector_kwargs)
```

Double marginalisation: over data resamples AND representations, so the
choice of lens is treated as a nuisance parameter rather than a decision you
have to defend. Same interface as `BootstrapMaskSelector`, plus:

- `.representation_agreement()` mean pairwise Jaccard between the regions
  each lens picks alone. Low agreement means the answer you would have
  reported depends on a preprocessing choice.

Pass `representations=` (plural) explicitly, or `n_representations=` to take
that many from `default_ensemble`.

### `SoftMaskSelector`

```
SoftMaskSelector(task="binary", segment=16, lam=0.03, l2=1e-3, lr=0.08,
                 n_iter=800, warmup_frac=0.3, mask_threshold=0.5,
                 gate="hardconcrete", tv=0.0, seed=0, fixed_mask=None,
                 representation=None, min_valid_frac=0.5, rho_min=0.1,
                 o_floor=0.2, n_min_hard=8, n_min_warn=30)
```

A single gated fit: the engine the other two resample. Useful for
diagnostics and for a fixed mask, but it produces no selection frequency, so
it is not stability selection on its own.

- `.fit(X, y, V=None)`, `.predict`, `.predict_proba`
- `.decision_function(X, V=None)` raw decisions, NaN rows below `rho_min`.
- `.prediction_valid(X, V=None)` which samples have enough evidence.
- `.selected_points()`, `.coverage()`
- `fixed_mask=` skips mask learning and fits the head against the mask you
  supply, which is how the full-signal baseline is built.

## 2. The pipeline: NestedCV and RSMResult

### `NestedCV`

```
NestedCV(estimator_factory=None, *, baseline=True, baseline_factory=None,
         k_outer=5, l_inner=5, repeated_outer_cv=1, segment=16,
         use_unlabelled_stats=False, mnar_guard=True, mnar_permutations=100,
         random_state=0, **default_estimator_kwargs)
```

Leakage-safe nested cross-validation. Every fitted statistic, including the
selection itself, lives inside its training fold. Extra keyword arguments
are forwarded to the default fold estimator, so
`NestedCV(segment=16, lam=0.05, n_bootstrap=30)` works.

- `.run(X, y, V=None, groups=None) -> RSMResult`. Pass `groups` when samples
  are not independent (repeated measurements of one specimen, several
  instruments per sample) so a group is never split across folds.
- `mnar_guard=True` runs the missingness-informativeness test per training
  fold and stamps the result.
- `use_unlabelled_stats=True` lets unlabelled TRAINING rows contribute to
  X-only statistics. A test-fold row never contributes to any statistic.

### `RSMResult`

A frozen dataclass returned by `NestedCV.run`. Fields include
`per_fold_scores`, `baseline_per_fold_scores`, `verdict`,
`oof_predictions`, `oof_count`, `n_refused`, `n_unlabelled`,
`selected_per_fold`, `selection_stability`,
`selection_stability_adjusted`, `mnar_reports`, `fold_records`,
`final_estimator`, `classes`, `config`.

- `.summary() -> dict` the headline numbers in one dict, with regression
  scores displayed as positive RMSE.
- `.display_score()` `(value, sd, name)`, regression sign flipped to RMSE.
- `.results_tables() -> dict` named tables as lists of dict rows.
- `.save(directory, prefix="rsm") -> Path` one CSV per table, plus a JSON
  summary and a pickle.
- `RSMResult.load(path)` read a saved result back.
- `.compare_to_baseline(scores)` paired verdict against an external
  baseline's per-fold scores.
- `.predict`, `.predict_proba` forwarded to `final_estimator`.
- `.mean_score`, `.std_score`, `.mnar_flag` properties.

## 3. Fitted attributes

What to read off a fitted selector. These exist only after `fit`.

**`BootstrapMaskSelector` and `RepresentationEnsembleSelector`**, all arrays
of length `n_segments` unless noted:

| attribute | meaning |
|---|---|
| `pi_` | selection frequency, computed only over resamples that could assess the segment. NaN, never 0, when support is below `a_min` |
| `support_` | the denominator of `pi_`: fraction of resamples that could assess the segment. **Always report beside `pi_`** |
| `verdicts_` | `"selected"`, `"rejected"` or `"unassessable"` per segment |
| `obs_frac_` | mean observation fraction per segment across samples |
| `selected_segments_` | ids above `tau_` |
| `rejected_segments_` | supported but below `tau_` |
| `unassessable_segments_` | not enough support to judge |
| `insufficient_evidence_` | segments that looked selectable but lacked support: the ones a naive tool would have reported |
| `tau_` | the threshold actually used (differs from `tau` under `target_coverage`) |
| `mask_` | the binary mask handed to the refit |
| `coef_`, `intercept_` | the refit model's parameters |
| `model_` | the refit `SoftMaskSelector` on the selected region |
| `resample_sets_` | selected set per resample, for stability metrics |
| `n_fits_`, `n_failed_fits_`, `fit_errors_` | resample-fit accounting |
| `degenerate_` | True when aggregation selected nothing and the fit fell back to the single best segment |
| `observation_groups_`, `strata_note_` | how resampling was stratified |

**`SoftMaskSelector`** additionally exposes `gate_params_` (raw gate
parameters), `assessable_segments_` (boolean), `obs_rate_`, `stats_`
(the frozen `StandardisationStats`), `final_loss_`, `head_loss_`,
`n_excluded_samples_` and `classes_`.

## 4. Lenses (representations)

A lens changes which segments rank highly, never the coordinate system the
answer is expressed in: the final refit always uses the plain mean lens, so
the deliverable is always intervals of your axis. Each lens exposes
`.transform(X, V, grid) -> (F, VF)` and a `.name`.

| lens | signature | what it responds to |
|---|---|---|
| `SegmentMean` | `SegmentMean()` | validity-weighted per-segment mean; the default |
| `SegmentStats` | `SegmentStats(min_count=2)` | mean and dispersion, texture-aware |
| `SavGolDerivative` | `SavGolDerivative(window=11, polyorder=2, deriv=1)` | Savitzky-Golay derivative energy, NaN-safe; the chemometrics baseline |
| `GaussianScale1D` | `GaussianScale1D(sigmas=(2.0, 8.0), truncate=4.0)` | the same lens at coarser scales: sharp peaks against broad envelopes |
| `RandomConv1D` | `RandomConv1D(n_filters=4, kernel=9, seed=0)` | random zero-mean filters, ROCKET-lite |

`default_ensemble(n_representations=4, seed=0) -> list` returns a
deliberately diverse ensemble for `RepresentationEnsembleSelector`.

## 5. Fold estimators and splitters

You only need these to customise what `NestedCV` fits per fold. All four
implement the `FoldEstimator` protocol, whose `fit` sees TRAIN data only.

| estimator | fits per fold |
|---|---|
| `BootstrapMaskFoldEstimator(**kwargs)` | stability selection; the `NestedCV` default |
| `EnsembleMaskFoldEstimator(n_representations=4, representations=None, **kwargs)` | the representation ensemble |
| `SoftMaskFoldEstimator(**kwargs)` | a single gated fit, no frequencies |
| `FullSignalFoldEstimator(segment=16, **kwargs)` | the matched baseline, every segment retained |

`FoldEstimator` is the protocol itself, for writing your own.

Splitter helpers, used internally and exposed for reproducing a split:

- `infer_task(y) -> str` infer `"binary"`, `"multiclass"` or `"regression"`
  from the LABELLED targets only.
- `make_outer_splitter(task, groups, n_splits, seed)` grouped, stratified or
  plain as the data requires.
- `make_inner_splitter(task, grouped, n_splits, seed)`

### Resampling helpers

The pieces the selectors use to draw resamples, exposed so a split can be
reproduced or a custom scheme built.

- `resample_indices(n, mode, rng, strat=None)` indices for one resample.
  `mode` is `"bootstrap"` or `"half"`. With `strat` labels the draw is
  stratified, taking a fixed count per stratum, which is why resample
  support has zero variance when observation groups are detectable.
- `complementary_pair(n, rng, strat=None)` two disjoint halves covering the
  sample: the Shah and Samworth complementary pairs scheme, reached from
  `BootstrapMaskSelector(subsample="complementary")`.
- `observation_groups(v_seg, max_groups=8, min_size=4)` exact
  validity-pattern groups, or `None` when the patterns do not form groups.
  This is the automatic fallback used to stratify resampling by observation
  pattern; there is no clustering guesswork, only exact pattern matching.

## 6. Tuning

Sparsity strength is per-dataset, so choosing it is a library feature rather
than something to guess.

- `lam_frontier(make_selector, lams) -> list` fit once per lam and tabulate
  coverage, chance-corrected stability, ambiguity, unassessable fraction,
  selection size, failed fits and degeneracy. No winner is chosen, because
  the trade-off is the result.
- `lam_for_coverage(make_selector, target, lo=1e-3, hi=1.0, n_iter=10, tol=0.05) -> dict`
  log-scale bisection to the lam whose coverage is nearest `target`. Returns
  `{"lam", "coverage", "target_reached", "history"}`. `target_reached` is
  False when the bracket cannot deliver the target, and the best point found
  is still returned, so a miss is never renamed a hit. A lam strong enough
  to make the selector refuse is recorded as `coverage` NaN with `refused`
  True and the search continues.

`make_selector` is a callable taking a lam and returning a FITTED selector.

## 7. Validity primitives

The masked-computation core. Use these directly when building your own lens
or diagnostic; every one of them treats unobserved entries as absent rather
than as a value.

- `as_validity(X, V=None) -> (X, V)` canonicalise a values-and-validity
  pair. With `V=None`, NaN in `X` means unobserved.
- `masked_mean(X, W, axis=None, strict=False) -> (value, ok)`
  validity-weighted mean, scale-safely, with a validity flag rather than a
  silent NaN.
- `masked_std(X, W, axis=None, min_count=2, strict=False) -> (value, ok)`
- `masked_standardise_fit(F, W, n_min_hard=8, n_min_warn=30, sigma_tol=1e-12, strict=False) -> StandardisationStats`
  fit standardisation over rows. Fit on a training fold only; using it
  anywhere that touches a test partition is leakage.
- `masked_standardise(F, W, stats)` apply frozen statistics. Unassessable
  features become exactly 0 and contribute nothing.
- `StandardisationStats` frozen per-feature statistics: `scale`, `mu`,
  `sigma`, `ess`, `assessable`, `low_support`, `degenerate_scale`.
- `renormalised_convolve1d(X, V, filt, min_frac=None, mode="reflect") -> (out, v_out)`
  partial 1D convolution measured against surviving evidence only. Level
  filters normalise by observed filter weight, contrast filters are
  re-centred over the surviving support and normalised by observed absolute
  mass. Reduces exactly to a plain convolution on a fully observed window.
- `renormalised_dot(U, gate, validity, rho_min=0.1)` the head's analogue:
  a per-sample renormalised gated sum.
- `is_contrast_filter(filt) -> bool` zero-sum filters measure contrast,
  nonzero-sum measure level. This decides which correction applies.
- `observation_support(V_seg, min_valid_frac=0.5)` fraction of samples
  observing each segment at that validity or better.
- `missingness_association(V_seg, y, task, n_permutations=200, n_splits=5, alpha=0.05, seed=0, top_k=5) -> dict`
  score how well the missingness pattern ALONE predicts the target, with a
  permutation test. Returns `verdict` in
  `{"informative", "not_informative", "could_not_assess"}` plus `score`,
  `p_value` and `top_segments`. This is the MNAR guard.
- `fill_zero(X, V)`, `fill_mean(X, V, baseline=None)` the fabricating
  strategies, present as the baselines to beat rather than as
  recommendations.
- `fabricated_edge(x_flat, V, filters, eps=1e-12) -> dict` how much response
  each fill strategy invents at a gap boundary on flat data, where the right
  answer is known to be zero.

### Gated-head primitives

The renormalised head as pure functions, for building a custom estimator or
checking the gradients. These are what make an unobserved segment receive
exactly zero gate gradient rather than a small one.

- `renormalised_gated_forward(U, v_seg, m, active, intercept, rho_min=0.1, r_override=None)`
  the renormalised gated head: `z = b + (M / D) * sum_k m_k v_ik u_ik`, a
  ratio estimator that reduces exactly to the plain head on complete data.
  Returns decisions, per-sample validity, and the renormaliser.
- `mask_data_gradient(U, dz, r, active)` the analytic stop-gradient
  `dLoss/dm`. Exactly zero for never-observed segments, which is the
  property that stops "unobserved" being learned as "unimportant".
- `loss_and_dz(task, Z, Y, valid)` task loss and `dLoss/dZ` over the valid
  samples only. Epsilon-free softplus and log-sum-exp forms, so the analytic
  gradient is the exact derivative of the reported loss.

## 8. Segment grid

`SegmentGrid(n_points, segment=16)` a regular partition of the shared axis.

- `SegmentGrid.from_signal_shape(shape, segment=16)` build one from an `X`
  shape.
- `.n_segments`, `.n_points`, `.segment`
- `.pool(X, V) -> (F, VF)` both `(n, S, C)`: the masked pooling.
- `.segment_validity(V) -> (n, S)` per-sample segment validity for gating.
- `.expand(seg_arr)` map a per-segment array back onto the point axis.
- `.segments_to_points(segment_ids)` point indices covered by those ids.

## 9. Metrics

- `score_predictions(task, y_true, y_pred=None, y_proba=None) -> float`
  task-appropriate score, always higher-is-better (regression returns
  negative RMSE).
- `rmse_from_score(score) -> float` convert that back to positive RMSE.
- `jaccard(set_a, set_b) -> float` with `J(empty, empty) := 1.0`.
- `expected_jaccard(size_a, size_b, n_total) -> float` the chance baseline.
- `adjusted_jaccard(set_a, set_b, n_total) -> float` chance-corrected. Use
  this rather than raw Jaccard, which rewards keeping everything.
- `mean_pairwise_jaccard(sets, min_size=1, n_total=None) -> float`
- `paired_comparison(scores, baseline_scores, alpha=0.05) -> BaselineComparison`
  Wilcoxon signed-rank (primary) plus a paired t-test. Folds where either
  side is NaN are excluded pairwise; with fewer than two comparable folds
  the outcome is `"undecidable"`.
- `BaselineComparison` fields `p_wilcoxon`, `p_ttest`, `mean_delta`,
  `outcome` in `{"preserved", "sig.better", "sig.worse", "undecidable"}`,
  plus `.to_dict()`.

## 10. Synthetic data and missingness generators

For controls, tests and demonstrations where the right answer is known.

- `make_signal_control(n=200, n_channels=1, n_points=256, task="binary", signal=2.0, noise=1.0, cell=8, region_cells=3, distractor_cells=3, n_classes=3, seed=0) -> SignalControl`
- `SignalControl` fields `X`, `y`, `informative` and `distractor` (boolean
  masks over the point axis), `task`; plus `.truth_segments(grid, min_overlap=0.5)`
  and `.distractor_segments(grid, min_overlap=0.5)`.

The generators return a boolean validity array, or a modified `X`:

| generator | pattern |
|---|---|
| `mask_scattered(shape, frac=0.2, seed=0)` | independent point missingness (MCAR) |
| `mask_dropout_stretches(shape, n_stretches=1, min_len=8, max_len=32, seed=0)` | contiguous runs, as a detector dropout |
| `mask_band_by_group(shape, groups, bands)` | whole bands missing per subgroup, as instrument ranges |
| `mask_saturation_censor(X, lower=None, upper=None)` | censored-at-limit points marked missing |
| `drop_labels(y, frac=0.2, seed=0)` | missing targets, stratified so no class is emptied |

## 11. Reproducibility

- `Seeds(base, n_bootstrap=0, n_representations=0)` deterministic per-stream
  seed derivation from one base seed, with `.bootstrap(i)`, `.inner(fold, repeat)`,
  `.outer(fold, repeat)`, `.representation(i)` and `.rng(seed)`. Streams are
  separated by additive offsets and the constructor asserts loudly if they
  could ever collide.
- `BOOTSTRAP_OFFSET`, `REPRESENTATION_OFFSET`, `INNER_OFFSET`,
  `REPEAT_STRIDE` the offsets themselves.
- `set_global_seed(seed)` seed every global RNG channel RSM can reach. Not
  needed for reproducibility of RSM itself, which derives its own streams;
  useful for making surrounding code reproducible too.

Results do not depend on `n_jobs` or on call order.

## 12. Configuration

- `RSMConfig(...)` a dataclass holding every user-facing knob with robust
  defaults, and `.to_dict()`. Convenient for recording a study's settings;
  the selectors take plain keyword arguments, so it is optional.
- `fit_count(k_outer, repeated_outer_cv, n_representations, n_bootstrap, n_iter, l_inner) -> int`
  total model-fit count, for estimating cost before starting.

## 13. Errors and warnings

- `InsufficientEvidenceError` raised when a requested quantity has no
  evidence behind it: too few samples for the effective-sample-size floor,
  an evidence threshold above what the data carries, or a penalty that
  closed every gate. It is a refusal, not a crash to work around; the fix is
  usually to lower a threshold deliberately or to declare a sparse dataset's
  evidence scale (`min_valid_frac`, `rho_min`).
- `MissingnessInformativeWarning` the observation pattern alone predicts the
  target, so the observation process is confounded with what you are
  measuring. RSM reports this and does not attempt to repair it. Do not
  silence it.

## 14. Refit model zoo

Selection is always done by the gated head, because the gates are trained by
gradient descent through it. The FINAL predictor on the chosen bands needs no
gradient, so it can be anything. `refit_model=` on either stability selector
swaps it, and the `alg` vocabulary matches RobustModelMaker so a study can
move between the two libraries.

| alg | model | tasks |
|---|---|---|
| `head` | the built-in renormalised gated head (default) | all |
| `lin` | least squares / logistic without penalty | all |
| `rdg` | ridge / ridge classifier | all |
| `las` | lasso / L1 logistic | all |
| `eln` | elastic net | all |
| `log` | logistic regression | classification only |
| `svm` | SVM with an RBF kernel | all |
| `rf` | random forest | all |
| `xgb` | xgboost if installed, else sklearn gradient boosting | all |
| `mlp` | multi-layer perceptron | all |
| `pls` | PLS regression, and PLS-DA for classification | all |

**The no-fabrication contract survives the swap, which is the point.** An
sklearn estimator cannot see a validity mask, so the selected region is
pooled to validity-weighted segment means and standardised with RSM's masked
standardisation, which sends unobserved entries to exactly 0 in standardised
space. That is the same convention the built-in head uses: an unmeasured band
contributes nothing rather than a fabricated value. Samples whose observed
evidence over the selected region falls below `rho_min` are REFUSED and come
back as NaN, exactly as the head refuses them, rather than as a confident
guess from a forest that would happily have produced one. Nothing is imputed.

Two consequences worth knowing:

- **Selection is unchanged by the choice.** `pi_`, `support_`, `verdicts_`
  and `selected_segments_` are identical whichever model you pick, which the
  test suite asserts. You are choosing a predictor, not a selector.
- **Under `NestedCV` the matched baseline uses the same model.** With
  `refit_model="rf"` the verdict compares a forest on the selected bands
  against a forest on all of them. Comparing a forest against the built-in
  head would confound region with model, which is what the verdict exists to
  isolate. Pass it to both factories, as in section 6 of the User Guide.

Functions and classes:

- `ALGORITHMS` the tuple of valid names above.
- `make_refit_model(alg, task, seed=0, **overrides) -> estimator` build one.
  Raises rather than substituting for an unknown name or an inapplicable
  task. When `xgb` falls back to sklearn it says so in `_rsm_note`.
- `PLSDA(n_components=2, scale=False)` PLS discriminant analysis, which
  scikit-learn does not ship: PLS regression onto one-hot targets, with
  softmax scores for `predict_proba`. `n_components` is clamped to what the
  data can support.
- `ExternalRefit(alg, task, grid, selected, seed=0, rho_min=0.1, ...)` the
  fit-and-refuse wrapper shared by the selector and the matched baseline.
- `refit_features(F, VF, selected, stats, rho_min=0.1) -> (Z, valid)` the
  feature construction, exposed so a custom estimator can honour the same
  refusal rule.

Fitted attributes when `refit_model != "head"`: `external_model_`,
`refit_stats_`, `refit_note_` (the xgboost fallback message, or None), and
`n_refit_excluded_`.
