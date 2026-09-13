# Changelog

All notable changes to RobustSignalMaker are documented here, following the
Keep a Changelog style used by RobustModelMaker. Versions follow
MAJOR.MINOR.PATCH; the single source of truth for the version string is
`robustsignalmaker/__init__.py`.

## v0.2.0 (2026-09-01, amended 2026-09-07 before first publication)

Nothing in v0.2.0 had been pushed or published when the amendments below were
made, so they are folded into this entry rather than given a version of their
own. The amendments changed no behaviour: they made every number the
repository quotes traceable to a file produced by this code, turned the
coverage floor into a real constraint, and completed the novelty defence.

### Amendments of 2026-09-07

- **Contrast-branch denominator re-examined and DELIBERATELY LEFT ALONE.** The
  sibling library RobustPixelMaker was fixed for the same count-based rescaling
  defect and, in the process, found that a re-centred contrast response is
  magnitude-preserved by the L2 ratio rather than the L1 (absolute mass) ratio
  this library uses; on RPM's criterion, fidelity of the convolution response
  with random zero-mean kernels, L2 beat count 79.9% against 93.2% RMS error.
  Porting it here was the obvious action and would have been wrong. The swap is
  decision-relevant (it changed the selected segment set in 8 of 12
  configurations), so it was scored on the criterion that governs THIS library,
  truth recovery of the selection: paired over 10 seeds, two dropout regimes and
  two contrast lenses, L1 beat L2 21 to 9 with 10 ties, mean F1 +0.019. The
  exchangeable-residual argument behind L2 does not hold for a Savitzky-Golay
  derivative, whose coefficients are smooth and antisymmetric, and RSM's default
  ensemble contains one. Count ties L1 (16 to 14, 10 ties) but is rejected as the
  rule that ignores filter weights and fabricates on the level branch. No code
  behaviour changed; the reasoning is recorded in `validity.py`, the experiment
  in FINDINGS sec.12, and both branch rules are now pinned by tests so neither
  library can drift into the other silently.


### Added

- `benchmarks/make_findings_table.py`, which GENERATES the consolidated
  cross-dataset table in FINDINGS sec.11 from `results/*.csv`. Nothing in
  that table is typed by hand. This is the structural fix for the defect the
  2026-09-02 audit found (eighteen stale figures in prose hand-copied from an
  earlier run): a generated table cannot drift from its data. The generator
  refuses to build a table whose per-dataset CSVs are more than an hour
  older than the run metadata, so it cannot silently mix two runs, and it
  prints the CSVs' own three-decimal precision rather than re-rounding.
- `benchmarks/results/run_meta.csv`, written by a FULL benchmark run only
  (a subset run leaves it alone, since a subset's metadata would shrink the
  universe the table is built from): per-dataset budget k, segment count,
  score name, truth segments and missingness.
- `tests/test_coverage_98.py`, 33 tests closing the last uncovered branches.
  They fall into three groups: public surface no benchmark happens to call
  (`decision_function`, `pi_points`, `selected_points`,
  `RSMResult.predict_proba`), documented alternative paths the default
  config never takes (`gate="sigmoid"`, `subsample="complementary"`,
  `use_unlabelled_stats=True`, the degenerate tau=1.0 fallback), and guards
  needing hostile input, including two third-party failure modes RSM must
  convert into "cannot assess" rather than a number: a rank test refusing a
  sample, and a CV splitter refusing a fold count. Each asserts the SHAPE of
  the refusal (nan, `could_not_assess`, a raise), never merely that the call
  returned.

- `RELATED_WORK.md` completed. It was a skeleton with 11 of 14 sections
  marked TODO; it now carries ten fields of prior art with checked citations
  (author, year, title, venue, with page ranges verified against the
  publisher record or flagged unchecked), a nine-row threat matrix naming
  the nearest prior art per claim and the experiment that would falsify it,
  and an explicit list of what is NOT claimed: superiority over imputation
  under ignorable missingness, error-control bounds under differential
  observation (Shah and Samworth's bounds assume exchangeable resampling
  over a fixed feature set and do not transfer to a per-feature denominator),
  and a stability gain from lens marginalisation where lenses agree. The
  falsified M6 marginalisation prediction is kept in the threat matrix as a
  falsification rather than dropped.

- **`lam_for_coverage` was unusable at its own defaults, and is fixed.** Found
  by executing the usage recipes in the PyPI README rather than by reading
  them. Its default bracket reaches `lam = 1.0`, which closes every gate on
  most datasets, and the selector then correctly REFUSES with
  `InsufficientEvidenceError`. That exception propagated straight out of a
  function whose entire job is to sweep lam, so the documented call crashed.
  A refusal is now recorded as `coverage` nan with `refused` True in the
  history and the search continues; the pre-existing `_distance` helper
  already scored nan as infinitely far, so a refusal can never win
  best-point tracking. Any other exception still propagates, because only
  the library's own "not enough evidence" is data.

  Catching it was not sufficient. A refusal at the strong end was then read
  as a failed bracket, so the search gave up and returned the WEAKEST lam:
  for a target coverage of 0.25 it reported 0.94 with `target_reached`
  False, when 0.25 was reachable at lam 0.02. A refusal means achievable
  coverage there is zero, so the target IS bracketed. Likewise, a refusal
  mid-search means the penalty is too strong and the bisection must weaken
  it; the original branch lumped refusals in with over-covering and
  strengthened, walking further into the refusing region. With both fixed
  the search now hits 0.25, 0.125 and 0.5 exactly on the README's own
  example, surviving a refusal in every run.

- **`docs/API_REFERENCE.md`, the reference the project did not have.** An
  audit of the repository documentation found that 45 of the 67 symbols in
  `__all__` were never mentioned in any guide or README: two thirds of the
  public surface was undocumented, including every lens except by name,
  all four missingness generators, the whole jaccard family and every
  masked primitive but two. The reference now covers all 67, grouped by
  module, with signatures taken by introspection rather than transcribed,
  a table of every fitted attribute (`pi_`, `support_`, `verdicts_`,
  `insufficient_evidence_` and the rest), and a statement of what is public
  and what may change. Completeness and every documented default are
  checked against the live API programmatically.

- **`refit_model=`: a model zoo for the final predictor
  (`robustsignalmaker/models.py`).** RSM had one predictor where RMM has
  nine, and the difference is architectural rather than an oversight: RSM's
  gates are trained by gradient descent THROUGH the head, so that head must
  be differentiable and hand-differentiated. The final refit needs no
  gradient, though, so it can be anything. `refit_model=` now takes RMM's
  own vocabulary (`lin`, `rdg`, `las`, `eln`, `log`, `svm`, `rf`, `xgb`,
  `mlp`) plus `pls`, which covers PLS regression and a hand-built PLS-DA
  since scikit-learn ships none and it is the standard chemometrics
  classifier. `xgb` uses xgboost when installed and announces the
  scikit-learn fallback in `refit_note_` rather than substituting silently.

  **The no-fabrication contract survives the swap, which was the design
  problem.** An sklearn estimator cannot see a validity mask. Rather than
  impute, the selected region is pooled to validity-weighted segment means
  and standardised so unobserved entries are exactly 0 in standardised
  space, the same convention the built-in head uses, and any sample below
  `rho_min` is refused and returns NaN. A test asserts that a random forest
  and PLS refuse precisely the samples the head refuses, on data where a
  fill-first pipeline would have scored all of them.

  Two invariants are pinned by tests: swapping the model leaves `pi_`,
  `support_`, `verdicts_` and the selected set bit-identical, so this is a
  choice of predictor and not of selector; and under `NestedCV` the matched
  baseline uses the same model, so the verdict compares a forest on the
  selected bands against a forest on all of them rather than confounding
  the region with the model.

### Changed

- **Documentation examples now run, and were executed rather than read.**
  All three User Guide code blocks previously failed on undefined names, so
  a reader could not paste them; they are now self-contained and run in
  sequence from section 3. The repo README gained a runnable quick start of
  the actual selection workflow, which it lacked (its only example was a
  validity primitive), with inline outputs matching a real run. The
  Interpretation Guide gained a worked example as its section 0, showing
  all three verdicts on one run including a genuinely unassessable band
  reported as `nan` with support 0.00 rather than as a confident rejection.
  All 14 python blocks across the README, the four guides and the PyPI
  README are executed as a check; 13 run, 1 needs reader-supplied groups.

- Coverage floor raised from `fail_under = 90` to `fail_under = 98`. Actual
  coverage is 100% on every module and on the package total, over 232 tests.
- `masking.py`: the "no segment is assessable" guard is marked
  `# pragma: no cover` with the reason. It is unreachable, not untested:
  `masked_standardise_fit(strict=True)` on the line above already raises
  whenever no (segment, channel) is assessable, and `_active` is that same
  array reduced over channels. Kept as an assertion of the invariant.

### Noted, not fixed here

- RobustPixelMaker carries the same partial-convolution defect this project
  fixed (FINDINGS sec.9 fix 1), and it is active there rather than latent.
  Reproduced 2026-09-07 against RPM `maskfill.renormalised_convolve`: on a
  flat image beside a straight gap edge a uniform 7x7 kernel is exact, a
  Gaussian 7x7 sigma 1.5 gives 74% error and a Gaussian 9x9 sigma 1.0 gives
  99.9% error. The contrast branch is wrong too and is the branch RPM
  actually exercises, since `RandomConvFeatures` builds random zero-mean
  filters: dropping the 8 of 25 pixels holding only 7% of the filter mass
  changed the response by 227%. Left for a separate RPM change at Amanda's
  direction; recorded here so the two siblings do not diverge silently.

### Verified (no changes needed)

- Full benchmark rerun of all nine datasets against the tagged code
  reproduces every CSV bit-identically, and every headline figure in
  FINDINGS sec.4, sec.6 and sec.8 matches its source CSV exactly (tecator
  rsm_masked 0.574 / 0.864, corn 0.283 / 0.432, ovarian 0.452 / 0.971,
  rruff 0.48 at AUC 0.994, basicmotions best stability 0.15).
- The threshold sensitivity sweep reproduces all five sec.10 verdicts and
  their measured drivers exactly, including the a_min sequence
  [9, 10] -> [10] -> refuses that refuted this project's own earlier claim
  that a_min could not bind.
- All six example notebooks re-executed end to end. Verified against their
  own printed output: tecator's "RMSE 4.17 against 5.05" (prints 4.170 /
  5.047); corn's lens agreement 0.205 and marginalised set [6 7 8 19 24];
  ovarian's censoring negative result (bands losing standing are NOT the
  most-clipped, and the second-most-clipped gains); RRUFF's grouped 0.9947
  against 0.9935 with the MNAR guard firing in 3 of 4 folds at p = 0.0244,
  diamond 1332 and calcite 1086 selected and beryl's 1067 in the adjacent
  unselected segment; BasicMotions windows [3, 4, 5] unchanged under
  gyroscope dropout.

### Fixed (documentation)

- Two stale claims the earlier audit had missed, both built by splicing
  numbers from two different runs, which is the defect class rather than any
  one number:
  - FINDINGS sec.6 described the corn mean lens as selecting "1100 to 1138
    and 2420 to 2498 nm". Measured: the m5 mean lens takes 1100 to 1138 and
    2460 to 2498 (one segment, not two) plus 1340 to 1418 and 1860 to 1898,
    so the sentence both widened the far edge and suppressed two
    mid-spectrum bands; "2420 to 2498" belonged to the GROUPED nested run,
    which contains no 1100 nm segment at all. Corrected in FINDINGS and in
    the corn notebook's own narrative, each naming its run.
  - FINDINGS sec.8 and the BasicMotions notebook quoted the single-split
    "AUC-OVR 0.97 vs 0.975" beside the nested verdict `preserved`, which
    reads as a tighter result than the nested run gives (0.921 vs 0.965).
    Both pairs are now stated and labelled, with the nested one called the
    honest reading.
- Artifact page: the published results page had transcribed basicmotions
  stability as 0.68 (its downstream score) instead of 0.15, led the RRUFF
  story with the optimistic ungrouped AUC that FINDINGS explicitly declines
  to claim, said the MNAR guard fired in 4 of 4 folds rather than 3, named a
  results file that does not exist, and footed itself as v0.1.0. The page is
  now BUILT from the result CSVs and the executed notebooks, embeds each
  notebook's figures, and carries each notebook's raw printed output beneath
  the prose so any quoted number can be checked in place.

### The v0.2.0 release itself (2026-09-01)

The real-data release. v0.1.0 completed the method on synthetic controls;
v0.2.0 takes it to five public datasets and a materials case study, executes
every example for real, and prepares the PyPI release path. The headline
science, in one paragraph: on gold nanoparticles the informative regions of
g(r), the bond-angle distribution and S(q) reorganise with the physical
question asked (size at low q, per-atom stability in the bond-angle motif
window, motif identity in the FCC signature angles), with five or six
selected bands rivalling 338 hand-crafted features; on corn NIR the lenses
disagree completely and the marginalised selection lands on the water-band
chemistry rather than scatter artefacts; and on the RRUFF Raman archive the
MNAR guard fired unprompted, catching a real coverage-vs-class confound that
any impute-first pipeline would have silently erased. Negative results are
kept: where missingness is ignorable and the signal univariate, mean-fill
plus a filter method remains accurate and more stable, and the tables say so.
The release closes with a deep bug sweep (28 reproduced findings, all fixed
or documented as measured scope; see Fixed below and FINDINGS sec.9), after
which every benchmark was re-run and every notebook re-executed.

### Added

- Real datasets in the benchmark registry: `tecator` (OpenML 505),
  `corn` (Eigenvector, instrument m5 for the benchmark, all three for the
  examples) and `ovarian` (FDA-NCI post-QAQC SELDI-TOF via the Internet
  Archive, per-sample m/z axes binned to a common grid and cached).
  Downloads cache under benchmarks/data/ (gitignored). The runner now
  handles truth-free datasets (declared budget k, no F1), regression
  downstream scoring (ridge R2), and dataset-declared selector
  configuration.
- FINDINGS sec.5/6: registered predictions and the real-data results.
  Finding A: stability and predictive value dissociate (tecator anova,
  stability 1.00 at R2 0.281); never rank by stability alone. Finding B:
  corn lens agreement is 0.205 (mean lens selects scatter edges, derivative
  lens the 1450 nm water band), the real-data lens disagreement the M6
  marginalisation claim was waiting for.
- Three example notebooks under examples/: Tecator quick start, Corn
  multi-instrument (groups, imposed instrument ranges, refusals, the MNAR
  alarm and its destruction by imputation, the lens story), ovarian
  SELDI-TOF (honesty reporting, imposed censoring, missing labels,
  provenance caveats).
- Tests: real-loader contract tests (skip when data is not cached).
- Gold nanoparticle example (user-supplied data, 4000 Au particles,
  examples/gold_nanoparticles/): notebook exploring which regions of g(r),
  the bond-angle distribution, and S(q) carry total formation energy,
  energy per atom, and the size-corrected excess stability. Findings
  recorded in benchmarks/FINDINGS.md sec.7: the modality ladder inverts
  with the target (low-q size vs bond-angle motif window vs medium-range
  order), five or six selected bands rival the 338 tabular features on the
  residual target, physically empty angle regions report unassessable, and
  the selected angle bands are identical across all nested-CV folds.

- Motif classification added to the gold nanoparticle example: a
  FCC-vs-twinned label derived transparently from the bimodal q6q6 interior
  crystallinity (threshold-robust, independent of the curves selected on).
  The angle distribution classifies perfectly from four bands at the
  90/109.5/120-degree FCC signatures, not the shared 60-degree band, and
  S(q) uses the Bragg region while ignoring the low-q size information it
  needed for total energy.
- Two more scientific examples with registered loaders (`rruff`,
  `basicmotions`): RRUFF Raman mineral identification (first multiclass
  example; real per-spectrum range differences as validity; specimen-grouped
  CV; the MNAR guard fired on the real archive, catching a
  coverage-vs-class confound, FINDINGS sec.8) and UEA BasicMotions (first
  real multi-channel joint-gating example; selected time windows unchanged
  under imposed gyroscope dropout for half the recordings).
- Coverage raised and enforced: targeted tests for guard/error/parallel
  branches (tests/test_coverage_gaps.py), every module now at 94% or
  higher (package 96%), and `fail_under = 90` in the coverage config.

- PyPI staging area (pypi_staging/) with a bespoke PyPI README, its own
  pyproject (same essentials as the root, urls added), scripted population
  (`prepare_staging.py` copies and verifies; nothing is synced by hand) and
  a parity test (tests/test_staging.py) guarding against the RMM
  manual-sync drift hazard. Wheel and sdist build from the staging area and
  pass twine check.
- Benchmark runner: multiclass downstream scoring (OVR AUC); full rerun of
  all nine datasets plus the threshold sweep recorded in
  benchmarks/results/ and FINDINGS sec.8.
- All six example notebooks executed for real with outputs saved in place.

### Fixed

The 2026-09-01 deep bug sweep (three parallel adversarial reviews plus
direct probing; every finding reproduced before fixing; the full record is
benchmarks/FINDINGS.md sec.9). Highlights:

- Renormalised convolution: the count-based rescale was exact only for
  uniform kernels and fabricated level shifts near gaps for Gaussian
  kernels (up to +39% on a flat signal, trusted); replaced with
  filter-weighted evidence normalisation, exact for any kernel. Every
  Gaussian-lens number under missingness changed slightly; results and
  notebooks were re-run.
- Payload hygiene in the public primitives: nan/inf at masked positions
  could reach arithmetic in masked_mean/std/standardise_fit,
  renormalised_convolve1d and renormalised_dot, returning nan WITH
  valid=True; all primitives now zero payloads first, and the payload
  tests cover nan/inf.
- Guards failing toward confidence: paired_comparison with nan fold scores
  said "preserved" (now pairwise exclusion plus an "undecidable" outcome);
  renormalised_dot's confident 0.0 at zero evidence; a_min=0 turning
  never-assessed segments into "rejected"; lam_for_coverage letting a nan
  coverage displace a within-tolerance hit.
- NestedCV: multiclass with grouped CV crashed when a class was
  group-confined (fold probabilities now align onto the engine's class
  list); repeated_outer_cv with deterministic GroupKFold duplicated folds
  and manufactured significance (forced to one pass, with a warning);
  object-dtype y with nan/None holes crashed opaquely (now excluded and
  counted, with a warning for "nan"-like string labels); constant y raises
  instead of burning a run.
- Fold adapters forwarded caller groups to resample stratification and now
  report the ASSESSABLE universe for chance-corrected stability (the raw
  segment count inflated it under heavy missingness).
- selection: fit-error diagnostics survive n_jobs > 1; ensemble
  (lens, resample) fit seeds are collision-free; per-lens agreement regions
  are thresholded at the same operating point as the reported answer.
- masking: string class labels survive predict (refusals read None);
  multiclass rejects single-class y; TV and L2 terms appear in the reported
  losses their gradients optimise; the fixed-mask path exposes final_loss_;
  Hard-Concrete docstrings no longer overclaim exact 0/1 test-time gates.
- Exports: both CSV writers quote properly (list- and comma-bearing cells
  corrupted every folds.csv row).
- Benchmarks: truth_segments no longer drops spike truth when band truth
  coexists (per-run rule); the Raman notebook's headline is the
  specimen-grouped nested comparison, not the selection-biased quick score.
- Threshold sensitivity rebuilt around bindability probes, because two of
  the five parameters had been swept over a regime where they provably
  could not act (a confirmation that cannot fail). Honest result in
  FINDINGS sec.10, which took three attempts to get right: o_floor is a
  genuine plateau; rho_min and o_min are live with deliberately loose
  defaults; min_valid_frac moved nothing across a range that straddles its
  driver; and a_min binds precisely when the missingness patterns cannot be
  grouped, which refuted this project's own earlier claim that it could
  never bind. Each parameter is now swept against the quantity it is
  compared with, with a probe that proves the sweep could fail.
- Measured scope decisions, documented rather than fixed: coefficients are
  not bitwise row-order invariant (BLAS reduction order; mask and selected
  set are; the contract is same seed AND same row order).
- RSMResult.summary() displayed the regression baseline score in the
  internal negative-RMSE convention while the method score was shown as
  positive RMSE; both now display alike.

## v0.1.0 (2026-09-01)

First feature-complete release: all eight planned milestones.

### Added

- Milestone 8: benchmarks, sensitivity sweep, guides.
  - `benchmarks/`: LOADERS registry (raman, xrd, ms, sensor domain controls
    with their characteristic missingness; a documented real-data loader
    template), SELECTORS registry (RSM masked and impute-then-select
    variants, ANOVA-F, Lasso, RF importance, iPLS, MC-UVE, full signal;
    every variant names its fill; METHOD_ORDER derives from the registry),
    `run_synthetic_benchmark.py`, `run_threshold_sensitivity.py`.
  - FINDINGS sec.3/4: predictions registered before the run, results and
    the scorecard after, including the falsified and inverted predictions
    (masked wins under dropout stretches; benign censoring makes
    interpolation legitimate reconstruction; the clean controls are the
    filter methods' home turf) and the two run-1 fixture/config lessons
    (shoulder truth under censoring; sparse data must declare its evidence
    scale via selector_config).
  - Threshold sensitivity: defaults confirmed on plateaus, one cliff
    (over-strict min_valid_frac) recorded.
  - docs/: USER_GUIDE.md, IMPLEMENTATION_GUIDE.md, INTERPRETATION_GUIDE.md
    (the RMM three-guide convention).
  - Tests: `test_benchmarks.py` (registry contracts, loader contract,
    fills, truth-segment rule).
  - Version 0.1.0.

- Milestone 7: missingness regimes and the novelty experiment.
  - `synthetic`: the generator family, `mask_dropout_stretches`,
    `mask_band_by_group`, `mask_saturation_censor` (clip and mark missing,
    the v1 censoring semantics), `drop_labels` (stratified). No generator
    ever writes a value.
  - benchmarks/FINDINGS.md sec.2: the scenario-by-scenario verdict. Masked
    computation refuses on never-observed bands where
    interpolate-then-screen confidently mis-selects fabrication (F 10x
    background on a segment nobody measured); the gated selector itself
    resists that fabrication; interpolation loses on recovery and stability
    under ignorable group bands while mean-impute honestly ties recovery
    and wins split-half stability there (the predicted 1/rho variance
    price); under MNAR the masked pipeline raises the alarm at the right
    segments and imputation destroys the alarm.
  - Tests: `test_missingness.py` (generator exactness; the three
    make-or-break scenarios pinned with the measured boundary documented
    in-line).

- Milestone 6: representation-ensemble marginalisation.
  - `selection`: `representation` parameter on `BootstrapMaskSelector`
    (single-lens bootstrap runs); `RepresentationEnsembleSelector`
    aggregating over lenses x resamples with the same support accounting,
    per-lens frequency maps, `representation_agreement()` in every
    stability report; shared `_frequency_map` helper.
  - `nested_cv`: `EnsembleMaskFoldEstimator`.
  - benchmarks/FINDINGS.md opened with the M6 negative result: the
    marginalised selection did NOT beat the best single lens on split-half
    stability on the synthetic control; it is near-best in both missingness
    conditions while the best lens's identity changes between them, beats
    the average lens, and insures against the worst. The stronger claim
    moves to M8 real-spectra testing.
  - Tests: `test_ensemble.py` (scripted per-lens map and agreement
    arithmetic, bitwise determinism, the reframed make-or-break with the
    falsification documented, NestedCV integration).

- Milestone 5: nested-CV wrap, verdict pipeline, results.
  - `nested_cv`: `NestedCV.run(X, y, V=None, groups=None)` with the
    structural leakage contract; missing-y rules (nan-target rows excluded
    from splits, fits and scoring, counted, optional
    `use_unlabelled_stats` pass-through); refusal-aware fold scoring;
    per-fold MNAR honesty stamps; built-in matched full-signal baseline
    and paired preserved / sig.better / sig.worse verdict; fold adapters
    `SoftMaskFoldEstimator`, `BootstrapMaskFoldEstimator`,
    `FullSignalFoldEstimator`; `infer_task` and group-safe splitters.
  - `results`: `RSMResult` (per-fold and baseline scores, verdict, OOF
    predictions, refusal and unlabelled counts, selection stability raw and
    chance-corrected, MNAR reports, fold records, final estimator),
    `summary()`, `results_tables()`, CSV/JSON/pickle `save()` and `load`.
  - Tests: `test_nested_cv.py` (make-or-break: verdict `preserved` at mean
    coverage < 0.3; missing-y bookkeeping; MNAR stamp points at the
    constructed bands; export round-trip) and `test_leakage.py` (canary
    proofs: no test row reaches fit, injected leaks trip, unlabelled rows
    reach neither fits nor test partitions, groups never span a split).

- Milestone 4: stability selection with support accounting, lam tuning.
  - `selection`: `BootstrapMaskSelector` (pi over ASSESSABLE resamples only;
    resample support and observation fraction reported beside pi;
    three-valued verdicts selected / rejected / unassessable with nan pi for
    unsupported segments; `insufficient_evidence_`; failed fits excluded
    from denominators, counted, and warned about above `max_failed_frac`;
    class x observation-group stratified resampling with
    `observation_groups` exact-pattern auto-detection; Shah-Samworth
    complementary pairs; tau targeting over the supported universe;
    all-closed aggregation raises instead of RPM's arbitrary-argmax
    fallback), plus `resample_indices` and `complementary_pair`.
  - `tuning`: `lam_frontier` (coverage / adjusted-stability / ambiguity
    table) and `lam_for_coverage` (log-bisection, honest about misses).
  - Tests: `test_selection.py` (scripted-inner-fit arithmetic pinning pi,
    support, verdicts and the failed-fit exclusion by hand-computable
    numbers; make-or-break bimodality + NaN stability incl. the
    threshold-free map-ordering assertion; band-by-group stratification;
    tuning honesty paths).

- Milestone 3: the gated engine.
  - `masking`: `SoftMaskSelector` (two-phase VTF fit ported from RPM;
    Hard-Concrete L0 and sigmoid L1 gates; TV-1D prior coupling assessable
    pairs only; renormalised head with stop-gradient renormaliser;
    observation-scaled sparsity penalty; rho_min sample exclusion with nan
    predictions; degenerate-mask flags; fixed-mask head refits; scale-safe
    y standardisation), plus the pure functions
    `renormalised_gated_forward`, `mask_data_gradient`, `loss_and_dz`.
  - Losses are exact (stable softplus / log-sum-exp), replacing RPM's
    epsilon-ed forms, so analytic gradients are exact derivatives of the
    reported loss and the numeric gradient checks pass at 1e-5.
  - Coverage policy: 90% per module, whole-package measurement only
    (config in pyproject); pytest-cov added to the dev extra. All modules
    at or above 90% (package total 94%).
  - Tests: `test_gates.py` (gradient checks for all three tasks, the
    stop-gradient property, HC helpers, TV contiguity, both make-or-break
    experiments, degeneracy and refusal paths).

- Milestone 2: selection units, lenses, synthetic control.
  - `segments`: `SegmentGrid` with validity-weighted pooling (per-sample
    denominators, unlike RPM's pre-normalised pooling matrix), segment
    validity for gating, expansion, ragged trailing segments, and
    `segment=1` as the exact point-wise mode.
  - `representations`: five validity-aware lenses (`SegmentMean`,
    `SegmentStats`, `SavGolDerivative`, `GaussianScale1D`, `RandomConv1D`)
    behind the `transform(X, V, grid) -> (F, FV)` contract, plus
    `default_ensemble`. Every lens reduces exactly to its unmasked form on
    complete data and never reads a masked payload.
  - `synthetic`: `make_signal_control` (jointly necessary informative cells,
    equal-variance distractor band, binary/multiclass/regression targets)
    and `mask_scattered`. Control uses non-negative amplitude latents and
    smooth in-cell bumps so it is fair to rectified-energy lenses (see
    module docstring for the symmetry argument).
  - Tests: `test_segments.py`, `test_representations.py`,
    `test_synthetic.py` including the M2 make-or-break (every lens separates
    the planted band, complete and under 20% scattered NaNs).

- Milestone 1 scaffold: package layout, MIT licence, root pyproject.toml with
  dynamic versioning (no staging copy step), py.typed.
- `reproducibility`: the family Seeds contract (additive documented offsets,
  collision asserts), numpy-only.
- `validity`: masked-computation primitives. `as_validity`, scale-safe masked
  mean/std, masked standardisation with effective-sample-size floors,
  `renormalised_convolve1d` (level vs contrast correction, fractional
  validity out), `renormalised_dot` (the renormalised head inner product),
  `observation_support`, `missingness_association` (the MNAR honesty guard,
  three-valued verdict), fill baselines and the 1D `fabricated_edge`
  diagnostic.
- `metrics`: score conventions, Jaccard family with the empty-set guard and
  the chance-corrected (assessable-universe) variant, `paired_comparison`.
- `config`: `RSMConfig` with the adopted evidence-threshold defaults, and the
  fit-cost model.
- Tests: `test_unit.py`, `test_reproducibility.py`, `test_validity.py`
  (make-or-break: flat-signal fabricated-edge, scale invariance at 1e-200 and
  1e300, masked-payload bitwise invariance, loud-failure guards).
