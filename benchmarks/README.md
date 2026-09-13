# RobustSignalMaker benchmarks

Findings live in FINDINGS.md (negative results included, predictions
registered before runs). Result CSVs land in results/ named per run target.

## Run

```
py benchmarks/run_synthetic_benchmark.py                    # every dataset
py benchmarks/run_synthetic_benchmark.py raman ms           # a subset
py benchmarks/run_synthetic_benchmark.py tecator corn ovarian rruff basicmotions
py benchmarks/run_threshold_sensitivity.py                  # threshold sweep
py benchmarks/make_findings_table.py --write                # refresh FINDINGS sec.11
```

A FULL run also writes `results/run_meta.csv` (per-dataset budget k,
segment count, score name, truth segments, missingness). A subset run leaves
that sidecar alone, because a subset's metadata would shrink the universe
the consolidated table is built from.

`make_findings_table.py` regenerates the cross-dataset table in FINDINGS
sec.11 from those CSVs, between its two sentinel comments. Nothing in that
table is typed by hand: the 2026-09-02 audit found eighteen stale figures in
prose copied from an earlier run, and generation is the structural fix. The
generator refuses to build a table whose per-dataset CSVs are more than an
hour older than run_meta.csv, so it cannot silently mix two runs.

## Datasets

Synthetic controls (known truth, F1 scored): raman, xrd, ms, sensor.

Real datasets (no ground truth; stability and downstream score at a declared
budget k): tecator (OpenML 505, fetched automatically), corn (Eigenvector
corn.mat), ovarian (FDA-NCI post-QAQC SELDI-TOF via the Internet Archive;
the loader bins the raw per-sample m/z axes and caches an npz), rruff
(RRUFF excellent-unoriented Raman, six-mineral multiclass with real
range-difference missingness and specimen groups) and basicmotions (UEA
multivariate, 6-channel wearable series). Downloads live under
benchmarks/data/, which stays out of git; the docstrings in datasets.py
carry each dataset's provenance and caveats (the ovarian batch-artefact
caveat is mandatory reading before quoting its numbers, and the RRUFF
coverage-vs-class confound is recorded in FINDINGS sec.8). The runner's
uniform downstream scorer covers regression (ridge R2), binary and
multiclass (OVR AUC), so every registered dataset produces a table row; the
notebooks remain the place where grouped CV and the honesty machinery are
exercised properly, which the flat table cannot do.

## What is measured

Per dataset x selector, at equal selection size k = |truth segments|:
truth-recovery F1, split-half stability (chance-corrected mean pairwise
Jaccard, 3 replicates), and a uniform downstream score on the selected
segment means of mean-filled data (5-fold CV: ridge R2 for regression sets,
logistic AUC for binary, OVR AUC for multiclass), identical for every
method, so the comparison is between selections, not predictors.

## Extending

- Dataset: write a loader returning `(X, V, y, meta)` (see
  `datasets.load_real_template` for the contract, including how missingness
  must be declared) and register it in `datasets.LOADERS` under an explicit
  name. Sparse or unusual data declares its evidence scale in
  `meta["selector_config"]`.
- Selector: `select(X, V, y, grid, task, seed, k, **cfg) -> segment ids`,
  registered in `competitors.SELECTORS` with a name that states its fill
  strategy explicitly. `METHOD_ORDER` derives from the registry; nothing is
  hardcoded elsewhere (the RPM harness lessons).

## Open item

The real-data applications (alloys spectra, biomarker signal feeds) have a
registered loader template waiting on data files; the FINDINGS sec.4 caveat
says why the clean synthetic controls flatter filter methods and what the
real data must re-test (registered predictions first, per house practice).
