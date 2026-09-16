# RobustSignalMaker

[![License: MIT](https://img.shields.io/badge/license-MIT-440154.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-414487.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.2.0-31688e.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-283-21918c.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-22a884.svg)](tests/)
[![PyPI](https://img.shields.io/pypi/v/robustsignalmaker.svg?color=2a788e)](https://pypi.org/project/robustsignalmaker/)

NaN-aware, leakage-free stability selection for scientific signals and
spectra. RSM identifies the important parts of a series (time-, space-,
mass-series and the like) to retain and removes the rest, returning a
reproducible region of the sampling axis and an honest estimate of what that
region can predict.

Third sibling in the RobustMaker family:

| Package | Selects | Data |
|---|---|---|
| RobustModelMaker (RMM) | columns | tabular features |
| RobustPixelMaker (RPM) | patches | scientific images |
| RobustSignalMaker (RSM) | points and bands | signals and spectra |

**What it is for.** The same job RMM does for tabular columns, done for the
sampling axis: select a small, reproducible set of bands and report honestly
what was lost by discarding the rest. RSM re-implements the shared
RobustMaker family contracts (deterministic seed arithmetic, tau = 0.7
stability thresholding,
leakage-safe nested cross-validation, coverage-and-stability reporting) so a
band set comes with a per-fold selection frequency and a paired verdict
against the full signal, not just a ranking.

**What makes it work on real instrument data.** Spectra arrive with gaps:
detector dropout, saturated peaks, instrument ranges that differ per
specimen. Stability selection needs a selection frequency per band, and that
quantity is not well defined when different resamples see different parts of
the axis. RSM's answer is to treat missing data as absent evidence rather
than something to fabricate: every pooling, lens, and head operation is
renormalised over the evidence that actually exists, and every frequency is
reported beside its observation support, so a band nobody measured reads as
"cannot assess" and never as "stably rejected". That machinery is the
enabling mechanism for the selection, not a separate feature.

## Status

All eight planned milestones are complete (v0.2.0): the
masked-computation core, selection units and lenses, the gated engine,
stability selection with three-valued verdicts, the leakage-safe nested-CV
wrap with the full-signal verdict, the representation ensemble, the
missingness regimes, the benchmark harness with per-domain controls and
competitors, and the four guides in docs/. Every experiment's honest
verdict, including the falsified predictions, is in benchmarks/FINDINGS.md.
The open item is the real-data applications, which await data files; 
benchmarks/README.md documents the loader contract.

**Does it do its job?** Every real dataset compresses hard and comes back
`preserved`, the verdict meaning nothing significant was lost against the
full signal under leakage-safe nested CV. Figures are from the executed
notebooks in examples/:

| dataset | bands kept | verdict, selected vs full signal |
|---|---|---|
| tecator | 7 of 20 | preserved, RMSE 5.047 vs 4.170 |
| corn (specimen-grouped) | 4 of 35 | preserved, RMSE 0.256 vs 0.184 |
| ovarian SELDI-TOF | 6 of 64 | preserved, AUC 0.983 vs 0.995 |
| RRUFF Raman (specimen-grouped) | 8.5 of 64 per fold | preserved, AUC-OVR 0.9947 vs 0.9935 |
| BasicMotions | 2.75 of 10 per fold | preserved, AUC-OVR 0.921 vs 0.965 |
| Au nanoparticles, residual stability | 4 of 18 angle bands | preserved, RMSE 0.0297 vs 0.0279 |

Read `preserved` as "nothing SIGNIFICANT lost", not "nothing lost": on
several of these the full signal is ahead on the point estimate and the
paired test cannot resolve it at five folds, which is why both means are
printed beside every verdict.

The other verdicts, kept honest: masked computation wins where missingness
is never-observed or non-ignorable (refusal instead of fabricated findings;
the MNAR alarm that imputation destroys), imputation does not lose where
missingness is ignorable and the signal is strongly univariate, and lens
marginalisation is insurance against choosing the wrong lens rather than a
guaranteed stability boost. A filter such as ANOVA will often match or beat
RSM's downstream score at a fixed budget; that is a comparison between
predictors, and the deliverable here is the reproducible band set with its
verdict. Tecator is the cautionary case in the other direction: ANOVA
reaches stability 1.000 at R2 0.281 against RSM's 0.574 at 0.864, a
perfectly reproducible wrong answer.

Coverage policy: 98% enforced package-wide in the whole-package coverage run
(`fail_under`), with per-module figures printed beside it; every module is
currently at 100% and so is the package total, over 283 tests. Every
quantitative claim in this repository has been audited against the data file
that produced it (FINDINGS sec.9 and sec.10 record the sweep, the audit, and
the claims of ours that they refuted).

## Install

Editable install from the repo root (no staging step):

```
py -m pip install -e .
```

## Environment

On this machine the scientific stack lives in the `py` (Python 3.9)
interpreter, not bare `python`. Run everything with `py`:

```
py -m pytest tests -q
```

## Quick start

The library in four lines, on a control whose answer is known so you can
check it. This runs as written.

```python
import numpy as np
from robustsignalmaker import BootstrapMaskSelector, NestedCV, make_signal_control

c = make_signal_control(n=200, n_points=256, task="binary", seed=0)
X, y = c.X.copy(), c.y             # X is (200, 1, 256); use your own here
X[:80, 0, 180:210] = np.nan        # a detector dropout on 80 of the samples

sel = BootstrapMaskSelector(task="binary", segment=16, lam=0.05,
                            n_bootstrap=50, seed=0).fit(X, y)
print(sel.selected_segments_)      # [4 5]
print(sorted(set(np.flatnonzero(c.informative) // 16)))   # [4, 5], the truth

# what is the selection worth? refit everything inside each training fold
result = NestedCV(k_outer=5, segment=16, lam=0.05, n_bootstrap=50,
                  random_state=0).run(X, y)
s = result.summary()
print(s["verdict"], round(s["score_mean"], 3), "vs full", round(s["baseline_score_mean"], 3))
# preserved 0.868 vs full 0.846, keeping 2 of 16 bands
```

Selection is always done by the gated head, but what PREDICTS on the chosen
bands is your choice: `refit_model=` takes the same algorithm vocabulary as
RobustModelMaker (`lin`, `rdg`, `las`, `eln`, `log`, `svm`, `rf`, `xgb`,
`mlp`) plus `pls` for PLS regression and PLS-DA. Swapping it never changes
the selection, and the refusal rule survives: a random forest refuses the
same thin samples the head refuses instead of predicting from imputed
values. See docs/API_REFERENCE.md section 14.

`preserved` means nothing SIGNIFICANT was lost, not that nothing was lost.
Both means are printed beside the verdict for exactly that reason. See
docs/INTERPRETATION_GUIDE.md before quoting any of these numbers.

## Quick taste (Milestone 1 primitives)

```python
import numpy as np
from robustsignalmaker import as_validity, renormalised_convolve1d

x = np.full(200, 3.7)          # a flat spectrum
x[80:120] = np.nan             # a detector dropout
x, v = as_validity(x)

deriv = np.array([-1.0, 0.0, 1.0]) / 2.0
out, v_out = renormalised_convolve1d(x, v, deriv)
print(np.abs(out).max())       # ~1e-16: the gap fabricates no edge
```

Zero-filling the same gap and convolving fabricates an edge of the order of
the signal level. That difference, propagated through segmentation, lenses,
the selection head, and the stability frequencies, is the point of the
library.

## Layout

- `robustsignalmaker/` the package, one module per concern
  (docs/IMPLEMENTATION_GUIDE.md maps them)
- `tests/` a file per milestone, plus the bug-sweep and coverage-gap suites
- `benchmarks/` registries, synthetic controls, real datasets (tecator,
  corn, ovarian SELDI-TOF), FINDINGS.md with registered predictions and
  unsoftened results
- `examples/` six notebooks: Tecator (quick start), Corn multi-instrument
  (groups, ranges, the MNAR alarm, the lens story), ovarian SELDI-TOF
  (sparse MS, censoring, missing labels, provenance caveats), gold
  nanoparticles (three structure characterisations x four questions: the
  modality ladder, legible band physics, FCC-vs-twinned motif
  classification, nested verdicts), RRUFF Raman mineral identification
  (multiclass, real range-difference missingness, specimen-grouped CV, and
  a real archive confound caught by the MNAR guard), and BasicMotions
  (multi-channel joint gating with channel dropout)
- `docs/` four guides: USER_GUIDE (how to run a study), API_REFERENCE
  (every public symbol), INTERPRETATION_GUIDE (what the numbers mean),
  IMPLEMENTATION_GUIDE (how the internals work)
- `RELATED_WORK.md` novelty defence against prior art: ten fields of
  related work with citations, a per-claim threat matrix naming what
  would falsify each claim, and an explicit list of what is NOT claimed

## Citation

There is no paper for RSM yet, so cite the software itself:

```bibtex
@software{barnard2026robustsignalmaker,
  author  = {Barnard, Amanda S},
  title   = {{RobustSignalMaker}: NaN-aware, leakage-free stability
             selection for scientific signals and spectra},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/amaxiom/RobustSignalMaker},
}
```

`CITATION.cff` carries the same metadata in Citation File Format, which is
what GitHub reads to offer the "Cite this repository" button.

The shared stability-selection and leakage-safe nested-CV framework that RSM
re-implements for the sampling axis is described in the RobustModelMaker
preprint, which is the right secondary citation if you are citing the method
rather than this implementation:

```bibtex
@misc{barnard2026robust,
  title  = {RobustModelMaker: Coupling Bootstrap Stability Selection with
            Leakage-Safe Nested Cross-Validation for Scientific Machine
            Learning},
  author = {Amanda S Barnard},
  year   = {2026},
  eprint = {2606.01566},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url    = {https://arxiv.org/abs/2606.01566},
}
```

## Licence

MIT. Copyright (c) 2026 Amanda S Barnard.
