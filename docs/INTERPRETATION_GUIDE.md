# RobustSignalMaker Interpretation Guide

How to read what RSM gives you, what each number is and is not, and what to
report in a paper. Companion to USER_GUIDE.md (how to run it),
API_REFERENCE.md (every public symbol) and IMPLEMENTATION_GUIDE.md (how it
works).

## Contents

0. A worked example to read along with
1. The three-valued verdict
2. Selection frequencies (pi) and what they are not
3. Support: the mandatory co-report
4. Coverage and the stability confound
5. The ambiguous fraction: go/no-go
6. The performance verdict
7. Refusals
8. The MNAR flag
9. Lens agreement
10. When imputation honestly wins
11. What to report in a paper

## 0. A worked example to read along with

Every section below refers to this run. It is a control with a planted
informative band at segments 4 and 5, plus two deliberate gaps: one band no
sample measured at all, and one measured by only half the cohort.

```python
import numpy as np
from robustsignalmaker import BootstrapMaskSelector, make_signal_control

c = make_signal_control(n=200, n_points=256, task="binary", seed=0)
X = c.X.copy()
X[:, 0, 128:160] = np.nan          # a band NOBODY measured
X[:100, 0, 200:230] = np.nan       # a band half the cohort measured

sel = BootstrapMaskSelector(task="binary", segment=16, lam=0.05,
                            n_bootstrap=50, seed=0).fit(X, c.y)

print("band   pi     support  verdict")
for k in range(sel.grid.n_segments):
    pi = "nan" if not np.isfinite(sel.pi_[k]) else f"{sel.pi_[k]:.2f}"
    print(f"{k:>4}  {pi:>5}   {sel.support_[k]:>5.2f}   {sel.verdicts_[k]}")
```

```
band   pi     support  verdict
   0   0.24    1.00   rejected
   1   0.08    1.00   rejected
   2   0.02    1.00   rejected
   3   0.14    1.00   rejected
   4   1.00    1.00   selected
   5   0.94    1.00   selected
   6   0.02    1.00   rejected
   7   0.16    1.00   rejected
   8    nan    0.00   unassessable
   9    nan    0.00   unassessable
  10   0.04    1.00   rejected
  11   0.00    1.00   rejected
  12   0.46    1.00   rejected
  13   0.06    1.00   rejected
  14   0.34    1.00   rejected
  15   0.02    1.00   rejected
```

Three things to notice before reading on. The planted band is recovered
(4 and 5, at pi 1.00 and 0.94). The band nobody measured comes back as
`nan` with support 0.00 and the verdict `unassessable`, **not** as pi 0.00
and `rejected`: that distinction is the whole point of the library. And
segment 12 sits at pi 0.46, neither in nor out, which is what the ambiguous
fraction counts.

`sel.stability_report()` summarises the same run:

```
coverage: 0.143              (2 of the 14 ASSESSABLE segments, not of 16)
assessable_universe: 14
n_segments: 16
unassessable_fraction: 0.125
ambiguous_fraction: 0.214
insufficient_evidence: []
degenerate: False
strata: stratified by class x observation group
```

Note `coverage` is 2/14, not 2/16. Never-observed segments were never
candidates, so counting them would flatter the compression.

## 1. The three-valued verdict

Every segment ends as one of:

- **selected**: chosen in at least `tau` of the resamples that could assess
  it, with enough support to trust that frequency.
- **rejected**: assessable, and reproducibly not chosen. This is an
  evidence-backed negative.
- **unassessable**: the data could not support a judgement (never observed,
  observed by too few samples, or standardisable in too few resamples).
  `pi` is nan there. UNASSESSABLE IS NOT REJECTED: absence of evidence is
  never evidence of absence. A band your instrument never covered can be
  neither in nor out of the answer.

Segments whose raw frequency clears `tau` but whose support does not appear
in `insufficient_evidence_`: near-findings the data cannot back. Report them
as exactly that.

## 2. Selection frequencies (pi) and what they are not

`pi_k` is the fraction of ASSESSABLE resamples on which segment k was
selected. It is a reproducibility statement about this dataset, pipeline and
sparsity strength. It is NOT a posterior probability that the segment is
"truly informative", not an effect size, and not transportable across
preprocessing choices. Two segments with pi 0.9 and 0.7 are both stable
findings; do not over-read the difference.

## 3. Support: the mandatory co-report

Beside every `pi_k` travel `support_` (fraction of resamples that could
assess the segment) and `obs_frac_` (fraction of samples observing it).
A high pi at support 0.55 is a weaker statement than the same pi at 1.0.
Never quote pi without its support; the report methods do this for you.

## 4. Coverage and the stability confound

Raw Jaccard stability is confounded with how much a method selects (a
selector retaining most of the axis overlaps with itself trivially; in the
RPM benchmarks this confound reached r = +0.99). Use the chance-corrected
stability (`n_total` = the ASSESSABLE universe) and always read stability
and coverage together. The empty-set hazard is guarded: a selector that
collapses to nothing scores nan, not a perfect 1.0.

## 5. The ambiguous fraction: go/no-go

`ambiguous_fraction()` is the fraction of supported segments with pi in the
undecided band (0.2 to 0.8). Bimodal pi (small ambiguous fraction) is the
structural signature of a reproducible region. A diffuse pi means no subset
reproducibly carries the signal at this sample size; no threshold can
manufacture one, and the honest conclusion is "not decidable from this
dataset", not a hopeful selection.

## 6. The performance verdict

`result.verdict.outcome` compares the selected region against the SAME head
using every segment, paired over folds (Wilcoxon primary):

- **preserved** is the success criterion: the region lost nothing the full
  signal had, at a fraction of the coverage.
- **sig.better** happens and is welcome, but preserved is the claim.
- **sig.worse** means the selection discarded signal; loosen the sparsity or
  distrust the region.

Caveat from RMM: at 5 folds the two-sided Wilcoxon floor is 0.0625, so
"preserved" can hide a real per-fold difference; the t-test p and mean
delta are reported alongside.

## 7. Refusals

Predictions can be nan: the sample's observed evidence fell below
`rho_min` and RSM refused to score it rather than amplify a sliver.
`result.n_refused` counts them. Many refusals mean your data and the
selected region barely overlap for part of the cohort; that is a data
statement, and silently imputing through it would have hidden it.

## 8. The MNAR flag

`result.mnar_flag` (and per-fold stamps) says the missingness PATTERN alone
predicted the target in training folds. Everything selected from observed
data is then confounded with the observation process: the region may encode
which instrument saw the sample, not the underlying signal. RSM warns and
carries on; you must decide whether the confound is benign (for example,
instrument assignment genuinely random) or fatal. Note that any
imputation-first pipeline would have DELETED this warning along with the
mask (FINDINGS sec.2, scenario C).

## 9. Lens agreement

For ensemble selections, `representation_agreement()` is the mean pairwise
Jaccard between the regions each lens would have chosen alone. Low agreement
means a single-lens result would have been lens-specific, and the
marginalised region is the defensible one; high agreement means one lens
would have sufficed (report that honestly). Benchmarks so far: on clean
controls where lenses agree, marginalisation is insurance, not a stability
boost (FINDINGS sec.1).

## 10. When imputation honestly wins

The benchmarks are deliberate about this (FINDINGS sec.2 and sec.4). When
missingness is IGNORABLE (independent of the target and of the informative
structure) and the signal is strongly univariate, mean-impute-then-filter is
accurate and often MORE stable: imputation is a variance reduction, and the
renormalised head pays a real variance price (the 1/rho inflation). Where
masked computation is the right tool: regions nobody observed (refusal
instead of fabricated findings), non-ignorable missingness (the alarm
survives), censoring and detection-limit semantics, and any setting where
"how much evidence was there?" must be part of the answer. Say which regime
your data is in; the MNAR flag and the support maps are the evidence.

## 11. What to report in a paper

Minimum honest set for a selected region:

1. The selected segments with pi AND support, and the unassessable set.
2. Coverage, chance-corrected stability, ambiguous fraction.
3. The verdict against the full-signal baseline, with fold scores.
4. The MNAR verdict per fold.
5. Refusal counts.
6. Sample size, segment size, sparsity strength (or target coverage), seeds.
7. For ensembles: the lens list and their agreement.

The one-line summary formats: "preserved full-signal performance
(AUC a +- b vs c +- d, Wilcoxon p = e) at f% coverage, selection stability
g (chance-corrected), h segments unassessable for lack of observation,
missingness not informative of the target (p = i)".
