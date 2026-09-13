# RELATED_WORK: the novelty defence for RobustSignalMaker

Structure follows RobustPixelMaker's RELATED_WORK: each section names the
nearest prior art, what RSM borrows from it, and what remains unclaimed. The
precise claim is sec.12 here, and sec.11 is the per-claim threat matrix that
a reviewer should attack first. docs/IMPLEMENTATION_GUIDE.md carries the
design rationale for the machinery itself.

Citation format is author, year, title, venue, with volume and pages where
they have been checked against the publisher record. Entries marked
`[pages unchecked]` carry a verified author, year, title and venue but a
page range that has not been confirmed; add DOIs from a reference manager
before submission rather than trusting this file.

Every experimental claim referenced below is backed by a numbered section of
benchmarks/FINDINGS.md, including the ones that went against us.

## 0. Positioning

**The purpose.** RSM is the third member of the RobustMaker family:
RobustModelMaker selects tabular columns, RobustPixelMaker selects image
patches, and RSM selects points and bands of signals and spectra. All three
do the same thing: leakage-free stability selection, delivering a small
reproducible feature set with a per-fold selection frequency and a paired
verdict against using everything. For RSM the features are intervals of a
physical sampling axis, which is the form a chemist or spectroscopist can
act on.

**Why it needed new machinery.** Stability selection's central quantity is a
per-feature selection frequency over resamples. On real instrument data that
quantity is not well defined, because different samples observe different
parts of the axis: detector dropout, saturated peaks, instrument ranges that
vary per specimen. A band observed in a fifth of the resamples has a
frequency computed over a fifth as much evidence as its neighbour, and
reporting the two on one scale makes "rarely measured" indistinguishable
from "measured and rejected". Validity propagation is the answer to that
problem, not an independent selling point: it is what makes the selection
frequency mean something when the axis is only partly observed.

**Why the join is new.** The three literatures RSM sits between have each
solved a different part and none has joined them. Chemometrics owns band
selection but assumes complete spectra on a common grid. Statistics owns
stability selection but assumes a complete design matrix. Machine learning
owns learned sparse gates and masked computation but has no
selection-stability semantics and no notion of refusing to pronounce. The
join is where the new failure modes live, which is what sections 1 to 10
document and section 11 puts up for attack.

## 1. Interval selection in chemometrics

- Nørgaard, Saudland, Wagner, Nielsen, Munck, Engelsen (2000). Interval
  partial least-squares regression (iPLS): a comparative chemometric study
  with an example from near-infrared spectroscopy. *Applied Spectroscopy*
  54(3), 413-419.
- Jiang, Berry, Siesler, Ozaki (2002). Wavelength interval selection in
  multicomponent spectral analysis by moving window partial least-squares
  regression, with applications to mid-infrared and near-infrared
  spectroscopic data. *Analytical Chemistry* 74. [pages unchecked]
- Leardi (2000). Application of genetic algorithm-PLS for feature selection
  in spectral data sets. *Journal of Chemometrics* 14. [pages unchecked]

These are the classic band-selection baselines, and iPLS is the direct
ancestor of the problem RSM solves: choose contiguous intervals of a
physical axis, not scattered points. RSM borrows the framing and implements
iPLS as a benchmark competitor (benchmarks/competitors.py).

What remains unclaimed: all of these assume fully observed spectra on a
common grid. None defines what an interval score means when different
samples observe different parts of the axis, and none reports how much of
the axis was measured beside the score it reports. The failure this permits
is concrete and is measured in FINDINGS sec.2: an interval that nobody
measured can be selected with confidence once a fill step has invented
values in it.

## 2. Variable elimination and sampling-based selection

- Centner, Massart, de Noord, de Jong, Vandeginste, Sterna (1996).
  Elimination of uninformative variables for multivariate calibration.
  *Analytical Chemistry* 68(21), 3851-3858.
- Cai, Li, Shao (2008). A variable selection method based on uninformative
  variable elimination for multivariate calibration of near-infrared
  spectra. *Chemometrics and Intelligent Laboratory Systems* 90(2), 188-194.
  (MC-UVE.)
- Li, Liang, Xu, Cao (2009). Key wavelengths screening using competitive
  adaptive reweighted sampling method for multivariate calibration.
  *Analytica Chimica Acta* 648(1), 77-84. (CARS.)

This family is the closest thing in chemometrics to a stability idea:
MC-UVE builds many models on resampled calibration sets and keeps variables
whose coefficients are stable across them. RSM shares the resampling
instinct and beats MC-UVE cheaply on the synthetic controls (FINDINGS
sec.11, table A).

What remains unclaimed: the selection is point-wise rather than banded,
complete-data throughout, and there is no leakage contract, so the
resampling statistic and the reported performance are computed on the same
data. Crucially, coefficient stability across resamples is not the same
quantity as selection frequency with an assessability denominator: if a
variable is missing in most resamples, MC-UVE's stability is computed over
whatever remains and reported as a number, where RSM reports `nan` and the
verdict `unassessable`.

## 3. Sparse, fused, and group penalties on spectra

- Tibshirani (1996). Regression shrinkage and selection via the lasso.
  *Journal of the Royal Statistical Society B* 58(1), 267-288.
- Tibshirani, Saunders, Rosset, Zhu, Knight (2005). Sparsity and smoothness
  via the fused lasso. *Journal of the Royal Statistical Society B* 67(1),
  91-108.
- Yuan, Lin (2006). Model selection and estimation in regression with
  grouped variables. *Journal of the Royal Statistical Society B* 68(1),
  49-67.
- Simon, Friedman, Hastie, Tibshirani (2013). A sparse-group lasso.
  *Journal of Computational and Graphical Statistics* 22(2), 231-245.
- Rudin, Osher, Fatemi (1992). Nonlinear total variation based noise removal
  algorithms. *Physica D* 60, 259-268.

The fused lasso and total variation are the direct ancestors of RSM's TV-1D
prior, which is why contiguity is encoded as a difference penalty on the
chain graph rather than as a post-hoc smoothing of a point-wise answer.

The borrow is deliberately displaced: RSM applies the fused penalty to the
learned GATES, not to regression coefficients. That matters because a fused
penalty on coefficients ties the magnitude of neighbouring effects, which is
a claim about the physics of the response; a fused penalty on gates ties
only whether neighbouring regions are retained, which is a claim about the
instrument's resolution. RSM couples only pairs where both segments are
assessable, so contiguity is never inferred across a gap that was never
measured.

What remains unclaimed: complete-data assumptions throughout, a single
selection rather than a frequency, and no report of observation support.

## 4. Functional data analysis

- Ramsay, Silverman (2005). *Functional Data Analysis*, 2nd edition.
  Springer.
- Ferraty, Vieu (2006). *Nonparametric Functional Data Analysis*. Springer.
- James, Wang, Zhu (2009). Functional linear regression that's
  interpretable. *Annals of Statistics* 37(5A), 2083-2108.
- Pini, Vantini (2016). The interval testing procedure: a general framework
  for inference on functional data. *Journal of Nonparametric Statistics*
  28. [pages unchecked]

FDA is the field that takes the sampling axis most seriously, and the
interval testing procedure is the nearest prior art to "which part of the
domain matters" with error control.

What remains unclaimed, and this is the sharpest disagreement in this
document: sparse FDA handles irregular sampling by smoothing and borrowing
strength across curves. Under the RSM contract that is fabrication. It is
principled fabrication with good asymptotics, and where missingness is
ignorable it is the right thing to do (FINDINGS sec.2 scenario B is honest
that mean-imputation wins split-half stability in exactly that regime). The
RSM position is narrower: when the observation process is confounded with
the target, or when a region was never observed by anyone, borrowing
strength launders the confound into the answer, and no amount of smoothing
theory recovers what was not measured. RSM refuses instead, and reports the
refusal.

## 5. Stability selection

- Meinshausen, Bühlmann (2010). Stability selection. *Journal of the Royal
  Statistical Society B* 72(4), 417-473.
- Shah, Samworth (2013). Variable selection with error control: another look
  at stability selection. *Journal of the Royal Statistical Society B*
  75(1), 55-80.
- Bach (2008). Bolasso: model consistent lasso estimation through the
  bootstrap. *ICML*.
- Nogueira, Sechidis, Brown (2018). On the stability of feature selection
  algorithms. *Journal of Machine Learning Research* 18. [pages unchecked]

RSM implements Shah and Samworth's complementary pairs as a resampling
scheme (`subsample="complementary"`) and inherits the tau = 0.7 family
convention. Nogueira et al. is why RSM's stability metric is chance
corrected: raw Jaccard rewards keeping everything.

What remains unclaimed, and it is the technical core of the paper: both
frameworks define the selection frequency

    pi_k = (number of resamples selecting k) / (number of resamples)

with a denominator that is the same for every feature, because every feature
is present in every resample. Under differential observation that
denominator is wrong. A band observed in a fifth of the resamples has a
selection frequency computed over a fifth as much evidence as its neighbour,
and reporting the two on the same scale is a category error: it makes
"rarely measured" look identical to "measured and rejected".

RSM's extension is to make the denominator per-feature and explicit. `pi_k`
is computed over the resamples in which segment k was ASSESSABLE (observed
above `o_min` and clearing the ESS floor), `support_` reports what that
denominator was, and when fewer than `a_min` of the resamples could assess a
segment, `pi_k` is `nan` and the verdict is `unassessable`, never `0`. The
output is three-valued (selected, rejected, unassessable), and the
chance-corrected Jaccard is taken over the assessable universe rather than
the full segment count.

The error-control results of Shah and Samworth do NOT transfer to this
estimator unchanged, and we do not claim them. Their bounds are stated for
an exchangeable resampling scheme over a fixed feature set; RSM's
denominator is a random variable that differs per feature. Establishing the
analogous bound under differential observation is open work, stated as such
as open work, and the library reports the empirical support rather than a
theoretical error rate.

## 6. Learned sparse gates

- Louizos, Welling, Kingma (2018). Learning sparse neural networks through
  L0 regularization. *ICLR*. arXiv:1712.01312. (Hard concrete gates.)
- Maddison, Mnih, Teh (2017). The concrete distribution: a continuous
  relaxation of discrete random variables. *ICLR*.
- Jang, Gu, Poole (2017). Categorical reparameterization with
  Gumbel-softmax. *ICLR*.
- Yamada, Lindenbaum, Negahban, Kluger (2020). Feature selection using
  stochastic gates. *ICML*. (STG.)
- Balın, Abid, Zou (2019). Concrete autoencoders for differentiable feature
  selection and reconstruction. *ICML*.

RSM borrows the hard-concrete gate and the two-phase fit (train the head,
freeze it, then prune the mask) from this line via RPM.

What remains unclaimed, and both additions are forced by missingness rather
than chosen:

1. **Observation-scaled penalty.** Sparsity pressure is full strength on
   every gate, but the data gradient reaching a gate scales with how often
   its segment was observed. A sparsely observed segment therefore pays full
   rent on a fraction of the income and is driven closed for being
   unmeasured rather than unimportant. RSM scales the penalty, not the
   gradient: `lam_k = lam * max(mean_obs_k, o_floor)`. Scaling the gradient
   would have added variance; scaling the penalty is deterministic.
2. **Stop-gradient renormaliser.** The head is a ratio estimator, so the
   normaliser depends on the gates. Differentiating through it gives a
   never-observed segment a nonzero gate gradient through the denominator
   alone. Stopping the gradient there makes that derivative exactly zero, so
   a segment nobody measured cannot be pushed open or closed by the fit.

Neither appears in the gate literature because complete data makes both
degenerate to the identity.

## 7. Missing data in chemometrics and machine learning

- Rubin (1976). Inference and missing data. *Biometrika* 63(3), 581-592.
- Dempster, Laird, Rubin (1977). Maximum likelihood from incomplete data via
  the EM algorithm. *Journal of the Royal Statistical Society B* 39(1),
  1-38.
- Little, Rubin (2019). *Statistical Analysis with Missing Data*, 3rd
  edition. Wiley.
- Nelson, Taylor, MacGregor (1996). Missing data methods in PCA and PLS:
  score calculations with incomplete observations. *Chemometrics and
  Intelligent Laboratory Systems* 35. [pages unchecked]
- Walczak, Massart (2001). Dealing with missing data: Part I. *Chemometrics
  and Intelligent Laboratory Systems* 58(1), 15-27; Part II, same volume,
  29-42.
- van Buuren, Groothuis-Oudshoorn (2011). mice: multivariate imputation by
  chained equations in R. *Journal of Statistical Software* 45(3).
- Stekhoven, Bühlmann (2012). MissForest: non-parametric missing value
  imputation for mixed-type data. *Bioinformatics* 28(1), 112-118.

Rubin's MCAR/MAR/MNAR taxonomy is the vocabulary RSM's claims are scoped in,
and the honest scoping matters: RSM does not claim to beat imputation
everywhere. FINDINGS sec.2 records that mean-imputation wins split-half
stability when missingness is ignorable, which is the predicted 1/rho
variance price and not a defect. The claim is scoped to non-ignorable and
never-observed regimes.

Nelson, Taylor and MacGregor, and Walczak and Massart, are the chemometrics
prior art closest in spirit: both compute model quantities from incomplete
observations rather than filling first. RSM's renormalised operations are in
that tradition.

What remains unclaimed: this literature estimates MODELS under missingness.
None of it does SELECTION with stability semantics, and none reports a
three-valued verdict. The imputation packages are the explicit competitors
in M7 and M8, and the result that matters is not that they score worse; it
is that they score CONFIDENTLY on bands nobody measured (FINDINGS sec.2
scenario A: interpolate-then-screen scores fabricated background at F = 31.8
against a genuine-background maximum of 3.3), and that imputing first
destroys the MNAR alarm (FINDINGS sec.6: the corn guard reports
`informative` on the masked pipeline and `not_informative` after
imputation).

## 8. Partial convolutions and masked computation

- Liu, Reda, Shih, Wang, Tao, Catanzaro (2018). Image inpainting for
  irregular holes using partial convolutions. *ECCV*.
- Uhrig, Schneider, Schneider, Franke, Brox, Geiger (2017). Sparsity
  invariant CNNs. *3DV*.
- Yu, Lin, Yang, Shen, Lu, Huang (2019). Free-form image inpainting with
  gated convolution. *ICCV*.
- Harley, Derpanis, Kokkinos (2017). Segmentation-aware convolutional
  networks using local attention masks. *ICCV*.
- Savitzky, Golay (1964). Smoothing and differentiation of data by
  simplified least squares procedures. *Analytical Chemistry* 36(8),
  1627-1639.

Liu et al. is the source of the partial convolution: numerator over kept
pixels, rescaled by how many were kept. RSM ports it to 1D and extends it
past the convolution into the pooling, the standardisation, the head (a
renormalised inner product) and the gate training, so no operation in the
pipeline ever meets a fabricated value.

Two divergences from the source construction, both load-bearing:

1. **Level versus contrast correction.** A zero-mean filter over a partial
   support no longer sums to zero, so a flat region produces a response at a
   mask boundary: a fabricated edge of exactly the kind the construction
   exists to prevent. RSM detects the filter class from `filt.sum()` and
   re-centres contrast filters over the surviving support.
2. **Rescaling by observed filter WEIGHT, not observed COUNT.** This is a
   correction to the partial-convolution formula as commonly implemented,
   and it was found by adversarial audit of our own code (FINDINGS sec.9).
   Count-based rescaling is exact only for uniform kernels. For a level
   filter the correct denominator is the observed filter weight
   `sum(m*f) / sum(f)`; for a contrast filter it is the observed absolute
   mass `sum(m*|f|) / sum(|f|)`. Before the fix, a Gaussian lens fabricated
   up to a 39% level shift on a flat signal beside a gap and reported it at
   high validity. Anyone porting the textbook partial convolution to
   non-uniform kernels inherits this, and the same defect has since been
   reproduced in RobustPixelMaker, where a Gaussian 9x9 kernel produced up
   to 99.9% error on a flat image.

What remains unclaimed by the prior art: it has no selection layer and no
stability layer. Partial convolutions exist to make an inpainting network
train well; they do not report which regions the evidence supports, and they
have no vocabulary for refusing.

## 9. Time-series feature machinery

- Dempster, Petitjean, Webb (2020). ROCKET: exceptionally fast and accurate
  time series classification using random convolutional kernels. *Data
  Mining and Knowledge Discovery* 34, 1454-1495.
- Dempster, Schmidt, Webb (2021). MiniRocket: a very fast (almost)
  deterministic transform for time series classification. *KDD*.
- Christ, Braun, Neuffer, Kempa-Liehr (2018). Time series feature extraction
  on basis of scalable hypothesis tests (tsfresh). *Neurocomputing* 307,
  72-77.
- Lubba, Sethi, Knaute, Schultz, Fulcher, Jones (2019). catch22: canonical
  time-series characteristics. *Data Mining and Knowledge Discovery* 33.
  [pages unchecked]
- Ye, Keogh (2009). Time series shapelets: a new primitive for data mining.
  *KDD*.

ROCKET's random convolutional kernels are borrowed directly as a lens
(`RandomConv1D`), under the lens invariant inherited from RPM: a lens
changes which segments rank highly, never the coordinate system in which the
answer is expressed. The final refit predictor always uses the plain
SegmentMean lens, so the deliverable is a set of intervals of the physical
axis rather than a set of filter responses.

What remains unclaimed: none of these handle missingness natively, and none
outputs a stable region. They output a feature space, and a feature space is
not an answer to "which part of the spectrum should I keep".

A negative result belongs here rather than in a footnote. RSM marginalises
over lenses because the choice of lens is a nuisance parameter, and the
registered prediction was that marginalisation would raise stability. On the
synthetic controls it did not (FINDINGS sec.1): lens agreement is high, so
there is little disagreement to average away, and the ensemble sits near but
not above the best single lens. The claim was therefore narrowed to
insurance against choosing the wrong lens, which is what the controls
support. On real spectra the premise does hold: corn lens agreement is
0.205, the mean lens takes both scatter edges, the derivative lens takes
neither and selects the water bands, and the marginalised selection drops
both edges (FINDINGS sec.6).

## 10. Missing y and semi-supervision

- Chapelle, Schölkopf, Zien (2006). *Semi-Supervised Learning*. MIT Press.
- Blum, Mitchell (1998). Combining labeled and unlabeled data with
  co-training. *COLT*.

Semi-supervised learning is a declared non-aim, and the boundary is worth
stating precisely because it is easy to drift across.

RSM's rule: only labelled rows enter splits, losses, resamples and scoring.
Unlabelled rows in the TRAINING partition may optionally contribute to
X-only statistics (`use_unlabelled_stats=True`, off by default). A test-fold
row never contributes to any fitted statistic, with or without a label. RSM
never propagates labels, never trains on pseudo-labels, and counts what it
excluded (`n_unlabelled` in the result summary).

The reason for the narrow line is the leakage contract, not modesty about
semi-supervision: the moment an unlabelled test-partition row informs a
fitted statistic, the fold estimate is no longer honest, and that is the one
property the whole library exists to protect.

## 11. Threat matrix

Per claim, the nearest prior art, and why it does not cover the claim. A
reviewer should attack this table first. The rightmost column names the
experiment that would falsify the claim, and where an experiment already
went against us it says so.

| # | Claim | Nearest prior art | Why it does not cover | Evidence, and what would falsify it |
|---|---|---|---|---|
| C1 | Selection frequencies conditioned on assessability, with a per-feature denominator | Meinshausen and Bühlmann 2010; Shah and Samworth 2013 | Both use a common denominator because every feature is present in every resample; neither defines pi under differential observation | FINDINGS sec.2 scenario A. Falsified if a common-denominator pi can be shown to rank differentially observed bands as well as the conditioned one |
| C2 | Absence of evidence is never evidence of absence: three-valued verdicts, pi = nan without support | None found in selection literature | Filter, wrapper and embedded selectors all return a score or a rank for every feature; there is no "cannot assess" state to return | tests/test_selection.py canaries. Falsified by prior art returning a genuine third state, not merely a missing-value flag |
| C3 | Validity propagated through every operation, never imputed | Liu et al. 2018 (convolution only); Nelson et al. 1996, Walczak and Massart 2001 (model fitting only) | Partial convolutions stop at the convolution; the chemometrics work stops at the model. Neither carries validity into a selection head or into gate training | FINDINGS sec.9; the M1 fabricated-edge experiment. Falsified by an end-to-end masked selection pipeline in prior art |
| C4 | Rescale partial convolutions by observed filter weight, not observed count | Liu et al. 2018 | The textbook formula rescales by count, which is exact only for uniform kernels | FINDINGS sec.9 fix 1: 39% fabricated level shift in RSM, 99.9% reproduced in RPM. Falsified by showing count-based rescaling is exact for non-uniform kernels, which it is not |
| C5 | Masked computation beats impute-then-select where missingness is never-observed or MNAR | Imputation pipelines (mice, missForest, EM-PLS) | These estimate models under missingness but select on fabricated values, and imputing first destroys the MNAR signal | FINDINGS sec.2 scenario A (F = 31.8 on fabricated background) and sec.6 (corn guard: informative, then not_informative after imputation). **Scoped, not universal**: sec.2 scenario B records imputation honestly winning under ignorable missingness |
| C6 | Observation-scaled penalty and stop-gradient renormaliser stop "unobserved" masquerading as "unimportant" | Louizos et al. 2018; Yamada et al. 2020 | Both degenerate to the identity on complete data, so the problem does not arise and the corrections are absent | tests/test_gates.py gradient checks; the no-attenuation-under-MCAR test. Falsified if an unscaled penalty recovers the same bands under differential observation |
| C7 | Missingness that predicts the target is reported, not silently absorbed | MNAR literature diagnoses; no selection tool ships a guard | Diagnostics exist in the missing-data literature but are not wired into selection tools, so a selection run does not emit them | FINDINGS sec.8: the guard fired unprompted on the RRUFF archive, p = 0.0244, in 3 of 4 folds. Falsified by a selection tool that ships an equivalent automatic guard |
| C8 | Marginalising over lenses beats the best single lens on stability | Ensemble feature selection | This one was FALSIFIED on the synthetic controls (FINDINGS sec.1) and the claim was narrowed to insurance against a bad lens choice. It holds in the weaker form on real spectra (sec.6, corn agreement 0.205) | Kept in the table as a falsification, not a claim |
| C9 | Error-control guarantees under differential observation | Shah and Samworth 2013 | **Not claimed.** Their bounds assume exchangeable resampling over a fixed feature set; RSM's denominator is a per-feature random variable | Open work, stated as such. The library reports empirical support, not a theoretical error rate |

## 12. The claim

**The deliverable** is leakage-free stability selection over signal bands: a
reproducible set of intervals of the sampling axis, each carrying a per-fold
selection frequency, with a paired verdict against the full signal, and no
statistic crossing a fold boundary anywhere in its construction. That is the
same contract RMM offers for tabular columns, and it is what a user comes
for.

**The novel part of the construction**, which is what makes the deliverable
attainable on real instrument data rather than a separate claim: the
validity mask is propagated (never imputed) through lenses, pooling,
standardisation, head and gate training; selection frequencies are
conditioned on assessability with a per-feature denominator; and observation
support is a mandatory co-report, so absence of evidence is never evidence
of absence.

The order matters for how the work should be read and reviewed. The
selection contract is the product and is demonstrated on six real datasets,
every one compressing hard and returning `preserved` (README table). The
validity machinery is the mechanism, and the experiments that isolate it
(FINDINGS sec.2, sec.6, sec.8) show what goes wrong without it rather than
constituting the reason to use the library.

To our knowledge this join is unclaimed. Sections 1 to 10 carry the
per-field defence and section 11 states what would falsify each part. What
is explicitly NOT claimed: superiority over imputation under ignorable
missingness (sec.7), error-control bounds under differential observation
(sec.5, C9), a stability gain from lens marginalisation on data where lenses
agree (sec.9, C8), or any treatment that repairs MNAR rather than reporting
it (sec.7).

## 13. Positioning for AI4Science

The argument for a scientific audience is not that RSM predicts better. On
clean, strongly univariate data it often does not, and the benchmark table
says so plainly (FINDINGS sec.11 table A: mean-fill plus a simple filter is
accurate and maximally stable there). The argument is that instrument data
is not clean, and that the standard pipeline's failure mode under real
missingness is silent.

Three results carry it, all from real archives rather than controls.

**Reproducibility of the region, not just the score.** On tecator, ANOVA
selection is perfectly stable (1.000) and predictively poor (R2 0.281),
against masked RSM at stability 0.574 and R2 0.864 (FINDINGS sec.6). A
perfectly reproducible wrong answer scores 1.000 on stability alone. Any
paper that reports a selected band set owes the reader both numbers, and a
tool that reports one invites the error.

**Instrument coverage is a confound, and it is invisible after imputation.**
The RRUFF Raman archive's per-spectrum coverage pattern alone predicts the
mineral, because which laser configuration measured a specimen correlates
with what the specimen is, and diamond's diagnostic 1332 line sits in the
variably covered region. RSM's guard fired unprompted (FINDINGS sec.8). An
impute-first pipeline fills the uncovered range and the confound disappears
from view while remaining in the answer. Archive-scale reuse of
heterogeneous instrument data is exactly the direction the field is moving,
and this is the failure it will meet.

**Selected regions can be read as physics.** On gold nanoparticles the
informative bands reorder with the question asked: size at low q in S(q),
per-atom stability in the bond-angle motif window, and motif identity in the
FCC signature angles (90, 109.5, 120 degrees) and pointedly not the 60
degree band both motifs share. Five or six bands of one measured curve rival
338 hand-crafted features (FINDINGS sec.7). Selection that lands on
interpretable structure is worth more to a scientist than a marginal
accuracy gain from an opaque feature space, and it is checkable against
known physics in a way that accuracy is not.

The open case studies are the alloys spectra (loader template registered,
awaiting data) and further biomarker feeds; the registered-prediction
discipline of FINDINGS sec.3 and sec.5 applies to both.
