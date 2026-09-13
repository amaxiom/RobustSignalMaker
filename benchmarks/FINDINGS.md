# FINDINGS

Benchmark and experiment findings for RobustSignalMaker, in the RPM
discipline: negative results are recorded deliberately and unsoftened,
because they redirect milestones. Numbers here come from seeded, reproducible
runs; the settings are stated so each entry can be rerun.

Sections run in the order the experiments were done, so the record shows what
was believed when. **For the cross-dataset table, jump to sec.11**, which is
GENERATED from the result CSVs rather than typed: the 2026-09-02 audit found
eighteen stale figures in hand-copied prose, and a generated table is the
structural fix for that class of error.

## 1. The M6 marginalisation criterion was falsified on the synthetic control (2026-09-01)

The registered M6 criterion required: marginalised selection at least
as stable as the best single lens at equal coverage, including under
missingness. Measured with split-half replication (4 replicates, selected
sets compared by chance-corrected mean pairwise Jaccard over 32 segments;
control n=240, T=256, signal 2.0, noise 1.0, seed 62; selectors at segment=8,
lam=0.08, n_bootstrap=6, n_iter=120, target coverage 4/32; NaN condition:
30% scattered, seed 63):

(PRE-SWEEP NUMBERS, superseded by the addendum at the end of this section:
the NaN column was inflated by the renormalisation bug of sec.9 fix 1.)

| configuration | complete | 30% NaN |
|---|---|---|
| mean lens alone | 0.68 | 0.57 |
| savgol lens alone | 0.14 | 0.18 |
| gauss lens alone | 0.65 | 0.68 |
| ensemble (marginalised) | 0.57 | 0.64 |

The ensemble does NOT beat the best single lens in either condition. Three
things it does do, and these are what test_ensemble.py pins:

1. The identity of the best lens CHANGES with the missingness condition
   (mean is best complete, gauss best under NaN), and nothing in advance
   says which. Marginalisation is near-best in both conditions without that
   oracle.
2. The worst lens (savgol derivative energy on this piecewise-smooth
   control) would alone produce junk (0.14 to 0.18); a lens-specific
   analysis is the real hazard the marginalisation guards against.
3. The ensemble always beats the average lens, i.e. picking a lens at
   random loses to integrating them out.

Interpretation, honestly: on a control where most lenses agree about a
strong band, marginalisation buys insurance, not extra stability. The
premise "marginalisation increases stability" needs lens DISAGREEMENT to
have something to average away, which RPM's real-image benchmarks had
(agreement 0.36 to 0.45) and this clean control largely lacks. The claim
must be re-tested on real spectra in M8 before the paper leans on it; the
insurance framing is what the method section should promise meanwhile.

Watch-item for M8: the savgol lens's poor showing here is partly the
control's piecewise-smooth construction (bump edges carry little
class-separating derivative energy relative to noise). On sharp-peaked
domains (XRD, MS) it should behave very differently; if it does not, the
default ensemble composition should be revisited.

**Post-sweep addendum (2026-09-01, after sec.9 fix 1).** Re-measured with
the corrected renormalised convolution: complete-data row unchanged (mean
0.68, savgol 0.14, gauss 0.65, ensemble 0.57); the 30% NaN row becomes
mean 0.57, savgol 0.19, gauss 0.52, ensemble 0.52. The original "gauss is
best under NaN" was PARTLY AN ARTEFACT OF THE BUG this table helped to
hide: the count-based renormalisation inflated Gaussian-lens features near
gaps. Point 1 above (the best lens's identity changes with the condition)
no longer holds on this control; points 2 and 3 (the ensemble beats the
average lens and insures against the worst) survive the fix, and the
insurance framing remains the claim.

## 2. The M7 novelty experiment: where masked computation wins, and where imputation honestly does not lose (2026-09-01)

Three scenarios (all seeded; settings in tests/test_missingness.py, which
pins every claim below).

**A. A band nobody ever observed** (bump on points 64..96 drives y; points
88..120, segments 11..14, missing for every sample):

- The masked pipeline refuses: verdicts unassessable, pi nan, selection
  confined to the observed truth segments.
- The standard pipeline (linear interpolation, then a univariate F screen)
  confidently mis-selects fabrication: interpolated segment 12, PURE
  background nobody measured, scores F = 31.8 against a genuine-background
  maximum of 3.3, and fabricated segments sit in the screen's top four.
  Interpolation ramps from the bump edge launder amplitude into the gap.
- Two findings in RSM's favour beyond the plan: our own gated selector
  RESISTS the same fabrication (pi = 0 on segments 12..14 even on
  interpolated data), because the frozen-head phase prunes segments that
  are redundant with the real band; and mean-filled constants trip the
  degenerate-scale guard, so even a user who imputes first gets
  "unassessable" rather than a number there.

**B. Ignorable group-band missingness** (half the samples lack points
64..80, two-thirds of the truth band; truth = segments 8, 9, 10):

- Truth recovery (full fit): masked F1 1.00, mean-impute 1.00,
  interpolate 0.50. Split-half stability (3 replicates, n=120 halves):
  masked 0.23, mean-impute 0.40, interpolate 0.13.
- The plan's blanket criterion "masked beats impute-then-select on recovery
  AND stability" is therefore HALF TRUE and recorded as such: interpolation
  loses on both counts, but mean-impute ties recovery and is MORE stable.
  When missingness is ignorable (independent of y), mean imputation is a
  legitimate variance reduction and the renormalised head pays a real
  variance price, exactly as the design analysis predicted (variance
  inflation by 1/rho). The masked method's case does not rest on this
  regime; it rests on A and C, where imputation fabricates or launders.

**C. Non-ignorable (MNAR) missingness** (class 1 never observes segments
20..23; no signal anywhere):

- Masked pipeline: MissingnessInformativeWarning, verdict "informative",
  top associated segments exactly 20..23 (permutation p about 0.02).
- After interpolation the validity pattern no longer exists, the guard sees
  a fully observed dataset, and the verdict is "not_informative".
  IMPUTATION DESTROYS THE ALARM: the pipeline that fabricates values also
  deletes the evidence that a warning was needed. This is the sharpest
  argument for validity as a first-class citizen and should lead the paper's
  motivation section.

## 3. M8 synthetic benchmark: predictions registered BEFORE the run (2026-09-01)

Setup: four domain controls (raman/dropout, xrd/saturation-censoring,
ms/zeros-as-missing, sensor/band-by-group), ten selectors from the registry,
equal selection size k = |truth segments|, split-half stability over 3
replicates, uniform downstream scorer. Predictions, written before any
result existed:

P1. rsm_masked recovers truth (F1 at least 0.8) on all four domains.
P2. On raman (ignorable dropout), mean-fill competitors match rsm_masked on
    F1, and rsm_mean_fill matches or beats rsm_masked on stability (the M7
    ignorable-regime trade).
P3. On xrd, where censoring clips exactly the informative reflections,
    rsm_masked beats every *_mean_fill and *_interp_fill on F1: filled
    constants at the clipped peaks attenuate the contrast the competitors
    see, while masked computation uses the surviving flanks.
P4. rf_mean_fill is the strongest non-RSM competitor overall (the RPM
    finding that Random Forest, not the filter methods, is the one to beat).
P5. ipls_mean_fill and mcuve_mean_fill do not beat rsm_masked on F1 on any
    domain.
P6. On ms (spike-like truth), anova_mean_fill is competitive with rsm_masked
    on F1 (localised univariate signal is the filter methods' best case).
P7. The threshold sensitivity sweep finds plateaus: F1 and stability move by
    less than 0.1 as each evidence threshold varies around its default on
    the raman control (defaults confirmed rather than revisited).

Results follow in sec.4 after the run, unedited.

## 4. M8 synthetic benchmark: results and the prediction scorecard (2026-09-01)

Full tables in benchmarks/results/benchmark_*.csv (F1 at equal selection
size, split-half chance-corrected stability over 3 replicates, uniform
downstream AUC). Run 1 exposed two fixture/configuration defects, fixed
before the final run and recorded: the xrd truth labelling was too narrow
(censoring clips the peak TIPS, so amplitude information genuinely lives in
the shoulders; selectors choosing shoulders scored F1 0 while their
downstream AUC beat the "truth" selections, 0.979 vs 0.854), and the MS
control refused everything under dense-signal evidence floors (a sample
observes about 4% of a sparse spike axis; the loader now DECLARES its
evidence scale, min_valid_frac 0.02 and rho_min 0.01, as explicit
selector_config; never a silent heuristic).

Headline numbers (F1 / stability), final run:

| method | raman | xrd | ms | sensor |
|---|---|---|---|---|
| rsm_masked | 1.00 / 0.86 | 0.75 / 0.52 | 0.86 / 0.71 | 0.88 / 0.39 |
| rsm_mean_fill | 0.75 / 0.61 | 1.00 / 0.69 | 1.00 / 1.00 | 0.93 / 0.75 |
| anova_mean_fill | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| rf_mean_fill | 1.00 / 1.00 | 1.00 / 0.72 | 1.00 / 1.00 | 1.00 / 1.00 |
| ipls_mean_fill | 1.00 / 1.00 | 1.00 / 0.72 | 1.00 / 1.00 | 1.00 / 0.83 |
| mcuve_mean_fill | 0.50 / 0.15 | 0.50 / 0.59 | 1.00 / 1.00 | 0.25 / 0.03 |

Scorecard against the section-3 predictions:

- P1 (masked F1 >= 0.8 everywhere): MOSTLY held (1.00 / 0.86 / 0.88), missed
  on xrd (0.75).
- P2 (raman: fills match masked): WRONG IN RSM'S FAVOUR. Under dropout
  stretches masked beat its own mean-fill variant on BOTH accuracy (1.00 vs
  0.75) and stability (0.86 vs 0.61): filling a dropout stretch with column
  means injects noise exactly where the lens pools.
- P3 (xrd: masked beats fills): FALSIFIED, and the reason matters. When
  censoring clips a SMOOTH peak's tip, the flanks constrain the missing
  values, so interpolation is legitimate reconstruction rather than
  fabrication, and the filled pipelines (F1 1.00) beat masked (0.75).
  Fabrication is harmful when the missing region is UNCONSTRAINED by its
  neighbours (the M7 scenario-A gap); this control's censoring is the
  benign kind. The claim is sharpened, not defended.
- P4 (RF the strongest non-RSM): NOT QUITE; anova ties or beats RF
  everywhere here. These controls are strongly univariate, the filter
  methods' best case; the RPM finding (RF beats filters) needs real data to
  re-test.
- P5 (ipls/mcuve never beat masked on F1): FALSIFIED for ipls (1.00 on
  three domains); held for mcuve except on ms.
- P6 (ms: anova competitive): held (1.00).
- P7 (threshold plateaus): SUPERSEDED. As originally run the sweep could
  not have failed for two of the five parameters, so its "defaults
  confirmed" was partly vacuous. Rewritten and re-run; the honest result is
  sec.10.

What these controls establish, honestly: on easy, strongly univariate
synthetic domains with ignorable missingness, mean-fill plus a simple filter
is accurate and maximally stable, and masked RSM is competitive but pays its
variance price; masked RSM wins outright where the missingness itself
corrupts filled pipelines (dropout stretches through a pooling lens). The
method's decisive advantages remain the M7 regimes (never-observed bands,
MNAR, alarm survival) and the honesty machinery (verdicts, support,
refusals), which no competitor here provides at all. The real-data test
(alloys, biomarker feeds) is the open item: these clean controls are the
filter methods' home turf, and lens disagreement plus non-univariate
structure is where the RPM experience says the ordering changes.

## 5. Real datasets: predictions registered BEFORE the run (2026-09-01)

Three public datasets wired (loaders in datasets.py; caches under
benchmarks/data/, out of git; results as benchmark_<name>.csv):

- tecator: 240 meat NIR spectra, 100 channels, fat regression (OpenML 505).
- corn: 80 NIR spectra x 700 channels, instrument m5, moisture regression
  (Eigenvector; the three-instrument stack is reserved for the examples).
- ovarian: 216 FDA-NCI post-QAQC SELDI-TOF serum spectra (121 cancer, 95
  normal), per-sample m/z axes binned to 4096 common bins over 700 to
  12000 Da (Internet Archive copy of the retired NCI-FDA site). Known
  batch/calibration caveats; any selected band is a candidate artefact
  detector as much as a biomarker.

No ground-truth bands exist, so the metrics are stability, downstream score
(uniform scorer: ridge R2 / logistic AUC), and selection size at the
declared budget k. Predictions, written before any result existed:

P8.  On real correlated spectra the filter methods lose their synthetic-
     control perfection: anova split-half stability drops below 1.0 on all
     three datasets, and below 0.6 on corn (700 highly collinear channels).
P9.  rsm_masked stability is within 0.15 of the best method on tecator and
     ovarian, and strictly ABOVE anova on corn.
P10. Downstream score of rsm_masked's selection is within 0.05 of the
     full-signal score on tecator and ovarian (the preserved property at
     k/S coverage of 20% or less); corn moisture, a famously hard
     700-channel problem at n=80, may lose more and is allowed to.
P11. On corn, the selected moisture bands include at least one segment
     covering the water-band channels (160 to 190 or 410 to 430; segments
     8 to 9 or 20 to 21 at segment=20).
P12. iPLS is the strongest competitor on the two NIR sets (its home
     literature), beating the other filled methods on downstream score or
     stability at least once.
P13. On ovarian, every method reaches AUC 0.9 or better at k=6 of 64
     segments (the dataset's notorious separability), so the DISCRIMINATING
     row is stability, where rsm_masked places in the top three.

Results follow in sec.6, unedited.

## 6. Real datasets: results, scorecard, and two findings that matter (2026-09-01)

Full tables in benchmarks/results/benchmark_{tecator,corn,ovarian}.csv.
Headline (stability / downstream, uniform scorer; full-signal reference in
brackets):

- tecator [R2 0.897]: rsm_masked 0.57 / 0.864 at 7 of 20 segments;
  lasso 0.70 / 0.883; anova 1.00 / 0.281; ipls 0.70 / 0.776.
- corn m5 moisture [R2 0.728]: rsm_masked 0.28 / 0.432 at 3 segments;
  anova 0.55 / 0.429; mcuve 0.10 / 0.627; nobody preserves the full
  signal with 5 of 35 segments at n=80 x 700 collinear channels.
- ovarian [AUC 0.993]: rsm_masked 0.45 / 0.971 at 6 of 64 segments;
  lasso 0.55 / 0.991; anova 0.55 / 0.853; rf 0.17 / 0.887.

Scorecard: P8 falsified on tecator, held on corn and ovarian. P9 falsified
on tecator and corn, held on ovarian. P10 held everywhere (tecator within
0.033 of full at 35% coverage, ovarian within 0.022 at 9%; corn was allowed
to lose more and did). P11 falsified FOR THE MEAN LENS and vindicated by the
derivative lens (below). P12 falsified (lasso strongest on tecator, MC-UVE's
lucky-unstable 0.627 on corn). P13 FALSIFIED on both halves (not every
method clears AUC 0.9, and rsm_masked is not top three on stability: the two
anova variants and lasso all sit at 0.549 against its 0.452, so it places
fourth at best). Corrected 2026-09-02 after an audit caught the original
"half held" softening a falsification.

**Finding A: stability and predictive value dissociate on real spectra.**
Tecator anova is PERFECTLY stable (1.00) and predictively poor (R2 0.281):
it reproducibly selects the same redundant, weakly-predictive band cluster.
Stability alone must never rank methods; the joint report (stability AND
score AND coverage) is the point, and any paper table needs both columns.
Related sanity check that passed: on complete data the masked and
impute-then-select RSM variants produce IDENTICAL rows on all three
datasets, confirming the no-fabrication machinery reduces exactly when
there is nothing to fabricate.

**Finding B: the M6 re-test arrived, and real spectra deliver the lens
disagreement the synthetic controls lacked.** On corn moisture the measured
lens agreement is 0.205 at the notebook settings (3 lenses, 25 resamples,
the number the executed notebook prints) and 0.0 in a 2-lens 10-resample
probe: far below the synthetic controls either way, and below RPM's
real-image range of 0.36 to 0.45. (An earlier draft of this section quoted
0.27, which no configuration reproduces; corrected 2026-09-02.) At the
notebook's m5 settings the mean lens takes BOTH spectrum edges, 1100 to 1138
and 2460 to 2498 nm, the classic multiplicative-scatter signature of
uncorrected NIR, plus 1340 to 1418 and 1860 to 1898; the Savitzky-Golay
derivative lens takes neither edge and instead selects 1420 to 1458, 1900 to
1978 and 2020 to 2098 nm, the water bands. And the MARGINALISED selection
lands on segments 6, 7, 8, 19 and 24 (1340 to 1458, 1860 to 1898 and 2060 to
2098 nm): the ensemble, given both lenses, puts four of its five bands on
the water-band chemistry and drops both scatter edges. On the synthetic
controls (sec.1) agreement was high and marginalisation bought only
insurance; on real spectra it is the
difference between reporting scatter artefacts and reporting water bands.
The corn example notebook carries this story; scatter correction in
chemometrics is a lens decision, not neutral preprocessing.

(Corrected 2026-09-02, second pass. The mean-lens sentence above previously
read "the mean lens selects the spectrum EDGES (1100 to 1138 and 2420 to
2498 nm)", which was wrong twice over. Its far-edge band is ONE segment,
2460 to 2498; the wider "2420 to 2458 and 2460 to 2498" belongs to the
GROUPED nested run, whose set is 1380 to 1418, 1860 to 1898, 2420 to 2458
and 2460 to 2498 and contains no 1100 nm segment at all. So the sentence had
spliced two different runs, and calling the mean lens edges-only also
suppressed its two mid-spectrum bands. The finding survives in the direction
that carries it: the mean lens takes both edges, the derivative lens takes
neither, and the marginalisation drops both. The notebook's own narrative
carried the same overstatement and is corrected too.)

Also noted: coverage targeting hits the nearest ACHIEVABLE coverage on the
pi grid (tecator: 7 of 20 against a 4-of-20 target), worth remembering when
comparing at "equal" selection size; and the ovarian caveat stands (its
notorious separability and batch artefacts mean AUC 0.97 at 9% coverage is
a claim about this dataset, not about ovarian cancer).

## 7. Gold nanoparticles: three characterisations, three questions (2026-09-01)

The user supplied 4000 Au nanoparticles (examples/gold_nanoparticles/):
g(r) over 0.1 to 50 A (500 points), bond-angle distribution (180 bins),
S(q) over 0.12 to 31.5 inverse A (512 points), 338 tabular features, and
total formation energy as the label (one particle unlabelled). Exploration
at a five-band budget per characterisation, three targets: E (total),
E/N, and the residual of E/N after the N^(-1/3) surface-fraction trend
(excess stability beyond size). References: N alone explains R2 0.82 of E
and 0.00 of the residual; the 338 tabular features explain 0.98 of E and
0.77 of the residual, so the residual is real structure signal.

Selected-band R2 at five (or six) bands, per characterisation x target:

| | E_total | E_per_atom | E_residual |
|---|---|---|---|
| g(r), 2 A bands | 0.82 | 0.51 | 0.39 |
| angles, 10 deg bands | 0.60 | 0.84 | 0.73 |
| S(q), 0.9 invA bands | 0.91 | 0.86 | 0.80 |

Findings:

1. **The modality ladder reorders with the physics asked.** S(q) wins E,
   selecting four of its five bands in the LOW-q form-factor region (q below
   4: size), plus one at q about 7. The angle distribution fails on E (0.60)
   because angles are size-blind, then CLIMBS to near-parity on per-atom
   stability (0.84 against S(q)'s 0.86, having been 0.31 behind on E), and
   on the size-corrected residual S(q) and angles both approach the full
   tabular feature set (0.80 and 0.73 against 0.77) from five or six bands
   of one measured curve. (Corrected 2026-09-02: this previously called the
   angle distribution the best per-atom predictor, which its own table
   contradicts; the finding is the reordering, not a new winner.)
2. **The selected bands are legible physics.** Residual-stability angle
   window: 41 to 90 degrees, covering the FCC 60 and 90 degree signatures
   and the 63 degree five-fold twin angle (the 72 degree twin angle falls
   just outside, in an unselected segment). g(r)
   residual bands: medium-range order (4 to 10, 16 to 20 A), NOT the first
   shell near 2.9 A that all gold shares. S(q) per-atom bands include the
   Au Bragg (111) region near q = 2.7.
3. **Honesty machinery exercised by real data, unprompted**: angles below
   31 degrees do not occur, and those segments report "unassessable"
   rather than "rejected"; the one unlabelled particle is excluded and
   counted by NestedCV.
4. **Nested verdict for the headline case** (angles predicting residual
   stability): `preserved` at 22% coverage with per-fold selection
   stability 1.0 (the SAME four angle bands in all five folds); the
   Wilcoxon sits exactly at the K=5 floor (0.0625) with the full signal
   slightly ahead each fold (RMSE 0.0297 vs 0.0279), so the honest reading
   is "nothing significant lost", with the K=5 caveat stated.
5. **Lens agreement is 0.25 on the bond-angle distribution** for the
   residual question (the executed notebook's measurement, 3 lenses at 10
   resamples): firmly in the lens-dependent regime, consistent with corn
   (sec.6) and unlike the synthetic controls. An earlier draft claimed
   "0.17 to 0.20 on all three characterisations"; the notebook measures
   agreement for the angle curve only, so the g(r) and S(q) figures never
   existed. Corrected 2026-09-02.

Fixed in passing: RSMResult.summary() displayed the method score as
positive RMSE but the baseline in the internal negative convention; both
now display alike.

**Motif classification addendum (2026-09-01).** A FCC-vs-twinned label was
derived transparently from the q6q6 bond-order features (strongly bimodal
interior crystallinity, mid-valley threshold at 6 connections per interior
atom, threshold-robust: the FCC fraction moves 0.70 to 0.74 as the
threshold sweeps 4 to 8; 2884 FCC vs 1116 twinned/disordered; the label
channel is coordinate-space analysis, independent of the three curves, so
no circularity). Results: the angle distribution separates the motifs
PERFECTLY (AUC 1.000) from four 10-degree bands at 81 to 90, 101 to 110,
121 to 130 and 151 to 160 degrees, i.e. the FCC 90 / tetrahedral 109.5 /
120 signatures, and pointedly NOT the 60-degree band both motifs share;
S(q) reaches 0.999 from the Bragg region (q 3 to 9) while IGNORING the
low-q size information it used for E_total (motif is size-independent and
the selection reflects that unprompted); g(r) manages 0.965 from
medium-range order. Nested: AUC 0.9999, verdict preserved, the same four
angle bands in every fold.

## 8. RRUFF Raman and BasicMotions: two more real datasets (2026-09-01)

**RRUFF mineral identification** (180 processed excellent-unoriented
spectra, six minerals, 80 specimens, four laser configurations, binned to
512 bins over 150 to 1400 inverse cm; loader `rruff`). Three results:

1. Eight selected bands land on the textbook fingerprints (diamond 1332,
   calcite 1086, the garnet and epidote bands below 1030; beryl's 1067 sits
   in the ADJACENT unselected segment, and its 686 line is not selected
   either). Under specimen-grouped nested CV they hold AUC-OVR 0.9947
   against the full spectrum's 0.9935, verdict `preserved` at an eight-fold
   compression. A quick ungrouped comparison in the notebook shows 0.999 vs
   0.989, but those bands were chosen using all the labels, so that number
   is optimistic and is NOT the claim (corrected 2026-09-02: this section
   previously led with it, contradicting sec.9's own record of fixing the
   same overclaim in the notebook).
2. THE MNAR GUARD FIRED ON A REAL ARCHIVE, UNPROMPTED: the per-spectrum
   coverage pattern alone predicts the mineral (permutation p 0.024, in 3
   of 4 folds), pointing at 1300 to 1400 inverse cm, where some instrument
   configurations stop early, and diamond's own diagnostic band lives in
   exactly that region. Which instrument measured a specimen correlates
   with what the specimen is: an archive-composition confound that any
   impute-first pipeline would have silently erased. This is the
   strongest single real-data vindication of the honesty machinery so far.
3. Specimen-grouped nested CV: AUC 0.995, verdict preserved, with the
   missingness_informative stamp carried in the summary.

**BasicMotions** (80 recordings, 6 channels x 100 steps, four activities;
loader `basicmotions`): the first real multi-channel joint-gating result.
Three mid-recording windows (steps 30 to 60, segments 3 to 5 at pi 0.84,
0.96 and 1.00) carry the activity signature, and imposing gyroscope dropout
for half the recordings changes the selected windows NOT AT ALL (identical
set at full support), the per-channel validity weighting absorbing the loss
below the shared gate, exactly as the M3 head algebra was designed to do.
Two separate measurements, kept apart (corrected 2026-09-02, second pass:
they were previously quoted as one): on a single split the three windows
score AUC-OVR 0.970 against the whole series' 0.975, while the four-fold
nested run gives 0.921 against 0.965, verdict `preserved`. The nested pair
is the honest one and the wider gap is what small folds cost; the earlier
text put the single-split pair next to the nested verdict, which reads as a
tighter result than the nested run actually delivers.

**Benchmark-runner rows for the two new datasets (2026-09-01, full rerun;
benchmark_rruff.csv, benchmark_basicmotions.csv).** The runner's uniform
scorer now covers multiclass (OVR AUC). rruff at k=8 of 64: rsm_masked
reaches AUC 0.994 at 10 segments (coverage targeting hit the nearest
achievable point) with split-half stability 0.48; lasso_mean_fill scores
0.999, anova_interp_fill is the stability leader (0.84 at AUC 0.982). Two
caveats the runner cannot see and the notebook carries: the runner does not
group by specimen (its replicates can split a specimen's spectra), and the
archive's coverage-vs-class confound means every filled competitor is
potentially exploiting laundered range information; the honest comparison
lives in the notebook's specimen-grouped, MNAR-stamped run. basicmotions at
n=80 is small-sample territory: every selector is unstable (best stability
0.15) and no selection beats the full signal on the joint of both columns,
though two do beat it on score alone (lasso 0.874 and mcuve 0.853 against
0.851; corrected 2026-09-02 from "the full signal wins outright"). The
joint-gating story, which is what the dataset is in the registry for, is a
notebook result, not a benchmark-table one.

## 9. The deep bug sweep (2026-09-01)

Three parallel adversarial reviews (numerical core; selection and CV layer;
harness, tests and notebooks) plus direct probing, every finding reproduced
before being fixed, in the MissLearn six-pass tradition. Twenty-eight
findings; the ones that mattered most:

1. WRONG-NUMBER SILENT, the sweep's worst: the renormalised convolution
   rescaled by observed COUNT, exact only for uniform kernels; a Gaussian
   level filter fabricated up to +39% level shift on a FLAT signal near a
   gap, trusted at validity 0.87. Fixed with filter-weighted evidence
   normalisation (exact for any kernel; level branch normalises by observed
   filter weight, contrast branch by observed absolute filter mass), a
   deliberate divergence from the RPM maskfill construction. Every
   GaussianScale-lens number under missingness changed slightly; results
   were re-run after the fix.
2. Payload leaks in the PUBLIC primitives: nan/inf payloads at masked
   positions reached arithmetic in masked_mean/std/standardise_fit,
   renormalised_convolve1d and renormalised_dot, returning nan WITH
   valid=True (the internal lens path was protected; the exported functions
   were not). All primitives now zero payloads before any arithmetic, and
   the payload tests cover nan/inf, not just finite values.
3. Guards failing toward confidence: paired_comparison returned "preserved"
   with nan fold scores (now pairwise exclusion and a fourth outcome,
   "undecidable"); renormalised_dot returned a confident 0.0 for a sample
   that observed nothing at rho_min=0; a_min=0 turned never-assessed
   segments into "rejected"; lam_for_coverage let a nan coverage displace a
   within-tolerance hit.
4. Engine crashes and false significance: multiclass with grouped CV
   crashed the out-of-fold accumulator when a class was group-confined
   (fold probabilities are now aligned onto the engine's class list);
   repeated_outer_cv with deterministic GroupKFold duplicated folds and
   manufactured Wilcoxon significance out of pseudo-replication (forced to
   one pass, loudly).
5. Silent contract violations: the fold adapters dropped caller-supplied
   groups instead of stratifying resamples by them; the engine's
   chance-corrected stability used the raw segment count as its universe
   (the metrics contract says assessable only); n_jobs > 1 lost all fit
   error diagnostics; ensemble (lens, resample) fits shared seeds whenever
   b + r matched.
6. Harness and reporting: exported CSVs were malformed by comma-bearing
   cells (both writers now use proper quoting); truth_segments dropped
   spike truth whenever band truth coexisted (per-run rule now); the
   sensitivity rows for a_min and o_floor were vacuous on the raman control
   because support never leaves 1.0 there (the sweep was rebuilt around
   bindability probes; sec.10 has the honest result, including that a_min
   cannot bind at all); string class labels crashed
   predict; the Raman notebook quoted a selection-biased comparison as its
   headline (the specimen-grouped nested numbers are the claim now); the
   gold nanoparticle notebook had a clobbered section header and a stale
   duplicate summary.

Scope decisions measured and documented rather than "fixed": fitted
coefficients are not bitwise row-order invariant (BLAS reduction order);
the mask and the selected set are, and the reproducibility contract is
stated as same seed AND same row order. Hard-Concrete test-time gates are
exactly 0/1 only once training saturates them; the docstrings now say so.

All benchmarks were re-run and all notebooks re-executed after these fixes.
Every number in this file reflects the post-sweep code EXCEPT the sec.1
table, which is deliberately kept as the pre-sweep record with its
correction in the addendum beneath it. A follow-up audit on 2026-09-02 then
caught nine stale or overstated figures elsewhere in this file, each now
corrected in place and marked, and the 2026-09-02 verification rerun then
caught TWO MORE that the audit had missed, both of the same kind: a claim
built by splicing numbers from two different runs (the corn mean-lens bands
in sec.6, and the BasicMotions single-split AUC pair quoted beside the
nested verdict in sec.8). Eleven in this file altogether. The lesson is that
prose which quotes a number from a run it does not name is the defect, not
any particular number, which is why the cross-dataset table in sec.11 is
generated and why both new corrections name their run explicitly; sec.10
documents the worst of them, which was
a claim of mine that the audit refuted outright.

## 10. Threshold sensitivity, done so it could fail (2026-09-01, corrected 2026-09-02)

This section was wrong twice before it was right, and the way it was wrong
is the finding worth keeping.

**v1** swept all five evidence thresholds on the raman control and reported
flat lines as "defaults confirmed". Two of the five could not act on that
control at all, so those confirmations could not have failed.

**v2** moved a_min and o_floor to the sensor control and concluded, in this
file, that "a_min cannot bind, and that is a fact about the design". That
conclusion was FALSE, and its stated mechanism ("the observation rate
concentrates tightly") was also false. An audit refuted it and the
refutation reproduces here: with observation patterns too varied for
`observation_groups` to group them, resample support takes intermediate
values and a_min changes the answer outright. The real mechanism is
sharper and is now documented in the sweep script: when the patterns DO
form detectable groups, `_strata` stratifies by class x observation group
and `resample_indices` draws exactly `len(idx)` rows per stratum, so the
observer count per resample has ZERO variance by construction and support is
identically 1.0. v2 had unwittingly built every one of its controls in that
regime. Two further v2 verdicts were artefacts: its `try` wrapped the
split-half replicates as well as the main fit, so a replicate crash was
recorded as the row's outcome, and its a_min probe (1.0) could not fail
because the test is `support >= a_min` with support identically 1.0.

**v3**, the current sweep, is falsifiable by construction: every parameter
has a DRIVER (the quantity it is compared against inside the library), the
driver is measured and reported, swept values are chosen to straddle it, the
main fit and the replicates have separate error handling, `target_coverage`
is off so a penalty change is free to move the answer, and warnings are not
silenced. Results (benchmarks/results/threshold_sensitivity.csv):

Each parameter is swept on the control where it can actually act, so the
control is named in every row and an F1 in one row is NOT comparable with an
F1 in another (added 2026-09-02: the CSV now carries a `control` column for
exactly this reason, because the table previously invited that comparison).

| parameter | control | driver (measured) | swept | verdict |
|---|---|---|---|---|
| min_valid_frac | partial_validity | 0.625 low-tail segment validity | 0.3 / 0.6 / 0.9 | inert on this control, probe 0.999 included |
| rho_min | raman | 0.883 min per-sample evidence | 0.1 / 0.5 / 0.9 | sensitive: 0.9 changes the set, 0.999 refuses |
| o_min | raman | 0.912 min per-segment observation | 0.1 / 0.5 / 0.9 | sensitive: 0.9 drops truth segments (F1 0.8 to 0.5) |
| o_floor | rare_band(rare=0.25) | 0.250 truth-segment observation | 0.1 / 0.25 / 0.5 | genuine plateau, probe 0.95 moves it |
| a_min | ungrouped_rare(share=0.12) | 0.600 intermediate resample support | 0.3 / 0.6 / 0.8 | sensitive: 0.3 keeps [9,10], 0.6 keeps [10], 0.8 refuses |

What this says about the defaults, honestly:

1. **rho_min and o_min are live and the defaults are deliberately loose.**
   Both bite hard when raised toward the data's own evidence level:
   o_min = 0.9 discards the two truth segments and halves F1. The defaults
   (0.1) sit far from that, which is the right side to err on, and the sweep
   now shows what over-tightening costs.
2. **o_floor is a genuine plateau** across 0.1 to 0.5 on a control where an
   informative band is observed by only a quarter of samples, with the probe
   at 0.95 pruning it. The default 0.2 is safe and the failure mode is
   visible.
3. **a_min is live in the ungrouped regime and inert in the grouped one**,
   and which regime you are in is decided by whether `observation_groups`
   can group your missingness patterns. That is now the documented
   description: not "a tuned threshold" and not "a fact about the design"
   that it cannot bind, but a threshold whose relevance is conditional on
   the missingness structure, and one that refuses loudly (0.8 and above on
   the test control) rather than silently.
4. **min_valid_frac did not move the selection on a control whose validity
   low tail it straddles.** With 22% of features partially observed and the
   threshold sweeping across their range, the selection was identical at
   0.3, 0.6, 0.9 and 0.999. The renormalised head absorbing eligibility
   changes is the plausible reason (dropping a partially observed feature
   only reweights evidence the head was already discounting), but this is
   reported as measured, not explained: the earlier claim of a "cliff at
   0.7" came from a run with `target_coverage` on, where the coverage
   targeting, not the threshold, moved the answer.

The transferable lesson, and the reason this section exists in three
versions: a sensitivity analysis must first prove the knob is connected, and
then prove the connection test itself could have failed. Sweeping a
parameter over a regime where it cannot act, and reading the flat line as
reassurance, is the same failure as reporting a stability figure with no
coverage beside it. Getting this wrong twice in one day, in a library whose
entire thesis is honest reporting, is the sharpest available reminder that
the discipline has to be applied to one's own claims too.

## 11. Consolidated results table (generated from the result CSVs)

Everything below the sentinel is written by
`py benchmarks/make_findings_table.py --write`, reading
`benchmarks/results/benchmark_*.csv` and `run_meta.csv`. Re-run it after any
benchmark run; do not hand-edit inside the sentinels. The generator refuses
to build a table whose CSVs are more than an hour apart from the run
metadata, so a table can never quietly mix two runs.

How to read it, and what it does NOT say:

- The two tables use different primary metrics because the two dataset
  families genuinely differ. Synthetic controls have planted truth, so F1 is
  meaningful. Real spectra have none, so their primary metric is split-half
  stability, and any F1 shown for them would be an invented truth set.
- **A high stability figure is not a quality claim.** Sec.6 records the
  clearest counter-example in this project: on tecator, ANOVA reaches
  stability 1.00 while its downstream R2 is 0.281, against 0.864 for masked
  RSM at stability 0.574. A perfectly reproducible wrong answer scores 1.00.
  That is why table B marks NOTHING as best: its primary metric is
  stability, and bolding a stability column would typographically recommend
  a method the prose warns against. Table A does mark a best, because with
  planted truth an F1 means what it says. Read both numbers in every cell.
- `full_signal` retains every segment. It is the no-compression reference
  for the downstream column, not a competitor, so it is excluded from table
  A's best-marking; its stability is 1.00 by construction.
- These are FLAT single-split numbers at a fixed budget, produced by a
  uniform scorer. They are deliberately not the honesty machinery: grouped
  CV, refusal accounting, three-valued verdicts and the MNAR guard cannot
  appear in a table of this shape, and the notebooks in `examples/` are
  where they are exercised. A reader who only reads sec.11 will conclude
  that a mean-filled filter is usually competitive, which is true on this
  table and is not the paper's claim (see sec.2 and sec.7).

<!-- BEGIN GENERATED TABLE (make_findings_table.py) -->

Generated by `benchmarks/make_findings_table.py` from the result CSVs of the run finishing 2026-09-02 16:42 AUS Eastern Standard Time. Do not edit inside the sentinels; re-run the generator instead.

**A. Synthetic controls (planted truth).** Primary metric is truth-recovery F1 at equal selection size; split-half chance-corrected stability over 3 replicates beside it.

| method | raman | xrd | ms | sensor |
|---|---|---|---|---|
| full_signal | 0.222 / 1.000 | 0.118 / 1.000 | 0.171 / 1.000 | 0.400 / 1.000 |
| rsm_masked | **1.000 / 0.856** | 0.750 / 0.516 | 0.857 / 0.714 | 0.875 / 0.392 |
| rsm_mean_fill | 0.750 / 0.605 | **1.000 / 0.691** | **1.000 / 1.000** | 0.933 / 0.748 |
| rsm_interp_fill | 0.667 / 0.325 | **1.000 / 1.000** | 0.857 / 0.589 | **1.000 / 0.822** |
| anova_mean_fill | **1.000 / 1.000** | **1.000 / 1.000** | **1.000 / 1.000** | **1.000 / 1.000** |
| anova_interp_fill | **1.000 / 1.000** | **1.000 / 1.000** | 0.667 / 1.000 | **1.000 / 1.000** |
| lasso_mean_fill | 0.750 / 0.476 | 0.750 / 0.587 | **1.000 / 1.000** | 0.875 / 0.229 |
| rf_mean_fill | **1.000 / 1.000** | **1.000 / 0.724** | **1.000 / 1.000** | **1.000 / 1.000** |
| ipls_mean_fill | **1.000 / 1.000** | **1.000 / 0.724** | **1.000 / 1.000** | **1.000 / 0.827** |
| mcuve_mean_fill | 0.500 / 0.150 | 0.500 / 0.587 | **1.000 / 1.000** | 0.250 / 0.034 |
| _budget_ | k=4 of 32 | k=4 of 64 | k=3 of 32 | k=8 of 32 |

Cells are F1 / stability. Bold marks the best F1 in the column among the selectors; full_signal is excluded from the marking because it retains every segment, making it the no-compression reference rather than a competitor.

**B. Real datasets (no ground truth).** No truth set exists, so the primary metric is split-half stability, with the uniform downstream score beside it (ridge R2 for tecator and corn, logistic or OVR AUC for the rest, computed on the selected segment means of mean-filled data so the comparison is between selections and not between predictors).

| method | tecator | corn | ovarian | rruff | basicmotions |
|---|---|---|---|---|---|
| full_signal | 1.000 / 0.897 | 1.000 / 0.728 | 1.000 / 0.993 | 1.000 / 0.993 | 1.000 / 0.851 |
| rsm_masked | 0.574 / 0.864 | 0.283 / 0.432 | 0.452 / 0.971 | 0.480 / 0.994 | 0.150 / 0.684 |
| rsm_mean_fill | 0.574 / 0.864 | 0.283 / 0.432 | 0.452 / 0.971 | 0.250 / 0.993 | 0.150 / 0.684 |
| rsm_interp_fill | 0.574 / 0.864 | 0.283 / 0.432 | 0.452 / 0.971 | 0.461 / 0.993 | 0.150 / 0.684 |
| anova_mean_fill | 1.000 / 0.281 | 0.553 / 0.429 | 0.549 / 0.853 | 0.635 / 0.957 | 0.150 / 0.786 |
| anova_interp_fill | 1.000 / 0.281 | 0.553 / 0.429 | 0.549 / 0.853 | 0.841 / 0.982 | 0.150 / 0.786 |
| lasso_mean_fill | 0.700 / 0.883 | 1.000 / 0.429 | 0.549 / 0.991 | 0.583 / 0.999 | -0.052 / 0.874 |
| rf_mean_fill | 0.550 / 0.852 | 0.381 / 0.429 | 0.167 / 0.887 | 0.424 / 0.992 | 0.069 / 0.839 |
| ipls_mean_fill | 0.700 / 0.776 | 0.338 / 0.526 | 0.159 / 0.906 | 0.635 / 0.987 | 0.150 / 0.764 |
| mcuve_mean_fill | 0.350 / 0.848 | 0.097 / 0.627 | 0.205 / 0.961 | 0.218 / 0.989 | 0.029 / 0.853 |
| _budget_ | k=4 of 20 | k=5 of 35 | k=6 of 64 | k=8 of 64 | k=3 of 10 |

Cells are stability / downstream score. Nothing is marked best: with no truth set the primary metric is stability, and a perfectly stable selection can be the predictively worst one (tecator anova here: stability 1.000 at R2 0.281). Read both numbers in a cell together. full_signal keeps every segment, so its stability is 1.00 by construction.

**Missingness present in each dataset**, since it decides which columns test the novelty axis at all:

| dataset | task-relevant missingness |
|---|---|
| raman | dropout stretches (ignorable) |
| xrd | saturation censoring (hits the signal) |
| ms | zeros-as-not-detected (loader flag) |
| sensor | band-by-group (ignorable: groups independent of y) |
| tecator | none (complete real spectra) |
| corn | none (complete real spectra) |
| ovarian | binned per-sample axes (complete at this resolution) |
| rruff | real per-spectrum wavenumber ranges (binned) |
| basicmotions | none (complete recordings) |

<!-- END GENERATED TABLE -->

## 12. The contrast-branch denominator: a sibling's fix that did not transfer (2026-09-07)

RobustPixelMaker was fixed today for the same count-based rescaling defect
this project fixed on 2026-09-01 (sec.9 fix 1). Fixing it there turned up a
question this project had not asked, and answering it here produced a
negative result worth recording, because the obvious action was wrong.

**The question.** Both libraries re-centre a contrast (zero-sum) filter over
the surviving support, then rescale. RSM rescales by observed ABSOLUTE (L1)
filter mass. RPM's measurement said that is not the magnitude-preserving
choice: a re-centred response is `sum_obs f_j r_j` over residuals, whose
magnitude for exchangeable residuals goes as `sqrt(sum_obs f^2)`, so the L2
ratio is the right one, while count and L1 both go as N/c for a random
observed subset and over-correct by roughly a square. On RPM's criterion,
recovering the full-data convolution response with random zero-mean 2D
kernels, L2 gave 79.9% RMS error against count's 93.2% under scattered
dropout and 57.3% against 75.2% under patch dropout. The obvious action was
to port L2 here.

**It is decision-relevant, so it could not be waved through.** Swapping L1
for L2 changed the selected segment set in 8 of 12 configurations (3 seeds x
2 dropout regimes x {SavGol derivative, RandomConv1D}). This is not a
cosmetic scaling difference.

**Measured on the criterion that governs RSM, L2 LOST.** Response fidelity is
RPM's criterion because its `renormalised_convolve` feeds no selection; RSM's
criterion is truth recovery of the SELECTION, which the synthetic controls
can score directly. Over 10 seeds x 2 dropout regimes x 2 contrast lenses
(n = 40 per rule), F1 against the planted truth:

| dropout | lens | count | L1 | L2 |
|---|---|---|---|---|
| scattered | savgol | 0.550 | 0.499 | 0.490 |
| scattered | randconv | 0.284 | 0.289 | 0.262 |
| stretches | savgol | 0.591 | 0.597 | 0.548 |
| stretches | randconv | 0.275 | 0.280 | 0.290 |
| **overall** | | **0.425** | **0.416** | **0.398** |

Paired per configuration: L1 beat L2 21 times, tied 10, lost 9, mean
difference +0.019 F1. L1 against count is a genuine tie (14 wins, 10 ties,
16 losses, mean -0.008).

**Why it does not transfer.** The exchangeable-residual argument behind L2
assumes the response is a roughly random combination of the window. That
holds for RPM's lens, which is random zero-mean kernels only, and the 1D
measurement agrees: on RandomConv1D, L2 does win the response-fidelity
comparison. It does not hold for a Savitzky-Golay derivative, whose
coefficients are smooth and antisymmetric and whose response on smooth
spectra is dominated by the local gradient rather than by iid variation;
there L1 wins on both criteria. RSM's default ensemble contains both lenses,
and RSM additionally requires a contrast window to be at least half observed
and then pools into segments, both of which damp whatever scaling difference
survives.

**Decision: RSM keeps L1, unchanged.** Not because it is theoretically
superior in general, but because it is the best of the three on this
library's own criterion and ties the third. Count is rejected despite tying,
because it ignores filter weights entirely and is the rule that fabricates on
the LEVEL branch; L1 is the principled generalisation of count to weighted
kernels. The two libraries now differ deliberately, each matching its own
lens set and criterion, and both are pinned by tests
(`test_coverage_98.py::test_contrast_branch_normalises_by_l1_mass_not_l2_or_count`
here) so neither can drift into the other silently.

**The general lesson, which is the reason this section exists.** A fix
verified in a sibling library is not verified here. The port arrived with a
real measurement behind it, on a real criterion, and was still the wrong
change, because the criterion was the sibling's rather than ours. Sec.9
records this project's habit of auditing its own claims; this records the
same scepticism applied to an incoming one.
