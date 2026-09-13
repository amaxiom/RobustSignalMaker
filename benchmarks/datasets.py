"""Benchmark datasets: per-domain synthetic controls with known truth.

Registry pattern per the RPM harness lessons: every dataset is a loader
returning ``(X, V, y, meta)`` registered in ``LOADERS`` by an explicit name;
tables and filenames derive from the registry, never from hardcoded lists.

Each loader plants informative structure whose ground truth is recorded in
``meta["truth_points"]``, applies the domain's characteristic missingness
with the Milestone-7 generators, and states its recommended segment size and
target selection size. Nothing is imputed here; loaders return validity.

The four domains and their missingness:

  raman   Lorentzian peaks on a smooth drifting baseline; detector-dropout
          stretches (ignorable).
  xrd     sharp reflections on a decaying background; saturation censoring
          (the tallest, most informative reflections clip).
  ms      sparse spiky high-dimensional m/z profiles; zeros are "not
          detected" and the loader decides, explicitly, that zeros are
          missing (the `zeros_are_missing` flag: no silent global rule).
  sensor  multi-channel series; band-by-group missingness (two instrument
          configurations covering different ranges), non-ignorable only if
          groups correlate with y, which here they do NOT.

Real datasets (loaders download once into benchmarks/data/, which stays out
of git, and cache a compact form):

  tecator  meat NIR, 240 samples x 100 channels, fat regression (OpenML 505)
  corn     Eigenvector corn NIR, 80 samples x 700 channels x 3 instruments,
           four property targets (benchmark uses instrument m5, moisture)
  ovarian  FDA-NCI ovarian SELDI-TOF serum spectra (post-QAQC set via the
           Internet Archive), 216 samples (121 cancer, 95 normal), raw
           per-sample m/z axes binned onto a common grid; known batch and
           calibration caveats travel in the meta

Real spectra carry no ground-truth bands: their meta has no ``truth_points``
and declares a selection budget ``k`` instead; the runner then reports
stability and downstream score without recovery F1. Further real feeds
(alloys) follow ``load_real_template``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"

from robustsignalmaker import (
    mask_band_by_group,
    mask_dropout_stretches,
    mask_saturation_censor,
)


def _lorentzian(t, centre, gamma):
    return gamma ** 2 / ((t - centre) ** 2 + gamma ** 2)


def _gaussian(t, centre, width):
    return np.exp(-0.5 * ((t - centre) / width) ** 2)


def _binary_from(latents, rng):
    agg = latents.sum(axis=1)
    return (agg > np.median(agg)).astype(int)


def load_raman(n: int = 240, n_points: int = 512, seed: int = 0):
    """Raman/FTIR-like: peaks on a drifting baseline, dropout stretches."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=float)
    # nuisance peaks shared by every sample, fixed amplitude
    nuisance = sum(_lorentzian(t, c, 6.0) for c in (60.0, 210.0, 430.0))
    # informative peaks: amplitudes are non-negative latents that drive y
    centres = (140.0, 320.0)
    A = np.abs(rng.standard_normal((n, len(centres))))
    truth = np.zeros(n_points, dtype=bool)
    X = np.zeros((n, 1, n_points))
    for k, c in enumerate(centres):
        peak = _lorentzian(t, c, 5.0)
        X[:, 0, :] += 3.0 * A[:, k, None] * peak[None, :]
        truth |= peak > 0.15
    drift = rng.standard_normal((n, 1, 1)) * (t / n_points)[None, None, :] ** 2
    X += 2.0 * nuisance[None, None, :] + 2.0 * drift + rng.normal(scale=0.4, size=X.shape)
    y = _binary_from(A, rng)
    V = mask_dropout_stretches(X.shape, n_stretches=1, min_len=20, max_len=60,
                               seed=seed + 1)
    meta = {"name": "raman", "task": "binary", "truth_points": truth,
            "segment": 16, "missingness": "dropout stretches (ignorable)"}
    return X, V, y, meta


def load_xrd(n: int = 240, n_points: int = 512, seed: int = 0):
    """XRD/SAXS-like: sharp reflections, the tallest of which saturate."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=float)
    nuisance = sum(_gaussian(t, c, 2.0) for c in (90.0, 250.0, 400.0, 470.0))
    centres = (170.0, 330.0)
    A = np.abs(rng.standard_normal((n, len(centres))))
    truth = np.zeros(n_points, dtype=bool)
    X = np.zeros((n, 1, n_points))
    for k, c in enumerate(centres):
        peak = _gaussian(t, c, 2.5)
        X[:, 0, :] += 6.0 * A[:, k, None] * peak[None, :]
        # truth includes the SHOULDERS (threshold 0.02, about +-7 points):
        # censoring clips the tips, so the amplitude information genuinely
        # lives in the flanks, and a selector choosing them is right, not
        # wrong (run-1 lesson: a 0.1 threshold mislabelled honest selections)
        truth |= peak > 0.02
    background = 3.0 * np.exp(-t / 200.0)
    X += 2.0 * nuisance[None, None, :] + background[None, None, :]
    X += rng.normal(scale=0.3, size=X.shape)
    y = _binary_from(A, rng)
    # the detector clips: exactly the informative tall reflections censor
    X, V = mask_saturation_censor(X, upper=float(np.quantile(X, 0.995)))
    meta = {"name": "xrd", "task": "binary", "truth_points": truth,
            "segment": 8, "missingness": "saturation censoring (hits the signal)"}
    return X, V, y, meta


def load_ms(n: int = 240, n_points: int = 1024, seed: int = 0,
            zeros_are_missing: bool = True):
    """Mass-spec-like: sparse spikes; zeros mean "not detected".

    ``zeros_are_missing`` is the loader's explicit, per-dataset decision
    (never a silent global rule): True treats a zero intensity as an
    unobserved m/z position; False treats it as a measured zero.
    """
    rng = np.random.default_rng(seed)
    library = rng.choice(n_points, size=40, replace=False)  # shared peak positions
    info_pos = library[:3]
    A = np.abs(rng.standard_normal((n, len(info_pos))))
    X = np.zeros((n, 1, n_points))
    for i in range(n):
        present = rng.random(len(library)) < 0.6  # stochastic detection
        for j, pos in enumerate(library):
            if not present[j]:
                continue
            amp = 1.0 + np.abs(rng.standard_normal())
            X[i, 0, pos] = amp
        for k, pos in enumerate(info_pos):  # informative peaks always present
            X[i, 0, pos] = 0.5 + 3.0 * A[i, k]
    truth = np.zeros(n_points, dtype=bool)
    truth[info_pos] = True
    y = _binary_from(A, rng)
    V = (X > 0) if zeros_are_missing else np.ones_like(X, dtype=bool)
    meta = {"name": "ms", "task": "binary", "truth_points": truth,
            "segment": 32, "zeros_are_missing": zeros_are_missing,
            "missingness": "zeros-as-not-detected (loader flag)",
            # sparse spike data observes a few percent of the axis per
            # sample: dense-signal evidence floors would refuse everything,
            # so the loader DECLARES the evidence scale explicitly (run-1
            # lesson; never a silent heuristic inside the selector)
            "selector_config": {"min_valid_frac": 0.02, "rho_min": 0.01}}
    return X, V, y, meta


def load_sensor(n: int = 240, n_channels: int = 4, n_points: int = 256,
                seed: int = 0):
    """Multi-channel series with band-by-group (instrument-range) missingness.

    Groups are assigned independently of y, so the missingness is ignorable
    by construction; the MNAR guard should stay quiet here.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=float)
    A = np.abs(rng.standard_normal((n, 2)))
    bump1 = _gaussian(t, 96.0, 8.0)
    bump2 = _gaussian(t, 176.0, 8.0)
    X = rng.normal(scale=1.0, size=(n, n_channels, n_points))
    X += 2.0 * A[:, 0, None, None] * bump1[None, None, :]
    X += 2.0 * A[:, 1, None, None] * bump2[None, None, :]
    truth = (bump1 > 0.1) | (bump2 > 0.1)
    y = _binary_from(A, rng)
    groups = rng.permutation(np.arange(n) % 2)  # independent of y
    V = mask_band_by_group(X.shape, groups, {0: [(0, 48)], 1: [(208, 256)]})
    meta = {"name": "sensor", "task": "binary", "truth_points": truth,
            "segment": 8, "groups": groups,
            "missingness": "band-by-group (ignorable: groups independent of y)"}
    return X, V, y, meta


def load_tecator(seed: int = 0):
    """Tecator meat NIR (OpenML 505): 100 absorbance channels, fat target.

    Complete data (V all True); the benchmark interest is band selection
    quality and stability on real, strongly collinear spectra. Wavelengths
    span 850 to 1050 nm at 2 nm steps.
    """
    from sklearn.datasets import fetch_openml

    d = fetch_openml(name="tecator", version=1, as_frame=True,
                     data_home=str(DATA_DIR / "openml"))
    cols = [c for c in d.data.columns if c.startswith("absorbance_")]
    cols.sort(key=lambda c: int(c.split("_")[1]))
    X = d.data[cols].to_numpy(dtype=float)[:, None, :]
    y = d.target.to_numpy(dtype=float)
    V = np.ones_like(X, dtype=bool)
    meta = {"name": "tecator", "task": "regression", "segment": 5, "k": 4,
            "missingness": "none (complete real spectra)",
            "axis": "850 to 1050 nm, 100 channels"}
    return X, V, y, meta


def load_corn(seed: int = 0, instrument: str = "m5", target: str = "Moisture"):
    """Eigenvector corn NIR: 80 samples x 700 channels, three instruments.

    The benchmark uses one instrument (default m5) so split-half replicates
    never place the same physical sample on both sides via another
    instrument's copy. ``instrument="all"`` stacks all three and returns
    instrument ids in ``meta["groups"]`` (use groups-aware CV then; the 80
    samples repeat across instruments). Wavelengths 1100 to 2498 nm at 2 nm
    steps; water bands sit near channels 175 (1450 nm) and 420 (1940 nm).
    """
    from scipy.io import loadmat

    path = DATA_DIR / "corn.mat"
    if not path.exists():  # pragma: no cover - needs network
        raise FileNotFoundError(
            f"{path} missing; download https://www.eigenvector.com/data/Corn/corn.mat")
    m = loadmat(str(path))

    def ds(key):
        return np.asarray(m[key]["data"][0, 0], dtype=float)

    targets = ["Moisture", "Oil", "Protein", "Starch"]
    y80 = ds("propvals")[:, targets.index(target)]
    if instrument == "all":
        X = np.concatenate([ds(f"{i}spec") for i in ("m5", "mp5", "mp6")])[:, None, :]
        y = np.tile(y80, 3)
        groups = np.repeat(np.arange(3), 80)
    else:
        X = ds(f"{instrument}spec")[:, None, :]
        y = y80
        groups = None
    meta = {"name": "corn", "task": "regression", "segment": 20, "k": 5,
            "target": target, "instrument": instrument,
            "missingness": "none (complete real spectra)",
            "axis": "1100-2498 nm, 2 nm steps"}
    if groups is not None:
        meta["groups"] = groups
        meta["note"] = "80 physical samples repeated per instrument; use groups"
    return X, np.ones_like(X, dtype=bool), y, meta


def load_ovarian(seed: int = 0, n_bins: int = 4096):
    """FDA-NCI ovarian SELDI-TOF serum spectra (post-QAQC), cancer vs normal.

    Raw files carry per-sample m/z axes, so intensities are binned onto a
    common grid over 700 to 12000 Da (mean of the measured points per bin:
    aggregation of observations, never fabrication; a bin a sample never hit
    would be missing, though at this resolution every bin is hit). Cached to
    an npz on first use; the 272 MB source zip comes from the Internet
    Archive copy of the retired NCI-FDA Clinical Proteomics site.

    Provenance caveat, which the analysis must carry: this dataset's
    batch/calibration artefacts are famous (they drove the reproducibility
    controversy around the original papers), so treat any selected band as a
    candidate ARTEFACT DETECTOR as much as a biomarker; that ambiguity is
    exactly what the honesty machinery is for.
    """
    cache = DATA_DIR / f"ovarian_postqaqc_{n_bins}.npz"
    if not cache.exists():
        _build_ovarian_cache(cache, n_bins)
    z = np.load(str(cache))
    X = z["X"][:, None, :]
    V = z["V"][:, None, :].astype(bool)
    y = z["y"].astype(int)
    meta = {"name": "ovarian", "task": "binary", "segment": 64, "k": 6,
            "missingness": "binned per-sample axes (complete at this resolution)",
            "axis": "700-12000 Da, common grid of %d bins" % n_bins,
            "note": "known batch/calibration artefact caveats; see docstring"}
    return X, V, y, meta


def _build_ovarian_cache(cache, n_bins):  # pragma: no cover - needs the zip
    import zipfile

    import pandas as pd

    src = DATA_DIR / "OvarianCD_PostQAQC.zip"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing; fetch OvarianCD_PostQAQC.zip via the Internet "
            f"Archive copy of home.ccr.cancer.gov/ncifdaproteomics/")
    z = zipfile.ZipFile(str(src))
    names = [n for n in z.namelist() if n.endswith(".txt")]
    edges = np.linspace(700.0, 12000.0, n_bins + 1)
    X, V, y = [], [], []
    for name in sorted(names):
        raw = pd.read_csv(z.open(name), sep="\t", header=None,
                          dtype=float).to_numpy()
        mz, inten = raw[:, 0], raw[:, 1]
        sums, _ = np.histogram(mz, bins=edges, weights=inten)
        counts, _ = np.histogram(mz, bins=edges)
        with np.errstate(invalid="ignore"):
            X.append(np.divide(sums, counts, out=np.zeros(n_bins),
                               where=counts > 0))
        V.append(counts > 0)
        y.append(1 if "/Cancer/" in name else 0)
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(str(cache), X=np.asarray(X), V=np.asarray(V),
                        y=np.asarray(y))


def load_rruff(seed: int = 0, n_bins: int = 512, n_classes: int = 6):
    """RRUFF Raman mineral identification (excellent unoriented, processed).

    The ``n_classes`` minerals with the most processed spectra (any laser)
    become a MULTICLASS problem. Spectra carry per-file wavenumber ranges, so
    binning onto the common 150 to 1400 inverse cm grid produces GENUINE
    band-by-sample missingness (a bin outside a spectrum's measured range is
    unobserved, never zero-filled). ``meta["groups"]`` carries the RRUFF
    specimen id: several spectra of one physical specimen exist, and grouped
    CV must keep a specimen on one side of a split. Intensities are scaled
    to unit maximum over the observed range per spectrum.

    Source: https://rruff.info/zipped_data_files/raman/excellent_unoriented.zip
    (cached under benchmarks/data/; a compact npz is built on first use).
    """
    cache = DATA_DIR / f"rruff_{n_classes}cls_{n_bins}.npz"
    if not cache.exists():
        _build_rruff_cache(cache, n_bins, n_classes)
    z = np.load(str(cache), allow_pickle=True)
    X = z["X"][:, None, :]
    V = z["V"][:, None, :].astype(bool)
    meta = {"name": "rruff", "task": "multiclass", "segment": 8, "k": 8,
            "classes": [str(c) for c in z["classes"]],
            "groups": z["groups"], "lasers": z["lasers"],
            "axis": "150-1400 inverse cm, %d bins" % n_bins,
            "missingness": "real per-spectrum wavenumber ranges (binned)"}
    return X, V, z["y"].astype(int), meta


def _build_rruff_cache(cache, n_bins, n_classes):  # pragma: no cover - needs the zip
    import collections
    import io
    import zipfile

    src = DATA_DIR / "rruff_excellent_unoriented.zip"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing; download rruff.info excellent_unoriented.zip")
    z = zipfile.ZipFile(str(src))
    proc = [n for n in z.namelist() if "Raman_Data_Processed" in n]
    counts = collections.Counter(n.split("__")[0] for n in proc)
    classes = [m for m, _ in counts.most_common(n_classes)]
    edges = np.linspace(150.0, 1400.0, n_bins + 1)
    X, V, y, groups, lasers = [], [], [], [], []
    for name in sorted(proc):
        mineral = name.split("__")[0]
        if mineral not in classes:
            continue
        parts = name.split("__")
        rows = [l for l in z.read(name).decode("ascii", "replace").splitlines()
                if l.strip() and not l.startswith("#")]
        try:
            arr = np.genfromtxt(io.StringIO("\n".join(rows)), delimiter=",")
        except ValueError:
            continue  # a malformed file is skipped, not silently zeroed
        if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 50:
            continue
        w, inten = arr[:, 0], arr[:, 1]
        sums, _ = np.histogram(w, bins=edges, weights=inten)
        cnts, _ = np.histogram(w, bins=edges)
        with np.errstate(invalid="ignore"):
            spec = np.divide(sums, cnts, out=np.zeros(n_bins), where=cnts > 0)
        obs = cnts > 0
        if obs.sum() < n_bins // 4:
            continue  # spectrum barely overlaps the grid
        peak = np.abs(spec[obs]).max()
        if peak <= 0:
            continue
        X.append(np.where(obs, spec / peak, 0.0))
        V.append(obs)
        y.append(classes.index(mineral))
        groups.append(parts[1])   # RRUFF specimen id
        lasers.append(parts[3])
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(str(cache), X=np.asarray(X), V=np.asarray(V),
                        y=np.asarray(y), classes=np.asarray(classes),
                        groups=np.asarray(groups), lasers=np.asarray(lasers))


def load_basicmotions(seed: int = 0):
    """UEA BasicMotions: 6-channel wearable series, four activities.

    The multivariate archive's beginner set: 80 recordings (train and test
    folds pooled; RSM does its own splitting), 6 channels (accelerometer and
    gyroscope, x/y/z) x 100 time steps, classes walking / running /
    badminton / standing. The example for JOINT GATING: one mask over time
    windows shared across all six sensors.

    Source: https://timeseriesclassification.com/aeon-toolkit/BasicMotions.zip
    """
    import zipfile

    src = DATA_DIR / "BasicMotions.zip"
    if not src.exists():  # pragma: no cover - needs network
        raise FileNotFoundError(f"{src} missing; download BasicMotions.zip")
    z = zipfile.ZipFile(str(src))

    def read_dim(dim, part):
        txt = z.read(f"BasicMotionsDimension{dim}_{part}.arff").decode("ascii")
        lines = txt.splitlines()
        start = next(i for i, l in enumerate(lines)
                     if l.strip().lower() == "@data")
        rows, labels = [], []
        for line in lines[start + 1:]:
            if not line.strip():
                continue
            *vals, lab = line.split(",")
            rows.append([float(v) for v in vals])
            labels.append(lab.strip().strip("'"))
        return np.asarray(rows), labels

    X_parts, labels_all = [], []
    for part in ("TRAIN", "TEST"):
        dims, part_labels = [], None
        for d in range(1, 7):
            rows, labels = read_dim(d, part)
            dims.append(rows)
            if part_labels is None:
                part_labels = labels
            elif labels != part_labels:
                # channels would be misaligned across samples: fail loudly
                raise ValueError(
                    f"BasicMotions dimension {d} {part} rows are ordered "
                    f"differently from dimension 1")
        X_parts.append(np.stack(dims, axis=1))  # (n, 6, 100)
        labels_all.extend(part_labels)
    X = np.concatenate(X_parts)
    classes = sorted(set(labels_all))
    y = np.array([classes.index(l) for l in labels_all])
    meta = {"name": "basicmotions", "task": "multiclass", "segment": 10,
            "k": 3, "classes": classes,
            "missingness": "none (complete recordings)",
            "axis": "100 time steps at 10 Hz"}
    return X, np.ones_like(X, dtype=bool), y, meta


def load_real_template(path):  # pragma: no cover - documentation stub
    """The contract for a real-data loader (alloys, biomarker feeds).

    Read the file(s) at ``path``; return ``(X, V, y, meta)`` with X shaped
    (n, C, T) float on a shared grid, V bool (True = observed; derive from
    NaNs via as_validity, from instrument ranges via mask_band_by_group,
    from detection limits via mask_saturation_censor, or from an explicit
    zeros_are_missing decision), y (n,) with nan for unlabelled samples, and
    meta at least {"name", "task", "segment"}; add "truth_points" only if a
    ground truth genuinely exists, "groups" for instrument/session ids.
    Register it in LOADERS under an explicit name.
    """
    raise NotImplementedError("provide the data files and adapt this template")


SYNTHETIC_LOADERS = {
    "raman": load_raman,
    "xrd": load_xrd,
    "ms": load_ms,
    "sensor": load_sensor,
}

REAL_LOADERS = {
    "tecator": load_tecator,
    "corn": load_corn,
    "ovarian": load_ovarian,
    "rruff": load_rruff,
    "basicmotions": load_basicmotions,
}

LOADERS = {**SYNTHETIC_LOADERS, **REAL_LOADERS}


def truth_segments(truth_points, grid, min_overlap: float = 0.5) -> np.ndarray:
    """Segments carrying enough truth to count as ground truth.

    The rule is PER CONTIGUOUS TRUTH RUN: a segment qualifies when it holds
    at least ``min_overlap`` of what that run could put into one segment,
    i.e. ``min_overlap * min(run_length, segment_length)`` points. A
    single-point spike therefore qualifies its segment (threshold under one
    point), a wide band qualifies the segments it substantially covers, and
    the two coexist. (The first release compared every segment against the
    GLOBAL fullest truth segment, which silently dropped spike truth
    whenever band truth existed anywhere; 2026-09-01 sweep.)
    """
    tp = np.asarray(truth_points, dtype=bool)
    if not tp.any():
        return np.array([], dtype=int)
    # label contiguous truth runs
    starts = np.flatnonzero(tp & ~np.roll(tp, 1))
    if tp[0]:
        starts = np.unique(np.concatenate([[0], starts]))
    ends = np.flatnonzero(tp & ~np.roll(tp, -1))
    if tp[-1]:
        ends = np.unique(np.concatenate([ends, [len(tp) - 1]]))
    qualified = np.zeros(grid.n_segments, dtype=bool)
    for run_start, run_end in zip(starts, ends):
        run_len = run_end - run_start + 1
        seg_ids = np.unique(grid.point_segment[run_start:run_end + 1])
        for s in seg_ids:
            pts = grid.segments_to_points([s])
            held = int(tp[pts].sum())
            thr = min_overlap * min(run_len, grid.segment_lengths[s])
            if held >= max(thr, 1.0):
                qualified[s] = True
    return np.flatnonzero(qualified)
