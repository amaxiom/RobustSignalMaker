"""Run the per-domain benchmark (synthetic controls and real datasets).

    py benchmarks/run_synthetic_benchmark.py [dataset ...]

For every dataset x selector: truth-recovery F1 at equal selection size
(synthetic controls only; real spectra carry no ground truth and declare a
selection budget ``k`` instead), split-half stability (chance-corrected mean
pairwise Jaccard across replicates), and a UNIFORM downstream score (5-fold
CV on the selected segment means of mean-filled data, logistic AUC for
classification and ridge R2 for regression, identical for every method, so
the comparison is between SELECTIONS, not between predictors).

Output: one CSV per dataset under benchmarks/results/, named by dataset (the
RPM harness lesson: fixed output filenames overwrote earlier runs), plus a
printed table ordered by the selector registry.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from competitors import METHOD_ORDER, SELECTORS, mean_fill, _segment_means  # noqa: E402
from datasets import LOADERS, truth_segments  # noqa: E402

from robustsignalmaker import (  # noqa: E402
    SegmentGrid,
    mean_pairwise_jaccard,
    resample_indices,
)

RESULTS = Path(__file__).resolve().parent / "results"


def _f1(selected, truth):
    s = set(int(v) for v in selected)
    tp = len(s & truth)
    return 2 * tp / (len(s) + len(truth)) if s else 0.0


def _downstream_score(F, y, selected, seed, task):
    """Uniform scorer: logistic AUC (binary / OVR multiclass) or ridge R2."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cols = np.asarray(sorted(set(int(v) for v in selected)), dtype=int)
    if task == "regression":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        pred = cross_val_predict(model, F[:, cols], y, cv=cv)
        return float(r2_score(y, pred))
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=seed))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, F[:, cols], y, cv=cv, method="predict_proba")
    if task == "multiclass":
        return float(roc_auc_score(y, proba, multi_class="ovr", average="weighted"))
    return float(roc_auc_score(y, proba[:, 1]))


def evaluate_dataset(name, selector_names=None, n_replicates=3, seed=0):
    X, V, y, meta = LOADERS[name](seed=seed)
    task = meta["task"]
    grid = SegmentGrid(X.shape[-1], segment=meta["segment"])
    truth = (set(truth_segments(meta["truth_points"], grid).tolist())
             if "truth_points" in meta else None)
    k = int(meta.get("k", len(truth) if truth else max(1, grid.n_segments // 8)))
    F_score = _segment_means(mean_fill(X, V), None, grid)  # uniform scorer features
    n = X.shape[0]
    strat = None if task == "regression" else y

    cfg = meta.get("selector_config", {})
    rows = []
    for method in (selector_names or METHOD_ORDER):
        fn = SELECTORS[method]
        t0 = time.perf_counter()
        try:
            selected = fn(X, V, y, grid, task, seed, k, **cfg)
            sets = []
            for r in range(n_replicates):
                rng = np.random.default_rng(1000 + r)
                idx = resample_indices(n, "half", rng, strat=strat)
                sets.append(fn(X[idx], V[idx], y[idx], grid, task,
                               seed + 1 + r, k, **cfg))
            stability = mean_pairwise_jaccard(sets, n_total=grid.n_segments)
            score = _downstream_score(F_score, y, selected, seed, task)
            rows.append({
                "dataset": name, "method": method,
                "n_selected": len(selected),
                "f1": round(_f1(selected, truth), 3) if truth else "",
                "stability_adj": round(float(stability), 3)
                if np.isfinite(stability) else "nan",
                "downstream": round(score, 3),
                "seconds": round(time.perf_counter() - t0, 1),
            })
        except Exception as exc:  # a failed method is a row, never a silent gap
            rows.append({"dataset": name, "method": method, "n_selected": "",
                         "f1": "", "stability_adj": "", "downstream": "",
                         "seconds": round(time.perf_counter() - t0, 1),
                         "error": f"{type(exc).__name__}: {exc}"})
    return rows, {"truth": sorted(truth) if truth else "none (real data)",
                  "k": k, "S": grid.n_segments,
                  "score_name": "R2" if task == "regression" else "AUC",
                  "missingness": meta["missingness"]}


def _write_csv(path, rows):
    """Proper CSV quoting: error strings and notes may contain commas."""
    import csv

    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})


def main(names=None):
    RESULTS.mkdir(exist_ok=True)
    meta_rows = []
    for name in (names or list(LOADERS)):
        rows, info = evaluate_dataset(name)
        out = RESULTS / f"benchmark_{name}.csv"
        _write_csv(out, rows)
        # per-run metadata sidecar: the consolidated FINDINGS table is
        # GENERATED from these files, so the budget, universe and score name
        # a table quotes come from the run that produced it, never retyped
        meta_rows.append({
            "dataset": name, "k": info["k"], "S": info["S"],
            "score_name": info["score_name"],
            "has_truth": "yes" if info["truth"] != "none (real data)" else "no",
            "truth_segments": (" ".join(str(v) for v in info["truth"])
                               if info["truth"] != "none (real data)" else ""),
            "missingness": info["missingness"],
        })
        print(f"\n=== {name}  (truth {info['truth']}, k={info['k']}, "
              f"S={info['S']}; {info['missingness']})")
        score_name = info["score_name"]
        print(f"{'method':20s} {'n_sel':>5s} {'F1':>6s} {'stab':>6s} "
              f"{score_name:>6s} {'sec':>6s}")
        for r in rows:
            print(f"{r['method']:20s} {str(r['n_selected']):>5s} {str(r['f1']):>6s} "
                  f"{str(r['stability_adj']):>6s} {str(r['downstream']):>6s} "
                  f"{str(r['seconds']):>6s}" + (f"  {r['error']}" if "error" in r else ""))
        print(f"-> {out}")

    # only a full run may replace the sidecar; a subset run would silently
    # shrink the universe the table is built from
    if names is None:
        _write_csv(RESULTS / "run_meta.csv", meta_rows)
        print(f"-> {RESULTS / 'run_meta.csv'}")
    elif meta_rows:
        print(f"(subset run: {RESULTS / 'run_meta.csv'} left untouched)")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
