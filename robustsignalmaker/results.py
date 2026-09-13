"""Result container and serialisation (Milestone 5).

The result object exposes the analysis directly (RMM sec.4.3 ethos): per-fold
scores WITH the matched full-signal baseline and the preserved / sig.better /
sig.worse verdict, out-of-fold predictions, per-fold selected segments with
their stability, the per-fold MNAR-guard stamps, refusal counts (samples the
model declined to score for lack of evidence), and the final refit estimator.

Export follows the RMM results_tables contract: ``results_tables()`` returns
named tables, ``save()`` writes one CSV per table plus a JSON summary and a
pickle. pandas is not required.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .metrics import paired_comparison, rmse_from_score


def _write_csv(path: Path, rows: list) -> None:
    """Well-formed CSV, whatever the cell payloads.

    Cells can contain commas (index lists, strata notes, error strings), so
    rows go through the csv module with proper quoting; columns are the
    union across rows, first-seen order, so no row silently drops a field.
    """
    import csv

    if not rows:
        path.write_text("")
        return
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


@dataclass
class RSMResult:
    task: str
    random_state: int
    per_fold_scores: np.ndarray
    baseline_per_fold_scores: Optional[np.ndarray]
    verdict: Optional[object]            # BaselineComparison or None
    oof_predictions: np.ndarray
    oof_count: np.ndarray
    n_refused: int
    n_unlabelled: int
    selected_per_fold: list
    selection_stability: float
    selection_stability_adjusted: float
    mnar_reports: list                   # one dict per fold (validity.missingness_association)
    fold_records: list
    final_estimator: object
    classes: Optional[np.ndarray] = None
    config: dict = field(default_factory=dict)

    # ---- convenience ------------------------------------------------------ #
    @property
    def mean_score(self) -> float:
        return float(np.nanmean(self.per_fold_scores))

    @property
    def std_score(self) -> float:
        return float(np.nanstd(self.per_fold_scores))

    @property
    def mnar_flag(self) -> bool:
        """True when any training fold's missingness pattern predicted y."""
        return any(r.get("verdict") == "informative" for r in self.mnar_reports)

    def display_score(self):
        """(value, sd, name) with the regression sign flipped back to RMSE."""
        if self.task == "regression":
            per = np.array([rmse_from_score(s) for s in self.per_fold_scores])
            return float(np.nanmean(per)), float(np.nanstd(per)), "RMSE"
        name = "AUC" if self.task == "binary" else "AUC-OVR"
        return self.mean_score, self.std_score, name

    def predict(self, X, V=None):
        return self.final_estimator.predict(X, V)

    def predict_proba(self, X, V=None):
        return self.final_estimator.predict_proba(X, V)

    def compare_to_baseline(self, baseline_per_fold_scores):
        """Paired verdict vs an external baseline's per-fold scores."""
        return paired_comparison(self.per_fold_scores, baseline_per_fold_scores)

    def summary(self) -> dict:
        val, sd, name = self.display_score()
        out = {
            "task": self.task,
            "score_name": name,
            "score_mean": val,
            "score_sd": sd,
            "n_folds": int(len(self.per_fold_scores)),
            "selection_stability_jaccard": self.selection_stability,
            "selection_stability_adjusted": self.selection_stability_adjusted,
            "mean_n_selected": float(
                np.mean([len(s) for s in self.selected_per_fold])
                if self.selected_per_fold else 0.0),
            "n_refused_predictions": int(self.n_refused),
            "n_unlabelled_excluded": int(self.n_unlabelled),
            "missingness_informative": self.mnar_flag,
            "random_state": self.random_state,
            **self.config,
        }
        if self.verdict is not None:
            out["verdict"] = self.verdict.outcome
            out["verdict_p_wilcoxon"] = self.verdict.p_wilcoxon
            out["verdict_mean_delta"] = self.verdict.mean_delta
            if self.baseline_per_fold_scores is not None:
                # displayed in the same units as score_mean (RMSE for
                # regression), not in the internal higher-is-better convention
                if self.task == "regression":
                    out["baseline_score_mean"] = float(np.nanmean(
                        [rmse_from_score(s) for s in self.baseline_per_fold_scores]))
                else:
                    out["baseline_score_mean"] = float(
                        np.nanmean(self.baseline_per_fold_scores))
        return out

    # ---- export ------------------------------------------------------------ #
    def results_tables(self) -> dict:
        """Named tables (list-of-dict rows), RMM export convention."""
        tables = {"overview": [self.summary()], "folds": list(self.fold_records)}
        sel_rows = []
        for rec, sel in zip(self.fold_records, self.selected_per_fold):
            for s in sel:
                sel_rows.append({"repeat": rec["repeat"], "fold": rec["fold"],
                                 "segment": int(s)})
        tables["selected_segments"] = sel_rows
        tables["mnar_guard"] = [
            {"repeat": rec["repeat"], "fold": rec["fold"],
             "verdict": rep.get("verdict"), "score": rep.get("score"),
             "p_value": rep.get("p_value")}
            for rec, rep in zip(self.fold_records, self.mnar_reports)
        ] if self.mnar_reports else []
        return tables

    def _json_safe(self) -> dict:
        d = self.summary()
        d["per_fold_scores"] = [float(s) for s in self.per_fold_scores]
        if self.baseline_per_fold_scores is not None:
            d["baseline_per_fold_scores"] = [
                float(s) for s in self.baseline_per_fold_scores]
        return d

    def save(self, directory, prefix: str = "rsm") -> Path:
        """One CSV per table + JSON summary + pickle (RMM export style)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{prefix}_summary.json").write_text(
            json.dumps(self._json_safe(), indent=2, default=str))
        for name, rows in self.results_tables().items():
            _write_csv(directory / f"{prefix}_{name}.csv", rows)
        with open(directory / f"{prefix}_result.pkl", "wb") as fh:
            pickle.dump(self, fh)
        return directory

    @staticmethod
    def load(path) -> "RSMResult":
        with open(path, "rb") as fh:
            return pickle.load(fh)
