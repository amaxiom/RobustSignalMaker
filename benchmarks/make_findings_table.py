"""Generate the consolidated FINDINGS results table from the result CSVs.

    py benchmarks/make_findings_table.py            # print the markdown
    py benchmarks/make_findings_table.py --write    # splice it into FINDINGS.md

Every number in the table is read from ``benchmarks/results/*.csv``; nothing
is retyped. That is a direct response to this project's own defect history:
the 2026-09-02 audit found eighteen stale or overstated figures in prose that
had been hand-copied from an earlier run, so the cross-dataset table is
generated and carries the CSV mtimes it was built from.

Two tables, because the two dataset families cannot share a primary metric:

  * synthetic controls have planted truth, so the primary metric is
    truth-recovery F1 at equal selection size, with stability beside it;
  * real spectra have no truth, so the primary metric is split-half
    stability, with the uniform downstream score beside it. Reporting an F1
    for real data would require inventing a truth set.

The block between the two SENTINEL comments in FINDINGS.md is what --write
replaces, so re-running after a benchmark run refreshes the table in place
and leaves the surrounding interpretation alone.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

BENCH = Path(__file__).resolve().parent
RESULTS = BENCH / "results"
FINDINGS = BENCH / "FINDINGS.md"

BEGIN = "<!-- BEGIN GENERATED TABLE (make_findings_table.py) -->"
END = "<!-- END GENERATED TABLE -->"

# display order and display names; derived from the registry so a new
# competitor cannot be silently omitted from the table
sys.path.insert(0, str(BENCH))
from competitors import METHOD_ORDER  # noqa: E402


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(row, key):
    """Cell value as float, or None when the run recorded no value."""
    v = (row.get(key) or "").strip()
    if v in ("", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load():
    meta_path = RESULTS / "run_meta.csv"
    if not meta_path.exists():
        raise SystemExit(
            "benchmarks/results/run_meta.csv is missing; run\n"
            "  py benchmarks/run_synthetic_benchmark.py\n"
            "(a full run, not a subset) to produce it")
    meta = {r["dataset"]: r for r in _read(meta_path)}

    data, stamps = {}, [meta_path]
    for name in meta:
        p = RESULTS / f"benchmark_{name}.csv"
        if not p.exists():
            raise SystemExit(f"{p} is missing but run_meta.csv lists {name}")
        if p.stat().st_mtime < meta_path.stat().st_mtime - 3600:
            raise SystemExit(
                f"{p.name} is more than an hour older than run_meta.csv; "
                "the table would mix two runs. Re-run the benchmark.")
        data[name] = {r["method"]: r for r in _read(p)}
        stamps.append(p)
    return meta, data, stamps


def _fmt(v, nd=3):
    """Three decimals: the precision the CSVs actually store.

    Re-rounding to two decimals added a second rounding layer, which made
    sec.4 (0.61) and this table (0.60) disagree about one identical stored
    value, 0.605. A traceability table must not introduce its own rounding.
    """
    return "n/a" if v is None else f"{v:.{nd}f}"


def _cell(row, primary, secondary):
    if row is None:
        return "absent"
    if row.get("error"):
        return "failed"
    a, b = _num(row, primary), _num(row, secondary)
    if a is None and b is None:
        return "n/a"
    return f"{_fmt(a)} / {_fmt(b)}"


def _table(meta, data, names, primary, secondary, primary_label, secondary_label,
           bold_best=True):
    """Markdown table: rows are methods, columns are datasets.

    ``bold_best`` marks the winning primary metric per column. It is OFF for
    the real-data table on purpose: there the primary metric is stability,
    and sec.6 of FINDINGS records a dataset (tecator) where the perfectly
    stable method is the predictively worst one. Bolding a stability column
    would recommend that method typographically while the prose warns
    against it, so the real-data table ranks nothing and forces the reader
    to use both numbers.
    """
    head = "| method | " + " | ".join(names) + " |"
    rule = "|---|" + "---|" * len(names)
    lines = [head, rule]

    # best value per column, over the SELECTORS only; full_signal keeps every
    # segment, so including it would make "best stability" meaningless
    best = {}
    for name in names if bold_best else []:
        vals = [_num(data[name].get(m), primary) for m in METHOD_ORDER
                if m != "full_signal"]
        vals = [v for v in vals if v is not None]
        best[name] = max(vals) if vals else None

    for m in METHOD_ORDER:
        cells = []
        for name in names:
            row = data[name].get(m)
            text = _cell(row, primary, secondary)
            v = _num(row, primary)
            if (bold_best and m != "full_signal" and v is not None
                    and best.get(name) is not None
                    and abs(v - best[name]) < 5e-4):
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    budgets = " | ".join(f"k={meta[n]['k']} of {meta[n]['S']}" for n in names)
    lines.append(f"| _budget_ | {budgets} |")
    lines.append("")
    note = f"Cells are {primary_label} / {secondary_label}. "
    if bold_best:
        note += (f"Bold marks the best {primary_label} in the column among "
                 f"the selectors; full_signal is excluded from the marking "
                 f"because it retains every segment, making it the "
                 f"no-compression reference rather than a competitor.")
    else:
        note += ("Nothing is marked best: with no truth set the primary "
                 "metric is stability, and a perfectly stable selection can "
                 "be the predictively worst one (tecator anova here: "
                 "stability 1.000 at R2 0.281). Read both numbers in a cell "
                 "together. full_signal keeps every segment, so its "
                 "stability is 1.00 by construction.")
    lines.append(note)
    return "\n".join(lines)


def build(meta, data, stamps):
    synth = [n for n in meta if meta[n]["has_truth"] == "yes"]
    real = [n for n in meta if meta[n]["has_truth"] != "yes"]

    newest = datetime.fromtimestamp(
        max(p.stat().st_mtime for p in stamps), tz=timezone.utc).astimezone()

    out = [BEGIN, ""]
    out.append(f"Generated by `benchmarks/make_findings_table.py` from the "
               f"result CSVs of the run finishing "
               f"{newest.strftime('%Y-%m-%d %H:%M %Z')}. Do not edit inside "
               f"the sentinels; re-run the generator instead.")
    out.append("")
    out.append("**A. Synthetic controls (planted truth).** Primary metric is "
               "truth-recovery F1 at equal selection size; split-half "
               "chance-corrected stability over 3 replicates beside it.")
    out.append("")
    out.append(_table(meta, data, synth, "f1", "stability_adj",
                      "F1", "stability"))
    out.append("")
    out.append("**B. Real datasets (no ground truth).** No truth set exists, "
               "so the primary metric is split-half stability, with the "
               "uniform downstream score beside it (ridge R2 for tecator and "
               "corn, logistic or OVR AUC for the rest, computed on the "
               "selected segment means of mean-filled data so the comparison "
               "is between selections and not between predictors).")
    out.append("")
    out.append(_table(meta, data, real, "stability_adj", "downstream",
                      "stability", "downstream score", bold_best=False))
    out.append("")
    out.append("**Missingness present in each dataset**, since it decides "
               "which columns test the novelty axis at all:")
    out.append("")
    out.append("| dataset | task-relevant missingness |")
    out.append("|---|---|")
    for n in list(synth) + list(real):
        out.append(f"| {n} | {meta[n]['missingness']} |")
    out.append("")
    out.append(END)
    return "\n".join(out)


def main(argv):
    meta, data, stamps = load()
    block = build(meta, data, stamps)
    if "--write" not in argv:
        print(block)
        return
    text = FINDINGS.read_text(encoding="utf-8")
    if BEGIN in text and END in text:
        pre = text.split(BEGIN)[0]
        post = text.split(END, 1)[1]
        FINDINGS.write_text(pre + block + post, encoding="utf-8")
        print(f"table refreshed in place: {FINDINGS}")
    else:
        raise SystemExit(
            f"sentinels not found in {FINDINGS.name}; add\n  {BEGIN}\n  {END}\n"
            "where the table belongs, then re-run with --write")


if __name__ == "__main__":
    main(sys.argv[1:])
