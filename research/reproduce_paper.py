"""Reproduce the paper's row-level results (M1).

The paper (ICOIICS 2025) reports 87.18% accuracy for an SVM. 87.18% is exactly
34/39, and 39 is the test size of an 80/20 split of the 195 recordings, so the
paper's protocol was a single random row-level 80/20 split. The original
notebook (and so its random seed) is not available, so this script reports:

1. A single 80/20 split with a seed fixed in advance (42), like the paper.
2. The same split repeated over many seeds, to show how much a single-split
   accuracy depends on which rows happen to land in the test set.
3. Stratified 10-fold row-level cross-validation, a steadier summary.

All three are ROW-LEVEL: recordings of the same person can be in both train and
test. That inflates accuracy; the subject-wise evaluation comes in M2.

Usage (from the repo root, after `python -m data.download`):
    python -m research.reproduce_paper
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split

from data.download import DEFAULT_DEST
from research.dataset import Dataset, load_dataset
from research.metrics import SCALAR_METRICS, evaluate_fitted
from research.models import MODEL_NAMES, build_model

PAPER_SVM_ACCURACY = 34 / 39  # 87.18%, as reported in the paper
TEST_SIZE = 0.2
SINGLE_SPLIT_SEED = 42  # chosen before looking at results; not searched for
N_REPEATS = 200
N_FOLDS = 10
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"


def single_split(ds: Dataset, model_name: str, seed: int) -> dict[str, float]:
    """Fit on a random 80/20 row-level split and score on the 20%."""
    X_tr, X_te, y_tr, y_te = train_test_split(ds.X, ds.y, test_size=TEST_SIZE, random_state=seed)
    model = build_model(model_name, random_state=seed).fit(X_tr, y_tr)
    return evaluate_fitted(model, X_te, y_te)


def repeated_splits(ds: Dataset, model_name: str, seeds: Iterable[int]) -> pd.DataFrame:
    """Run `single_split` once per seed; one row of metrics per seed."""
    return pd.DataFrame([single_split(ds, model_name, s) for s in seeds])


def row_level_cv(ds: Dataset, model_name: str, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Stratified k-fold CV at row level; one row of metrics per fold."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    rows = []
    for train_idx, test_idx in cv.split(ds.X, ds.y):
        model = build_model(model_name, random_state=0)
        model.fit(ds.X.iloc[train_idx], ds.y.iloc[train_idx])
        rows.append(evaluate_fitted(model, ds.X.iloc[test_idx], ds.y.iloc[test_idx]))
    return pd.DataFrame(rows)


def _summary(values: pd.Series) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        # "nearest" keeps percentiles on values a split can actually produce
        # (accuracy on 39 test recordings moves in steps of 1/39).
        "p5": float(values.quantile(0.05, interpolation="nearest")),
        "p95": float(values.quantile(0.95, interpolation="nearest")),
        "max": float(values.max()),
    }


def run(ds: Dataset, n_repeats: int = N_REPEATS, n_folds: int = N_FOLDS) -> dict:
    """Run all three protocols for every model and return JSON-serialisable results."""
    results: dict = {
        "dataset": {
            "sha256": ds.sha256,
            "n_recordings": len(ds.y),
            "n_subject_ids": int(ds.groups.nunique()),
            "n_parkinsons": int(ds.y.sum()),
            "n_healthy": int((ds.y == 0).sum()),
        },
        "config": {
            "scikit_learn": sklearn.__version__,
            "test_size": TEST_SIZE,
            "single_split_seed": SINGLE_SPLIT_SEED,
            "n_repeats": n_repeats,
            "n_folds": n_folds,
            "paper_svm_accuracy": PAPER_SVM_ACCURACY,
        },
        "single_split": {},
        "repeated_splits": {},
        "row_level_cv": {},
    }
    for name in MODEL_NAMES:
        results["single_split"][name] = single_split(ds, name, SINGLE_SPLIT_SEED)

        reps = repeated_splits(ds, name, range(n_repeats))
        results["repeated_splits"][name] = {
            **{m: _summary(reps[m]) for m in SCALAR_METRICS},
            "share_equal_to_paper": float(np.isclose(reps["accuracy"], PAPER_SVM_ACCURACY).mean()),
            "share_at_least_paper": float((reps["accuracy"] >= PAPER_SVM_ACCURACY - 1e-9).mean()),
        }

        folds = row_level_cv(ds, name, n_folds)
        results["row_level_cv"][name] = {m: _summary(folds[m]) for m in SCALAR_METRICS}
    return results


def _pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.2%}"


def _mean_std(s: dict[str, float]) -> str:
    if math.isnan(s["mean"]):
        return "n/a"
    return f"{s['mean']:.2%} ± {s['std']:.2%}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def to_markdown(results: dict) -> str:
    """Render the results dict as the Markdown report."""
    d, c = results["dataset"], results["config"]
    svm_rep = results["repeated_splits"]["SVM"]["accuracy"]
    svm_eq = results["repeated_splits"]["SVM"]["share_equal_to_paper"]
    svm_ge = results["repeated_splits"]["SVM"]["share_at_least_paper"]
    base = results["single_split"]["Majority baseline"]["accuracy"]
    cv = {k: v for k, v in results["row_level_cv"].items() if k != "Majority baseline"}
    best = max(cv, key=lambda k: cv[k]["accuracy"]["mean"])
    svm_cv = results["row_level_cv"]["SVM"]

    single_rows = []
    for name, m in results["single_split"].items():
        single_rows.append(
            [name]
            + [_pct(m[k]) for k in SCALAR_METRICS]
            + [f"{m['tn']} / {m['fp']} / {m['fn']} / {m['tp']}"]
        )
    rep_rows = []
    for name, r in results["repeated_splits"].items():
        a = r["accuracy"]
        rep_rows.append(
            [
                name,
                _mean_std(a),
                f"{_pct(a['p5'])} – {_pct(a['p95'])}",
                f"{_pct(a['min'])} – {_pct(a['max'])}",
                _mean_std(r["balanced_accuracy"]),
                _mean_std(r["roc_auc"]),
                _pct(r["share_at_least_paper"]),
            ]
        )
    cv_rows = [
        [name] + [_mean_std(r[k]) for k in SCALAR_METRICS]
        for name, r in results["row_level_cv"].items()
    ]
    metric_header = ["Accuracy", "Bal. acc.", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]

    return f"""# Paper reproduction (row-level)

_Generated by `research/reproduce_paper.py` — do not edit by hand._

- Dataset: {d['n_recordings']} recordings ({d['n_parkinsons']} Parkinson's, {d['n_healthy']} healthy), {d['n_subject_ids']} subject IDs. SHA-256 `{d['sha256'][:12]}…`
- scikit-learn {c['scikit_learn']}. Every model is `StandardScaler → classifier` in one Pipeline, with scikit-learn default settings (Logistic Regression: `max_iter` raised so it converges).
- Positive class = Parkinson's, so recall is sensitivity.

> **All results on this page are row-level:** recordings of the same person can appear in both the training and the test set, which inflates scores. Subject-wise results are reported separately (M2).

## Summary

The paper reports **{_pct(c['paper_svm_accuracy'])}** SVM accuracy (34 of 39 test recordings correct, i.e. one 80/20 split). The original notebook and its random seed are not available, so the exact split cannot be recovered. Over {c['n_repeats']} random 80/20 splits, the same SVM setup scores **{_mean_std(svm_rep)}** (5th–95th percentile {_pct(svm_rep['p5'])} – {_pct(svm_rep['p95'])}); {_pct(svm_eq)} of splits give exactly {_pct(c['paper_svm_accuracy'])} and {_pct(svm_ge)} give at least that. The paper's figure is therefore consistent with this setup, but a single split cannot pin a model's accuracy down to better than several percentage points. Always predicting "Parkinson's" scores {_pct(base)} on the seed-{c['single_split_seed']} test split.

With default settings, the best model by {c['n_folds']}-fold row-level CV accuracy is **{best}** ({_mean_std(cv[best]['accuracy'])}), against SVM's {_mean_std(svm_cv['accuracy'])}. SVM's high accuracy hides weak performance on the healthy group: its specificity is {_mean_std(svm_cv['specificity'])} and its balanced accuracy {_mean_std(svm_cv['balanced_accuracy'])}, because about 75% of recordings are Parkinson's.

## Table 1a — single 80/20 split (seed {c['single_split_seed']}, chosen in advance)

Test set: 39 recordings. Confusion matrix as TN / FP / FN / TP.

{_table(["Model", *metric_header, "TN / FP / FN / TP"], single_rows)}

## Table 1b — {c['n_repeats']} random 80/20 splits (seeds 0–{c['n_repeats'] - 1})

{_table(["Model", "Accuracy (mean ± std)", "Accuracy 5th–95th pct", "Accuracy min – max", "Bal. acc.", "ROC-AUC", f"Splits ≥ {_pct(c['paper_svm_accuracy'])}"], rep_rows)}

## Table 1c — stratified {c['n_folds']}-fold cross-validation (row-level)

Mean ± std across folds.

{_table(["Model", *metric_header], cv_rows)}
"""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run everything and write JSON + Markdown results."""
    parser = argparse.ArgumentParser(description="Reproduce the paper's row-level results.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = parser.parse_args(argv)

    results = run(load_dataset(args.data), args.n_repeats, args.n_folds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "paper_reproduction.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out_dir / "paper_reproduction.md").write_text(to_markdown(results))
    print(f"Wrote {args.out_dir / 'paper_reproduction.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
