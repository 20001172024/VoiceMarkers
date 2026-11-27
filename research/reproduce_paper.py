"""Reproduce the paper's row-level results (M1).

The paper (Arora et al., ICOIICS 2025, Section III) describes its protocol as: z-score
scaling, a stratified 80/20 train/test split, and an RBF SVM whose C and gamma
were tuned by grid search with 5-fold cross-validation. It reports 88.46% training
and 87.18% test accuracy (138/156 and 34/39 recordings). The original notebook
(and so its random seed and grid) is not available, so this script reports:

1. A single stratified 80/20 split with a seed fixed in advance (42).
2. The same split repeated over many seeds, to show how much a single-split
   accuracy depends on which rows happen to land in the test set.
3. Stratified 10-fold row-level cross-validation, a steadier summary.
4. The paper's protocol for the SVM in two variants, default settings and the
   described grid search, compared with the reported training and test accuracy.

One deliberate difference: the paper fitted the scaler on all 195 recordings
before splitting; here scaling sits inside the Pipeline, fitted on training rows.

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from data.download import DEFAULT_DEST
from research.dataset import Dataset, load_dataset
from research.metrics import SCALAR_METRICS, evaluate_fitted
from research.models import MODEL_NAMES, build_model

PAPER_SVM_ACCURACY = 34 / 39  # 87.18% test accuracy, as reported in the paper
PAPER_SVM_TRAIN_ACCURACY = 138 / 156  # 88.46% training accuracy, as reported
# The paper tunes C and gamma by grid search but does not list the grid; this is a
# conventional one, fixed before running.
PAPER_SVM_GRID = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1]}
N_TUNED_REPEATS = 100
TEST_SIZE = 0.2
SINGLE_SPLIT_SEED = 42  # chosen before looking at results; not searched for
N_REPEATS = 200
N_FOLDS = 10
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"


def _split(ds: Dataset, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 row-level split, as in the paper."""
    return train_test_split(ds.X, ds.y, test_size=TEST_SIZE, stratify=ds.y, random_state=seed)


def single_split(ds: Dataset, model_name: str, seed: int) -> dict[str, float]:
    """Fit on a stratified 80/20 row-level split; score on the 20% (and on the 80%)."""
    X_tr, X_te, y_tr, y_te = _split(ds, seed)
    model = build_model(model_name, random_state=seed).fit(X_tr, y_tr)
    return {
        **evaluate_fitted(model, X_te, y_te),
        "train_accuracy": float(model.score(X_tr, y_tr)),
    }


def tuned_svm_split(ds: Dataset, seed: int) -> dict[str, float]:
    """The paper's tuned variant: grid search over C and gamma with 5-fold CV on the 80%."""
    X_tr, X_te, y_tr, y_te = _split(ds, seed)
    search = GridSearchCV(
        build_model("SVM", random_state=seed),
        PAPER_SVM_GRID,
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        n_jobs=-1,
    ).fit(X_tr, y_tr)
    return {
        "accuracy": float(search.score(X_te, y_te)),
        "train_accuracy": float(search.best_estimator_.score(X_tr, y_tr)),
    }


def paper_protocol(ds: Dataset, default_runs: pd.DataFrame, n_tuned: int) -> dict:
    """Compare default and tuned SVMs with the paper's reported train/test accuracy."""
    tuned = pd.DataFrame([tuned_svm_split(ds, s) for s in range(n_tuned)])

    def describe(runs: pd.DataFrame) -> dict:
        test_match = np.isclose(runs["accuracy"], PAPER_SVM_ACCURACY)
        train_match = np.isclose(runs["train_accuracy"], PAPER_SVM_TRAIN_ACCURACY)
        return {
            "n_splits": len(runs),
            "train_accuracy": _summary(runs["train_accuracy"]),
            "accuracy": _summary(runs["accuracy"]),
            "n_test_equal_to_paper": int(test_match.sum()),
            "n_both_equal_to_paper": int((test_match & train_match).sum()),
        }

    return {
        "reported": {"train_accuracy": PAPER_SVM_TRAIN_ACCURACY, "accuracy": PAPER_SVM_ACCURACY},
        "grid": {k.removeprefix("clf__"): v for k, v in PAPER_SVM_GRID.items()},
        "default": describe(default_runs),
        "tuned": describe(tuned),
    }


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


def run(
    ds: Dataset,
    n_repeats: int = N_REPEATS,
    n_folds: int = N_FOLDS,
    n_tuned_repeats: int = N_TUNED_REPEATS,
) -> dict:
    """Run every protocol for every model and return JSON-serialisable results."""
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
            "n_tuned_repeats": n_tuned_repeats,
            "stratified": True,
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
        if name == "SVM":
            results["paper_protocol"] = paper_protocol(ds, reps, n_tuned_repeats)
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
    pp = results["paper_protocol"]
    rep_train, rep_test = pp["reported"]["train_accuracy"], pp["reported"]["accuracy"]

    def distance(v: dict) -> float:
        return abs(v["train_accuracy"]["mean"] - rep_train) + abs(v["accuracy"]["mean"] - rep_test)

    closer = "default" if distance(pp["default"]) <= distance(pp["tuned"]) else "tuned"
    pp_rows = [["Paper (Table I)", _pct(rep_train), _pct(rep_test), "–", "–"]]
    for key, label in (
        ("default", "SVM, default settings"),
        ("tuned", "SVM, grid search (as described)"),
    ):
        v = pp[key]
        pp_rows.append(
            [
                f"{label}, {v['n_splits']} splits",
                _mean_std(v["train_accuracy"]),
                _mean_std(v["accuracy"]),
                f"{v['n_test_equal_to_paper']} / {v['n_splits']}",
                f"{v['n_both_equal_to_paper']} / {v['n_splits']}",
            ]
        )
    grid = ", ".join(f"`{k}` ∈ {v}" for k, v in pp["grid"].items())
    closer_text = (
        "The reported pair is consistent with an SVM with **default settings**: its average "
        "training and test accuracy are close to the paper's, and some splits reproduce both "
        "numbers exactly. With the grid search the paper describes, the same protocol fits the "
        "training data almost perfectly and scores higher on the test set, so the reported "
        "figures most likely come from an untuned model."
        if closer == "default"
        else "The reported pair is closer to the tuned SVM than to the default one."
    )

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

The paper reports **{_pct(c['paper_svm_accuracy'])}** SVM test accuracy (34 of 39 test recordings correct) from one stratified 80/20 split. The original notebook and its random seed are not available, so the exact split cannot be recovered. Over {c['n_repeats']} random stratified 80/20 splits, the same SVM setup scores **{_mean_std(svm_rep)}** (5th–95th percentile {_pct(svm_rep['p5'])} – {_pct(svm_rep['p95'])}); {_pct(svm_eq)} of splits give exactly {_pct(c['paper_svm_accuracy'])} and {_pct(svm_ge)} give at least that. The paper's figure is therefore consistent with this setup, but a single split cannot pin a model's accuracy down to better than several percentage points. Always predicting "Parkinson's" scores {_pct(base)} on the seed-{c['single_split_seed']} test split.

With default settings, the best model by {c['n_folds']}-fold row-level CV accuracy is **{best}** ({_mean_std(cv[best]['accuracy'])}), against SVM's {_mean_std(svm_cv['accuracy'])}. SVM's high accuracy hides weak performance on the healthy group: its specificity is {_mean_std(svm_cv['specificity'])} and its balanced accuracy {_mean_std(svm_cv['balanced_accuracy'])}, because about 75% of recordings are Parkinson's.

## Matching the paper's protocol

The paper (Section III) scales the features, makes a stratified 80/20 split, and tunes an RBF SVM's C and gamma by grid search with 5-fold cross-validation; it reports **{_pct(rep_train)} training** and **{_pct(rep_test)} test** accuracy. The grid itself is not given; the one used here was fixed before running: {grid}.

{_table(["Setup", "Training accuracy", "Test accuracy", "Splits with test = paper", "Splits with both = paper"], pp_rows)}

{closer_text} Choosing one of the matching splits and calling it "the paper's split" would be selecting a seed after seeing the results, so this report does not do that.

## Table 1a — single 80/20 split (seed {c['single_split_seed']}, chosen in advance)

Test set: 39 recordings. Confusion matrix as TN / FP / FN / TP.

{_table(["Model", *metric_header, "TN / FP / FN / TP"], single_rows)}

## Table 1b — {c['n_repeats']} random stratified 80/20 splits (seeds 0–{c['n_repeats'] - 1})

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
    parser.add_argument("--n-tuned-repeats", type=int, default=N_TUNED_REPEATS)
    args = parser.parse_args(argv)

    results = run(load_dataset(args.data), args.n_repeats, args.n_folds, args.n_tuned_repeats)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "paper_reproduction.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out_dir / "paper_reproduction.md").write_text(to_markdown(results))
    print(f"Wrote {args.out_dir / 'paper_reproduction.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
