"""Reproduce the paper's row-level results (M1).

The paper (Arora et al., ICOIICS 2025) reports 88.46% training and 87.18% test
accuracy for an SVM (138/156 and 34/39 recordings). Its original notebook,
`research/original/Parkinsons.ipynb`, shows exactly how those numbers were made:

- an unstratified 80/20 split, `train_test_split(..., test_size=0.2, random_state=2)`;
- `StandardScaler` fitted on the training rows only;
- `SVC(kernel="linear")` with default C=1 and no hyperparameter search.

This script reports, all at row level:

1. An exact reproduction of the notebook (same split, scaler and model).
2. The notebook's linear SVM over many random splits, to show how much a
   single-split accuracy depends on which rows land in the test set.
3. The protocol the paper's *text* describes (Section III: stratified split, RBF
   SVM with C and gamma tuned by 5-fold grid search), with default and tuned
   settings, compared with the reported numbers.
4. The model comparison (Tables 1a-1c): every model on the notebook's split, over
   many splits, and with stratified 10-fold CV.

ROW-LEVEL means recordings of the same person can be in both train and test.
That inflates accuracy; the subject-wise evaluation comes in M2.

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data.download import DEFAULT_DEST
from research.dataset import Dataset, load_dataset
from research.metrics import SCALAR_METRICS, evaluate_fitted
from research.models import MODEL_NAMES, build_model

PAPER_SVM_ACCURACY = 34 / 39  # 87.18% test accuracy, as reported in the paper
PAPER_SVM_TRAIN_ACCURACY = 138 / 156  # 88.46% training accuracy, as reported
NOTEBOOK_SEED = 2  # random_state of the notebook's train_test_split (cell 21)
TEST_SIZE = 0.2
# The paper's text tunes C and gamma by grid search but does not list the grid;
# this is a conventional one, fixed before running.
PAPER_SVM_GRID = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1]}
N_TUNED_REPEATS = 100
N_REPEATS = 200
N_FOLDS = 10
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"


def _split(
    ds: Dataset, seed: int, stratify: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """80/20 row-level split; unstratified like the notebook unless `stratify`."""
    return train_test_split(
        ds.X, ds.y, test_size=TEST_SIZE, random_state=seed, stratify=ds.y if stratify else None
    )


def notebook_svm() -> Pipeline:
    """The notebook's model: scaler fitted on training rows, then a linear SVM (C=1)."""
    return Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="linear"))])


def _fit_and_score(model: Pipeline, split: tuple) -> dict[str, float]:
    X_tr, X_te, y_tr, y_te = split
    model.fit(X_tr, y_tr)
    return {**evaluate_fitted(model, X_te, y_te), "train_accuracy": float(model.score(X_tr, y_tr))}


def single_split(ds: Dataset, model_name: str, seed: int) -> dict[str, float]:
    """Fit one model on an unstratified 80/20 split; score on the 20% (and the 80%)."""
    return _fit_and_score(build_model(model_name, random_state=seed), _split(ds, seed))


def repeated_splits(ds: Dataset, model_name: str, seeds: Iterable[int]) -> pd.DataFrame:
    """Run `single_split` once per seed; one row of metrics per seed."""
    return pd.DataFrame([single_split(ds, model_name, s) for s in seeds])


def notebook_reproduction(ds: Dataset, n_repeats: int) -> dict:
    """The notebook's exact run, plus its linear SVM over `n_repeats` random splits."""
    exact = _fit_and_score(notebook_svm(), _split(ds, NOTEBOOK_SEED))
    runs = pd.DataFrame([_fit_and_score(notebook_svm(), _split(ds, s)) for s in range(n_repeats)])
    return {
        "seed": NOTEBOOK_SEED,
        "exact": exact,
        "matches_paper": bool(
            np.isclose(exact["accuracy"], PAPER_SVM_ACCURACY)
            and np.isclose(exact["train_accuracy"], PAPER_SVM_TRAIN_ACCURACY)
        ),
        "over_splits": {
            "n_splits": n_repeats,
            "train_accuracy": _summary(runs["train_accuracy"]),
            **{m: _summary(runs[m]) for m in SCALAR_METRICS},
        },
    }


def tuned_svm_split(ds: Dataset, seed: int) -> dict[str, float]:
    """The paper text's tuned variant: stratified split, grid search with 5-fold CV."""
    X_tr, X_te, y_tr, y_te = _split(ds, seed, stratify=True)
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


def described_protocol(ds: Dataset, n_default: int, n_tuned: int) -> dict:
    """The protocol in the paper's text: stratified split, RBF SVM, default vs tuned."""
    default = pd.DataFrame(
        [
            _fit_and_score(build_model("SVM", random_state=s), _split(ds, s, stratify=True))
            for s in range(n_default)
        ]
    )
    tuned = pd.DataFrame([tuned_svm_split(ds, s) for s in range(n_tuned)])

    def describe(runs: pd.DataFrame) -> dict:
        return {
            "n_splits": len(runs),
            "train_accuracy": _summary(runs["train_accuracy"]),
            "accuracy": _summary(runs["accuracy"]),
        }

    return {
        "grid": {k.removeprefix("clf__"): v for k, v in PAPER_SVM_GRID.items()},
        "rbf_default": describe(default),
        "rbf_tuned": describe(tuned),
    }


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
    """Run every protocol and return JSON-serialisable results."""
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
            "single_split_seed": NOTEBOOK_SEED,
            "n_repeats": n_repeats,
            "n_folds": n_folds,
            "n_tuned_repeats": n_tuned_repeats,
            "paper_svm_accuracy": PAPER_SVM_ACCURACY,
            "paper_svm_train_accuracy": PAPER_SVM_TRAIN_ACCURACY,
        },
        "notebook": notebook_reproduction(ds, n_repeats),
        "described_protocol": described_protocol(ds, n_repeats, n_tuned_repeats),
        "single_split": {},
        "repeated_splits": {},
        "row_level_cv": {},
    }
    for name in MODEL_NAMES:
        results["single_split"][name] = single_split(ds, name, NOTEBOOK_SEED)
        reps = repeated_splits(ds, name, range(n_repeats))
        results["repeated_splits"][name] = {
            **{m: _summary(reps[m]) for m in SCALAR_METRICS},
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
    nb, dp = results["notebook"], results["described_protocol"]
    ex, over = nb["exact"], nb["over_splits"]
    rep_train, rep_test = c["paper_svm_train_accuracy"], c["paper_svm_accuracy"]
    base = results["single_split"]["Majority baseline"]["accuracy"]
    cv = {k: v for k, v in results["row_level_cv"].items() if k != "Majority baseline"}
    best = max(cv, key=lambda k: cv[k]["accuracy"]["mean"])

    exact_table = _table(
        ["", "Training accuracy", "Test accuracy"],
        [
            ["Paper (Table I) and notebook output", _pct(rep_train), _pct(rep_test)],
            ["This reproduction", _pct(ex["train_accuracy"]), _pct(ex["accuracy"])],
        ],
    )
    exact_metrics = _table(
        ["Bal. acc.", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "TN / FP / FN / TP"],
        [
            [
                _pct(ex[k])
                for k in (
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "specificity",
                    "f1",
                    "roc_auc",
                )
            ]
            + [f"{ex['tn']} / {ex['fp']} / {ex['fn']} / {ex['tp']}"]
        ],
    )
    match_text = (
        "**The reproduction matches the paper exactly.**"
        if nb["matches_paper"]
        else "**The reproduction does not match the paper** — see the numbers above."
    )
    described_rows = [
        [
            "Paper (Table I) / notebook",
            "linear SVM, unstratified, seed 2",
            _pct(rep_train),
            _pct(rep_test),
        ],
        [
            f"Notebook's model over {over['n_splits']} splits",
            "linear SVM, unstratified",
            _mean_std(over["train_accuracy"]),
            _mean_std(over["accuracy"]),
        ],
        [
            f"Described, default settings ({dp['rbf_default']['n_splits']} splits)",
            "RBF SVM, stratified",
            _mean_std(dp["rbf_default"]["train_accuracy"]),
            _mean_std(dp["rbf_default"]["accuracy"]),
        ],
        [
            f"Described, grid search ({dp['rbf_tuned']['n_splits']} splits)",
            "RBF SVM, stratified, tuned C and gamma",
            _mean_std(dp["rbf_tuned"]["train_accuracy"]),
            _mean_std(dp["rbf_tuned"]["accuracy"]),
        ],
    ]
    grid = ", ".join(f"`{k}` ∈ {v}" for k, v in dp["grid"].items())

    single_rows = [
        [name]
        + [_pct(m[k]) for k in SCALAR_METRICS]
        + [f"{m['tn']} / {m['fp']} / {m['fn']} / {m['tp']}"]
        for name, m in results["single_split"].items()
    ]
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
- scikit-learn {c['scikit_learn']}. Positive class = Parkinson's, so recall is sensitivity.
- Original notebook: [`research/original/Parkinsons.ipynb`](../original/Parkinsons.ipynb).

> **All results on this page are row-level:** recordings of the same person can appear in both the training and the test set, which inflates scores. Subject-wise results are reported separately (M2).

## 1. Exact reproduction of the original notebook

The notebook splits 80/20 without stratification (`random_state={nb['seed']}`), fits `StandardScaler` on the training rows, and trains `SVC(kernel='linear')` with the default `C=1` (cells 21–38).

{exact_table}

{match_text} The notebook computes accuracy only; the precision, recall and F1 in the paper's Table I, and its rows for the other models, are not in it, so they cannot be traced to code. The same model on that test split:

{exact_metrics}

On this split the model recognises {_pct(ex['specificity'])} of healthy recordings, and always predicting "Parkinson's" would already score {_pct(base)}.

## 2. How much one split can vary

Over {over['n_splits']} random unstratified 80/20 splits the notebook's linear SVM scores **{_mean_std(over['accuracy'])}** test accuracy (5th–95th percentile {_pct(over['accuracy']['p5'])} – {_pct(over['accuracy']['p95'])}) and {_mean_std(over['balanced_accuracy'])} balanced accuracy. The paper's {_pct(rep_test)} is a typical draw from this range; a single split cannot pin accuracy down to better than several percentage points.

## 3. The protocol described in the paper's text

The paper's Section III describes a stratified split and an RBF-kernel SVM whose C and gamma were tuned by grid search with 5-fold cross-validation. The notebook instead used a linear kernel with default settings and an unstratified split. For comparison, the described protocol was run as written; the grid was not published, so a conventional one was fixed before running ({grid}).

{_table(["Setup", "Model", "Training accuracy", "Test accuracy"], described_rows)}

The reported figures come from the notebook's linear SVM. Run as described, the tuned RBF SVM scores {_mean_std(dp['rbf_tuned']['accuracy'])} on the test set at row level, but this is still a row-level estimate (see M2).

## 4. Model comparison (default settings)

The models below are each `StandardScaler → classifier` in one Pipeline with scikit-learn default settings (SVM = RBF kernel; Logistic Regression: `max_iter` raised so it converges). By {c['n_folds']}-fold row-level CV the best is **{best}** ({_mean_std(cv[best]['accuracy'])}).

### Table 1a — the notebook's split (seed {c['single_split_seed']})

Test set: 39 recordings. Confusion matrix as TN / FP / FN / TP.

{_table(["Model", *metric_header, "TN / FP / FN / TP"], single_rows)}

### Table 1b — {c['n_repeats']} random unstratified 80/20 splits (seeds 0–{c['n_repeats'] - 1})

{_table(["Model", "Accuracy (mean ± std)", "Accuracy 5th–95th pct", "Accuracy min – max", "Bal. acc.", "ROC-AUC", f"Splits ≥ {_pct(rep_test)}"], rep_rows)}

### Table 1c — stratified {c['n_folds']}-fold cross-validation (row-level)

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
