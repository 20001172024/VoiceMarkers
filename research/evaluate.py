"""Subject-wise re-evaluation of every model (M2).

Each subject contributes ~6 recordings. A row-level split puts the same person
in train and test, so a model can score well by recognising the *speaker*
rather than the disease. This script compares, on identical outer folds:

- ``row_default``      row-level CV, default settings (the paper's setup, leaky)
- ``subject_default``  subject-wise CV, default settings (leak removed)
- ``subject_tuned``    subject-wise CV, settings tuned by an inner subject-wise
                       grid search on the outer training set only (nested CV)

so the gap between the first two is the leakage effect and the gap between the
last two is the tuning effect. Outer CV is 5 folds x 10 repeats: with only 8
healthy subjects, each test fold holds 1-2 of them, so a single 5-fold run is
too noisy to rank models. Leave-one-subject-out (LOSO) is reported as a
cross-check, with metrics pooled over all held-out subjects (each LOSO test fold
contains one class only, so per-fold balanced accuracy and ROC-AUC are undefined).

Every (model, protocol) run is logged to MLflow (local SQLite store).

Usage (from the repo root, after `python -m data.download`):
    python -m research.evaluate
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from app.audio.features import REPRODUCIBLE_FEATURES
from data.download import DEFAULT_DEST
from research.dataset import Dataset, load_dataset
from research.metrics import SCALAR_METRICS, compute_metrics, evaluate_fitted, positive_scores
from research.models import BASELINE, MODEL_NAMES, build_model

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "voicemarkers-evaluation"

N_SPLITS = 5
N_REPEATS = 10
INNER_SPLITS = 4
# Parallel workers for the inner grid search (-1 = all cores). Tests set this to 1:
# for tiny datasets, starting worker processes costs more than the fits.
N_JOBS = -1
TUNING_METRIC = "balanced_accuracy"  # plain accuracy would reward "always Parkinson's"

# Small grids on purpose: with ~150 training recordings from ~25 people, a big
# search mostly fits noise in the inner folds.
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    BASELINE: {},
    "SVM": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.1],
        "clf__class_weight": [None, "balanced"],
    },
    "Random Forest": {
        "clf__max_depth": [None, 4],
        "clf__min_samples_leaf": [1, 3],
        "clf__class_weight": [None, "balanced"],
    },
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": [None, "balanced"],
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 11, 15],
        "clf__weights": ["uniform", "distance"],
    },
    "Gradient Boosting": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [2, 3],
        "clf__learning_rate": [0.05, 0.1],
    },
}
PROTOCOLS: tuple[str, ...] = ("row_default", "subject_default", "subject_tuned")
FEATURE_SETS: tuple[str, ...] = ("all", "reproducible")


@dataclass(frozen=True)
class Fold:
    """One outer train/test split."""

    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


def outer_folds(
    ds: Dataset, subject_wise: bool, n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS
) -> Iterator[Fold]:
    """Yield outer folds; repeat r always uses seed r, so protocols are comparable.

    Subject-wise folds keep every recording of a subject on one side of the split,
    and stratify so each test fold holds both classes.
    """
    for r in range(n_repeats):
        if subject_wise:
            splits = StratifiedGroupKFold(n_splits, shuffle=True, random_state=r).split(
                ds.X, ds.y, ds.groups
            )
        else:
            splits = StratifiedKFold(n_splits, shuffle=True, random_state=r).split(ds.X, ds.y)
        for f, (train, test) in enumerate(splits):
            yield Fold(r, f, train, test)


def fit_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    tune: bool,
    seed: int,
    inner_splits: int = INNER_SPLITS,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit one model on training data only; tune it with an inner subject-wise CV.

    Returns the fitted pipeline and the chosen settings (empty when not tuned).
    The inner CV groups by subject too, otherwise tuning itself would leak.
    """
    grid = PARAM_GRIDS[name]
    if not tune or not grid:
        return build_model(name, random_state=seed).fit(X, y), {}
    search = GridSearchCV(
        build_model(name, random_state=seed),
        grid,
        cv=StratifiedGroupKFold(inner_splits, shuffle=True, random_state=seed),
        scoring=TUNING_METRIC,
        n_jobs=N_JOBS,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_


def cross_validate(
    ds: Dataset,
    name: str,
    protocol: str,
    *,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    inner_splits: int = INNER_SPLITS,
) -> dict[str, Any]:
    """Run one model under one protocol on the shared outer folds.

    Returns per-fold metrics, the settings chosen in each fold, and out-of-fold
    predictions from repeat 0 (each recording predicted exactly once) for the
    pooled confusion matrix and ROC curve.
    """
    subject_wise = protocol != "row_default"
    tune = protocol == "subject_tuned"
    fold_rows, chosen = [], []
    oof_pred = np.full(len(ds.y), -1)
    oof_score = np.full(len(ds.y), np.nan)
    for fold in outer_folds(ds, subject_wise, n_splits, n_repeats):
        model, params = fit_model(
            name,
            ds.X.iloc[fold.train],
            ds.y.iloc[fold.train],
            ds.groups.iloc[fold.train],
            tune=tune,
            seed=fold.repeat,
            inner_splits=inner_splits,
        )
        X_te, y_te = ds.X.iloc[fold.test], ds.y.iloc[fold.test]
        fold_rows.append(evaluate_fitted(model, X_te, y_te))
        chosen.append(params)
        if fold.repeat == 0:
            oof_pred[fold.test] = model.predict(X_te)
            oof_score[fold.test] = positive_scores(model, X_te)
    folds = pd.DataFrame(fold_rows)
    return {
        "summary": {
            m: {"mean": folds[m].mean(), "std": folds[m].std(ddof=0)} for m in SCALAR_METRICS
        },
        "folds": folds.to_dict(orient="list"),
        "chosen_params": _param_counts(chosen),
        "oof_repeat0": {
            "pred": oof_pred.tolist(),
            "score": oof_score.tolist(),
            "pooled": compute_metrics(ds.y, oof_pred, oof_score),
        },
    }


def leave_one_subject_out(
    ds: Dataset, name: str, *, tune: bool, inner_splits: int = INNER_SPLITS
) -> dict[str, float]:
    """LOSO CV; metrics are computed once on predictions pooled over all subjects.

    Note: decision-function scores from different fold models are pooled for
    ROC-AUC, which assumes they are on comparable scales (true for these models
    in practice, but it is an approximation).
    """
    pred = np.empty(len(ds.y), dtype=int)
    score = np.empty(len(ds.y))
    for train, test in LeaveOneGroupOut().split(ds.X, ds.y, ds.groups):
        model, _ = fit_model(
            name,
            ds.X.iloc[train],
            ds.y.iloc[train],
            ds.groups.iloc[train],
            tune=tune,
            seed=0,
            inner_splits=inner_splits,
        )
        pred[test] = model.predict(ds.X.iloc[test])
        score[test] = positive_scores(model, ds.X.iloc[test])
    return compute_metrics(ds.y, pred, score)


def feature_importance(
    ds: Dataset, name: str, *, n_splits: int = N_SPLITS, inner_splits: int = INNER_SPLITS
) -> dict[str, dict[str, float]]:
    """Permutation importance of the tuned model, measured on outer test folds only.

    For each fold of repeat 0, the tuned model is fitted on the training subjects
    and each feature is shuffled in the held-out subjects' recordings; the drop
    in balanced accuracy is that feature's importance. Averaged over folds.
    """
    per_fold = []
    for fold in outer_folds(ds, subject_wise=True, n_splits=n_splits, n_repeats=1):
        model, _ = fit_model(
            name,
            ds.X.iloc[fold.train],
            ds.y.iloc[fold.train],
            ds.groups.iloc[fold.train],
            tune=True,
            seed=0,
            inner_splits=inner_splits,
        )
        result = permutation_importance(
            model,
            ds.X.iloc[fold.test],
            ds.y.iloc[fold.test],
            scoring=TUNING_METRIC,
            n_repeats=20,
            random_state=0,
        )
        per_fold.append(result.importances_mean)
    imp = np.array(per_fold)
    order = np.argsort(-imp.mean(axis=0))
    return {
        ds.X.columns[i]: {"mean": float(imp[:, i].mean()), "std": float(imp[:, i].std())}
        for i in order
    }


def _param_counts(chosen: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count how often each settings combination was chosen across folds."""
    keys = Counter(json.dumps(p, sort_keys=True, default=str) for p in chosen if p)
    return [{"params": json.loads(k), "folds": n} for k, n in keys.most_common()]


def _log_to_mlflow(results: dict[str, Any], tracking_uri: str) -> None:
    """Log one MLflow run per (model, protocol), plus one per LOSO run."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow  # imported lazily: slow to import, and only needed here

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    common = {**results["config"], "dataset_sha256": results["dataset"]["sha256"]}
    for name, by_protocol in results["cv"].items():
        for protocol, res in by_protocol.items():
            with mlflow.start_run(run_name=f"{name} | {protocol}"):
                mlflow.log_params({**common, "model": name, "protocol": protocol})
                mlflow.set_tag("feature_set", results["config"]["feature_set"])
                for m, s in res["summary"].items():
                    mlflow.log_metric(f"{m}_mean", s["mean"])
                    mlflow.log_metric(f"{m}_std", s["std"])
                mlflow.log_dict(res["chosen_params"], "chosen_params.json")
    for name, by_tuning in results["loso"].items():
        for tuning, metrics in by_tuning.items():
            with mlflow.start_run(run_name=f"{name} | loso_{tuning}"):
                mlflow.log_params({**common, "model": name, "protocol": f"loso_{tuning}"})
                mlflow.log_metrics({k: v for k, v in metrics.items() if not np.isnan(v)})


def run(
    ds: Dataset,
    *,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    inner_splits: int = INNER_SPLITS,
    models: tuple[str, ...] = MODEL_NAMES,
    feature_set: str = "all",
) -> dict[str, Any]:
    """Run every protocol for every model, LOSO, and feature importance.

    `feature_set` is a label recorded in the results; the features used are
    whatever columns `ds.X` holds.
    """
    results: dict[str, Any] = {
        "dataset": {
            "sha256": ds.sha256,
            "n_recordings": len(ds.y),
            "n_subject_ids": int(ds.groups.nunique()),
            "n_parkinsons": int(ds.y.sum()),
            "n_healthy": int((ds.y == 0).sum()),
            "n_parkinsons_subjects": int(ds.groups[ds.y == 1].nunique()),
            "n_healthy_subjects": int(ds.groups[ds.y == 0].nunique()),
            "y": ds.y.tolist(),
        },
        "config": {
            "scikit_learn": sklearn.__version__,
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "inner_splits": inner_splits,
            "tuning_metric": TUNING_METRIC,
            "feature_set": feature_set,
            "features": list(ds.X.columns),
        },
        "param_grids": {k: v for k, v in PARAM_GRIDS.items() if k in models},
        "cv": {},
        "loso": {},
    }
    for name in models:
        print(f"Evaluating {name}...", flush=True)
        results["cv"][name] = {
            p: cross_validate(
                ds, name, p, n_splits=n_splits, n_repeats=n_repeats, inner_splits=inner_splits
            )
            for p in PROTOCOLS
        }
        results["loso"][name] = {
            t: leave_one_subject_out(ds, name, tune=(t == "tuned"), inner_splits=inner_splits)
            for t in ("default", "tuned")
        }

    candidates = [m for m in models if m != BASELINE]
    best = max(
        candidates,
        key=lambda m: results["cv"][m]["subject_tuned"]["summary"][TUNING_METRIC]["mean"],
    )
    print(f"Feature importance for {best}...", flush=True)
    results["best_model"] = best
    results["feature_importance"] = feature_importance(
        ds, best, n_splits=n_splits, inner_splits=inner_splits
    )
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: evaluate, write JSON, log to MLflow."""
    parser = argparse.ArgumentParser(description="Subject-wise evaluation of every model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--features",
        choices=FEATURE_SETS,
        default="all",
        help="all 22 UCI features, or only the 15 reproducible from audio (M3)",
    )
    parser.add_argument("--out", type=Path, help="default: results/evaluation[_reproducible].json")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--inner-splits", type=int, default=INNER_SPLITS)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args(argv)

    ds = load_dataset(args.data)
    if args.features == "reproducible":
        ds = ds.select(REPRODUCIBLE_FEATURES)
    out = args.out or RESULTS_DIR / (
        "evaluation.json" if args.features == "all" else "evaluation_reproducible.json"
    )
    results = run(
        ds,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        inner_splits=args.inner_splits,
        feature_set=args.features,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=float) + "\n")
    print(f"Wrote {out}")
    if not args.no_mlflow:
        _log_to_mlflow(results, args.tracking_uri)
        print(f"Logged runs to MLflow ({args.tracking_uri})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
