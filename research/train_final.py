"""Train and save the deployed model on the audio-reproducible features (M3).

Model choice: Logistic Regression, unless it is clearly worse than the best
model on the reproducible features, where "clearly" means its mean subject-wise
tuned balanced accuracy is more than one standard error below the best model's.
Logistic Regression is preferred because each feature's contribution to a
prediction is exactly coefficient x scaled value (an honest, simple
explanation). M2 showed no model is clearly better than the others, so
simplicity breaks the tie.

Probabilities: the tuned pipeline is wrapped in Platt (sigmoid) calibration,
fitted on out-of-fold scores from a subject-wise CV, so every model (SVM
included) outputs a probability and the calibration never sees a subject's own
recordings. SVC's built-in `probability=True` would calibrate on row-level
folds, which leaks subjects.

The final model is tuned once more on all 32 subjects (subject-wise grid
search) and refitted on all data. Its performance estimate is NOT that inner
score (it is optimistic); it is the nested-CV estimate from
`research/results/evaluation_reproducible.json`, copied into the metadata.

Usage (from the repo root, after `python -m research.evaluate --features reproducible`):
    python -m research.train_final
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from app.audio.features import REPRODUCIBLE_FEATURES
from data.download import DEFAULT_DEST
from research.dataset import Dataset, load_dataset
from research.evaluate import PARAM_GRIDS, TUNING_METRIC
from research.metrics import SCALAR_METRICS
from research.models import BASELINE, build_model

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "backend" / "models"
RESULTS_DIR = REPO_ROOT / "research" / "results"
PREFERRED_MODEL = "Logistic Regression"
MODEL_VERSION = "1.0.0"
FINAL_CV_SPLITS = 5
PERCENTILES = (5, 25, 50, 75, 95)


def select_model(evaluation: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Apply the selection rule to nested-CV results; return (model name, rationale)."""
    cv = evaluation["cv"]
    n_outer = evaluation["config"]["n_splits"] * evaluation["config"]["n_repeats"]
    scores = {m: cv[m]["subject_tuned"]["summary"][TUNING_METRIC] for m in cv if m != BASELINE}
    best = max(scores, key=lambda m: scores[m]["mean"])
    standard_error = scores[best]["std"] / math.sqrt(n_outer)
    gap = scores[best]["mean"] - scores[PREFERRED_MODEL]["mean"]
    chosen = PREFERRED_MODEL if gap <= standard_error else best
    return chosen, {
        "rule": (
            f"{PREFERRED_MODEL} unless its mean subject-wise tuned "
            f"{TUNING_METRIC.replace('_', ' ')} is more than "
            "one standard error below the best model's"
        ),
        "best_model": best,
        "best_mean": scores[best]["mean"],
        "preferred_mean": scores[PREFERRED_MODEL]["mean"],
        "gap": gap,
        "standard_error": standard_error,
        "note": (
            "Standard error = std / sqrt(number of outer folds). Repeated CV folds are not "
            "independent, so this is a lenient threshold."
        ),
    }


def feature_distributions(ds: Dataset) -> dict[str, dict[str, dict[str, float]]]:
    """Percentiles of each feature overall and per class, for the UI's range comparison."""
    groups = {"all": ds.y >= 0, "parkinsons": ds.y == 1, "healthy": ds.y == 0}
    return {
        f: {
            g: {
                f"p{p}": float(v)
                for p, v in zip(
                    PERCENTILES, np.percentile(ds.X.loc[mask, f], PERCENTILES), strict=True
                )
            }
            for g, mask in groups.items()
        }
        for f in ds.X.columns
    }


def train(ds: Dataset, model_name: str) -> tuple[Any, dict[str, Any], float]:
    """Tune on all subjects with a subject-wise grid search and refit on all data."""
    search = GridSearchCV(
        build_model(model_name, random_state=0),
        PARAM_GRIDS[model_name],
        cv=StratifiedGroupKFold(FINAL_CV_SPLITS, shuffle=True, random_state=0),
        scoring=TUNING_METRIC,
        n_jobs=-1,
    )
    search.fit(ds.X, ds.y, groups=ds.groups)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def calibrate(model: Any, ds: Dataset) -> CalibratedClassifierCV:
    """Wrap the tuned pipeline so it outputs calibrated probabilities.

    With ``ensemble=False``, out-of-fold scores from subject-wise folds fit the
    sigmoid, then one copy of the pipeline is refitted on all data.
    """
    splits = list(
        StratifiedGroupKFold(FINAL_CV_SPLITS, shuffle=True, random_state=1).split(
            ds.X, ds.y, ds.groups
        )
    )
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=splits, ensemble=False)
    return calibrated.fit(ds.X, ds.y)


def build_metadata(
    ds: Dataset,
    model_name: str,
    params: dict[str, Any],
    inner_score: float,
    rationale: dict[str, Any],
    evaluation: dict[str, Any],
    evaluation_full: dict[str, Any] | None,
) -> dict[str, Any]:
    """Everything the API and UI need to describe the model honestly."""

    def summary(ev: dict[str, Any]) -> dict[str, dict[str, float]]:
        s = ev["cv"][model_name]["subject_tuned"]["summary"]
        return {m: {"mean": s[m]["mean"], "std": s[m]["std"]} for m in SCALAR_METRICS}

    return {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(UTC).date().isoformat(),
        "model_type": model_name,
        "params": {k.removeprefix("clf__"): v for k, v in params.items()},
        "features": list(ds.X.columns),
        "positive_class": "Parkinson's (status = 1)",
        "probability": (
            "Platt (sigmoid) calibration fitted on out-of-fold scores from "
            f"{FINAL_CV_SPLITS}-fold subject-wise CV"
        ),
        "scikit_learn": sklearn.__version__,
        "training_data": {
            "dataset": "UCI Parkinson's (Little et al. 2007), CC BY 4.0",
            "sha256": ds.sha256,
            "n_recordings": len(ds.y),
            "n_subject_ids": int(ds.groups.nunique()),
        },
        "selection": rationale,
        "evaluation": {
            "protocol": (
                f"Nested subject-wise CV: {evaluation['config']['n_splits']} folds x "
                f"{evaluation['config']['n_repeats']} repeats, inner "
                f"{evaluation['config']['inner_splits']}-fold grid search"
            ),
            "reproducible_features": summary(evaluation),
            "all_22_features": summary(evaluation_full) if evaluation_full else None,
            "final_inner_cv_score": inner_score,
            "final_inner_cv_note": (
                "Score of the chosen settings in the final grid search. It is optimistic "
                "(settings were picked to maximise it); use the nested-CV numbers instead."
            ),
        },
        "feature_distributions": feature_distributions(ds),
        "limitations": [
            "Trained on 195 recordings from 32 subject IDs (8 healthy).",
            "Features are computed with Praat; the training data used MDVP. Values are "
            "comparable in range, not identical.",
            "Microphones and recording conditions differ from the training data, so "
            "predictions on new recordings are not reliable.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: select, train and save the model with its metadata."""
    parser = argparse.ArgumentParser(description="Train and save the deployed model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--evaluation", type=Path, default=RESULTS_DIR / "evaluation_reproducible.json"
    )
    parser.add_argument("--evaluation-full", type=Path, default=RESULTS_DIR / "evaluation.json")
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)

    evaluation = json.loads(args.evaluation.read_text())
    if evaluation["config"]["features"] != list(REPRODUCIBLE_FEATURES):
        raise SystemExit(f"{args.evaluation} was not run on the reproducible features.")
    evaluation_full = (
        json.loads(args.evaluation_full.read_text()) if args.evaluation_full.exists() else None
    )

    ds = load_dataset(args.data).select(REPRODUCIBLE_FEATURES)
    model_name, rationale = select_model(evaluation)
    tuned, params, inner_score = train(ds, model_name)
    model = calibrate(tuned, ds)
    metadata = build_metadata(
        ds, model_name, params, inner_score, rationale, evaluation, evaluation_full
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / "model.joblib")
    (args.out_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    print(f"Selected {model_name} ({rationale['best_model']} was best by mean).")
    print(f"Wrote {args.out_dir / 'model.joblib'} and model_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
