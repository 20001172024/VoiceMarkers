"""Load the deployed model once and turn feature values into a prediction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_FILE = "model.joblib"
METADATA_FILE = "model_metadata.json"

LABEL_PARKINSONS = "parkinsons_like"
LABEL_HEALTHY = "healthy_like"
LABEL_TEXT = {
    LABEL_PARKINSONS: "Voice features are more similar to the Parkinson's group in the training data.",
    LABEL_HEALTHY: "Voice features are more similar to the healthy group in the training data.",
}


class ModelNotLoaded(RuntimeError):
    """The model artifact is missing or unreadable."""


@dataclass(frozen=True)
class Prediction:
    """Model output for one recording."""

    probability: float  # calibrated P(Parkinson's group)
    label: str  # LABEL_PARKINSONS or LABEL_HEALTHY


@dataclass
class LoadedModel:
    """The calibrated model plus its training metadata."""

    estimator: Any  # sklearn CalibratedClassifierCV wrapping the tuned pipeline
    metadata: dict[str, Any]

    @property
    def features(self) -> list[str]:
        """Feature names, in the order the model expects."""
        return list(self.metadata["features"])

    @cached_property
    def _uncalibrated(self) -> Any:
        # With ensemble=False there is exactly one calibrated classifier, whose
        # `.estimator` is the tuned pipeline refitted on all training data.
        return self.estimator.calibrated_classifiers_[0].estimator

    def frame(self, rows: np.ndarray | list[list[float]]) -> pd.DataFrame:
        """Wrap raw rows in a DataFrame with the training column names."""
        return pd.DataFrame(np.asarray(rows, dtype=float), columns=self.features)

    def probability(self, rows: np.ndarray) -> np.ndarray:
        """Calibrated P(Parkinson's group) for each row (used by SHAP too)."""
        return self.estimator.predict_proba(self.frame(rows))[:, 1]

    def predict(self, features: dict[str, float]) -> Prediction:
        """Predict one recording from a {feature name: value} mapping.

        The label comes from the tuned model's own prediction, which was trained
        with balanced class weights; that is the operating point the reported
        balanced accuracy describes. Thresholding the calibrated probability at
        0.5 instead would call almost everyone "Parkinson's-like", because the
        training data is ~75% Parkinson's.
        """
        row = [[features[name] for name in self.features]]
        probability = float(self.probability(np.asarray(row))[0])
        positive = int(self._uncalibrated.predict(self.frame(row))[0]) == 1
        return Prediction(probability, LABEL_PARKINSONS if positive else LABEL_HEALTHY)


def load_model(model_dir: Path = DEFAULT_MODEL_DIR) -> LoadedModel:
    """Load the artifact and its metadata from `model_dir`.

    Raises:
        ModelNotLoaded: if either file is missing or cannot be read.
    """
    try:
        estimator = joblib.load(model_dir / MODEL_FILE)
        metadata = json.loads((model_dir / METADATA_FILE).read_text())
    except (OSError, ValueError) as exc:
        raise ModelNotLoaded(f"Could not load the model from {model_dir}: {exc}") from exc
    return LoadedModel(estimator, metadata)


def is_finite(value: float) -> bool:
    """True for a real, finite number."""
    return isinstance(value, int | float) and math.isfinite(value)
