"""The classifiers compared in the paper, each wrapped in a leakage-safe Pipeline."""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASELINE = "Majority baseline"
MODEL_NAMES: tuple[str, ...] = (
    BASELINE,
    "SVM",
    "Random Forest",
    "Logistic Regression",
    "KNN",
    "Gradient Boosting",
)


def _classifier(name: str, random_state: int) -> ClassifierMixin:
    # scikit-learn defaults throughout (no tuning in M1; tuning happens inside CV in M2).
    # Only exception: LogisticRegression's max_iter is raised so it converges.
    match name:
        case "Majority baseline":
            return DummyClassifier(strategy="most_frequent")
        case "SVM":
            return SVC(random_state=random_state)
        case "Random Forest":
            return RandomForestClassifier(random_state=random_state)
        case "Logistic Regression":
            return LogisticRegression(max_iter=5000, random_state=random_state)
        case "KNN":
            return KNeighborsClassifier()
        case "Gradient Boosting":
            return GradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Unknown model: {name!r}. Choose from {MODEL_NAMES}")


def build_model(name: str, random_state: int = 0) -> Pipeline:
    """Return `StandardScaler -> classifier` as one Pipeline.

    Putting the scaler inside the pipeline means it is fitted on training rows
    only, every time the model is fitted, so test-set statistics never leak into
    training. (Scaling the whole dataset before splitting would leak them.)
    """
    return Pipeline([("scale", StandardScaler()), ("clf", _classifier(name, random_state))])
