"""Turn feature values into the full API prediction (prediction + explanation + ranges)."""

from __future__ import annotations

from app.api.schemas import (
    ContributionOut,
    FeatureRangeOut,
    ModelRef,
    PredictionOut,
    PredictionResponse,
)
from app.ml.explain import Explainer
from app.ml.model import LABEL_TEXT, LoadedModel
from app.ml.ranges import compare


def build_prediction(
    model: LoadedModel, explainer: Explainer, features: dict[str, float]
) -> PredictionResponse:
    """Predict, explain and compare one set of features."""
    prediction = model.predict(features)
    explanation = explainer.explain(features)
    ranges = compare(features, model.metadata)

    warnings = [
        "Predictions on new recordings are not reliable: your microphone and room differ "
        "from the training recordings."
    ]
    outside = [r.feature for r in ranges if not r.within_training_range]
    if outside:
        warnings.append(
            f"{len(outside)} feature(s) fall outside the range seen in training "
            f"({', '.join(outside)}); the model is extrapolating there."
        )

    meta = model.metadata
    return PredictionResponse(
        prediction=PredictionOut(
            label=prediction.label,
            label_text=LABEL_TEXT[prediction.label],
            probability=prediction.probability,
            base_rate=meta["base_rate"],
            reference_probability=explanation.base_value,
        ),
        contributions=[ContributionOut(**vars(c)) for c in explanation.contributions],
        feature_ranges=[FeatureRangeOut(**vars(r)) for r in ranges],
        features={f: features[f] for f in model.features},
        warnings=warnings,
        model=ModelRef(type=meta["model_type"], version=meta["model_version"]),
    )
