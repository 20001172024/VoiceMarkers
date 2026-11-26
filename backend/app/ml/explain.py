"""Per-prediction feature contributions with SHAP.

KernelSHAP estimates, for one recording, how much each feature moved the
predicted probability away from the average prediction over a reference set
(50 training recordings stored in the model metadata). The contributions plus
that average add up exactly to the predicted probability.

Caveat shown to users: contributions explain what *this model* does with the
values; they are not evidence about the voice itself, and correlated features
(the jitter and shimmer variants) share credit between them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shap

from app.ml.model import LoadedModel

N_SAMPLES = 1000  # coalition samples; ~0.15 s per prediction, stable to ~0.001


@dataclass(frozen=True)
class Contribution:
    """One feature's effect on this prediction's probability."""

    feature: str
    value: float
    contribution: float  # change in probability, positive = towards Parkinson's-like


@dataclass(frozen=True)
class Explanation:
    """All contributions, largest effect first, plus the reference probability."""

    base_value: float  # average predicted probability over the reference set
    contributions: list[Contribution]


class Explainer:
    """Wraps a SHAP KernelExplainer built once per loaded model."""

    def __init__(self, model: LoadedModel) -> None:
        self._model = model
        background = np.asarray(model.metadata["explanation_background"], dtype=float)
        self._kernel = shap.KernelExplainer(model.probability, background)

    def explain(self, features: dict[str, float]) -> Explanation:
        """Explain one prediction (seeded, so the same input gives the same output)."""
        row = np.asarray([[features[f] for f in self._model.features]])
        np.random.seed(0)  # KernelSHAP samples coalitions with NumPy's global RNG
        values = self._kernel.shap_values(row, nsamples=N_SAMPLES, silent=True)[0]
        contributions = [
            Contribution(f, float(features[f]), float(v))
            for f, v in zip(self._model.features, values, strict=True)
        ]
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
        return Explanation(float(self._kernel.expected_value), contributions)
