"""Request and response models for the API (they also document it in /docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.constants import DISCLAIMER, PRIVACY_NOTICE


class WithDisclaimer(BaseModel):
    """Every response carries the medical disclaimer."""

    disclaimer: str = DISCLAIMER


class ErrorDetail(BaseModel):
    """A machine-readable code and a message for the person using the app."""

    code: str
    message: str


class ErrorResponse(WithDisclaimer):
    """Body of every 4xx/5xx response."""

    error: ErrorDetail


class FeaturesRequest(BaseModel):
    """Precomputed feature values, keyed by their UCI column names."""

    features: dict[str, float] = Field(
        description="All features listed by GET /model/info, e.g. {'MDVP:Fo(Hz)': 150.0, ...}"
    )


class PredictionOut(BaseModel):
    """The model's output, worded for a non-expert."""

    label: Literal["parkinsons_like", "healthy_like"]
    label_text: str
    probability: float = Field(description="Calibrated P(Parkinson's group), 0-1")
    base_rate: float = Field(description="Share of Parkinson's recordings in the training data")
    reference_probability: float = Field(
        description="Average predicted probability over the training reference set"
    )


class ContributionOut(BaseModel):
    """One feature's SHAP contribution to the probability."""

    feature: str
    value: float
    contribution: float = Field(description="Change in probability; + means towards Parkinson's")


class FeatureRangeOut(BaseModel):
    """Where one value sits relative to both training groups."""

    feature: str
    value: float
    parkinsons: dict[str, float] = Field(description="p5/p25/p50/p75/p95 of the Parkinson's group")
    healthy: dict[str, float] = Field(description="p5/p25/p50/p75/p95 of the healthy group")
    within_training_range: bool
    closer_to: Literal["parkinsons", "healthy", "neither"]


class ModelRef(BaseModel):
    """Which model produced the prediction."""

    type: str
    version: str


class PredictionResponse(WithDisclaimer):
    """Response of both /predict endpoints."""

    prediction: PredictionOut
    contributions: list[ContributionOut] = Field(description="Largest effect first")
    feature_ranges: list[FeatureRangeOut]
    features: dict[str, float]
    warnings: list[str]
    model: ModelRef
    privacy: str = PRIVACY_NOTICE


class ModelInfoResponse(WithDisclaimer):
    """Everything needed to describe the model honestly."""

    model_type: str
    model_version: str
    trained_at: str
    features: list[str]
    params: dict[str, object]
    positive_class: str
    probability: str
    label_rule: str
    base_rate: float
    training_data: dict[str, object]
    selection: dict[str, object]
    evaluation: dict[str, object]
    limitations: list[str]
