"""FastAPI dependencies: the loaded model, the explainer and the rate limiter."""

from __future__ import annotations

from fastapi import Request

from app.api.errors import APIError
from app.ml.explain import Explainer
from app.ml.model import LoadedModel
from app.ratelimit import client_id


def get_model(request: Request) -> LoadedModel:
    """The model loaded at startup, or 503 if it failed to load."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise APIError(503, "model_unavailable", "The model is not available right now.")
    return model


def get_explainer(request: Request) -> Explainer:
    """The SHAP explainer built at startup alongside the model."""
    get_model(request)
    return request.app.state.explainer


def rate_limit(request: Request) -> None:
    """Count this request against the client's per-minute allowance."""
    request.app.state.limiter.hit(client_id(request))
