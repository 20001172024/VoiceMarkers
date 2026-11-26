"""Model information endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_model
from app.api.schemas import ErrorResponse, ModelInfoResponse
from app.ml.model import LoadedModel

router = APIRouter(tags=["model"])


@router.get(
    "/model/info", response_model=ModelInfoResponse, responses={503: {"model": ErrorResponse}}
)
def model_info(model: Annotated[LoadedModel, Depends(get_model)]) -> ModelInfoResponse:
    """Model type, version, training date, features, evaluation and limitations."""
    meta = model.metadata
    fields = set(ModelInfoResponse.model_fields) - {"disclaimer"}
    return ModelInfoResponse(**{k: meta[k] for k in fields})
