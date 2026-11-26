"""Health check router."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app import __version__
from app.constants import DISCLAIMER

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Liveness response. Includes the disclaimer, like every API response."""

    status: str
    version: str
    model_loaded: bool
    build: str
    disclaimer: str


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report that the API process is up, and whether the model loaded."""
    loaded = getattr(request.app.state, "model", None) is not None
    return HealthResponse(
        status="ok",
        version=__version__,
        model_loaded=loaded,
        build=request.app.state.settings.build_sha,
        disclaimer=DISCLAIMER,
    )
