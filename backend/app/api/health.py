"""Health check router."""

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.constants import DISCLAIMER

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Liveness response. Includes the disclaimer, like every API response."""

    status: str
    version: str
    disclaimer: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the API process is up."""
    return HealthResponse(status="ok", version=__version__, disclaimer=DISCLAIMER)
