"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import health
from app.config import get_settings
from app.constants import DISCLAIMER


def create_app() -> FastAPI:
    """Build and configure the FastAPI app."""
    settings = get_settings()
    app = FastAPI(
        title="VoiceMarkers API",
        version=__version__,
        description=(
            "Voice-feature Parkinson's research demo (ICOIICS 2025).\n\n"
            f"**Disclaimer:** {DISCLAIMER}"
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    return app


app = create_app()
