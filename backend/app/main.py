"""FastAPI application entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import health, model_info, predict
from app.api.errors import install_error_handlers
from app.config import Settings, get_settings
from app.constants import DISCLAIMER
from app.ml.explain import Explainer
from app.ml.model import DEFAULT_MODEL_DIR, ModelNotLoaded, load_model
from app.ratelimit import RateLimiter

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI app.

    The model is loaded once, at startup. If it cannot be loaded the API still
    starts: /health reports `model_loaded: false` and predictions return 503.
    """
    settings = settings or get_settings()
    model_dir: Path = settings.model_path or DEFAULT_MODEL_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.model = load_model(model_dir)
            app.state.explainer = Explainer(app.state.model)
            logger.info("Loaded model %s", app.state.model.metadata["model_version"])
        except ModelNotLoaded:
            logger.exception("Model failed to load; prediction endpoints will return 503")
            app.state.model = app.state.explainer = None
        yield

    app = FastAPI(
        title="VoiceMarkers API",
        version=__version__,
        description=(
            "Voice-feature Parkinson's research demo (ICOIICS 2025).\n\n"
            f"**Disclaimer:** {DISCLAIMER}"
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(model_info.router)
    app.include_router(predict.router)
    return app


app = create_app()
