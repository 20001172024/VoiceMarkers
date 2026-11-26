"""Prediction endpoints: from raw audio, or from precomputed features."""

from __future__ import annotations

import math
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool

from app.api.deps import get_explainer, get_model, rate_limit
from app.api.errors import APIError
from app.api.schemas import ErrorResponse, FeaturesRequest, PredictionResponse
from app.audio.features import extract_features
from app.audio.load import load_audio
from app.audio.validate import validate_audio
from app.ml.explain import Explainer
from app.ml.model import LoadedModel
from app.ml.service import build_prediction

router = APIRouter(prefix="/predict", tags=["predict"], dependencies=[Depends(rate_limit)])

ERRORS = {
    s: {"model": ErrorResponse} for s in (400, 413, 415, 422, 429, 503)
}  # documented in /docs
# Browsers record WebM (Chrome, Firefox) or MP4 (Safari); octet-stream covers
# clients that don't label the file. The bytes are checked by decoding anyway.
ALLOWED_TYPE_PREFIXES = ("audio/",)
ALLOWED_TYPES = {"video/webm", "video/mp4", "application/octet-stream"}

Model = Annotated[LoadedModel, Depends(get_model)]
Explain = Annotated[Explainer, Depends(get_explainer)]


async def read_body_capped(request: Request, max_bytes: int) -> bytes:
    """Read the request body into memory, refusing anything over `max_bytes`.

    Reading the raw body (instead of a multipart upload) keeps the audio in
    memory: Starlette spools multipart uploads over 1 MB to a temporary file.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise _too_large(max_bytes)
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > max_bytes:
            raise _too_large(max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


def _too_large(max_bytes: int) -> APIError:
    return APIError(
        413,
        "file_too_large",
        f"The file is larger than {max_bytes / 1e6:g} MB. Record 3-5 seconds.",
    )


def _analyse_audio(data: bytes, model: LoadedModel, explainer: Explainer) -> PredictionResponse:
    """CPU-bound part of /predict/audio (run in a worker thread)."""
    audio = load_audio(data)
    validate_audio(audio)
    return build_prediction(model, explainer, extract_features(audio))


@router.post(
    "/audio",
    response_model=PredictionResponse,
    responses=ERRORS,
    openapi_extra={
        "requestBody": {
            "required": True,
            "description": "Raw audio bytes (WAV, FLAC, OGG, WebM or MP4) as the request body.",
            "content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}},
        }
    },
)
async def predict_audio(request: Request, model: Model, explainer: Explain) -> PredictionResponse:
    """Analyse a sustained-vowel recording sent as the raw request body.

    Example: `curl -X POST --data-binary @aaah.wav -H "Content-Type: audio/wav" .../predict/audio`.
    The audio is processed in memory and never stored.
    """
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if not (content_type.startswith(ALLOWED_TYPE_PREFIXES) or content_type in ALLOWED_TYPES):
        raise APIError(
            415,
            "unsupported_media_type",
            f"Expected an audio file, got '{content_type or 'no content type'}'.",
        )
    data = await read_body_capped(request, int(request.app.state.settings.max_upload_mb * 1e6))
    if not data:
        raise APIError(400, "empty_body", "No audio was received.")
    return await run_in_threadpool(_analyse_audio, data, model, explainer)


@router.post("/features", response_model=PredictionResponse, responses=ERRORS)
def predict_features(body: FeaturesRequest, model: Model, explainer: Explain) -> PredictionResponse:
    """Predict from precomputed feature values (all features from GET /model/info)."""
    expected = set(model.features)
    missing = sorted(expected - set(body.features))
    unknown = sorted(set(body.features) - expected)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        if unknown:
            parts.append(f"not used by this model: {', '.join(unknown)}")
        raise APIError(422, "invalid_features", "Feature names do not match. " + "; ".join(parts))
    bad = [f for f, v in body.features.items() if not math.isfinite(v) or (v < 0 and f != "HNR")]
    if bad:
        raise APIError(
            422,
            "invalid_features",
            f"Values must be finite and non-negative (HNR may be negative): {', '.join(bad)}",
        )
    return build_prediction(model, explainer, body.features)
