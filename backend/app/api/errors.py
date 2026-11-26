"""Error handling: every error response has the same shape and the disclaimer."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.schemas import ErrorDetail, ErrorResponse
from app.audio.errors import AudioError
from app.ratelimit import RateLimitExceeded

# Audio problems the person can fix by re-recording are 422; an unreadable or
# unsupported file is a bad upload (400 / 415).
AUDIO_STATUS = {"decode_failed": 400, "unsupported_format": 415}


class APIError(Exception):
    """An error with an HTTP status, a stable code and a user-facing message."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def error_response(
    status: int, code: str, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    """Build the standard error body."""
    body = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    """Register handlers so no error escapes without the disclaimer."""

    @app.exception_handler(APIError)
    async def _api(_: Request, exc: APIError) -> JSONResponse:
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(AudioError)
    async def _audio(_: Request, exc: AudioError) -> JSONResponse:
        return error_response(AUDIO_STATUS.get(exc.code, 422), exc.code, exc.message)

    @app.exception_handler(RateLimitExceeded)
    async def _rate(_: Request, exc: RateLimitExceeded) -> JSONResponse:
        return error_response(
            429,
            "rate_limited",
            f"Too many requests. Please wait {exc.retry_after} s and try again.",
            headers={"Retry-After": str(exc.retry_after)},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()))
        return error_response(422, "invalid_request", f"{where}: {first.get('msg', 'invalid')}")

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, __: Exception) -> JSONResponse:
        # No details: they could include request content. The traceback is still
        # logged by the server (without the audio bytes).
        return error_response(500, "internal_error", "Something went wrong on our side.")
