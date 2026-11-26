"""Application settings loaded from environment variables (and `.env`)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration for the backend.

    Values come from environment variables; a local `.env` file is read if present.
    Field names map to upper-case env vars (e.g. `cors_origins` -> `CORS_ORIGINS`).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    # Comma-separated list of allowed frontend origins.
    cors_origins: str = "http://localhost:5173"
    max_upload_mb: float = 5.0
    # Directory holding model.joblib + model_metadata.json. None = backend/models/.
    model_path: Path | None = None
    # Prediction requests allowed per client IP per minute (0 disables the limit).
    rate_limit_per_minute: int = 10
    # Number of trusted reverse proxies in front of the app. With N > 0 the client
    # IP is the N-th entry from the right of X-Forwarded-For (the entry the
    # nearest trusted proxy added); entries further left are client-controlled.
    # 0 = ignore the header and use the connection's peer address.
    forwarded_hops: int = 0
    # Identifies the deployed build (set to the git commit by the deploy job).
    build_sha: str = "dev"

    @property
    def cors_origin_list(self) -> list[str]:
        """Return `cors_origins` split into a clean list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (use as a FastAPI dependency)."""
    return Settings()
