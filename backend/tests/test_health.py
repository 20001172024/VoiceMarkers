"""Tests for the /health endpoint and app wiring."""

from fastapi.testclient import TestClient

from app import __version__
from app.config import Settings
from app.constants import DISCLAIMER
from app.main import app

client = TestClient(app)


def test_health_returns_ok_with_disclaimer() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["disclaimer"] == DISCLAIMER
    assert isinstance(body["model_loaded"], bool)


def test_openapi_description_includes_disclaimer() -> None:
    schema = client.get("/openapi.json").json()
    assert DISCLAIMER in schema["info"]["description"]


def test_cors_allows_configured_origin() -> None:
    response = client.options(
        "/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_origin_list_splits_and_strips() -> None:
    settings = Settings(cors_origins=" http://a.com , http://b.com,, ")
    assert settings.cors_origin_list == ["http://a.com", "http://b.com"]
