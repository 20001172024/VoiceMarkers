"""API tests with the FastAPI test client, against the committed model artifact.

Audio comes from the committed SYNTHETIC samples and the synthetic generator.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio.features import REPRODUCIBLE_FEATURES
from app.config import Settings
from app.constants import DISCLAIMER, PRIVACY_NOTICE
from app.main import create_app
from app.ml.model import LABEL_HEALTHY, LABEL_PARKINSONS, load_model
from app.ml.ranges import compare
from app.ratelimit import RateLimiter, RateLimitExceeded
from data.make_samples import SAMPLES_DIR, VoiceSpec, synth_vowel, to_wav_bytes

STEADY = (SAMPLES_DIR / "synthetic_steady.wav").read_bytes()
ROUGH = (SAMPLES_DIR / "synthetic_rough.wav").read_bytes()
WAV = {"Content-Type": "audio/wav"}


def make_client(**settings: object) -> TestClient:
    return TestClient(create_app(Settings(rate_limit_per_minute=0, **settings)))


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with make_client() as c:  # the context manager runs startup (model loading)
        yield c


def assert_error(response, status: int, code: str) -> None:  # type: ignore[no-untyped-def]
    assert response.status_code == status, response.text
    body = response.json()
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    assert body["disclaimer"] == DISCLAIMER


# --------------------------------------------------------------- happy path


def test_health_reports_model_loaded(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["disclaimer"] == DISCLAIMER


def test_predict_audio_returns_full_response(client: TestClient) -> None:
    response = client.post("/predict/audio", content=STEADY, headers=WAV)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["disclaimer"] == DISCLAIMER
    assert body["privacy"] == PRIVACY_NOTICE
    assert list(body["features"]) == list(REPRODUCIBLE_FEATURES)
    pred = body["prediction"]
    assert 0 <= pred["probability"] <= 1
    assert pred["label"] in {LABEL_PARKINSONS, LABEL_HEALTHY}
    assert "training data" in pred["label_text"]
    assert pred["base_rate"] == pytest.approx(147 / 195)
    assert len(body["contributions"]) == len(body["feature_ranges"]) == 15
    assert any("not reliable" in w for w in body["warnings"])
    assert body["model"]["version"]


def test_contributions_add_up_to_the_probability(client: TestClient) -> None:
    body = client.post("/predict/audio", content=ROUGH, headers=WAV).json()
    total = body["prediction"]["reference_probability"] + sum(
        c["contribution"] for c in body["contributions"]
    )
    assert total == pytest.approx(body["prediction"]["probability"], abs=1e-6)
    sizes = [abs(c["contribution"]) for c in body["contributions"]]
    assert sizes == sorted(sizes, reverse=True)  # largest effect first


def test_explanations_are_deterministic(client: TestClient) -> None:
    a = client.post("/predict/audio", content=STEADY, headers=WAV).json()["contributions"]
    b = client.post("/predict/audio", content=STEADY, headers=WAV).json()["contributions"]
    assert a == b


def test_audio_and_features_endpoints_agree(client: TestClient) -> None:
    from_audio = client.post("/predict/audio", content=ROUGH, headers=WAV).json()
    from_features = client.post("/predict/features", json={"features": from_audio["features"]})
    assert from_features.status_code == 200
    assert from_features.json()["prediction"] == from_audio["prediction"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_predict_accepts_browser_webm(client: TestClient, tmp_path: Path) -> None:
    src, dst = tmp_path / "in.wav", tmp_path / "out.webm"
    src.write_bytes(STEADY)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(src), "-c:a", "libopus", str(dst)], check=True
    )
    response = client.post(
        "/predict/audio", content=dst.read_bytes(), headers={"Content-Type": "audio/webm"}
    )
    assert response.status_code == 200, response.text


def test_label_follows_the_models_own_decision_not_a_half_threshold() -> None:
    """The steady sample: probability > 0.5 but below the base rate, labelled healthy-like."""
    model = load_model()
    from app.audio.features import extract_features
    from app.audio.load import load_audio

    features = extract_features(load_audio(STEADY))
    prediction = model.predict(features)
    expected = int(
        model._uncalibrated.predict(model.frame([[features[f] for f in model.features]]))[0]
    )
    assert prediction.label == (LABEL_PARKINSONS if expected == 1 else LABEL_HEALTHY)
    assert prediction.probability > 0.5 and prediction.label == LABEL_HEALTHY


def test_model_info(client: TestClient) -> None:
    body = client.get("/model/info").json()
    assert body["features"] == list(REPRODUCIBLE_FEATURES)
    assert body["model_type"] and body["model_version"] and body["trained_at"]
    assert "balanced_accuracy" in body["evaluation"]["reproducible_features"]
    assert body["limitations"]
    assert body["disclaimer"] == DISCLAIMER
    assert "explanation_background" not in body  # internal data stays internal


# ------------------------------------------------------------ audio errors


def test_rejects_non_audio_content_type(client: TestClient) -> None:
    r = client.post("/predict/audio", content=b"hi", headers={"Content-Type": "text/plain"})
    assert_error(r, 415, "unsupported_media_type")


def test_rejects_unreadable_audio(client: TestClient) -> None:
    assert_error(
        client.post("/predict/audio", content=b"not audio", headers=WAV), 400, "decode_failed"
    )


def test_rejects_empty_body(client: TestClient) -> None:
    assert_error(client.post("/predict/audio", content=b"", headers=WAV), 400, "empty_body")


def test_rejects_oversized_upload() -> None:
    with make_client(max_upload_mb=0.1) as small:
        assert_error(
            small.post("/predict/audio", content=STEADY, headers=WAV), 413, "file_too_large"
        )


@pytest.mark.parametrize(
    ("samples", "code"),
    [
        (np.zeros(44_100 * 3), "silent"),
        (synth_vowel(VoiceSpec(duration=1.0)), "too_short"),
        (0.3 * np.random.default_rng(0).standard_normal(44_100 * 3), "not_voiced"),
    ],
    ids=["silent", "too-short", "noise"],
)
def test_rejects_unusable_recordings(client: TestClient, samples: np.ndarray, code: str) -> None:
    r = client.post("/predict/audio", content=to_wav_bytes(samples), headers=WAV)
    assert_error(r, 422, code)


# --------------------------------------------------------- features errors


def _features(client: TestClient) -> dict[str, float]:
    return client.post("/predict/audio", content=STEADY, headers=WAV).json()["features"]


def test_features_missing_one(client: TestClient) -> None:
    features = _features(client)
    features.pop("HNR")
    r = client.post("/predict/features", json={"features": features})
    assert_error(r, 422, "invalid_features")
    assert "HNR" in r.json()["error"]["message"]


def test_features_unknown_name(client: TestClient) -> None:
    features = {**_features(client), "PPE": 0.2}
    r = client.post("/predict/features", json={"features": features})
    assert_error(r, 422, "invalid_features")
    assert "PPE" in r.json()["error"]["message"]


def test_features_negative_value(client: TestClient) -> None:
    features = {**_features(client), "MDVP:Jitter(%)": -0.01}
    assert_error(
        client.post("/predict/features", json={"features": features}), 422, "invalid_features"
    )


def test_features_wrong_type(client: TestClient) -> None:
    r = client.post("/predict/features", json={"features": {"HNR": "loud"}})
    assert_error(r, 422, "invalid_request")


# ------------------------------------------------------ rate limit, startup


def test_rate_limit_returns_429_with_retry_after() -> None:
    with TestClient(create_app(Settings(rate_limit_per_minute=2))) as limited:
        for _ in range(2):
            assert limited.post("/predict/audio", content=b"", headers=WAV).status_code == 400
        r = limited.post("/predict/audio", content=STEADY, headers=WAV)
        assert_error(r, 429, "rate_limited")
        assert int(r.headers["Retry-After"]) >= 1
        assert limited.get("/health").status_code == 200  # health is never limited


def test_rate_limiter_window_resets() -> None:
    now = [0.0]
    limiter = RateLimiter(limit=1, window_s=60, clock=lambda: now[0])
    limiter.hit("a")
    limiter.hit("b")  # limits are per client
    with pytest.raises(RateLimitExceeded):
        limiter.hit("a")
    now[0] = 61
    limiter.hit("a")


def test_missing_model_gives_503_but_app_starts(tmp_path: Path) -> None:
    with make_client(model_path=tmp_path) as broken:
        assert broken.get("/health").json()["model_loaded"] is False
        assert_error(
            broken.post("/predict/audio", content=STEADY, headers=WAV), 503, "model_unavailable"
        )
        assert_error(broken.get("/model/info"), 503, "model_unavailable")


# ------------------------------------------------------------------- ranges


def test_compare_places_values_between_groups() -> None:
    dist = {"p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
    metadata = {
        "features": ["x"],
        "feature_distributions": {
            "x": {
                "all": {**dist, "p5": 1.0, "p95": 9.0},
                "parkinsons": {**dist, "p50": 8.0},
                "healthy": {**dist, "p50": 2.0},
            }
        },
    }
    assert compare({"x": 7.5}, metadata)[0].closer_to == "parkinsons"
    assert compare({"x": 2.5}, metadata)[0].closer_to == "healthy"
    assert compare({"x": 5.0}, metadata)[0].closer_to == "neither"
    assert compare({"x": 10.0}, metadata)[0].within_training_range is False
    assert math.isclose(compare({"x": 5.0}, metadata)[0].value, 5.0)
