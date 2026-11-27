"""Tests for deploy/deploy_space.py (staging only: no network, no token)."""

from __future__ import annotations

from pathlib import Path

from deploy import deploy_space as ds


def test_stage_contains_exactly_what_the_space_needs(tmp_path: Path) -> None:
    files = ds.stage(tmp_path / "space")
    assert {"Dockerfile", "README.md", "pyproject.toml", ".dockerignore"} <= set(files)
    assert {"models/model.joblib", "models/model_metadata.json", "app/main.py"} <= set(files)
    assert "app/audio/features.py" in files
    # Nothing that doesn't belong in the image.
    assert not any("__pycache__" in f or f.startswith("tests") for f in files)
    assert not any(f.startswith(("research", "data", "frontend")) for f in files)


def test_space_readme_declares_a_docker_space_on_the_app_port(tmp_path: Path) -> None:
    ds.stage(tmp_path)
    readme = (tmp_path / "README.md").read_text()
    front_matter = readme.split("---")[1]
    assert "sdk: docker" in front_matter
    assert "app_port: 7860" in front_matter
    assert "Not a medical device" in readme
    assert "EXPOSE 7860" in (tmp_path / "Dockerfile").read_text()


def test_space_variables() -> None:
    assert ds.space_variables("abc", None) == {"BUILD_SHA": "abc", "FORWARDED_HOPS": "1"}
    assert ds.space_variables("abc", "https://a.vercel.app")["CORS_ORIGINS"] == (
        "https://a.vercel.app"
    )


def test_dry_run_needs_no_token(capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert ds.main(["--dry-run"]) == 0
    assert "models/model.joblib" in capsys.readouterr().out


def test_real_deploy_requires_a_token(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert ds.main([]) == 1


# ------------------------------------------------------------ wait_for_space

from deploy import wait_for_space as wfs  # noqa: E402

SHA = "abc123"
LIVE = {"build": SHA, "model_loaded": True}


def test_live_when_health_reports_the_new_build() -> None:
    assert wfs.decide({"stage": "RUNNING"}, LIVE, SHA, 0).state == "live"


def test_waits_while_an_old_build_is_still_serving() -> None:
    old = {"build": "old", "model_loaded": True}
    assert wfs.decide({"stage": "BUILDING"}, old, SHA, 0).state == "wait"


def test_waits_if_new_build_has_not_loaded_its_model() -> None:
    assert (
        wfs.decide({"stage": "RUNNING"}, {"build": SHA, "model_loaded": False}, SHA, 0).state
        == "wait"
    )


def test_fails_fast_on_build_error_with_huggingface_message() -> None:
    d = wfs.decide({"stage": "BUILD_ERROR", "errorMessage": "pip failed"}, None, SHA, 0)
    assert d.state == "fail" and "pip failed" in d.message


def test_paused_is_final_only_after_several_polls() -> None:
    runtime = {"stage": "PAUSED", "errorMessage": "Quota exceeded for flavor cpu-basic"}
    assert wfs.decide(runtime, None, SHA, 1).state == "wait"
    d = wfs.decide(runtime, None, SHA, wfs.PAUSED_POLLS_BEFORE_FAILING)
    assert d.state == "fail" and "Quota exceeded" in d.message


def test_waits_when_nothing_is_reachable() -> None:
    assert wfs.decide(None, None, SHA, 0).state == "wait"


def test_wait_stops_on_paused_space(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    paused = {"stage": "PAUSED", "errorMessage": "Quota exceeded"}
    monkeypatch.setattr(
        wfs, "_get_json", lambda url, timeout=15.0: paused if "/runtime" in url else None
    )
    monkeypatch.setattr(wfs.time, "sleep", lambda s: None)
    d = wfs.wait("o/s", "https://x.hf.space", SHA, timeout_s=60, interval_s=1)
    assert d.state == "fail" and "Quota exceeded" in d.message
