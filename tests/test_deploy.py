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
