"""Deploy the backend to a Hugging Face Docker Space.

Assembles exactly what the Space needs in a staging folder (the Dockerfile, the
app package, the model artifact and the Space README) and uploads it as one
commit with `huggingface_hub`, which handles large/binary files for us. Files
in the Space that are not part of the new build are deleted, so the Space
always mirrors the repository.

Space variables (non-secret settings such as CORS_ORIGINS) are set through the
API; changing them restarts the Space.

Usage (from the repo root; HF_TOKEN must hold a write token):
    python -m deploy.deploy_space --dry-run          # stage only, list files
    python -m deploy.deploy_space --build-sha "$GITHUB_SHA" --cors-origins https://x.vercel.app
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"
SPACE_README = Path(__file__).resolve().parent / "huggingface" / "README.md"
DEFAULT_REPO_ID = "arora2006/voicemarkers-api"
# The Space sits behind one Hugging Face proxy, which appends the client IP.
DEFAULT_FORWARDED_HOPS = "1"

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")


def stage(dest: Path) -> list[str]:
    """Copy the files the Space needs into `dest`; return their relative paths."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("Dockerfile", ".dockerignore", "pyproject.toml"):
        shutil.copy2(BACKEND / name, dest / name)
    shutil.copytree(BACKEND / "app", dest / "app", ignore=_IGNORE)
    (dest / "models").mkdir()
    for artifact in ("model.joblib", "model_metadata.json"):
        shutil.copy2(BACKEND / "models" / artifact, dest / "models" / artifact)
    shutil.copy2(SPACE_README, dest / "README.md")
    return sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())


def space_variables(build_sha: str, cors_origins: str | None) -> dict[str, str]:
    """Non-secret settings for the running app (see backend/app/config.py)."""
    variables = {"BUILD_SHA": build_sha, "FORWARDED_HOPS": DEFAULT_FORWARDED_HOPS}
    if cors_origins:
        variables["CORS_ORIGINS"] = cors_origins
    return variables


def deploy(repo_id: str, token: str, build_sha: str, cors_origins: str | None) -> None:
    """Set the Space's variables, then upload the staged build as one commit."""
    from huggingface_hub import HfApi  # deploy-time only dependency

    api = HfApi(token=token)
    for key, value in space_variables(build_sha, cors_origins).items():
        api.add_space_variable(repo_id, key, value)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=tmp,
            commit_message=f"Deploy {build_sha[:12]} from GitHub",
            delete_patterns=["*"],  # remove files that are no longer part of the build
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Deploy the backend to a Hugging Face Space.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--build-sha", default="dev")
    parser.add_argument("--cors-origins", default=None, help="comma-separated frontend origins")
    parser.add_argument("--dry-run", action="store_true", help="stage and list files only")
    args = parser.parse_args(argv)

    if args.dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            for path in stage(Path(tmp)):
                print(path)
        print(space_variables(args.build_sha, args.cors_origins))
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("error: HF_TOKEN is not set", file=sys.stderr)
        return 1
    deploy(args.repo_id, token, args.build_sha, args.cors_origins)
    print(f"Deployed {args.build_sha} to https://huggingface.co/spaces/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
