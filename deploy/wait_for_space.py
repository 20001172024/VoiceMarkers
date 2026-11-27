"""Wait until a Hugging Face Space serves a given build, failing fast on errors.

After `deploy_space.py` uploads a new build, the Space rebuilds its Docker image.
This script polls two things:

- the Space's runtime status from the public Hugging Face API, so a build error,
  a crash or a Space stuck in PAUSED (for example "Quota exceeded") fails the
  deploy within a minute, with Hugging Face's own error message;
- the app's `/health`, which reports the deployed commit as `build`, so success
  means the *new* code is live, not just that an old build is still running.

Usage (from the repo root):
    python -m deploy.wait_for_space --repo-id arora2006/voicemarkers-api \\
        --api-url https://arora2006-voicemarkers-api.hf.space --build-sha "$GITHUB_SHA"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

HF_API = "https://huggingface.co/api/spaces"
# Stages that will not fix themselves: stop waiting and report.
ERROR_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}
# Right after an upload the Space may still report its old (paused) state for a
# moment; only treat PAUSED as final once it has been seen this many times in a row.
PAUSED_POLLS_BEFORE_FAILING = 3


@dataclass(frozen=True)
class Decision:
    """What to do after one poll."""

    state: str  # "live", "wait" or "fail"
    message: str = ""


def decide(
    runtime: dict[str, Any] | None,
    health: dict[str, Any] | None,
    build_sha: str,
    paused_polls: int,
) -> Decision:
    """Decide from one poll's results. `paused_polls` counts consecutive PAUSED polls."""
    if health and health.get("build") == build_sha and health.get("model_loaded"):
        return Decision("live", f"Build {build_sha[:12]} is live and the model is loaded.")
    if runtime:
        stage = runtime.get("stage", "")
        error = runtime.get("errorMessage") or "no error message"
        if stage in ERROR_STAGES:
            return Decision("fail", f"Space stage {stage}: {error}")
        if stage == "PAUSED" and paused_polls >= PAUSED_POLLS_BEFORE_FAILING:
            return Decision(
                "fail",
                f"Space is PAUSED: {error}. Resolve it in the Space's Settings on "
                "Hugging Face (e.g. hardware quota or billing), then re-run this workflow.",
            )
    return Decision("wait")


def _get_json(url: str, timeout: float = 15.0) -> dict[str, Any] | None:
    """GET a JSON document; None on any network or HTTP error."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def wait(
    repo_id: str, api_url: str, build_sha: str, timeout_s: float, interval_s: float
) -> Decision:
    """Poll until the build is live, a failure is certain, or `timeout_s` passes."""
    deadline = time.monotonic() + timeout_s
    paused_polls = 0
    while time.monotonic() < deadline:
        runtime = _get_json(f"{HF_API}/{repo_id}/runtime")
        paused_polls = paused_polls + 1 if runtime and runtime.get("stage") == "PAUSED" else 0
        health = _get_json(f"{api_url.rstrip('/')}/health")
        decision = decide(runtime, health, build_sha, paused_polls)
        stage = runtime.get("stage") if runtime else "unknown"
        print(f"stage={stage} health_build={(health or {}).get('build')}", flush=True)
        if decision.state != "wait":
            return decision
        time.sleep(interval_s)
    return Decision(
        "fail", f"Build {build_sha[:12]} was not live within {timeout_s / 60:.0f} minutes."
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit code 0 only when the new build is live."""
    parser = argparse.ArgumentParser(description="Wait for a Hugging Face Space build.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--build-sha", required=True)
    parser.add_argument("--timeout-minutes", type=float, default=30)
    parser.add_argument("--interval-seconds", type=float, default=20)
    args = parser.parse_args(argv)

    decision = wait(
        args.repo_id, args.api_url, args.build_sha, args.timeout_minutes * 60, args.interval_seconds
    )
    stream = sys.stdout if decision.state == "live" else sys.stderr
    print(decision.message, file=stream)
    return 0 if decision.state == "live" else 1


if __name__ == "__main__":
    raise SystemExit(main())
