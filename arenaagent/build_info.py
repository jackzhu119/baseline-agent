from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

BUILD_NAME = "competition-runtime-v4"
_REVISION_ENV_VARS = ("AGENT_BUILD_SHA", "GITHUB_SHA", "CI_COMMIT_SHA")


@lru_cache(maxsize=1)
def source_revision() -> str:
    """Return a deploy-injected or checkout-derived revision without failing startup."""
    for key in _REVISION_ENV_VARS:
        value = str(os.getenv(key) or "").strip()
        if value:
            return value[:12]

    repository = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--short=12", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else "unknown"


def agent_build() -> str:
    return f"{BUILD_NAME}+{source_revision()}"
