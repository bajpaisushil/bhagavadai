"""Machine-readable run reports with provenance metadata.

Reports deliberately omit the hostname and user name: they may be committed as evidence and
should not leak personal details.
"""

from __future__ import annotations

import subprocess
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bhagavatam_ai import __version__
from bhagavatam_ai.utils.io import write_json


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def git_revision(cwd: Path) -> dict[str, Any]:
    """Return the current commit and whether the working tree is dirty (best effort)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def write_report(
    payload: Mapping[str, Any], *, reports_dir: Path, name: str, run_id: str, project_root: Path
) -> Path:
    """Write ``payload`` to ``reports_dir/<name>/<run_id>.json`` plus a ``latest.json`` copy."""
    document = {
        "_meta": {
            "report": name,
            "run_id": run_id,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "package_version": __version__,
            "git": git_revision(project_root),
        },
        **payload,
    }
    target_dir = reports_dir / name
    path = target_dir / f"{run_id}.json"
    write_json(path, document)
    write_json(target_dir / "latest.json", document)
    return path
