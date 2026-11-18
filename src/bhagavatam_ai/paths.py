"""Project directory layout, resolved once from the repository root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_ENV_VAR = "BHAGAVATAM_ROOT"
_MARKERS = ("pyproject.toml", "configs")


class ProjectRootNotFoundError(RuntimeError):
    """Raised when the repository root cannot be located."""


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root.

    Resolution order: the ``BHAGAVATAM_ROOT`` environment variable, then the nearest parent of
    ``start`` (default: the current working directory) that contains both ``pyproject.toml`` and
    ``configs/``.
    """
    env_root = os.environ.get(ROOT_ENV_VAR)
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not all((root / marker).exists() for marker in _MARKERS):
            raise ProjectRootNotFoundError(f"{ROOT_ENV_VAR}={root} does not look like the project root")
        return root

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in _MARKERS):
            return candidate
    raise ProjectRootNotFoundError(
        f"Could not find the project root from {current}. Run inside the repository or set {ROOT_ENV_VAR}."
    )


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical locations of project directories. All paths are absolute."""

    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> ProjectPaths:
        return cls(root=find_project_root(start))

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def manifests(self) -> Path:
        return self.data / "manifests"

    @property
    def review(self) -> Path:
        return self.data / "review"

    @property
    def reference(self) -> Path:
        return self.data / "reference"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    def resolve(self, path: str | Path) -> Path:
        """Resolve ``path`` relative to the project root unless it is already absolute."""
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else (self.root / candidate).resolve()
