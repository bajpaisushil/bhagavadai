"""Shared fixtures and opt-in switches for slow tests.

* ``--run-model`` / BHAGAVATAM_RUN_MODEL_TESTS=1: tests that load the real 0.8B model (~1.7 GB download).
* ``--run-network`` / BHAGAVATAM_RUN_NETWORK_TESTS=1: tests that need internet access.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-model", action="store_true", help="run tests that load the real model")
    parser.addoption("--run-network", action="store_true", help="run tests that need network access")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_model = config.getoption("--run-model") or os.environ.get("BHAGAVATAM_RUN_MODEL_TESTS") == "1"
    run_network = config.getoption("--run-network") or os.environ.get("BHAGAVATAM_RUN_NETWORK_TESTS") == "1"
    for item in items:
        if "model" in item.keywords and not run_model:
            item.add_marker(pytest.mark.skip(reason="needs --run-model"))
        if "network" in item.keywords and not run_network:
            item.add_marker(pytest.mark.skip(reason="needs --run-network"))


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated copy of the project skeleton (configs + reference data) in a temp directory."""
    root = tmp_path / "project"
    (root / "data" / "manifests").mkdir(parents=True)
    (root / "data" / "reference").mkdir(parents=True)
    (root / "data" / "raw").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "configs", root / "configs")
    shutil.copy(REPO_ROOT / "pyproject.toml", root / "pyproject.toml")
    shutil.copy(
        REPO_ROOT / "data" / "reference" / "speakers.yaml", root / "data" / "reference" / "speakers.yaml"
    )
    monkeypatch.setenv("BHAGAVATAM_ROOT", str(root))
    monkeypatch.chdir(root)
    return root
