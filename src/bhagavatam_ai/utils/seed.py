"""Reproducibility: seed every random number generator in use."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed Python, and torch (all devices) when it is installed."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - data-only installs
        return
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
