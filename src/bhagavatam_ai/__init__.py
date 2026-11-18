"""Bhagavatam AI: adapting a pretrained LLM to Srimad-Bhagavatam through training (not retrieval)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bhagavatam-ai")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
