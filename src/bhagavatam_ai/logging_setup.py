"""Logging configuration: readable console output plus optional JSON-lines log files."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

_NOISY_LOGGERS: Final = (
    "httpx",
    "httpx2",
    "httpcore",
    "urllib3",
    "filelock",
    "huggingface_hub.file_download",
)
_OWNED: Final = "_bhagavatam_owned"
_RESERVED: Final = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; ``extra={...}`` fields are included as top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}
        )
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: str = "INFO", *, json_console: bool = False, log_file: Path | None = None
) -> None:
    """Configure the root logger. Safe to call more than once: handlers installed by an earlier call are
    replaced; handlers installed by someone else (a host application, pytest) are left alone."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        if getattr(handler, _OWNED, False):
            root.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        JsonFormatter()
        if json_console
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    setattr(console, _OWNED, True)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        setattr(file_handler, _OWNED, True)
        root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
