"""Parser interface and registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ParserError(ValueError):
    """The raw file does not have the structure the parser expects."""


@dataclass
class ParsedVerse:
    canto: int
    chapter: int
    verse: int
    text: str
    locator: str
    part: int | None = None
    variant: int = 0
    speaker_raw: str | None = None
    speaker_explicit: bool = False
    flags: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    parser: str
    parser_version: str
    verses: list[ParsedVerse]
    header: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


ParserFn = Callable[[Path], ParsedDocument]
_REGISTRY: dict[str, ParserFn] = {}


def register(name: str) -> Callable[[ParserFn], ParserFn]:
    def decorator(fn: ParserFn) -> ParserFn:
        if name in _REGISTRY:
            raise ValueError(f"parser {name!r} registered twice")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_parser(name: str) -> ParserFn:
    # Import parser modules so their @register decorators run.
    from bhagavatam_ai.data.parsers import gretil  # noqa: F401

    try:
        return _REGISTRY[name]
    except KeyError:
        raise ParserError(f"unknown parser {name!r}; available: {sorted(_REGISTRY)}") from None
