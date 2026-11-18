"""Speaker attributions ("śrī-śuka uvāca") -> canonical speaker names.

Sanskrit speaker lines vary by edition and sandhi ("śrīśuka uvāca", "śrī-śuka uvāca",
"śrībādarāyaṇir uvāca", "śrīrājovāca"). :func:`speaker_key` reduces them to a stem; the reviewed
table in ``data/reference/speakers.yaml`` maps stems to names. Ambiguous stems (e.g. "rājā", "the
king", is Parīkṣit in most chapters but Rahūgaṇa in 5.10) are marked so that no identity is guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

from bhagavatam_ai.data.normalize.iast import normalize_iast

_VERBS: Final = ("uvācaḥ", "uvāca", "ūcatuḥ", "ūcuḥ", "ūcur", "ūcuś", "ūcus")


def speaker_key(raw: str) -> str:
    """Reduce a Sanskrit speaker line to a sandhi-independent stem, e.g. ``"śrīrājovāca" -> "rāja"``."""
    s = normalize_iast(raw).lower()
    s = re.sub(r"[\s\-.,;:/|'0-9]", "", s)
    s = s.replace("śṛī", "śrī").removeprefix("śrī")
    for verb in _VERBS:
        if s.endswith(verb) and len(s) > len(verb):
            s = s[: -len(verb)]
            break
    else:
        if s.endswith("ovāca"):  # a/ā + uvāca -> ovāca (sandhi): "rājovāca" -> "rāja"
            s = s[:-5] + "a"
    s = re.sub(r"[rḥsś]$", "", s)  # visarga sandhi before u-: "bādarāyaṇir" -> "bādarāyaṇi"
    if s.endswith("y"):  # ī + u -> y: "rukmiṇy" -> "rukmiṇī"
        s = s[:-1] + "ī"
    return s


@dataclass(frozen=True)
class Speaker:
    canonical: str
    ambiguous: bool = False
    note: str = ""


class SpeakerTable:
    def __init__(self, mapping: dict[str, Speaker]) -> None:
        self._mapping = mapping

    @classmethod
    def load(cls, path: Path) -> SpeakerTable:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mapping: dict[str, Speaker] = {}
        for entry in document.get("speakers", []):
            speaker = Speaker(entry["canonical"], bool(entry.get("ambiguous", False)), entry.get("note", ""))
            for key in entry["keys"]:
                normalized = normalize_iast(key)
                if normalized in mapping:
                    raise ValueError(f"speaker key {key!r} is mapped twice")
                mapping[normalized] = speaker
        return cls(mapping)

    def resolve(self, raw: str) -> tuple[str | None, bool, str]:
        """Return ``(canonical_name_or_None, ambiguous, key)``. Ambiguous stems return no name."""
        key = speaker_key(raw)
        speaker = self._mapping.get(key)
        if speaker is None:
            return None, False, key
        return (None if speaker.ambiguous else speaker.canonical), speaker.ambiguous, key

    def __len__(self) -> int:
        return len(self._mapping)
