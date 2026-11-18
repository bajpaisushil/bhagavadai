from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bhagavatam_ai.data.normalize.iast import normalize_iast
from bhagavatam_ai.data.normalize.speakers import SpeakerTable, speaker_key


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("śrī-śuka uvāca", "śuka"),
        ("śrīśuka uvāca", "śuka"),
        ("śṛīśuka uvāca", "śuka"),  # typo in source
        ("śrībādarāyaṇir uvāca", "bādarāyaṇi"),
        ("śrīrājovāca", "rāja"),
        ("brahmovāca", "brahma"),
        ("ṛṣaya ūcuḥ", "ṛṣaya"),
        ("śrīrukmiṇy uvāca", "rukmiṇī"),
        ("pṛthur uvāca", "pṛthu"),
        ("śrī-bhagavān uvāca", "bhagavān"),
        ("sunanda-nandāv ūcatuḥ", "sunandanandāv"),
    ],
)
def test_speaker_key(raw: str, key: str) -> None:
    assert speaker_key(raw) == key


def test_real_table_resolves_and_withholds_ambiguous(repo_root: Path) -> None:
    table = SpeakerTable.load(repo_root / "data" / "reference" / "speakers.yaml")
    assert len(table) > 150
    assert table.resolve("śrī-śuka uvāca") == ("Śukadeva Gosvāmī", False, "śuka")
    assert table.resolve("śrībādarāyaṇir uvāca")[0] == "Śukadeva Gosvāmī"
    name, ambiguous, _ = table.resolve("śrīrājovāca")
    assert name is None and ambiguous  # Parīkṣit in most chapters, but not all - never guessed
    assert table.resolve("nonexistent uvāca") == (None, False, "nonexistent")


def test_real_table_keys_are_normalized(repo_root: Path) -> None:
    document = yaml.safe_load(
        (repo_root / "data" / "reference" / "speakers.yaml").read_text(encoding="utf-8")
    )
    keys = [k for entry in document["speakers"] for k in entry["keys"]]
    assert len(keys) == len(set(keys))
    for key in keys:
        assert key == normalize_iast(key) and key == key.lower() and " " not in key


def test_duplicate_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("speakers:\n- {canonical: A, keys: [x]}\n- {canonical: B, keys: [x]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapped twice"):
        SpeakerTable.load(path)
