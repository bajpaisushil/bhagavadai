"""Leakage-free train/validation split at the chapter level.

The split depends only on the seed and the canonical 335-chapter structure, never on the data, so
every source (Sanskrit, each English translation, later SFT examples) assigns a given chapter to the
same split. A chapter's verses can therefore never appear in training through one translation and in
validation through another.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from bhagavatam_ai.config import DedupConfig, SplitConfig
from bhagavatam_ai.data.dedup import exact_duplicates, near_duplicates
from bhagavatam_ai.data.reference import ChapterRef, all_chapters

Split = Literal["train", "val"]


@dataclass(frozen=True)
class SplitPlan:
    seed: int
    val_fraction: float
    val_chapters: tuple[str, ...]

    def split_of(self, canto: int, chapter: int) -> Split:
        return "val" if f"{canto}.{chapter}" in self._val else "train"

    @property
    def _val(self) -> frozenset[str]:
        return frozenset(self.val_chapters)

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "val_fraction": self.val_fraction, "val_chapters": list(self.val_chapters)}


def plan_split(cfg: SplitConfig, chapters: Iterable[ChapterRef] | None = None) -> SplitPlan:
    """Stratified by canto: each canto contributes ``round(n * val_fraction)`` chapters (>=1 if n >= 10)."""
    by_canto: dict[int, list[ChapterRef]] = defaultdict(list)
    for ref in chapters if chapters is not None else all_chapters():
        by_canto[ref.canto].append(ref)
    chosen: set[str] = {str(ChapterRef.parse(c)) for c in cfg.forced_val_chapters}
    if cfg.val_fraction > 0:
        for canto, refs in sorted(by_canto.items()):
            k = round(len(refs) * cfg.val_fraction)
            if k == 0 and len(refs) >= 10:
                k = 1
            rng = random.Random(f"{cfg.seed}:{canto}")
            chosen.update(str(r) for r in rng.sample(sorted(refs), k))
    ordered = sorted(chosen, key=lambda s: tuple(int(x) for x in s.split(".")))
    return SplitPlan(seed=cfg.seed, val_fraction=cfg.val_fraction, val_chapters=tuple(ordered))


def leakage_report(train: Mapping[str, str], val: Mapping[str, str], cfg: DedupConfig) -> dict[str, Any]:
    """Exact and near-duplicate text shared between splits (keys are record ids)."""
    combined = {f"train|{k}": v for k, v in train.items()} | {f"val|{k}": v for k, v in val.items()}
    exact_cross = [g for g in exact_duplicates(combined) if {x.split("|")[0] for x in g} == {"train", "val"}]
    near = near_duplicates(train, cfg, cross=val)
    return {
        "exact_cross_split_groups": len(exact_cross),
        "exact_examples": exact_cross[:20],
        "near_duplicate_pairs": len(near),
        "near_examples": [{"train": n.a, "val": n.b, "jaccard": n.jaccard} for n in near[:20]],
    }
