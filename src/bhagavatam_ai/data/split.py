"""Leakage-free train/validation split at the chapter level.

The split depends only on the seed and the canonical 335-chapter structure, never on the data, so
every source (Sanskrit, each English translation, later SFT examples) assigns a given chapter to the
same split. A chapter's verses can therefore never appear in training through one translation and in
validation through another.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
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


def equivalence_groups(texts: Mapping[str, str], cfg: DedupConfig) -> list[list[str]]:
    """Keys whose texts are identical or near-identical, grouped transitively (union-find)."""
    parent = {key: key for key in texts}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for group in exact_duplicates(texts):
        for key in group[1:]:
            parent[find(key)] = find(group[0])
    for pair in near_duplicates(texts, cfg):
        parent[find(pair.b)] = find(pair.a)
    groups: dict[str, list[str]] = defaultdict(list)
    for key in texts:
        groups[find(key)].append(key)
    return sorted(sorted(g) for g in groups.values() if len(g) > 1)


def _chapter_of(ref_key: str) -> tuple[int, int]:
    canto, chapter, *_ = (int(x) for x in ref_key.split("."))
    return canto, chapter


def decontaminate(
    train: Sequence[Mapping[str, Any]],
    val: Sequence[Mapping[str, Any]],
    cfg: DedupConfig,
    *,
    equivalent_refs: Iterable[Sequence[str]] = (),
    is_train: Callable[[int, int], bool],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    """Drop validation rows that the training split already teaches.

    A row leaves validation when (a) its verse belongs to an equivalence group (the edition repeats the
    verse elsewhere) that has a member in a training chapter - this holds across languages, since every
    translation of a repeated verse renders the same Sanskrit - or (b) its text is an exact or near
    duplicate of a training row. Training rows are never dropped: the model should still learn every
    verse. Returns ``(kept_val_rows, removed)``.
    """
    leaked_refs: set[str] = set()
    for group in equivalent_refs:
        if any(is_train(*_chapter_of(ref)) for ref in group):
            leaked_refs.update(ref for ref in group if not is_train(*_chapter_of(ref)))
    removed: list[dict[str, Any]] = []
    kept: list[Mapping[str, Any]] = []
    for row in val:
        if row["ref_key"] in leaked_refs:
            removed.append(
                {"record_id": row["record_id"], "ref": row["ref"], "reason": "equivalent_verse_in_train"}
            )
        else:
            kept.append(row)
    train_texts = {str(r["record_id"]): str(r["text"]) for r in train}
    val_texts = {str(r["record_id"]): str(r["text"]) for r in kept}
    combined = {f"t|{k}": v for k, v in train_texts.items()} | {f"v|{k}": v for k, v in val_texts.items()}
    contaminated = {
        key[2:]
        for group in exact_duplicates(combined)
        if any(k.startswith("t|") for k in group)
        for key in group
        if key.startswith("v|")
    }
    contaminated |= {pair.b for pair in near_duplicates(train_texts, cfg, cross=val_texts)}
    final: list[Mapping[str, Any]] = []
    for row in kept:
        if row["record_id"] in contaminated:
            removed.append(
                {"record_id": row["record_id"], "ref": row["ref"], "reason": "text_duplicate_in_train"}
            )
        else:
            final.append(row)
    return final, removed


def summarize_removed(removed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "removed_from_val": len(removed),
        "by_reason": dict(Counter(str(r["reason"]) for r in removed)),
        "examples": [dict(r) for r in removed[:20]],
    }
