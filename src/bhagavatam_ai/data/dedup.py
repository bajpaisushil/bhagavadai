"""Exact and near-duplicate detection (MinHash + LSH over character shingles).

Used to detect repeated passages within a source and, more importantly, leakage of near-identical
text between the training and validation splits.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from bhagavatam_ai.config import DedupConfig

_MERSENNE = np.uint64((1 << 61) - 1)
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def normalize_for_dedup(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


def shingles(text: str, k: int) -> set[str]:
    norm = normalize_for_dedup(text)
    if len(norm) <= k:
        return {norm} if norm else set()
    return {norm[i : i + k] for i in range(len(norm) - k + 1)}


def _hash64(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little") >> 3


@dataclass(frozen=True)
class NearDuplicate:
    a: str
    b: str
    jaccard: float


class MinHasher:
    def __init__(self, num_perm: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, int(_MERSENNE), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_MERSENNE), size=num_perm, dtype=np.uint64)

    def signature(self, tokens: set[str]) -> np.ndarray:
        if not tokens:
            return np.full(self.a.shape, np.iinfo(np.uint64).max, dtype=np.uint64)
        hashes = np.fromiter((_hash64(t) for t in tokens), dtype=np.uint64, count=len(tokens))
        # (a*h + b) mod p, computed in uint64; values are < 2^61 so products wrap - acceptable for
        # hashing because the same wrap is applied to every item.
        values = (self.a[:, None] * hashes[None, :] + self.b[:, None]) % _MERSENNE
        return values.min(axis=1)


def exact_duplicates(texts: Mapping[str, str]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for key, text in texts.items():
        groups[normalize_for_dedup(text)].append(key)
    return [sorted(keys) for norm, keys in groups.items() if norm and len(keys) > 1]


def near_duplicates(
    texts: Mapping[str, str], cfg: DedupConfig, *, cross: Mapping[str, str] | None = None
) -> list[NearDuplicate]:
    """Near-duplicate pairs within ``texts`` or, if ``cross`` is given, between ``texts`` and ``cross``."""
    if cfg.num_perm % cfg.bands:
        raise ValueError("num_perm must be divisible by bands")
    rows = cfg.num_perm // cfg.bands
    hasher = MinHasher(cfg.num_perm)
    items = {f"a:{k}": v for k, v in texts.items()}
    if cross is not None:
        items.update({f"b:{k}": v for k, v in cross.items()})
    sets = {key: shingles(text, cfg.shingle_size) for key, text in items.items()}
    buckets: dict[tuple[int, bytes], list[str]] = defaultdict(list)
    for key, tokens in sets.items():
        if not tokens:
            continue
        signature = hasher.signature(tokens)
        for band in range(cfg.bands):
            buckets[(band, signature[band * rows : (band + 1) * rows].tobytes())].append(key)
    candidates: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i, x in enumerate(members):
            for y in members[i + 1 :]:
                pair = (x, y) if x < y else (y, x)
                if cross is None or pair[0][0] != pair[1][0]:
                    candidates.add(pair)
    results = []
    for x, y in sorted(candidates):
        inter = len(sets[x] & sets[y])
        jaccard = inter / len(sets[x] | sets[y])
        if jaccard >= cfg.near_duplicate_threshold:
            results.append(NearDuplicate(x[2:], y[2:], round(jaccard, 4)))
    return results
