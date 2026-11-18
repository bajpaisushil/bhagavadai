"""Tokenizer and chat-template inspection.

Answers the questions that shape data preparation and SFT formatting:

* are the chat control tokens single tokens (so LoRA need not train embeddings)?
* what exactly does the chat template render, including Qwen3.5's empty ``<think>`` block?
* how expensive is IAST-transliterated Sanskrit, and do spelling variants change tokenisation?
  Measured on Qwen3.5: the tokenizer applies NFC itself, so NFC and NFD input tokenise identically;
  but the two anusvara spellings (``ṁ`` dot-above vs ``ṃ`` dot-below) are different characters and
  tokenise differently, which is why the corpus is normalised to one convention.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from typing import Any, Final

from bhagavatam_ai.prompts import SYSTEM_PROMPT

CONTROL_TOKENS: Final = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")

IAST_TERMS: Final = (
    "Śrīmad-Bhāgavatam",
    "Bhāgavata Purāṇa",
    "Śukadeva Gosvāmī",
    "Sūta Gosvāmī",
    "Maitreya Ṛṣi",
    "Parīkṣit",
    "Kṛṣṇa",
    "Nārada",
    "Vyāsadeva",
    "Prahlāda",
    "Dhruva",
    "Uddhava",
    "Vidura",
    "bhakti",
    "dharma",
    "ātmā",
    "Brahman",
    "Paramātmā",
    "Bhagavān",
    "māyā",
    "guṇa",
    "saṅkīrtana",
    "Vaikuṇṭha",
    "Naimiṣāraṇya",
)

# (label, text) pairs that differ only in Unicode form or transliteration convention.
NORMALIZATION_PROBES: Final = (
    ("kṛṣṇa NFC", unicodedata.normalize("NFC", "kṛṣṇa")),
    ("kṛṣṇa NFD", unicodedata.normalize("NFD", "kṛṣṇa")),
    ("krishna ASCII", "krishna"),
    ("kRSNa Harvard-Kyoto", "kRSNa"),
    ("saṃsāra dot-below anusvara", "saṃsāra"),
    ("saṁsāra dot-above anusvara", "saṁsāra"),
)

ENGLISH_SAMPLE: Final = (
    "The narrator answers the king's questions patiently, explaining how attachment and duty shape a "
    "person's life, and why the sages valued hearing about the Lord above all other pursuits."
)

SAMPLE_CONVERSATION: Final = (
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Who is Parīkṣit?"},
    {"role": "assistant", "content": "Parīkṣit is the king who hears the Bhāgavatam from Śukadeva."},
    {"role": "user", "content": "Where did they meet?"},
)


def _ids(tokenizer: Any, text: str) -> list[int]:
    ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
    return ids


def inspect_tokenizer(tokenizer: Any) -> dict[str, Any]:
    template: str = tokenizer.chat_template or ""
    control = {}
    for token in CONTROL_TOKENS:
        ids = _ids(tokenizer, token)
        control[token] = {"ids": ids, "single_token": len(ids) == 1}

    iast_rows = []
    for term in IAST_TERMS:
        ids = _ids(tokenizer, term)
        iast_rows.append(
            {
                "term": term,
                "tokens": len(ids),
                "chars_per_token": round(len(term) / len(ids), 2),
                "round_trip_exact": tokenizer.decode(ids) == term,
            }
        )

    probes = []
    for label, text in NORMALIZATION_PROBES:
        ids = _ids(tokenizer, text)
        probes.append({"label": label, "codepoints": len(text), "tokens": len(ids), "ids": ids})
    nfc_ids, nfd_ids = probes[0]["ids"], probes[1]["ids"]

    english_ids = _ids(tokenizer, ENGLISH_SAMPLE)
    rendered = tokenizer.apply_chat_template(
        list(SAMPLE_CONVERSATION), tokenize=False, add_generation_prompt=True
    )
    without_prompt = tokenizer.apply_chat_template(
        list(SAMPLE_CONVERSATION), tokenize=False, add_generation_prompt=False
    )
    total_iast_chars = sum(len(r["term"]) for r in iast_rows)
    total_iast_tokens = sum(r["tokens"] for r in iast_rows)
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size_with_added": len(tokenizer),
        "model_max_length": tokenizer.model_max_length,
        "special_tokens": {
            "bos": tokenizer.bos_token,
            "eos": tokenizer.eos_token,
            "eos_id": tokenizer.eos_token_id,
            "pad": tokenizer.pad_token,
            "pad_id": tokenizer.pad_token_id,
            "padding_side": tokenizer.padding_side,
        },
        "control_tokens": control,
        "control_tokens_all_single": all(v["single_token"] for v in control.values()),
        "chat_template": {
            "sha256": hashlib.sha256(template.encode()).hexdigest(),
            "supports_enable_thinking": "enable_thinking" in template,
            "has_generation_markers": "{% generation %}" in template,
            "generation_prompt_suffix": rendered[len(without_prompt) :],
            "rendered_sample": rendered,
        },
        "iast": {
            "terms": iast_rows,
            "mean_chars_per_token": round(total_iast_chars / total_iast_tokens, 2),
            "all_round_trip_exact": all(r["round_trip_exact"] for r in iast_rows),
        },
        "normalization_probes": probes,
        "nfc_and_nfd_tokenize_identically": nfc_ids == nfd_ids,
        "anusvara_variants_tokenize_identically": probes[4]["ids"] == probes[5]["ids"],
        "english": {
            "words": len(ENGLISH_SAMPLE.split()),
            "tokens": len(english_ids),
            "tokens_per_word": round(len(english_ids) / len(ENGLISH_SAMPLE.split()), 2),
            "chars_per_token": round(len(ENGLISH_SAMPLE) / len(english_ids), 2),
        },
    }


def embedding_row_norms(
    snapshot: Any, embed_key: str, token_ids: dict[str, int], *, sample: int = 4096, seed: int = 0
) -> dict[str, Any]:
    """L2 norms of selected embedding rows vs the vocabulary's typical norm, read lazily from safetensors.

    An untrained (near-zero or initial-scale) row for a control token means LoRA cannot learn to use
    it; Qwen states <|im_start|>/<|im_end|> were trained, but <think> rows are unverified.
    """
    import json
    from pathlib import Path

    import torch
    from safetensors import safe_open

    root = Path(snapshot)
    index = root / "model.safetensors.index.json"
    filename = (
        json.loads(index.read_text())["weight_map"][embed_key] if index.exists() else "model.safetensors"
    )
    with safe_open(str(root / filename), "pt") as handle:
        table = handle.get_slice(embed_key)
        vocab = table.get_shape()[0]
        gen = torch.Generator().manual_seed(seed)
        rows = torch.randint(0, vocab, (sample,), generator=gen).tolist()
        sampled = torch.stack([table[r : r + 1][0].float() for r in rows])
        selected = {name: float(table[i : i + 1][0].float().norm()) for name, i in token_ids.items()}
        vectors = {name: table[i : i + 1][0].float() for name, i in token_ids.items()}
    norms = sampled.norm(dim=-1)
    median = float(norms.median())
    unit_sample = sampled / norms.clamp(min=1e-8)[:, None]
    hidden = sampled.shape[1]
    tokens: dict[str, Any] = {}
    for name, vec in vectors.items():
        unit = vec / vec.norm().clamp(min=1e-8)
        cosines = unit_sample @ unit
        tokens[name] = {
            "norm": round(selected[name], 4),
            "ratio_to_median": round(selected[name] / median, 3),
            "max_cosine_to_sample": round(float(cosines.max()), 3),
        }
    names = list(vectors)
    pairwise = {
        f"{a}|{b}": round(float(torch.nn.functional.cosine_similarity(vectors[a], vectors[b], dim=0)), 3)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
    return {
        "vocab_rows": vocab,
        "sampled_rows": sample,
        "median_norm": round(median, 4),
        "p05_norm": round(float(norms.quantile(0.05)), 4),
        # For an untrained N(0, 0.02^2) row: norm ~ 0.02*sqrt(hidden) and |cosine| to other rows ~ 0
        # (max over the sample ~ sqrt(2 ln(sample) / hidden)). Norm alone cannot tell trained from random.
        "random_init_reference": {
            "norm": round(0.02 * hidden**0.5, 3),
            "max_abs_cosine": round((2 * math.log(sample) / hidden) ** 0.5, 3),
        },
        "tokens": tokens,
        "pairwise_cosine": pairwise,
    }
