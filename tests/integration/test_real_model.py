"""Tests against the real Qwen/Qwen3.5-0.8B-Base weights (opt-in: --run-model; ~1.7 GB download)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from bhagavatam_ai.config import load_model_config

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def model_cfg() -> Any:
    return load_model_config(Path(__file__).resolve().parents[2] / "configs/model.yaml")


@pytest.fixture(scope="module")
def loaded(model_cfg: Any) -> Any:
    from bhagavatam_ai.modeling.loader import load_model, load_tokenizer, snapshot_dir

    snapshot = snapshot_dir(model_cfg)
    model, report = load_model(model_cfg, torch.device("cpu"), torch.float32, snapshot=snapshot)
    return model, load_tokenizer(model_cfg, snapshot=snapshot), report


def test_integrity(loaded: Any) -> None:
    _, _, report = loaded
    assert report.model_class == "Qwen3_5ForCausalLM"
    assert report.missing_keys == [] and all(report.spot_checks.values()) and report.tied_embeddings
    assert report.layer_types == {"linear_attention": 18, "full_attention": 6}
    assert 750e6 < report.num_parameters < 755e6


def test_forward_is_finite_and_knows_english(loaded: Any) -> None:
    model, tokenizer, _ = loaded
    ids = tokenizer(
        "The sages asked the narrator to describe the qualities of the Lord.", return_tensors="pt"
    ).input_ids
    with torch.inference_mode():
        out = model(input_ids=ids, labels=ids)
    assert torch.isfinite(out.logits).all() and out.loss.item() < 5.0


def test_tokenizer_report(loaded: Any) -> None:
    from bhagavatam_ai.modeling.tokenizer_inspect import inspect_tokenizer

    report = inspect_tokenizer(loaded[1])
    assert report["control_tokens_all_single"]
    assert report["iast"]["all_round_trip_exact"]
    assert report["nfc_and_nfd_tokenize_identically"]  # Qwen3_5Tokenizer applies NFC itself
    assert not report["anusvara_variants_tokenize_identically"]  # so ṁ -> ṃ normalisation matters
    assert (
        report["chat_template"]["generation_prompt_suffix"]
        == "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_smoke_generation(loaded: Any) -> None:
    from bhagavatam_ai.inference.smoke import run_smoke

    result = run_smoke(loaded[0], loaded[1], max_new_tokens=6)
    assert result["all_ok"]


def test_cached_decoding_matches_full_forward(loaded: Any) -> None:
    """The hardware check's generation test on the real model: finite logits, and the cached decode
    (DeltaNet recurrent + conv state, KV cache) agrees with a no-cache teacher-forced forward pass."""
    from bhagavatam_ai.hardware.benchmark import _generation

    result = _generation(loaded[0], loaded[1], torch.device("cpu"), 12)
    assert result["logits_finite"] and result["cache_consistency"] >= 0.9 and result["ok"], result


def test_lora_step_on_real_model(loaded: Any, model_cfg: Any) -> None:
    """Runs last: trains adapters in place (no 3 GB deepcopy), then unloads them."""
    from bhagavatam_ai.config import LoraSettings
    from bhagavatam_ai.modeling.lora import build_lora_model, make_optimizer, trainable_parameters
    from bhagavatam_ai.training.losses import causal_lm_loss

    model = build_lora_model(
        loaded[0], LoraSettings(r=4, alpha=8), model_cfg.lora_target_modules, gradient_checkpointing=True
    )
    params = trainable_parameters(model)
    optimizer = make_optimizer(params, 1e-3, torch.device("cpu"))
    ids = loaded[1]("Śukadeva spoke to Parīkṣit on the bank of the Gaṅgā.", return_tensors="pt").input_ids
    losses = []
    for _ in range(3):
        loss = causal_lm_loss(model, ids, mode="chunked", chunk_size=16)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    base = model.unload()
    base.activation_checkpointing_disable()
    assert not any("lora_" in name for name, _ in base.named_parameters())
    assert losses[-1] < losses[0]
