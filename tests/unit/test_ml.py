"""ML-side tests on a tiny randomly initialised Qwen3.5 (real architecture code, no download)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from bhagavatam_ai.config import LoraSettings, ModelExpectations, load_model_config
from bhagavatam_ai.hardware.benchmark import _verdict, theoretical_memory
from bhagavatam_ai.hardware.device import (
    DeviceUnavailableError,
    collect_environment,
    dtype_for_device,
    is_out_of_memory,
    select_device,
)
from bhagavatam_ai.hardware.memory import MemoryMonitor, peak_rss_bytes, take_snapshot
from bhagavatam_ai.hardware.ops_check import _classify, _max_rel_error, descriptions, run_op_checks
from bhagavatam_ai.inference.generate import GenerationParams, stop_token_ids
from bhagavatam_ai.inference.smoke import repeated_ngram_ratio
from bhagavatam_ai.modeling.loader import (
    ModelIntegrityError,
    checkpoint_key,
    spot_check_weights,
    verify_architecture,
    verify_loading_info,
)
from bhagavatam_ai.modeling.lora import build_lora_model, make_optimizer, trainable_parameters
from bhagavatam_ai.training.losses import causal_lm_loss, chunked_causal_lm_loss, fused_causal_lm_loss

EXPECTED = ModelExpectations(
    model_class="Qwen3_5ForCausalLM",
    num_hidden_layers=4,
    vocab_size=512,
    tie_word_embeddings=True,
    allowed_unexpected_key_patterns=(r"^mtp\.", r"^model\.visual\."),
    checkpoint_key_prefix="model.language_model.",
    integrity_spot_check_keys=(
        "model.embed_tokens.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.3.self_attn.q_proj.weight",
    ),
)


@pytest.fixture(scope="module")
def tiny_model() -> Any:
    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        max_position_embeddings=256,
        tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(cfg).eval()


# --------------------------------------------------------------------------- losses


def test_loss_modes_match_transformers(tiny_model: Any) -> None:
    ids = torch.randint(0, 512, (2, 37), generator=torch.Generator().manual_seed(1))
    reference = tiny_model(input_ids=ids, labels=ids).loss
    for mode in ("standard", "chunked", "fused"):
        assert torch.allclose(
            causal_lm_loss(tiny_model, ids, mode=mode, chunk_size=8), reference, atol=1e-5
        ), mode
    with pytest.raises(ValueError, match="unknown loss mode"):
        causal_lm_loss(tiny_model, ids, mode="bogus")


def test_loss_gradients_match_and_ignore_index() -> None:
    gen = torch.Generator().manual_seed(0)
    hidden = torch.randn(2, 30, 16, generator=gen, requires_grad=True)
    weight = (torch.randn(100, 16, generator=gen) * 0.1).requires_grad_(True)
    labels = torch.randint(0, 100, (2, 30), generator=gen)
    labels[0, :10] = -100
    ref = torch.nn.functional.cross_entropy(
        (hidden[:, :-1] @ weight.T).reshape(-1, 100), labels[:, 1:].reshape(-1), ignore_index=-100
    )
    ref_grads = torch.autograd.grad(ref, (hidden, weight))
    for fn in (
        lambda: chunked_causal_lm_loss(hidden, weight, labels, chunk_size=7),
        lambda: fused_causal_lm_loss(hidden, weight, labels),
    ):
        loss = fn()
        assert torch.allclose(loss, ref, atol=1e-6)
        for got, want in zip(torch.autograd.grad(loss, (hidden, weight)), ref_grads, strict=True):
            assert torch.allclose(got, want, atol=1e-6)


def test_chunked_loss_validation() -> None:
    with pytest.raises(ValueError, match="expected hidden_states"):
        chunked_causal_lm_loss(torch.zeros(2, 3), torch.zeros(4, 3), torch.zeros(2, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="shape mismatch"):
        chunked_causal_lm_loss(torch.zeros(1, 4, 3), torch.zeros(5, 3), torch.zeros(1, 5, dtype=torch.long))
    with pytest.raises(ValueError, match="chunk_size"):
        chunked_causal_lm_loss(
            torch.zeros(1, 4, 3), torch.zeros(5, 3), torch.zeros(1, 4, dtype=torch.long), chunk_size=0
        )
    all_ignored = chunked_causal_lm_loss(torch.zeros(1, 4, 3), torch.zeros(5, 3), torch.full((1, 4), -100))
    assert all_ignored.item() == 0.0  # no NaN from 0/0


# --------------------------------------------------------------------------- LoRA wiring


def test_configured_lora_targets_exist_and_training_reduces_loss(tiny_model: Any, repo_root: Path) -> None:
    import copy

    targets = load_model_config(repo_root / "configs/model.yaml").lora_target_modules
    linear_names = {n.split(".")[-1] for n, m in tiny_model.named_modules() if isinstance(m, torch.nn.Linear)}
    assert set(targets) <= linear_names  # every configured target exists in the real architecture
    model = build_lora_model(
        copy.deepcopy(tiny_model), LoraSettings(r=4, alpha=8), targets, gradient_checkpointing=True
    )
    params = trainable_parameters(model)
    assert params and all(p.dtype == torch.float32 for p in params)
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    optimizer = make_optimizer(params, 5e-3, torch.device("cpu"))
    ids = torch.randint(0, 512, (1, 32), generator=torch.Generator().manual_seed(2))
    losses = []
    for _ in range(8):
        loss = causal_lm_loss(model, ids, mode="chunked", chunk_size=8)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.05 and all(math.isfinite(x) for x in losses)


# --------------------------------------------------------------------------- loader integrity


def test_verify_loading_info() -> None:
    assert verify_loading_info({"unexpected_keys": {"mtp.fc.weight", "model.visual.x"}}, EXPECTED) == [
        "model.visual.x",
        "mtp.fc.weight",
    ]
    with pytest.raises(ModelIntegrityError, match="missing keys"):
        verify_loading_info({"missing_keys": {"model.layers.0.mlp.up_proj.weight"}}, EXPECTED)
    with pytest.raises(ModelIntegrityError, match="unexpected keys"):
        verify_loading_info({"unexpected_keys": {"model.mystery.weight"}}, EXPECTED)
    with pytest.raises(ModelIntegrityError, match="shape-mismatched"):
        verify_loading_info({"mismatched_keys": [("a", (1,), (2,))]}, EXPECTED)


def test_checkpoint_key() -> None:
    assert (
        checkpoint_key("model.layers.0.mlp.up_proj.weight", EXPECTED)
        == "model.language_model.layers.0.mlp.up_proj.weight"
    )
    with pytest.raises(ValueError):
        checkpoint_key("lm_head.weight", EXPECTED)


def test_verify_architecture(tiny_model: Any) -> None:
    assert verify_architecture(tiny_model, EXPECTED) == {"linear_attention": 3, "full_attention": 1}
    with pytest.raises(ModelIntegrityError, match="vocab_size"):
        verify_architecture(tiny_model, EXPECTED.model_copy(update={"vocab_size": 9}))


def test_spot_check_weights_detects_tampering(tiny_model: Any, tmp_path: Path) -> None:
    from safetensors.torch import save_file

    state = {
        checkpoint_key(k, EXPECTED): v.contiguous()
        for k, v in tiny_model.state_dict().items()
        if k.startswith("model.")
    }
    save_file(state, str(tmp_path / "model.safetensors"))
    assert all(spot_check_weights(tiny_model, tmp_path, EXPECTED).values())
    state["model.language_model.embed_tokens.weight"] = state["model.language_model.embed_tokens.weight"] + 1
    save_file(state, str(tmp_path / "model.safetensors"))
    with pytest.raises(ModelIntegrityError, match="differ from the checkpoint"):
        spot_check_weights(tiny_model, tmp_path, EXPECTED)
    index = {"weight_map": dict.fromkeys(state, "model.safetensors")}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ModelIntegrityError):
        spot_check_weights(tiny_model, tmp_path, EXPECTED)


# --------------------------------------------------------------------------- hardware


def test_op_checks_pass_on_cpu_fp32() -> None:
    results = run_op_checks(torch.device("cpu"), [torch.float32])
    assert {r.name for r in results} == set(descriptions())
    assert all(r.status == "pass" for r in results), [r for r in results if r.status != "pass"]


def test_op_check_helpers() -> None:
    assert _classify(NotImplementedError("x")) == "unsupported"
    assert (
        _classify(RuntimeError("The operator 'aten::foo' is not currently implemented for the MPS device"))
        == "unsupported"
    )
    assert _classify(RuntimeError("shape mismatch")) == "error"
    assert _max_rel_error([torch.tensor([float("nan")])], [torch.tensor([1.0])]) == math.inf
    assert _max_rel_error([], []) is None
    with pytest.raises(RuntimeError, match="count mismatch"):
        _max_rel_error([torch.zeros(1)], [])


def test_device_selection() -> None:
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in {"cpu", "mps", "cuda"}
    if not torch.backends.mps.is_available():
        with pytest.raises(DeviceUnavailableError, match="MPS requested"):
            select_device("mps")
    from bhagavatam_ai.config import DTypePolicy

    assert dtype_for_device(torch.device("cpu"), DTypePolicy()) == torch.float32
    assert is_out_of_memory(RuntimeError("MPS backend out of memory (MPS allocated: 9 GB)"))
    assert not is_out_of_memory(RuntimeError("other"))


def test_environment_report_has_no_personal_details() -> None:
    import getpass
    import socket

    env = json.dumps(collect_environment())
    assert socket.gethostname() not in env
    assert str(Path.home()) not in env
    user = getpass.getuser()
    if len(user) > 3:
        assert user not in env
    assert '"torch"' in env


def test_memory_monitor_cpu() -> None:
    assert peak_rss_bytes() > 0
    snapshot = take_snapshot()
    assert snapshot.device_allocated_bytes is None and snapshot.system_total_bytes > 0
    with MemoryMonitor(torch.device("cpu"), interval_s=0.01) as monitor:
        block = torch.ones(4_000_000)
        monitor.sample_device()
    del block
    summary = monitor.summary()
    assert summary["peak_process_rss_gb"] > 0 and summary["samples"] >= 1 and "swap_growth_gb" in summary


def test_theoretical_memory() -> None:
    est = theoretical_memory(752_390_000, 254_279_680, 10_220_000, weight_bytes=2)
    assert 1.5 < est["lora_static_gb"] < 1.6
    assert 11.0 < est["full_ft_all_params_gb"] < 11.3


def _report(device: str = "mps", **training: Any) -> dict[str, Any]:
    sweep = training.get(
        "sweep",
        [
            {
                "seq_len": 1024,
                "status": "ok",
                "tokens_per_s": 500.0,
                "hours_per_million_tokens": 0.56,
                "memory_pressure": False,
                "loss_mode": "chunked",
            }
        ],
    )
    return {
        "device": device,
        "dtype": "bfloat16",
        "op_checks": [{"name": "x", "dtype": "bfloat16", "status": "pass"}],
        "load": {"ok": True},
        "numerics": {"ok": True},
        "generation": {"ok": True},
        "training_sanity": {"ok": True},
        "training": {"sweep": sweep, "loss_mode_comparison": training.get("comparison", [])},
    }


def test_verdict_logic() -> None:
    from bhagavatam_ai.config import HardwareCheckConfig

    cfg = HardwareCheckConfig()
    ready = _verdict(_report(), cfg)
    assert ready["ready_for_phase2"] and ready["max_seq_len_without_memory_pressure"] == 1024
    cpu = _verdict(_report(device="cpu"), cfg)
    assert not cpu["ready_for_phase2"] and "CPU" in cpu["note"]
    failing_op = _report()
    failing_op["op_checks"].append(
        {"name": "gated_delta_rule_chunked", "dtype": "bfloat16", "status": "unsupported"}
    )
    assert not _verdict(failing_op, cfg)["ready_for_phase2"]
    fp16_only_failure = _report()
    fp16_only_failure["op_checks"].append({"name": "y", "dtype": "float16", "status": "numeric_mismatch"})
    assert _verdict(fp16_only_failure, cfg)["ready_for_phase2"]  # fp16 is not used for training
    faster = _report(
        comparison=[{"loss_mode": "fused", "status": "ok", "tokens_per_s": 650.0, "memory_pressure": False}]
    )
    assert _verdict(faster, cfg)["fastest_loss_mode"] == "fused"
    pressure = _report(
        sweep=[
            {
                "seq_len": 256,
                "status": "ok",
                "tokens_per_s": 1.0,
                "hours_per_million_tokens": 1,
                "memory_pressure": True,
                "loss_mode": "chunked",
            }
        ]
    )
    assert not _verdict(pressure, cfg)["lora_training_ok"]


# --------------------------------------------------------------------------- inference helpers


class _StubTokenizer:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_end|>": 248046, "<|endoftext|>": 248044}.get(token, 0)


def test_stop_tokens_and_params() -> None:
    assert stop_token_ids(_StubTokenizer()) == [248046, 248044]
    assert stop_token_ids(_StubTokenizer(), ["<missing>"]) == []
    greedy = GenerationParams(max_new_tokens=5).to_generate_kwargs()
    assert greedy == {"max_new_tokens": 5, "do_sample": False, "repetition_penalty": 1.05}
    sampled = GenerationParams(do_sample=True, temperature=0.5).to_generate_kwargs()
    assert sampled["temperature"] == 0.5 and sampled["top_p"] == 0.9


def test_repeated_ngram_ratio() -> None:
    assert repeated_ngram_ratio("") == 0.0
    assert repeated_ngram_ratio("a b c d e f") == 0.0
    assert repeated_ngram_ratio("a b c a b c a b c") > 0.5
