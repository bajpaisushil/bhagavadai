"""Op-level compatibility checks for the exact code paths Qwen3.5 runs off-CUDA.

The CUDA-only ``flash-linear-attention`` and ``causal-conv1d`` kernels are unavailable on Apple
Silicon, so transformers runs Qwen3.5's 18 Gated DeltaNet layers through reference PyTorch code.
Each check below exercises one of those paths (forward, and backward where training needs it) on
the target device and compares it with a float32 CPU reference computed from identical inputs.

A ``NotImplementedError`` (or an "not implemented/supported for MPS" runtime error) is reported as
``unsupported``; run without ``PYTORCH_ENABLE_MPS_FALLBACK`` so such ops surface here instead of
silently running on the CPU.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from bhagavatam_ai.hardware.device import synchronize

Status = Literal["pass", "numeric_mismatch", "unsupported", "error"]
Inputs = dict[str, torch.Tensor]
OpFn = Callable[[Inputs, torch.device, torch.dtype], list[torch.Tensor]]

# Max error relative to the reference tensor's largest magnitude.
DEFAULT_TOLERANCE: dict[torch.dtype, float] = {torch.float32: 2e-3, torch.bfloat16: 6e-2, torch.float16: 2e-2}
_UNSUPPORTED_MARKERS = ("not implemented", "not currently implemented", "not supported", "unsupported")
_CPU = torch.device("cpu")


@dataclass(frozen=True)
class OpCheckResult:
    name: str
    dtype: str
    status: Status
    max_rel_error: float | None
    tolerance: float
    seconds: float | None
    detail: str | None = None


@dataclass(frozen=True)
class OpCheck:
    name: str
    description: str
    make_inputs: Callable[[torch.Generator], Inputs]
    fn: OpFn
    dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.bfloat16, torch.float16)
    tolerance: dict[torch.dtype, float] | None = None


def _place(inputs: Inputs, device: torch.device, dtype: torch.dtype, grad: Sequence[str] = ()) -> Inputs:
    placed: Inputs = {}
    for key, value in inputs.items():
        tensor = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        placed[key] = tensor.detach().requires_grad_(True) if key in grad else tensor
    return placed


def _grad(t: torch.Tensor) -> torch.Tensor:
    if t.grad is None:
        raise RuntimeError("expected a gradient but none was produced")
    return t.grad


def _randn(gen: torch.Generator, *shape: int, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, generator=gen) * scale


# --------------------------------------------------------------------------- checks


def _matmul(x: Inputs, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    p = _place(x, device, dtype)
    return [p["a"] @ p["b"]]


def _lora_linear_backward(x: Inputs, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    p = _place(x, device, dtype, grad=("lora_a", "lora_b"))
    y = F.linear(p["x"], p["w"]) + F.linear(F.linear(p["x"], p["lora_a"]), p["lora_b"]) * 2.0
    torch.autograd.backward(y.float().square().mean())
    return [y, _grad(p["lora_a"]), _grad(p["lora_b"])]


def _causal_conv1d(x: Inputs, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    # Mirrors transformers' reference ``causal_conv1d_fn`` (depthwise, left padding, SiLU).
    p = _place(x, device, dtype, grad=("x",))
    channels, seq_len = p["x"].shape[1], p["x"].shape[2]
    out = F.conv1d(p["x"], p["w"].unsqueeze(1), None, padding=p["w"].shape[-1] - 1, groups=channels)
    out = F.silu(out[..., :seq_len])
    torch.autograd.backward(out.float().square().mean())
    return [out, _grad(p["x"])]


def _gated_delta_rule_chunked(x: Inputs, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    p = _place(x, device, dtype, grad=("q", "k", "v"))
    out, _ = torch_chunk_gated_delta_rule(
        p["q"], p["k"], p["v"], g=p["g"].float(), beta=p["beta"], use_qk_l2norm_in_kernel=True
    )
    torch.autograd.backward(out.float().square().mean())
    return [out, _grad(p["q"]), _grad(p["k"]), _grad(p["v"])]


def _gated_delta_rule_recurrent(x: Inputs, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

    p = _place(x, device, dtype)
    out, state = torch_recurrent_gated_delta_rule(
        p["q"],
        p["k"],
        p["v"],
        g=p["g"].float(),
        beta=p["beta"],
        initial_state=p["state"].float(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    if state is None:
        raise RuntimeError("recurrent rule returned no final state")
    return [out, state]


def _sdpa_gqa(x: Inputs, device: torch.device, dtype: torch.dtype, *, masked: bool) -> list[torch.Tensor]:
    # Qwen3.5 full-attention layers: 8 query heads, 2 KV heads, head_dim 256. transformers passes
    # ``enable_gqa=True`` to SDPA on MPS with torch >= 2.13.
    p = _place(x, device, dtype, grad=("q",))
    if masked:
        seq_len = p["q"].shape[-2]
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        out = F.scaled_dot_product_attention(p["q"], p["k"], p["v"], attn_mask=mask, enable_gqa=True)
    else:
        out = F.scaled_dot_product_attention(p["q"], p["k"], p["v"], is_causal=True, enable_gqa=True)
    torch.autograd.backward(out.float().square().mean())
    return [out, _grad(p["q"])]


def _rmsnorm_gated(x: Inputs, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    # Mirrors Qwen3_5RMSNormGated.forward.
    p = _place(x, device, dtype, grad=("h",))
    hidden = p["h"].to(torch.float32)
    hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + 1e-6)
    out = (p["w"] * hidden.to(dtype)) * F.silu(p["z"].to(torch.float32))
    out = out.to(dtype)
    torch.autograd.backward(out.float().square().mean())
    return [out, _grad(p["h"])]


def _cross_entropy_large_vocab(x: Inputs, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    p = _place(x, device, dtype, grad=("h",))
    loss = F.cross_entropy(F.linear(p["h"], p["w"]).float(), p["targets"])
    torch.autograd.backward(loss)
    return [loss.reshape(1), _grad(p["h"])]


def _embedding_backward(x: Inputs, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    p = _place(x, device, dtype, grad=("table",))
    emb = F.embedding(p["ids"], p["table"])
    torch.autograd.backward(emb.float().square().sum())
    rows = torch.unique(p["ids"])
    return [emb, _grad(p["table"])[rows]]


def _adamw_step(x: Inputs, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    param = torch.nn.Parameter(x["param"].to(device=device, dtype=dtype).clone())
    # Fused AdamW on accelerators; plain loop on the CPU reference.
    opt = (
        torch.optim.AdamW([param], lr=1e-2, weight_decay=0.01, foreach=False)
        if device.type == "cpu"
        else torch.optim.AdamW([param], lr=1e-2, weight_decay=0.01, fused=True)
    )
    for step in range(3):
        param.grad = (x["grad"] * (step + 1)).to(device=device, dtype=dtype)
        opt.step()
    return [param.detach()]


def _autocast_matmul(x: Inputs, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    a, b = x["a"].to(device), x["b"].to(device)
    if dtype == torch.float32:
        return [a @ b]
    with torch.autocast(device_type=device.type, dtype=dtype):
        out = a @ b
    if out.dtype != dtype:
        raise RuntimeError(f"autocast produced {out.dtype}, expected {dtype}")
    return [out]


def _sampling(x: Inputs, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    logits = x["logits"].to(device=device, dtype=dtype)
    probs = torch.softmax(logits.float(), dim=-1)
    values, _ = torch.topk(probs, k=50, dim=-1)
    sample = torch.multinomial(probs, num_samples=1)
    vocab = logits.shape[-1]
    if not bool(((sample >= 0) & (sample < vocab)).all()):
        raise RuntimeError("multinomial returned out-of-range token ids")
    return [values]


VOCAB = 248_320


def _check_list() -> list[OpCheck]:
    f32 = (torch.float32,)
    return [
        OpCheck(
            "matmul_batched",
            "batched matmul",
            lambda g: {"a": _randn(g, 8, 64, 128), "b": _randn(g, 8, 128, 64)},
            _matmul,
        ),
        OpCheck(
            "lora_linear_backward",
            "linear + LoRA branch, backward",
            lambda g: {
                "x": _randn(g, 2, 32, 256),
                "w": _randn(g, 256, 256, scale=0.05),
                "lora_a": _randn(g, 16, 256, scale=0.05),
                "lora_b": _randn(g, 256, 16, scale=0.05),
            },
            _lora_linear_backward,
        ),
        OpCheck(
            "causal_conv1d",
            "depthwise causal conv1d + SiLU (Gated DeltaNet), backward",
            lambda g: {"x": _randn(g, 1, 384, 96), "w": _randn(g, 384, 4, scale=0.5)},
            _causal_conv1d,
        ),
        OpCheck(
            "gated_delta_rule_chunked",
            "transformers torch_chunk_gated_delta_rule (training/prefill), backward",
            lambda g: {
                "q": _randn(g, 1, 96, 4, 32),
                "k": _randn(g, 1, 96, 4, 32),
                "v": _randn(g, 1, 96, 4, 32),
                "g": -F.softplus(_randn(g, 1, 96, 4)) * 0.5,
                "beta": torch.sigmoid(_randn(g, 1, 96, 4)),
            },
            _gated_delta_rule_chunked,
        ),
        OpCheck(
            "gated_delta_rule_recurrent",
            "transformers torch_recurrent_gated_delta_rule (cached decoding)",
            lambda g: {
                "q": _randn(g, 1, 1, 4, 32),
                "k": _randn(g, 1, 1, 4, 32),
                "v": _randn(g, 1, 1, 4, 32),
                "g": -F.softplus(_randn(g, 1, 1, 4)) * 0.5,
                "beta": torch.sigmoid(_randn(g, 1, 1, 4)),
                "state": _randn(g, 1, 4, 32, 32, scale=0.1),
            },
            _gated_delta_rule_recurrent,
        ),
        OpCheck(
            "sdpa_gqa_causal",
            "SDPA, 8q/2kv heads, head_dim 256, is_causal, enable_gqa, backward",
            lambda g: {
                "q": _randn(g, 1, 8, 64, 256),
                "k": _randn(g, 1, 2, 64, 256),
                "v": _randn(g, 1, 2, 64, 256),
            },
            lambda x, d, t: _sdpa_gqa(x, d, t, masked=False),
        ),
        OpCheck(
            "sdpa_gqa_masked",
            "SDPA with explicit boolean mask + enable_gqa, backward",
            lambda g: {
                "q": _randn(g, 1, 8, 64, 256),
                "k": _randn(g, 1, 2, 64, 256),
                "v": _randn(g, 1, 2, 64, 256),
            },
            lambda x, d, t: _sdpa_gqa(x, d, t, masked=True),
        ),
        OpCheck(
            "rmsnorm_gated",
            "gated RMSNorm (fp32 upcast), backward",
            lambda g: {"h": _randn(g, 64, 128), "w": 1 + _randn(g, 128, scale=0.1), "z": _randn(g, 64, 128)},
            _rmsnorm_gated,
        ),
        OpCheck(
            "cross_entropy_large_vocab",
            "cross-entropy over the 248,320-token vocabulary, backward",
            lambda g: {
                "h": _randn(g, 32, 64),
                "w": _randn(g, VOCAB, 64, scale=0.05),
                "targets": torch.randint(0, VOCAB, (32,), generator=g),
            },
            _cross_entropy_large_vocab,
        ),
        OpCheck(
            "embedding_backward",
            "embedding lookup + backward on a 248,320-row table",
            lambda g: {
                "table": _randn(g, VOCAB, 64, scale=0.02),
                "ids": torch.randint(0, VOCAB, (2, 128), generator=g),
            },
            _embedding_backward,
        ),
        OpCheck(
            "adamw_fused",
            "fused AdamW optimizer step (fp32 LoRA parameters)",
            lambda g: {"param": _randn(g, 4096), "grad": _randn(g, 4096, scale=0.1)},
            _adamw_step,
            dtypes=f32,
        ),
        OpCheck(
            "autocast_matmul",
            "torch.autocast mixed precision matmul",
            lambda g: {"a": _randn(g, 64, 128), "b": _randn(g, 128, 64)},
            _autocast_matmul,
        ),
        OpCheck(
            "sampling_topk_multinomial",
            "softmax/top-k/multinomial over the full vocabulary",
            lambda g: {"logits": _randn(g, 2, VOCAB)},
            _sampling,
        ),
    ]


def _max_rel_error(outputs: Sequence[torch.Tensor], reference: Sequence[torch.Tensor]) -> float | None:
    if len(outputs) != len(reference):
        raise RuntimeError(f"output count mismatch: {len(outputs)} vs {len(reference)}")
    worst: float | None = None
    for out, ref in zip(outputs, reference, strict=True):
        actual = out.detach().to(_CPU, torch.float32)
        if actual.shape != ref.shape:
            raise RuntimeError(f"shape mismatch {tuple(actual.shape)} vs {tuple(ref.shape)}")
        if not bool(torch.isfinite(actual).all()):
            return float("inf")
        scale = float(ref.abs().max().clamp(min=1e-6))
        err = float((actual - ref).abs().max()) / scale
        worst = err if worst is None else max(worst, err)
    return worst


def _classify(exc: BaseException) -> Status:
    if isinstance(exc, NotImplementedError):
        return "unsupported"
    message = str(exc).lower()
    return "unsupported" if any(marker in message for marker in _UNSUPPORTED_MARKERS) else "error"


def run_op_checks(
    device: torch.device, dtypes: Sequence[torch.dtype], *, seed: int = 0
) -> list[OpCheckResult]:
    """Run every op check on ``device`` for each dtype it supports."""
    results: list[OpCheckResult] = []
    for check in _check_list():
        inputs = check.make_inputs(torch.Generator().manual_seed(seed))
        try:
            reference = [t.detach().to(_CPU, torch.float32) for t in check.fn(inputs, _CPU, torch.float32)]
        except Exception as exc:  # the CPU reference must work; failure here is a harness problem
            results.append(
                OpCheckResult(
                    check.name, "float32", "error", None, 0.0, None, f"CPU reference failed: {exc!r}"
                )
            )
            continue
        tolerances = {**DEFAULT_TOLERANCE, **(check.tolerance or {})}
        for dtype in dtypes:
            if dtype not in check.dtypes:
                continue
            tol = tolerances[dtype]
            dtype_name = str(dtype).removeprefix("torch.")
            try:
                start = time.perf_counter()
                outputs = check.fn(inputs, device, dtype)
                synchronize(device)
                seconds = time.perf_counter() - start
                err = _max_rel_error(outputs, reference)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"[:600]
                results.append(OpCheckResult(check.name, dtype_name, _classify(exc), None, tol, None, detail))
                continue
            status: Status = "pass" if err is None or err <= tol else "numeric_mismatch"
            results.append(OpCheckResult(check.name, dtype_name, status, err, tol, round(seconds, 4)))
    return results


def descriptions() -> dict[str, str]:
    return {check.name: check.description for check in _check_list()}
