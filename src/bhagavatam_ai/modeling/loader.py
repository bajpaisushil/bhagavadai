"""Download and load the Qwen3.5 text decoder with integrity verification.

``Qwen/Qwen3.5-0.8B-Base`` is a vision-language checkpoint (``Qwen3_5ForConditionalGeneration``).
For text-only training we load ``Qwen3_5ForCausalLM`` through ``AutoModelForCausalLM``: transformers
maps checkpoint keys ``model.language_model.*`` onto the text decoder and skips the vision tower
(``model.visual.*``) and multi-token-prediction head (``mtp.*``).

A silent key-mapping failure would leave layers randomly initialised while still producing output,
so loading here is strict: missing or unexpected keys are errors, and a sample of tensors is compared
byte-for-byte with the checkpoint on disk.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bhagavatam_ai.config import ModelConfig, ModelExpectations
from bhagavatam_ai.utils.hashing import sha256_file

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class ModelIntegrityError(RuntimeError):
    """Loaded weights do not match what the configuration expects."""


@dataclass
class LoadReport:
    model_class: str
    revision: str
    dtype: str
    device: str
    load_seconds: float
    num_parameters: int
    num_embedding_parameters: int
    tied_embeddings: bool
    missing_keys: list[str]
    unexpected_keys: list[str]
    mismatched_keys: list[str]
    spot_checks: dict[str, bool]
    layer_types: dict[str, int]
    fast_path_kernels: dict[str, bool]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fast_path_kernels() -> dict[str, bool]:
    """Whether the CUDA-only kernels that accelerate Gated DeltaNet are importable."""
    return {name: importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d")}


def snapshot_dir(cfg: ModelConfig, *, local_files_only: bool = False) -> Path:
    """Return the local snapshot directory for the pinned revision, downloading it if needed."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(cfg.name_or_path, revision=cfg.revision, local_files_only=local_files_only))


def download_and_verify(cfg: ModelConfig) -> dict[str, Any]:
    """Download the pinned snapshot and verify each LFS file's SHA-256 against the Hub metadata."""
    from huggingface_hub import HfApi

    started = time.perf_counter()
    path = snapshot_dir(cfg)
    info = HfApi().model_info(cfg.name_or_path, revision=cfg.revision, files_metadata=True)
    files: list[dict[str, Any]] = []
    for sibling in info.siblings or []:
        local = path / sibling.rfilename
        entry: dict[str, Any] = {"file": sibling.rfilename, "size": sibling.size, "present": local.exists()}
        expected_sha = sibling.lfs.sha256 if sibling.lfs is not None else None
        if expected_sha and local.exists():
            entry["sha256_expected"] = expected_sha
            entry["sha256_ok"] = sha256_file(local) == expected_sha
        files.append(entry)
    failed = [f["file"] for f in files if not f["present"] or f.get("sha256_ok") is False]
    if failed:
        raise ModelIntegrityError(f"Snapshot verification failed for: {failed}")
    total = sum(int(f["size"] or 0) for f in files)
    return {
        "model": cfg.name_or_path,
        "revision": cfg.revision,
        "resolved_commit": info.sha,
        "files": files,
        "total_size_gb": round(total / 1024**3, 3),
        "seconds": round(time.perf_counter() - started, 1),
        # Only the cache-relative location: absolute paths can contain the user name.
        "cache_relative_path": str(Path(*path.parts[-4:])),
    }


def load_tokenizer(cfg: ModelConfig, *, snapshot: Path | None = None) -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer

    source = snapshot or snapshot_dir(cfg)
    return AutoTokenizer.from_pretrained(str(source))


def _checkpoint_files(snapshot: Path) -> dict[str, Path]:
    """Map checkpoint tensor name -> safetensors file containing it."""
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        weight_map: dict[str, str] = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return {key: snapshot / filename for key, filename in weight_map.items()}
    single = snapshot / "model.safetensors"
    if not single.exists():
        raise ModelIntegrityError(f"No safetensors weights found in {snapshot}")
    from safetensors import safe_open

    with safe_open(str(single), "pt") as handle:
        return dict.fromkeys(handle.keys(), single)


def checkpoint_key(model_key: str, expected: ModelExpectations) -> str:
    """``model.layers.0...`` (text-only model) -> ``model.language_model.layers.0...`` (checkpoint)."""
    if not model_key.startswith("model."):
        raise ValueError(f"unexpected model key {model_key}")
    return expected.checkpoint_key_prefix + model_key.removeprefix("model.")


def verify_loading_info(info: dict[str, Any], expected: ModelExpectations) -> list[str]:
    """Raise on missing/mismatched keys or unexpected keys outside the allow-list."""
    missing = sorted(str(k) for k in info.get("missing_keys", ()))
    mismatched = sorted(str(k) for k in info.get("mismatched_keys", ()))
    unexpected = sorted(str(k) for k in info.get("unexpected_keys", ()))
    allowed = [re.compile(p) for p in expected.allowed_unexpected_key_patterns]
    disallowed = [k for k in unexpected if not any(p.search(k) for p in allowed)]
    problems = []
    if missing:
        problems.append(f"{len(missing)} missing keys (randomly initialised!): {missing[:8]}")
    if mismatched:
        problems.append(f"{len(mismatched)} shape-mismatched keys: {mismatched[:8]}")
    if disallowed:
        problems.append(f"{len(disallowed)} unexpected keys: {disallowed[:8]}")
    if problems:
        raise ModelIntegrityError("; ".join(problems))
    return unexpected


def verify_architecture(model: PreTrainedModel, expected: ModelExpectations) -> dict[str, int]:
    config = model.config
    checks = {
        "model_class": (type(model).__name__, expected.model_class),
        "num_hidden_layers": (config.num_hidden_layers, expected.num_hidden_layers),
        "vocab_size": (config.vocab_size, expected.vocab_size),
        "tie_word_embeddings": (
            bool(getattr(config, "tie_word_embeddings", False)),
            expected.tie_word_embeddings,
        ),
    }
    wrong = {name: f"got {got!r}, expected {want!r}" for name, (got, want) in checks.items() if got != want}
    if wrong:
        raise ModelIntegrityError(f"Architecture mismatch: {wrong}")
    return dict(Counter(getattr(config, "layer_types", None) or []))


def spot_check_weights(
    model: PreTrainedModel, snapshot: Path, expected: ModelExpectations
) -> dict[str, bool]:
    """Compare selected loaded tensors with the checkpoint, exactly (after casting to the model dtype)."""
    import torch
    from safetensors import safe_open

    if not expected.integrity_spot_check_keys:
        return {}
    files = _checkpoint_files(snapshot)
    state = model.state_dict()
    results: dict[str, bool] = {}
    for key in expected.integrity_spot_check_keys:
        ckpt_key = checkpoint_key(key, expected)
        if key not in state or ckpt_key not in files:
            results[key] = False
            continue
        with safe_open(str(files[ckpt_key]), "pt") as handle:
            reference = handle.get_tensor(ckpt_key)
        loaded = state[key].detach().cpu()
        results[key] = bool(torch.equal(loaded, reference.to(loaded.dtype)))
    if not all(results.values()):
        raise ModelIntegrityError(f"Loaded weights differ from the checkpoint: {results}")
    return results


def load_model(
    cfg: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    snapshot: Path | None = None,
    verify: bool = True,
) -> tuple[PreTrainedModel, LoadReport]:
    """Load the text decoder, verify it, move it to ``device`` and set eval mode."""
    from transformers import AutoModelForCausalLM

    source = snapshot or snapshot_dir(cfg)
    started = time.perf_counter()
    model, info = AutoModelForCausalLM.from_pretrained(
        str(source),
        dtype=dtype,
        attn_implementation=cfg.attn_implementation,
        output_loading_info=True,
    )
    unexpected = verify_loading_info(info, cfg.expected)
    layer_types = verify_architecture(model, cfg.expected)
    spot = spot_check_weights(model, source, cfg.expected) if verify else {}
    model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - started

    embed = model.get_input_embeddings().weight
    tied = embed.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    if cfg.expected.tie_word_embeddings and not tied:
        raise ModelIntegrityError("lm_head is not tied to the input embeddings after loading")
    kernels = fast_path_kernels()
    warnings = []
    if not all(kernels.values()):
        warnings.append(
            "Gated DeltaNet runs on the reference PyTorch path (flash-linear-attention / causal-conv1d "
            "are CUDA-only and not installed): correct but slower."
        )
    report = LoadReport(
        model_class=type(model).__name__,
        revision=cfg.revision,
        dtype=str(dtype).removeprefix("torch."),
        device=str(device),
        load_seconds=round(load_seconds, 2),
        num_parameters=sum(p.numel() for p in model.parameters()),
        num_embedding_parameters=embed.numel(),
        tied_embeddings=tied,
        missing_keys=[],
        unexpected_keys=unexpected,
        mismatched_keys=[],
        spot_checks=spot,
        layer_types=layer_types,
        fast_path_kernels=kernels,
        warnings=warnings,
    )
    logger.info(
        "Loaded %s (%.1fM params) on %s as %s in %.1fs",
        report.model_class,
        report.num_parameters / 1e6,
        device,
        report.dtype,
        load_seconds,
    )
    return model, report
