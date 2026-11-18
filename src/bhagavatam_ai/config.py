"""Typed, validated configuration loaded from YAML files under ``configs/``.

Every YAML file holds one top-level section (``model:``, ``hardware_check:``, ``data:``). Unknown keys
are rejected (``extra="forbid"``) so that a typo in a config file fails loudly instead of silently
falling back to a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DTypeName = Literal["float32", "bfloat16", "float16"]
DevicePreference = Literal["auto", "mps", "cuda", "cpu"]
LossMode = Literal["standard", "chunked", "fused"]


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing or invalid."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- model


class DTypePolicy(StrictModel):
    """Compute dtype per device type."""

    mps: DTypeName = "bfloat16"
    cuda: DTypeName = "bfloat16"
    cpu: DTypeName = "float32"


class ModelExpectations(StrictModel):
    """Facts about the checkpoint that loading must confirm; any mismatch is an error."""

    model_class: str
    num_hidden_layers: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    tie_word_embeddings: bool
    # Checkpoint keys we deliberately do not load into the text-only model (vision tower, MTP head).
    allowed_unexpected_key_patterns: tuple[str, ...] = ()
    # The checkpoint stores the text decoder under this prefix; the text-only model uses "model.".
    checkpoint_key_prefix: str = "model."
    # Tensors compared byte-for-byte between the loaded model and the checkpoint file.
    integrity_spot_check_keys: tuple[str, ...] = ()


class ModelConfig(StrictModel):
    name_or_path: str
    # A full 40-character commit hash: branch or tag names can move, so they are rejected.
    revision: str = Field(pattern=r"^[0-9a-f]{40}$", description="Pinned commit hash for reproducibility.")
    attn_implementation: Literal["sdpa", "eager"] = "sdpa"
    dtype: DTypePolicy = DTypePolicy()
    expected: ModelExpectations
    lora_target_modules: tuple[str, ...] = Field(min_length=1)


# --------------------------------------------------------------------------- hardware check


class OpChecksConfig(StrictModel):
    enabled: bool = True
    dtypes: tuple[DTypeName, ...] = ("float32", "bfloat16", "float16")


class NumericsConfig(StrictModel):
    compare_with_cpu_reference: bool = True
    max_abs_loss_diff: float = Field(default=0.08, gt=0)
    min_top1_agreement: float = Field(default=0.85, ge=0, le=1)


class GenerationCheckConfig(StrictModel):
    max_new_tokens: int = Field(default=32, gt=0)


class LoraSettings(StrictModel):
    r: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.0, ge=0, lt=1)


class TrainingCheckConfig(StrictModel):
    enabled: bool = True
    seq_lens: tuple[Annotated[int, Field(ge=2)], ...] = Field(default=(256, 512, 1024, 2048), min_length=1)
    batch_size: int = Field(default=1, gt=0)
    warmup_steps: int = Field(default=1, ge=0)
    measured_steps: int = Field(default=3, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    gradient_checkpointing: bool = True
    loss: LossMode = "chunked"
    loss_chunk_size: int = Field(default=1024, gt=0)
    # After the sequence-length sweep, also time these loss implementations at the longest sequence
    # length that fit, to pick the fastest one that fits in memory.
    compare_loss_modes: tuple[LossMode, ...] = ()
    # Optional sustained run (minutes) at the longest fitting sequence length, to measure thermal
    # throttling on fanless machines. 0 disables it.
    sustained_minutes: float = Field(default=0.0, ge=0)
    sanity_overfit_steps: int = Field(default=6, ge=0)
    lora: LoraSettings = LoraSettings()


class HardwareCheckConfig(StrictModel):
    device: DevicePreference = "auto"
    seed: int = 1234
    # Cap MPS allocations at this fraction of Metal's recommendedMaxWorkingSetSize so that running out
    # of memory raises an error instead of pushing a 16 GB machine into heavy swap. None disables it.
    mps_memory_fraction: float | None = Field(default=1.0, gt=0, le=2)
    # Treat a run as "under memory pressure" beyond these limits (unified memory machines swap silently).
    max_swap_growth_gb: float = Field(default=0.5, ge=0)
    min_system_available_gb: float = Field(default=1.0, ge=0)
    op_checks: OpChecksConfig = OpChecksConfig()
    numerics: NumericsConfig = NumericsConfig()
    generation: GenerationCheckConfig = GenerationCheckConfig()
    training: TrainingCheckConfig = TrainingCheckConfig()


# --------------------------------------------------------------------------- data


class DownloadConfig(StrictModel):
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=4, ge=0)
    backoff_s: float = Field(default=2.0, ge=0)
    polite_delay_s: float = Field(default=1.5, ge=0)
    respect_robots_txt: bool = True
    user_agent: str = "bhagavatam-ai-data-pipeline/0.1 (research; non-commercial)"
    max_bytes: int = Field(default=500 * 1024 * 1024, gt=0)


class SplitConfig(StrictModel):
    seed: int = 20261118
    val_fraction: float = Field(default=0.05, ge=0, lt=0.5)
    # Chapters forced into validation regardless of sampling, e.g. ["3.25", "11.7"].
    forced_val_chapters: tuple[str, ...] = ()


class DedupConfig(StrictModel):
    shingle_size: int = Field(default=5, gt=0)
    num_perm: int = Field(default=64, gt=0)
    bands: int = Field(default=16, gt=0)
    near_duplicate_threshold: float = Field(default=0.85, gt=0, le=1)

    @model_validator(mode="after")
    def _bands_divide_perms(self) -> Self:
        if self.num_perm % self.bands:
            raise ValueError(f"num_perm ({self.num_perm}) must be divisible by bands ({self.bands})")
        return self


class ValidationConfig(StrictModel):
    fail_on_error: bool = True
    min_chars_per_verse: int = Field(default=8, ge=0)


class DataConfig(StrictModel):
    manifest: Path = Path("data/manifests/sources.yaml")
    expected_structure: Path = Path("data/reference/expected_structure.json")
    verse_equivalences: Path = Path("data/reference/verse_equivalences.json")
    # Added to the always-blocked BBT domains (bhagavatam_ai.data.manifest.ALWAYS_BLOCKED).
    blocked_domains: tuple[str, ...] = ()

    @field_validator("blocked_domains")
    @classmethod
    def _normalise_domains(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(d for d in (v.strip().lower().strip(".") for v in value) if d)

    download: DownloadConfig = DownloadConfig()
    split: SplitConfig = SplitConfig()
    dedup: DedupConfig = DedupConfig()
    validation: ValidationConfig = ValidationConfig()


# --------------------------------------------------------------------------- loading

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return loaded


def load_section(path: Path, section: str, model_cls: type[ConfigT]) -> ConfigT:
    """Load ``section`` from the YAML file at ``path`` and validate it as ``model_cls``."""
    document = load_yaml(path)
    if section not in document:
        raise ConfigError(f"{path} has no top-level '{section}' section")
    try:
        return model_cls.model_validate(document[section])
    except ValidationError as exc:
        raise ConfigError(f"Invalid '{section}' config in {path}:\n{exc}") from exc


def load_model_config(path: Path) -> ModelConfig:
    return load_section(path, "model", ModelConfig)


def load_hardware_config(path: Path) -> HardwareCheckConfig:
    return load_section(path, "hardware_check", HardwareCheckConfig)


def load_data_config(path: Path) -> DataConfig:
    return load_section(path, "data", DataConfig)
