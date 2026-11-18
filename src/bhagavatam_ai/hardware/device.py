"""Device selection, dtype policy and environment introspection (Apple MPS first)."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Final

import psutil

from bhagavatam_ai.config import DevicePreference, DTypeName, DTypePolicy

if TYPE_CHECKING:
    import torch

_PACKAGES: Final = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "huggingface-hub",
    "tokenizers",
    "numpy",
)
# Reported by presence only: values may contain local paths that include the user name.
_ENV_VARS_PRESENCE: Final = ("HF_HOME", "HF_TOKEN", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
_ENV_VARS_VALUE: Final = (
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
    "OMP_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
)


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested device is not usable."""


def select_device(preference: DevicePreference = "auto") -> torch.device:
    """Pick the compute device. ``auto`` prefers MPS, then CUDA, then CPU."""
    import torch

    mps_ok = torch.backends.mps.is_available()
    cuda_ok = torch.cuda.is_available()
    if preference == "auto":
        if mps_ok:
            return torch.device("mps")
        return torch.device("cuda") if cuda_ok else torch.device("cpu")
    if preference == "mps" and not mps_ok:
        raise DeviceUnavailableError(
            f"MPS requested but unavailable (built={torch.backends.mps.is_built()}). "
            "MPS needs Apple Silicon, macOS 12.3+ and an arm64 (not Rosetta) Python."
        )
    if preference == "cuda" and not cuda_ok:
        raise DeviceUnavailableError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(preference)


def dtype_from_name(name: DTypeName) -> torch.dtype:
    import torch

    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"Unknown dtype {name}")
    return dtype


def dtype_for_device(device: torch.device, policy: DTypePolicy) -> torch.dtype:
    name: DTypeName = getattr(policy, device.type, "float32")
    return dtype_from_name(name)


def synchronize(device: torch.device) -> None:
    """Block until queued kernels finish, so wall-clock timings are honest."""
    import torch

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def empty_cache(device: torch.device) -> None:
    import torch

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def is_out_of_memory(exc: BaseException) -> bool:
    """True for allocator OOM errors on any backend (MPS raises a plain RuntimeError)."""
    message = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in message


def _run(cmd: list[str], timeout: float = 20.0) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _apple_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"chip": _run(["sysctl", "-n", "machdep.cpu.brand_string"])}
    raw = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=30.0)
    if raw:
        try:
            displays = json.loads(raw).get("SPDisplaysDataType", [])
        except json.JSONDecodeError:
            displays = []
        if displays:
            info["gpu_model"] = displays[0].get("sppci_model")
            info["gpu_cores"] = displays[0].get("sppci_cores")
    perf = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
    eff = _run(["sysctl", "-n", "hw.perflevel1.physicalcpu"])
    info["performance_cores"] = int(perf) if perf and perf.isdigit() else None
    info["efficiency_cores"] = int(eff) if eff and eff.isdigit() else None
    info["low_power_mode"] = _pmset_value(_run(["pmset", "-g"]), "lowpowermode")
    info["on_ac_power"] = _on_ac_power(_run(["pmset", "-g", "batt"]))
    return info


def _pmset_value(output: str | None, key: str) -> bool | None:
    """Parse a boolean ``key  value`` line from ``pmset -g`` output."""
    if not output:
        return None
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() == key.lower():
            return parts[1] == "1"
    return None


def _on_ac_power(output: str | None) -> bool | None:
    if not output:
        return None
    return "AC Power" in output.splitlines()[0] if output.splitlines() else None


def thermal_state() -> dict[str, Any] | None:
    """macOS thermal limits (``pmset -g therm``); ``CPU_Speed_Limit < 100`` means throttling."""
    if platform.system() != "Darwin":
        return None
    output = _run(["pmset", "-g", "therm"])
    if output is None:
        return None
    state: dict[str, Any] = {"raw": output[-500:]}
    for line in output.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            value = value.strip()
            state[key.strip()] = int(value) if value.isdigit() else value
    return state


def _linux_cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:  # noqa: PTH123 - procfs
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_environment() -> dict[str, Any]:
    """Describe OS, hardware and software. Deliberately excludes hostname, user name and paths."""
    vm = psutil.virtual_memory()
    env: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "macos_version": platform.mac_ver()[0] or None,
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_is_arm64_native": platform.machine() == "arm64" if platform.system() == "Darwin" else None,
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": round(vm.total / 1024**3, 2),
        "memory_available_gb": round(vm.available / 1024**3, 2),
        "packages": package_versions(),
        "env_vars": {name: os.environ.get(name) for name in _ENV_VARS_VALUE},
        "env_vars_set": {name: name in os.environ for name in _ENV_VARS_PRESENCE},
    }
    if platform.system() == "Darwin":
        env["apple"] = _apple_hardware()
    else:
        env["cpu_model"] = _linux_cpu_model()

    try:
        import torch
    except ImportError:
        env["torch"] = None
        return env
    torch_info: dict[str, Any] = {
        "version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.backends.mps.is_available():
        torch_info["mps_recommended_max_memory_gb"] = round(torch.mps.recommended_max_memory() / 1024**3, 2)
    if torch.cuda.is_available():
        torch_info["cuda_device"] = torch.cuda.get_device_name(0)
    env["torch"] = torch_info
    return env
