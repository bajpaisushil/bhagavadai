"""Cross-platform memory measurement for CPU, Apple MPS (unified memory) and CUDA.

On Apple Silicon the GPU shares physical RAM with the CPU, so three numbers matter:

* ``device_allocated``: bytes held by live MPS tensors (``torch.mps.current_allocated_memory``);
* ``device_driver``: bytes the Metal driver has allocated for this process, including the
  allocator's cache (``torch.mps.driver_allocated_memory``) - the true GPU-side footprint;
* ``system_available`` / swap: whether the machine as a whole is under memory pressure.
"""

from __future__ import annotations

import resource
import sys
import threading
from dataclasses import asdict, dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    import torch

GiB = 1024**3


def peak_rss_bytes() -> int:
    """Process high-water mark RSS. ``ru_maxrss`` is bytes on macOS but KiB on Linux."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maxrss if sys.platform == "darwin" else maxrss * 1024)


@dataclass(frozen=True)
class MemorySnapshot:
    process_rss_bytes: int
    system_available_bytes: int
    system_total_bytes: int
    swap_used_bytes: int
    device_allocated_bytes: int | None = None
    device_driver_bytes: int | None = None

    def to_gb(self) -> dict[str, float | None]:
        return {k: (round(v / GiB, 3) if isinstance(v, int) else None) for k, v in asdict(self).items()}


def device_memory(device: torch.device | None) -> tuple[int | None, int | None]:
    """Return ``(allocated, driver_or_reserved)`` bytes for ``device`` (``None`` on CPU)."""
    if device is None or device.type == "cpu":
        return None, None
    import torch

    if device.type == "mps":
        return int(torch.mps.current_allocated_memory()), int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device)), int(torch.cuda.memory_reserved(device))
    return None, None


def take_snapshot(device: torch.device | None = None) -> MemorySnapshot:
    vm = psutil.virtual_memory()
    allocated, driver = device_memory(device)
    return MemorySnapshot(
        process_rss_bytes=int(psutil.Process().memory_info().rss),
        system_available_bytes=int(vm.available),
        system_total_bytes=int(vm.total),
        swap_used_bytes=int(psutil.swap_memory().used),
        device_allocated_bytes=allocated,
        device_driver_bytes=driver,
    )


@dataclass
class MemoryPeaks:
    peak_rss_bytes: int = 0
    min_system_available_bytes: int | None = None
    max_swap_used_bytes: int = 0
    peak_device_allocated_bytes: int | None = None
    peak_device_driver_bytes: int | None = None
    samples: int = 0

    def update_host(self, rss: int, available: int, swap: int) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.min_system_available_bytes = (
            available
            if self.min_system_available_bytes is None
            else min(self.min_system_available_bytes, available)
        )
        self.max_swap_used_bytes = max(self.max_swap_used_bytes, swap)
        self.samples += 1

    def update_device(self, allocated: int | None, driver: int | None) -> None:
        if allocated is not None:
            self.peak_device_allocated_bytes = max(self.peak_device_allocated_bytes or 0, allocated)
        if driver is not None:
            self.peak_device_driver_bytes = max(self.peak_device_driver_bytes or 0, driver)


@dataclass
class MemoryMonitor:
    """Track peak memory over a code block.

    Host metrics (RSS, available RAM, swap) are sampled from a background thread. Device allocator
    statistics are sampled only from the calling thread via :meth:`sample_device` (call it at step
    boundaries); on CUDA the allocator's own peak counter is used as well.
    """

    device: torch.device | None = None
    interval_s: float = 0.05
    baseline: MemorySnapshot | None = None
    peaks: MemoryPeaks = field(default_factory=MemoryPeaks)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _sample_host(self) -> None:
        vm = psutil.virtual_memory()
        rss = int(psutil.Process().memory_info().rss)
        swap = int(psutil.swap_memory().used)
        with self._lock:
            self.peaks.update_host(rss, int(vm.available), swap)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample_host()

    def sample_device(self) -> None:
        allocated, driver = device_memory(self.device)
        if self.device is not None and self.device.type == "cuda":
            import torch

            allocated = int(torch.cuda.max_memory_allocated(self.device))
        with self._lock:
            self.peaks.update_device(allocated, driver)

    def __enter__(self) -> MemoryMonitor:
        if self.device is not None and self.device.type == "cuda":
            import torch

            torch.cuda.reset_peak_memory_stats(self.device)
        self.baseline = take_snapshot(self.device)
        self._sample_host()
        self.sample_device()
        self._thread = threading.Thread(target=self._run, name="memory-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample_host()
        if exc_type is None:
            self.sample_device()

    def summary(self) -> dict[str, Any]:
        """Peaks and deltas versus the baseline, in GiB."""
        base = self.baseline

        def gb(value: int | None) -> float | None:
            return None if value is None else round(value / GiB, 3)

        result: dict[str, Any] = {
            "peak_process_rss_gb": gb(self.peaks.peak_rss_bytes),
            "min_system_available_gb": gb(self.peaks.min_system_available_bytes),
            "peak_device_allocated_gb": gb(self.peaks.peak_device_allocated_bytes),
            "peak_device_driver_gb": gb(self.peaks.peak_device_driver_bytes),
            "samples": self.peaks.samples,
        }
        if base is not None:
            result["baseline"] = base.to_gb()
            result["swap_growth_gb"] = gb(max(0, self.peaks.max_swap_used_bytes - base.swap_used_bytes))
            if self.peaks.min_system_available_bytes is not None:
                result["system_memory_consumed_gb"] = gb(
                    max(0, base.system_available_bytes - self.peaks.min_system_available_bytes)
                )
        return result
