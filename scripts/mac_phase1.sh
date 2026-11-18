#!/usr/bin/env bash
# Phase 1 checks on the Apple Silicon training machine (MacBook Air M5, 16 GB).
#
#   scripts/mac_phase1.sh                       # full check (~15-30 min)
#   scripts/mac_phase1.sh --sustained-minutes 20 # add a thermal-throttling measurement
#
# Close other heavy apps first: macOS itself uses several GB of the 16 GB.
set -euo pipefail
cd "$(dirname "$0")/.."

[[ "$(uname -s)" == "Darwin" ]] || { echo "error: run this on the Mac" >&2; exit 1; }
[[ "$(uname -m)" == "arm64" ]] || { echo "error: shell is not arm64 (Rosetta?); MPS needs native arm64" >&2; exit 1; }
command -v uv >/dev/null || { echo "error: install uv first: brew install uv" >&2; exit 1; }

uv sync --frozen --extra ml --extra dev
uv run --frozen python - <<'PY'
import platform, torch
assert platform.machine() == "arm64", f"Python is {platform.machine()}, need native arm64"
assert torch.backends.mps.is_available(), "MPS is not available to PyTorch"
print(f"torch {torch.__version__}, MPS available, recommended max memory "
      f"{torch.mps.recommended_max_memory() / 2**30:.1f} GiB")
PY

# Unsupported MPS ops must fail loudly during the check, not silently fall back to the CPU.
unset PYTORCH_ENABLE_MPS_FALLBACK

run() { uv run --frozen bhagavatam "$@"; }
run env
run model download
run model inspect
caffeinate -dimsu uv run --frozen bhagavatam hardware check --strict "$@" || status=$?
run model smoke
run data download
run data build

echo
echo "Phase 1 checks finished. Key report: reports/hardware_check/latest.json"
echo "ready_for_phase2 = $(uv run --frozen python -c 'import json;print(json.load(open("reports/hardware_check/latest.json"))["verdict"]["ready_for_phase2"])')"
exit "${status:-0}"
