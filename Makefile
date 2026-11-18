# Common tasks. Requires uv (https://docs.astral.sh/uv/): `brew install uv` on macOS.
.DEFAULT_GOAL := help
SHELL := /bin/bash
UV ?= uv
RUN := $(UV) run --frozen
BHAGAVATAM := $(RUN) bhagavatam
# Keep a fanless Mac from sleeping or down-clocking (App Nap) during long GPU runs.
CAFFEINATE := $(shell command -v caffeinate >/dev/null 2>&1 && echo caffeinate -dimsu)

.PHONY: help setup lock lint format typecheck test test-model check env model-download model-inspect \
        model-smoke hardware-ops hardware-check hardware-check-sustained data-download data-build \
        data-validate phase1 clean

help: ## List targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}'

setup: ## Install locked dependencies (ml + dev extras) and git hooks
	$(UV) sync --frozen --extra ml --extra dev
	$(RUN) pre-commit install

lock: ## Re-resolve uv.lock after editing dependencies
	$(UV) lock

lint: ## Ruff lint + format check
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

format: ## Apply ruff fixes and formatting
	$(RUN) ruff check --fix src tests
	$(RUN) ruff format src tests

typecheck: ## mypy --strict
	$(RUN) mypy

test: ## Unit tests (no model download)
	$(RUN) pytest --cov

test-model: ## Integration tests against the real model (~1.7 GB download, ~5 GB RAM)
	$(RUN) pytest tests/integration --run-model

check: lint typecheck test ## Everything CI runs

env: ## Environment and hardware report
	$(BHAGAVATAM) env

model-download: ## Download the pinned model snapshot and verify checksums
	$(BHAGAVATAM) model download

model-inspect: ## Tokenizer / chat template / architecture report
	$(BHAGAVATAM) model inspect

model-smoke: ## Small end-to-end inference test of the base model
	$(BHAGAVATAM) model smoke

hardware-ops: ## Fast op-level backend compatibility checks
	$(BHAGAVATAM) hardware ops

hardware-check: ## Full hardware check (MPS ops, numerics, generation, LoRA memory/speed)
	$(CAFFEINATE) $(BHAGAVATAM) hardware check

hardware-check-sustained: ## Hardware check + 20-minute sustained run (thermal throttling)
	$(CAFFEINATE) $(BHAGAVATAM) hardware check --sustained-minutes 20

data-download: ## Download raw files for sources whose licence permits it
	$(BHAGAVATAM) data download

data-build: ## Parse, validate and build the (licence-gated) corpus
	$(BHAGAVATAM) data build

data-validate: ## Re-run validation on parsed data
	$(BHAGAVATAM) data validate

phase1: env model-download model-inspect hardware-check model-smoke data-download data-build ## Full Phase 1 sequence

clean: ## Remove caches (keeps data, reports, checkpoints)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
