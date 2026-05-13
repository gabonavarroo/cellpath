# CellPath — convenience entry points.
# Targets are documented inline; run `make help` for the list.

PYTHON ?= python
UV ?= uv
DOCKER ?= docker

.DEFAULT_GOAL := help

.PHONY: help setup data vae pairs dynamics rl evaluate pipeline test lint format \
        docker-build docker-cpu docker-cuda tensorboard clean nuke notebooks

help:  ## Show this help message.
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | awk -F':.*?## ' '{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

setup:  ## Create .venv with uv and install all deps in editable mode.
	$(UV) venv .venv --python 3.11
	. .venv/bin/activate && $(UV) pip install -e ".[dev]"
	. .venv/bin/activate && pre-commit install
	@echo ""
	@echo "  Done. Activate with:  source .venv/bin/activate"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

data:  ## Download Norman 2019 (pertpy primary, GEO fallback) + DepMap.
	bash scripts/download_data.sh

# ---------------------------------------------------------------------------
# Training entry points (each accepts CONFIG=name; e.g. `make vae CONFIG=baseline`)
# ---------------------------------------------------------------------------

CONFIG ?= default

vae:  ## Train scVI VAE.
	$(PYTHON) scripts/train_vae.py --config-name $(CONFIG)

pairs:  ## Build OT pseudo-pairs (train/val/ood/combo). Run after make vae.
	$(PYTHON) scripts/build_pairs.py --config-name $(CONFIG)

dynamics:  ## Train residual MLP dynamics model.
	$(PYTHON) scripts/train_dynamics.py --config-name $(CONFIG)

rl:  ## Train MaskablePPO RL agent (refuses unless dynamics gate passed).
	$(PYTHON) scripts/train_rl.py --config-name $(CONFIG)

evaluate:  ## Run full evaluation suite (DepMap enrichment, latent metrics, trajectories).
	$(PYTHON) scripts/evaluate.py --config-name $(CONFIG)

pipeline:  ## End-to-end pipeline: data → vae → dynamics → rl → evaluate.
	$(PYTHON) -m src.pipeline run --config-name $(CONFIG)

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

test:  ## Run pytest (mock data only, fast).
	pytest tests/ -v --no-cov -k "not slow"

test-all:  ## Run pytest including slow integration tests.
	pytest tests/ -v -m "slow or not slow"

lint:  ## Run ruff lint.
	ruff check src tests scripts

format:  ## Run ruff format.
	ruff format src tests scripts

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build: docker-cpu docker-cuda  ## Build both Docker images.

docker-cpu:  ## Build CPU image for CI / smoke.
	$(DOCKER) build -f Dockerfile.cpu -t cellpath:cpu .

docker-cuda:  ## Build CUDA image for cluster.
	$(DOCKER) build -f Dockerfile.cuda -t cellpath:cuda .

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

tensorboard:  ## Launch TensorBoard on artifacts/.
	tensorboard --logdir artifacts --host 0.0.0.0 --port 6006

notebooks:  ## Launch Jupyter Lab.
	jupyter lab notebooks/

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:  ## Remove __pycache__ and .pytest_cache.
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -exec rm -rf {} +

nuke: clean  ## Remove artifacts/ and .venv (dangerous, asks first).
	@read -p "  This deletes artifacts/ and .venv. Confirm [y/N]: " ans; \
	if [ "$$ans" = "y" ]; then rm -rf artifacts .venv; echo "  Nuked."; else echo "  Aborted."; fi
