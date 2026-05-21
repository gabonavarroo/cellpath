# CellPath — convenience entry points.
# Targets are documented inline; run `make help` for the list.

PYTHON ?= python
UV ?= uv
DOCKER ?= docker

.DEFAULT_GOAL := help

.PHONY: help setup data vae pairs dynamics rl evaluate pipeline test lint format \
        docker-build docker-cpu docker-cuda tensorboard clean nuke notebooks \
        rl-eval rl-random rl-summary aggregate visualize depmap-compare \
        final-v3c final-v3c-eval final-v3c-demo final-v3c-baseline final-v3c-figures final-v3c-audit

# ---------------------------------------------------------------------------
# V3C final pipeline (champion-default — cheap eval, NOT retraining)
# ---------------------------------------------------------------------------

final-v3c-eval:  ## Re-run champion 7-cell evaluation (cheap, ~10 min).
	$(PYTHON) scripts/run_final_v3c_pipeline.py --mode eval

final-v3c-demo:  ## Fast 1-cell champion demo (~2 min).
	$(PYTHON) scripts/run_final_v3c_pipeline.py --mode demo --n-episodes 50

final-v3c-baseline:  ## Re-run V2 anchor baseline for side-by-side.
	$(PYTHON) scripts/run_final_v3c_pipeline.py --mode baseline

final-v3c-figures:  ## Regenerate presentation figures from existing aggregator outputs.
	$(PYTHON) scripts/run_final_v3c_pipeline.py --mode figures

final-v3c-audit:  ## Re-run V3C utility audit on champion dynamics field.
	$(PYTHON) scripts/run_final_v3c_pipeline.py --mode audit

final-v3c: final-v3c-eval final-v3c-figures  ## Default: champion eval + figures.

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

pipeline:  ## V2 primary pipeline: data → vae → pairs → dynamics → rl → evaluate (default composition).
	$(PYTHON) -m src.pipeline run --config-name $(CONFIG)

pipeline-final:  ## FINAL MODEL pipeline (V3C champion): composes config/experiments/final.yaml.
	$(PYTHON) -m src.pipeline run --config-name experiments/final

eval-final:  ## Evaluate the V3C champion only (no retraining; reuses on-disk checkpoints).
	$(PYTHON) -m src.pipeline run --config-name experiments/final --from evaluate

# ---------------------------------------------------------------------------
# RL evaluation utilities (need EXTRA= for paths / overrides).
# Each writes metadata.json + rollouts.parquet + action_freq.json into its out_dir.
# ---------------------------------------------------------------------------

EXTRA ?=

rl-eval:  ## Evaluate an existing PPO checkpoint (det + stoch). Pass EXTRA="+eval_rl.ppo_path=... +eval_rl.out_dir=...".
	$(PYTHON) scripts/evaluate_rl.py --config-name $(CONFIG) rl.train.skip_gate=true $(EXTRA)

rl-random:  ## Run a random-policy baseline matched to PPO eval env. EXTRA= must include +random_policy.out_dir=...
	$(PYTHON) scripts/run_random_policy.py --config-name $(CONFIG) rl.train.skip_gate=true $(EXTRA)

rl-summary:  ## Summarize an RL run dir. Usage: make rl-summary RUN_DIR=path/to/eval_deterministic [RAND_DIR=...].
	@if [ -z "$(RUN_DIR)" ]; then echo "ERROR: set RUN_DIR=<path>"; exit 2; fi
	$(PYTHON) scripts/summarize_rl_run.py --run-dir $(RUN_DIR) $(if $(RAND_DIR),--random-baseline-dir $(RAND_DIR))

# ---------------------------------------------------------------------------
# Phase 5 — aggregation, visualization, full evaluate.
# ---------------------------------------------------------------------------

aggregate:  ## Compose artifacts/eval/{summary,results_table,caveats} from existing runs.
	$(PYTHON) scripts/aggregate_eval.py --config-name $(CONFIG) rl.train.skip_gate=true $(EXTRA)

visualize:  ## Render every defense figure under artifacts/eval/figures/.
	$(PYTHON) scripts/visualize.py --config-name $(CONFIG) rl.train.skip_gate=true $(EXTRA)

depmap-compare:  ## Run DepMap gene-score comparison (PPO vs random vs action universe).
	$(PYTHON) scripts/evaluate.py --config-name $(CONFIG) rl.train.skip_gate=true \
	    +evaluate.skip_aggregate=true +evaluate.skip_latent_quality=true \
	    +evaluate.skip_depmap=true $(EXTRA)

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
