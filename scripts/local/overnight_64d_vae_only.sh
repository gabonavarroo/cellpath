#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
source .venv/bin/activate
export CELLPATH_ROOT="$(pwd)"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/overnight_64d_vae_${STAMP}"
mkdir -p "$LOG_DIR"

echo "[$(date)] Starting isolated 64D VAE training" | tee "$LOG_DIR/run.log"
echo "Artifacts: $(pwd)/artifacts_64" | tee -a "$LOG_DIR/run.log"

python scripts/train_vae.py --config-name default \
  vae.n_latent=64 \
  paths.artifacts="$(pwd)/artifacts_64" \
  2>&1 | tee "$LOG_DIR/01_vae64.log"

echo "[$(date)] DONE 64D VAE" | tee -a "$LOG_DIR/run.log"
