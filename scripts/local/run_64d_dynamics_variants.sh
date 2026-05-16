#!/usr/bin/env bash
set -u -o pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
source .venv/bin/activate
export CELLPATH_ROOT="$(pwd)"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/dynamics64_variants_${STAMP}"
mkdir -p "$LOG_DIR"

ART="$(pwd)/artifacts_64"
OUT_BASE="$ART/dynamics_variants"
mkdir -p "$OUT_BASE"

COMMON=(
  "--config-name" "default"
  "vae.n_latent=64"
  "paths.artifacts=$ART"
  "dynamics.lr=1e-4"
  "dynamics.max_epochs=300"
  "dynamics.early_stop_patience=35"
  "dynamics.selection_metric=gate_margin"
  "dynamics.lambda_mse_delta=0.0"
  "+force=true"
)

run_variant () {
  local name="$1"
  local state_linear="$2"
  local gene_bias="$3"

  echo ""
  echo "============================================================"
  echo "[$(date)] Running variant: $name"
  echo "state_linear=$state_linear | gene_bias=$gene_bias"
  echo "============================================================"

  python scripts/train_dynamics.py "${COMMON[@]}" \
    "paths.dynamics_dir=$OUT_BASE/$name" \
    "dynamics.use_state_linear_skip=$state_linear" \
    "dynamics.use_gene_delta_bias=$gene_bias" \
    2>&1 | tee "$LOG_DIR/${name}.log"

  local code=${PIPESTATUS[0]}
  echo "[$(date)] Variant $name exited with code $code" | tee -a "$LOG_DIR/run.log"
}

run_variant "baseline_plain" "false" "false"
run_variant "state_linear" "true" "false"
run_variant "gene_bias" "false" "true"
run_variant "state_linear_gene_bias" "true" "true"

echo ""
echo "[$(date)] All variants attempted. Logs in $LOG_DIR"
