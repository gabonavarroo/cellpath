#!/usr/bin/env bash
# Proposal 1 driver — pairing-noise sweep + RoR dynamics on V1 32D latents.
# All outputs land under artifacts_proposal1/; nothing else is touched.
# Usage: source .venv/bin/activate && bash artifacts_proposal1/run_proposal1.sh STEP
#   STEP ∈ {pairs, noise, dynamics, all}
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

ROOT=artifacts_proposal1
STEP="${1:-all}"

build_pairs() {
    local name="$1" method="$2" eps="$3"
    echo "[build_pairs] name=$name method=$method ot_epsilon=$eps"
    python scripts/build_pairs.py --config-name default \
        paths.pairs_dir=$ROOT/pairs_$name \
        pairing.method=$method \
        pairing.ot_epsilon=$eps \
        +dry_run=false 2>&1 | tail -5
}

noise() {
    local name="$1"
    echo "[noise] $name"
    python scripts/diagnose_pairing_noise.py \
        --pairs_dir $ROOT/pairs_$name \
        --out $ROOT/diagnostics/noise_$name.json 2>&1 | tail -10
}

train_dyn() {
    local name="$1"
    echo "[dynamics] pairs=$name"
    python scripts/train_dynamics.py --config-name default \
        paths.pairs_dir=$ROOT/pairs_$name \
        paths.dynamics_dir=$ROOT/dynamics_${name}_ror \
        dynamics.use_residual_over_ridge=true \
        dynamics.use_state_linear_skip=false \
        dynamics.lambda_corr=0.10 \
        +force=true +dry_run=false 2>&1 | tail -10
}

if [[ "$STEP" == "pairs" || "$STEP" == "all" ]]; then
    build_pairs ot_eps001       ot      0.01
    build_pairs ot_eps002       ot      0.02
    build_pairs soft_ot_eps001  soft_ot 0.01
    build_pairs soft_ot_eps005  soft_ot 0.05
fi

if [[ "$STEP" == "noise" || "$STEP" == "all" ]]; then
    noise ot_eps001
    noise ot_eps002
    noise soft_ot_eps001
    noise soft_ot_eps005
fi

if [[ "$STEP" == "dynamics" || "$STEP" == "all" ]]; then
    # Train RoR+corr0.10 on the four new pair sets
    train_dyn ot_eps001
    train_dyn ot_eps002
    train_dyn soft_ot_eps001
    train_dyn soft_ot_eps005
    # Also train RoR on the V1 pairs as the apples-to-apples baseline
    echo "[dynamics] V1 OT eps=0.05 (baseline)"
    python scripts/train_dynamics.py --config-name default \
        paths.pairs_dir=artifacts/pairs \
        paths.dynamics_dir=$ROOT/dynamics_v1ot_ror \
        dynamics.use_residual_over_ridge=true \
        dynamics.use_state_linear_skip=false \
        dynamics.lambda_corr=0.10 \
        +force=true +dry_run=false 2>&1 | tail -10
fi

echo "[done] step=$STEP"
