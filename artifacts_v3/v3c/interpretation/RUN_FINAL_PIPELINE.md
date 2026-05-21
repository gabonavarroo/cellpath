# RUN_FINAL_PIPELINE — V3C reproduction quickstart

> **Default behavior**: evaluate the locked champion (no retraining). Expensive operations (PPO training, dynamics training, full sweep) require explicit flags.

---

## 0. One-time environment

```bash
make setup                     # creates .venv with uv, installs all deps (Python 3.11)
source .venv/bin/activate
```

Frozen tiers (`artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts/rl_sweeps/`) are not modified by V3C — all outputs land under `artifacts_v3/v3c/`.

## 1. Quickstart — champion evaluation (~10 min CPU)

The default re-runs the 7-cell PPO_BCD + reward-aware greedy_dyn_K + random/noop baseline evaluation against the locked champion:

```bash
python scripts/run_final_v3c_pipeline.py --mode eval
# or:
make final-v3c-eval
```

Outputs land in `artifacts_v3/v3c/eval_final_champion_repro_default/`. Each cell × policy writes a `<cell>/<policy>_summary.json` with raw success, mean_final_distance, mean_steps, tox_path, common_essential, unc_path_max, action_freq.

## 2. Fast demo (~2 min CPU)

Single-cell demo at K=2/bin8-10/OOD (the V3B Phase 4 binding non-saturated cell):

```bash
python scripts/run_final_v3c_pipeline.py --mode demo --n-episodes 50
```

## 3. Anchor baseline for side-by-side

```bash
python scripts/run_final_v3c_pipeline.py --mode baseline
# or:
make final-v3c-baseline
```

Runs the V3B Phase 4 V2 anchor PPO_BCD against same baselines for the same 7-cell matrix. Use the two output directories to diff per-cell deltas.

## 4. Track L / Track N references (Phase 4 4-seed)

Already trained; aggregator re-runs only:

```bash
# Track L 4-seed
python scripts/aggregate_v3b_phase4.py \
  --eval_dir artifacts_v3/v3c/rl_final/track_l_4seed_locked/eval \
  --out_dir  artifacts_v3/v3c/rl_final/track_l_4seed_locked/agg

# Track N 4-seed (500k checkpoint)
python scripts/aggregate_v3b_phase4.py \
  --eval_dir artifacts_v3/v3c/rl_final/track_n_4seed_locked/eval_500k \
  --out_dir  artifacts_v3/v3c/rl_final/track_n_4seed_locked/agg_500k

# Track N 4-seed (1M checkpoint)
python scripts/aggregate_v3b_phase4.py \
  --eval_dir artifacts_v3/v3c/rl_final/track_n_4seed_locked/eval_1M \
  --out_dir  artifacts_v3/v3c/rl_final/track_n_4seed_locked/agg_1M
```

## 5. Utility-audit re-run on champion

```bash
python scripts/run_final_v3c_pipeline.py --mode audit --n-episodes 200
```

Or per-bucket:

```bash
python scripts/audit_dynamics_utility_v3c.py reachability \
  --field-id artifacts_v3__dynamics_n64_legacy_ror_corr010 --n-episodes 200 --force
python scripts/audit_dynamics_utility_v3c.py contraction \
  --field-id artifacts_v3__dynamics_n64_legacy_ror_corr010 --force
```

## 6. Regenerate figures from existing outputs

```bash
python scripts/run_final_v3c_pipeline.py --mode figures
# or:
make final-v3c-figures
# or call individually:
python scripts/generate_v3c_figures.py pipeline_overview
python scripts/generate_v3c_figures.py final_leaderboard
```

Figures land in `artifacts_v3/v3c/figures/*.png`.

## 7. Cross-field aggregator (rebuild summary CSVs)

```bash
python scripts/aggregate_v3c_utility_audit.py
```

Rewrites:
- `artifacts_v3/v3c/utility_audit/prediction_metrics.csv`
- `artifacts_v3/v3c/utility_audit/reachability_matrix.csv`
- `artifacts_v3/v3c/utility_audit/greedy_saturation_matrix.csv`
- `artifacts_v3/v3c/utility_audit/contraction_geometry.csv`
- `artifacts_v3/v3c/utility_audit/action_heterogeneity.csv`
- `artifacts_v3/v3c/utility_audit/reward_leverage_fused.csv`
- `artifacts_v3/v3c/utility_audit/ppo_preconditions.csv`
- `artifacts_v3/v3c/utility_audit/utility_summary.md`
- `artifacts_v3/v3c/utility_audit/candidate_ranking.md`

## 8. Expensive operations (explicit flags only)

### Retrain a Phase 2.5 contraction-aware variant

```bash
.venv/bin/python scripts/train_dynamics.py \
  paths.vae_dir=artifacts_v3/vae_n64_legacy \
  paths.pairs_dir=artifacts_v3/pairs_n64_legacy \
  paths.dynamics_dir=artifacts_v3/v3c/dynamics_candidates/contraction_aware_<tag> \
  vae.n_latent=64 \
  dynamics.contraction_aware.enabled=true \
  dynamics.contraction_aware.lambda_excessive_alignment=<λ_ea> \
  dynamics.contraction_aware.tau_excessive_alignment=<τ_ea> \
  dynamics.contraction_aware.lambda_universal_attractor=<λ_ua> \
  dynamics.contraction_aware.tau_universal_attractor=<τ_ua> \
  dynamics.contraction_aware.lambda_action_diversity=<λ_ad> \
  dynamics.contraction_aware.tau_action_diversity=<τ_ad> \
  +force=true
```

Wall time: ~5–8 min on Apple M1/M2 MPS (84–92 epochs with early stop).

### Retrain PPO_BCD smoke on a new dynamics field

```bash
.venv/bin/python scripts/train_rl_v3b.py \
  --mode biorealistic_fused \
  --dynamics_dir <path> \
  --vae_dir <path> \
  --pairs_dir <path> \
  --total_timesteps 500000 \
  --seed 42 \
  --epsilon_value <per-VAE p15> \
  --output_dir artifacts_v3/v3c/rl_smokes/<tag>
```

Wall time: ~3–5 min training + ~5–10 min evaluation per cell at n_eps=200.

## 9. Expected output locations

| What | Where |
|---|---|
| Final champion manifest | `artifacts_v3/v3c/final_champion_manifest.json` |
| Champion eval (default mode) | `artifacts_v3/v3c/eval_final_champion_repro_default/` |
| Per-field utility audits | `artifacts_v3/v3c/utility_audit/<field_id>/` |
| Cross-field summaries | `artifacts_v3/v3c/utility_audit/*.csv` and `*.md` |
| PPO smokes (Phase 1) | `artifacts_v3/v3c/rl_smokes/<field>_<seed>_<horizon>/` |
| Phase 4 4-seed escalations | `artifacts_v3/v3c/rl_final/<field>_4seed_locked/` |
| Phase 2 / 2.5 dynamics candidates | `artifacts_v3/v3c/dynamics_candidates/contraction_aware_<tag>/` |
| Presentation figures | `artifacts_v3/v3c/figures/*.png` |
| Interpretation docs | `artifacts_v3/v3c/interpretation/*.md` |

## 9b. Replogle 2022 K562 essential CRISPRi held-out audit (Bucket-C)

Bucket-C post-hoc action-overlap audit. No retraining; reads existing eval `summary.json`
files. Total wall time: ~10 s for both stages.

```bash
# Stage 2 — download Harmonizome gene-set artifacts and rebuild processed JSON/CSV
python scripts/download_replogle_heldout.py            # ~5 MB raw .gz to data/raw/replogle/
python scripts/download_replogle_heldout.py --skip-download  # reuse cached raw files

# Stage 4 — action-overlap audit for the V3C champion
python scripts/audit_v3c_replogle_heldout.py
python scripts/audit_v3c_replogle_heldout.py --primary-cell-only
```

Reads:
- `data/processed/replogle/replogle_norman_intersection.json` (Stage 2 output)
- `artifacts_v3/v3c/rl_smokes/contraction_aware_v2_aggressive_seed{42,0,1,7}_500k/eval/<cell>/<policy>/summary.json`
- `artifacts_v3/v3c/rl_final/track_l_4seed_locked/eval/seed{42,0,1,7}/<cell>/<policy>/summary.json`
- `artifacts_v3/v3c/rl_smokes/anchor_v2_ror_corr010_1M_reused/eval/<cell>/<policy>/summary.json`

Writes:
- `data/processed/replogle/{replogle_essential_genes.json, replogle_essential_genes.csv,
  replogle_norman_intersection.json, replogle_only_essential_genes.json, source_metadata.json}`
- `artifacts_v3/v3c/replogle_heldout_action_overlap.csv` (277-row long table)
- `artifacts_v3/v3c/interpretation/v3c_final_replogle_heldout_audit_metrics.json`
- `artifacts_v3/v3c/figures/replogle_heldout_action_overlap.png` (generated separately
  via the snippet at the end of `v3c_final_replogle_heldout_audit.md` §10)

Full report: `artifacts_v3/v3c/interpretation/v3c_final_replogle_heldout_audit.md`.

## 10. Sacred-rule check

After any session:

```bash
git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/  # must be clean
PYTHONPATH=. .venv/bin/pytest -q                                          # must show 0 failures
```
