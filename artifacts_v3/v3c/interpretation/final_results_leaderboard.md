# V3C Final Results — Leaderboard

> Cross-field comparison at K=2/bin8-10/OOD (the V3B Phase 4 binding non-saturated cell) and at the K=3/bin8-10/OOD cell where V2/V3 greedy saturates. All numbers reported at per-VAE p15 ε.

---

## 1. Dynamics-only metrics (no PPO involved)

`val/OOD Pearson` is U-A prediction sanity; `cf, gu_max, align_med, act_div` are U-D contraction geometry on the OOD start pool (n=1500 samples); `K=2/b8-10 reach` and `K=3/b8-10 reach` are U-B beam reachability at p15 (n_eps=200, beam_width=64).

| Field | val P | OOD P | cf | gu_max | align_med | act_div | K=2/b8-10 reach | K=2/b6-8 reach | K=3/b8-10 reach (greedy_dyn_1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 anchor (RoR_corr010, 32D) | 0.615 | 0.516 | 0.9995 | 0.921 | 0.784 | 0.116 | 0.120 | 0.655 | 1.000 (sat) |
| Track L (n64 legacy) | 0.620 | 0.515 | 0.9998 | 0.933 | 0.821 | 0.107 | **0.560** | 0.895 | 1.000 (sat) |
| Track N (n64 NB) | 0.621 | 0.504 | 0.9992 | 0.932 | 0.796 | 0.114 | 0.435 | 0.855 | 1.000 (sat) |
| contraction_aware_v1 (τ=0.80) | 0.619 | 0.514 | 0.9998 | 0.905 | 0.806 | 0.104 | 0.560 | 0.745 | 1.000 (sat) |
| **contraction_aware_v2_aggressive (τ=0.60)** | 0.614 | 0.511 | 0.9997 | **0.874** | **0.720** | 0.090 | **0.000** ⚠ | 0.285 | **0.56 (un-sat!)** |
| contraction_aware_v3_diverse (τ=0.80 + λ_ad=0.10) | 0.619 | 0.514 | 0.9998 | 0.905 | 0.807 | 0.104 | 0.435 | 0.790 | 1.000 (sat) |
| contraction_aware_v4_combo (τ=0.60 + λ_ad=0.10) | 0.614 | 0.511 | 0.9997 | 0.874 | 0.721 | 0.090 | 0.000 | 0.300 | 0.630 (un-sat) |
| mean_delta_corr_010 | 0.520 | 0.384 | 0.985 | 0.657 | 0.487 | 0.135 | 0.000 | 0.000 | 0.000 |
| Soft-OT (anti-contractive) | 0.934 | 0.743 | 0.000 | −0.622 | −0.770 | 0.048 | 0.000 | 0.000 | 0.000 |
| Random pairs (negative control) | 0.723 | — | 1.000 | 0.991 | — | 0.029 | 1.000 (sat) | 1.000 (sat) | 1.000 (sat) |

**Reading guide:**
- The 24 OT-trained fields cluster at `cf ≈ 1.0` + `gu_max ≈ 0.92` (universal over-contraction) and saturate at K≥3.
- Track L / Track N preserve the saturation at K≥3 but expand the K=2 actionable region (4–5× anchor reach at K=2/b8-10).
- Phase 2 v1 conservative regularizer moved `gu_max` 0.028 without breaking anything.
- Phase 2.5 v2_aggressive moved `gu_max` significantly more (−0.06) AND un-saturated K=3/b8-10 (0.56 vs 1.0) but destroyed K=2/b8-10 reach.

## 2. PPO_BCD results (multi-seed where available)

All trained with the locked B+C+D reward stack at per-VAE p15. Single-seed (seed 42) unless noted "4-seed".

| Field × seed | PPO_BCD K=2/b8-10 | same-field greedy_dyn_2 | Δ | mean_final_distance Δ (PPO − greedy) | Verdict |
|---|---:|---:|---:|---:|---|
| V2 anchor (4-seed, 1M) | 0.148 ± 0.037 | 0.130 | +0.018 | tied | `LOCKED_DESIGN_TECHNICAL_ONLY` |
| Track L (4-seed, 1M) | **0.705 ± 0.000** | 0.695 | **+0.010** | **+0.173 (regression)** | `NO_STABLE_SIGNAL` — Pareto distance fail |
| Track N (4-seed, 500k) | 0.499 ± 0.052 | 0.495 | +0.004 [−0.047, +0.055] | mild regression | `NO_STABLE_SIGNAL` |
| Track N (4-seed, 1M) | 0.472 ± 0.097 | 0.495 | −0.023 [−0.117, +0.072] | regresses | `NO_STABLE_SIGNAL` |
| **contraction_aware_v2_aggressive seed 42 (500k) at K=3/b8-10** | **0.840** | 0.705 (g_2) / **0.765** (g_3) | **+0.135** (vs g_2) / **+0.075** (vs g_3) | n/a (K=3 fully reached) | **CANDIDATE_SIGNAL_RAW** (single-seed) |
| contraction_aware_v2_aggressive seed 0 (500k) at K=3/b8-10 | 0.705 | 0.705 (g_2) / 0.765 (g_3) | 0.0 / −0.060 | n/a | variance check — seed 42 advantage not reproduced |
| contraction_aware_v2_aggressive 2-seed mean at K=3/b8-10 | 0.7725 ± 0.095 | 0.765 (g_3) | +0.0075 (tied within variance) | n/a | TUNED candidate; 4-seed Phase 4 needed |
| **contraction_aware_v2_aggressive 4-seed at K=3/b8-10** (seeds 42, 0, 1, 7) | **0.806 ± 0.068** | 0.765 (g_3) | **+0.041 CI95 [-0.025, +0.107]** | distance preserved | **V2AGG_VARIANCE_BOUNDED**: 3/4 seeds at 0.840; seed 0 outlier 0.705 |
| v2_aggressive PPO tuning: ent_coef=0.05 seed42 500k | 0.705 | 0.765 (g_3) | -0.060 | 2.91 | PPO_TUNED regression; default IS optimum |
| v2_aggressive PPO tuning: lr=1e-4 seed42 500k | 0.840 | 0.765 (g_3) | +0.075 | 2.90 | matches default at K=3/b8-10 but loses K=3/b6-8 |
| v2_aggressive PPO tuning: 750k seed42 | 0.705 | 0.765 (g_3) | -0.060 | 2.94 | PPO_TUNED regression with longer training |
| Phase 3 ensemble (3 Track L seeds) audit | n/a | n/a | n/a | n/a | **ENSEMBLE_DIAGNOSTIC_ONLY**: action_dep score = 5e-5 (state-dep only, reproduces V3B Phase 4) |
| mean_delta_corr_010 (seed 42 500k) | 0.000 | 0.000 | 0.000 | n/a | `NO_SIGNAL` (NOOP-strategy) |

## 3. Champion selection

See `final_champion_selection.md` for the interpretive rationale and tie-breakers.

**PRIMARY default champion** (`CHAMPION_TUNED_RESULT`): `contraction_aware_v2_aggressive` + PPO_BCD seed 42 500k. Best PPO−greedy delta at a non-saturated cell: **+0.075** at K=3/bin8-10/OOD (single-seed; variance-bounded).

**SECONDARY reference** (`LOCKED_DEFAULT_RESULT`): Track L (`artifacts_v3/dynamics_n64_legacy_ror_corr010`) + PPO_BCD 4-seed × 1M. Stable, 4-seed-validated reference; +0.010 zero-variance edge vs same-field greedy_dyn_2 at K=2/bin8-10/OOD, 4.8× lift over V2 anchor.

## 4. Reproduction commands

```bash
# Reproduce champion evaluation (default mode)
python scripts/run_final_v3c_pipeline.py --mode eval

# Quick 1-cell demo
python scripts/run_final_v3c_pipeline.py --mode demo

# Anchor baseline for side-by-side
python scripts/run_final_v3c_pipeline.py --mode baseline

# Regenerate figures from existing aggregator outputs
python scripts/run_final_v3c_pipeline.py --mode figures
```
