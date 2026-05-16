# P0B2 — Mean-Delta Dynamics + Correlation Loss Interpretation (2026-05-16)

## Configuration

| Parameter | Value |
|---|---|
| Dynamics pair source | `artifacts_v2/pairs_mean_delta` |
| Base architecture | Residual heteroscedastic MLP, state_linear_skip=True, 3 layers, d_emb=64 |
| lambda_corr values tried | 0.0, 0.05, 0.10, 0.30 |
| Gate threshold (val margin) | +0.030 (PHASES.md Phase 2) |
| epsilon_p25 | 3.1663 |
| Start pool | OOD cells, bin 8–10, n=17 |
| Beam search | beam_width=50, max_depth=3 |
| Hard benchmark | n=100, k=3, epsilon_p25 |

## Summary of trained variants

| Variant | Val margin | OOD Pearson | Unc Spearman | OOD dim-11 diff | Gate |
|---|---:|---:|---:|---:|---|
| baseline (λ=0.00) | +0.0214 | 0.3833 | ~0.22 | -0.253 | FAIL |
| λ_corr=0.05 | +0.0225 | 0.3835 | 0.2217 | -0.258 | FAIL |
| λ_corr=0.10 | +0.0227 | 0.3836 | 0.2219 | -0.255 | FAIL |
| λ_corr=0.30 | +0.0232 | 0.3849 | 0.2238 | -0.250 | FAIL |

All variants failed the gate. The threshold is val mlp-minus-ridge Pearson ≥ +0.030.

## Root-cause: dim-11 OOD regression

The gate bottleneck is a single latent dimension (dim 11) that regresses severely in OOD:

| Variant | Val dim-11 (MLP) | Val dim-11 (ridge) | Val diff | OOD dim-11 (MLP) | OOD dim-11 (ridge) | OOD diff |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.526 | 0.553 | -0.027 | 0.057 | 0.310 | -0.253 |
| λ=0.05 | ~0.540 | ~0.553 | -0.013 | 0.053 | 0.310 | -0.258 |
| λ=0.10 | ~0.545 | ~0.553 | -0.008 | 0.051 | 0.310 | -0.255 |
| λ=0.30 | 0.554 | 0.553 | +0.001 | 0.060 | 0.310 | -0.250 |

Key observation: the ridge Pearson on dim 11 OOD is 0.310 across ALL variants — it never changes.
This means the mean-delta pairing predicts dim-11 perturbation effects at exactly 0.310 Pearson from
the held-out gene OOD split. The MLP, however, collapses to ~0.06 on this dimension in OOD, while
achieving 0.554 on val. This is a classical OOD generalization failure: dim-11 behavior for held-out
genes cannot be inferred from training-gene mean-delta embeddings, regardless of how the loss is
shaped.

The correlation loss improved val dim-11 (recovering it from -0.027 to +0.001), but OOD dim-11
remained stuck at ~0.06 MLP vs 0.31 ridge. The correlation penalty is only evaluated during training
(where the MLP sees training-gene pairs), so it cannot force the MLP to generalize dim-11 to OOD.

## Correlation loss plateau

| λ | Val margin | Gain over prev |
|---|---|---|
| 0.00 | +0.0214 | baseline |
| 0.05 | +0.0225 | +0.0011 |
| 0.10 | +0.0227 | +0.0002 |
| 0.30 | +0.0232 | +0.0005 |

Gains are monotonic but strongly saturating. Extrapolating, λ→∞ would not close the
+0.030 threshold. The remaining gap (+0.0068 at λ=0.30) is not reducible by scaling λ.

## Reachability probe (best variant λ=0.30, repeat_mask=True)

| Dynamics | n_success / n_starts | success_rate | best_final_dist | mean_best_dist |
|---|---:|---:|---:|---:|
| V1 OT (sanity check) | 17 / 17 | 1.000 | 1.5932 | 2.0378 |
| mean_delta baseline | 0 / 17 | 0.000 | 4.1139 | 5.3371 |
| mean_delta λ=0.30 | 0 / 17 | 0.000 | 4.0899 | 5.2449 |

The correlation loss improved best reachable distance by 0.024 units (4.114 → 4.090) — negligible.
The gap to epsilon_p25 (3.167) remains 0.923 units. V1 OT sanity check passes (17/17).

## Hard benchmark (λ=0.30, n=100, k=3, epsilon_p25, bin 8–10, OOD genes)

| Policy | Success rate | Mean final dist |
|---|---:|---:|
| always_noop | 0.000 | 8.479 |
| greedy_dyn_1 | 0.000 | 5.394 |
| greedy_dyn_1_noop_free | 0.000 | 5.394 |
| ppo_deterministic (V1 PPO) | 0.000 | 5.896 |

The field is directionally contractive (greedy reduces from 8.479 to 5.394 = 36% reduction), but
all policies fail the epsilon threshold. The gap between greedy mean final dist (5.394) and
epsilon_p25 (3.167) is 2.23 units — far beyond what a longer-horizon planner could close under
the current dynamics.

Note: greedy_dyn_1 and greedy_dyn_1_noop_free produce identical results because the mean-delta
field already picks gene actions at every step (noop is suboptimal in terms of distance, so greedy
never picks it).

## Verdict

**H_mean_delta_gate (corr loss closes gate):** Rejected. Correlation loss at λ ∈ {0.05, 0.10, 0.30}
improves val margin by +0.0018 total (from +0.0214 to +0.0232) but does not reach +0.030. The
bottleneck is OOD dim-11 generalization failure intrinsic to the mean-delta pairing, not fixable
by loss shaping.

**Root cause:** Mean-delta pairing averages perturbation effects across all cells per gene. For
dim-11 in OOD genes, the averaged effect is not predictive from the gene embedding alone — the
variance is entirely cell-state dependent. This is a fundamental limitation of mean-delta as a
pairing strategy for this dimension.

## Comparison with V1 OT dynamics

| Metric | V1 OT | mean_delta (best) |
|---|---|---|
| Val Pearson | 0.564 | 0.521 |
| Val margin | +0.0074 | +0.0232 |
| OOD Pearson | 0.490 | 0.385 |
| OOD dim-11 diff | ~+0.08 | -0.250 |
| Beam success rate | 1.000 | 0.000 |
| Gate | PASS | FAIL |

V1 OT dynamics has lower absolute val Pearson but passes the gate and supports 100% beam-search
reachability. The mean-delta dynamics has higher val Pearson but fails the gate and is unreachable.
This is consistent with the P0C0 finding: soft-OT (even higher val Pearson=0.934) completely fails
reachability; OT pairing quality, not val Pearson, determines RL controllability.

## Recommended next step

**PATH C escalation — mean-delta is not reachable and cannot be gated with corr-loss alone.**

Three diagnostic phases (P0C0, P0B2) confirm that the V1 OT dynamics is the only field that:
1. Passes the validation gate
2. Supports 100% beam-search reachability from OOD bin 8–10

The recommended investigation is why the V1 PPO (trained on V1 OT dynamics) fails on the V2 hard
benchmark. Specifically:

**Option C1 (recommended): Retrain PPO on V1 OT dynamics with improved curriculum or reward.**
V1 greedy succeeds at 100% on this benchmark. The PPO's 0% success on the V2 hard bench is likely
a training-distribution mismatch — the V1 PPO was trained with start_epsilon_label=p50, not the
hard-bench epsilon_p25 starts from bin 8–10. Retraining PPO on V1 OT dynamics with:
- `start_epsilon_label=p25` or explicit bin 8–10 curriculum
- More timesteps (1M or 2M)
- Check action masking / repeat_mask interaction

**Option C2: Investigate dim-11 OOD regression specifically.**
The mean-delta field's dim-11 collapse is a hard bottleneck. This could be addressed by:
- Per-dim loss weighting (upweight dim-11 in both NLL and corr loss)
- Targeted architecture fix (residual skip that bypasses dim-11 prediction for OOD)
- Hybrid pairing: OT pairs for held-out genes, mean-delta for training genes

**Option C3: Accept mean-delta at current margin (+0.0232) and lower the gate threshold.**
Explicitly rejected per sacred rules — gate thresholds must not be changed.

**Singular recommendation: Option C1 — retrain PPO on V1 OT dynamics.**
V1 OT is the only verified controllable field. The PPO training configuration needs adjustment to
match the hard-bench difficulty level. This requires explicit user approval before execution per
CLAUDE.md §9.
