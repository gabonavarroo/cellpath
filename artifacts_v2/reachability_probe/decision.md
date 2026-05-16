# P0C0 — Decision Document (2026-05-16)

## D1–D4 Numerical Results

### D1: Per-gene contraction (n_starts=500)

| Dynamics | Gini | Entropy fraction | Mean fraction_positive | Mean improvement |
|---|---:|---:|---:|---:|
| V1 OT (P0A) | 0.1748 | 0.9858 | 0.955 | +0.52 (est.) |
| mean_delta | 0.1386 | 0.9668 | **0.826** | **+0.405** |
| soft_ot | 0.1333 | 0.9994 | **0.000** | **−1.425** |

soft_ot max(fraction_positive) = 0.002 (2 out of 52,500 gene×start pairs are distance-reducing).
**soft_ot is completely anti-contractive — every gene action increases distance at every start.**

### D2: NoopFreeGreedy hard benchmark (primary cell k=3, n=100)

| Dynamics | always_noop | greedy_dyn_1 | greedy_dyn_1_noop_free |
|---|---:|---:|---:|
| V1 OT | sr=0.000, dist=8.479 | sr=1.000, dist=2.850 | sr=1.000, dist=2.835 |
| mean_delta | sr=0.000, dist=8.479 | sr=0.000, dist=5.489 | sr=0.000, dist=5.489 |
| soft_ot | sr=0.000, dist=8.479 | sr=0.000, dist=8.479 | sr=0.000, dist=21.999 |

Key observations:
- mean_delta greedy_dyn_1: reduces distance from 8.48 → 5.49 (−2.99 in 3 steps). Greedy never
  picks noop (noop_free result is identical), confirming field is contractive.
- soft_ot noop_free: distance explodes from 8.48 → 22.0. Forcing gene picks actively harms cells.
  V1 greedy=noop because noop is genuinely the best 1-step action under soft-OT dynamics.

### D3: Beam-search probe (beam=50, depth=3, repeat_mask=True)

| Dynamics | n_success/17 | success_rate | best_dist | mean_best |
|---|---:|---:|---:|---:|
| V1 OT | 17/17 | 1.000 | 1.593 | 2.038 |
| mean_delta | 0/17 | 0.000 | **4.114** | 5.337 |
| soft_ot | 0/17 | 0.000 | **16.974** | 22.195 |

V1 OT sanity check: PASSED (17/17 success). Probe is not buggy.

### D4: ε-feasibility (repeat_mask=True)

| Dynamics | ε for 10% success | ε for 25% success | ε for 50% success |
|---|---:|---:|---:|
| V1 OT | 1.752 | 1.910 | 2.051 |
| mean_delta | 4.323 | 5.016 | 5.501 |
| soft_ot | 18.123 | 18.679 | 20.352 |

epsilon_p25 = 3.166, epsilon_p50 = 3.531.
Mean-delta 3-step beam needs ε=5.0 for 25% success, but this is a 3-step upper bound.
With k=8 RL steps, ~1 unit/step contraction → 8.5 − 8 = 0.5 (well below epsilon).

---

## Decision Rule Matrix Application

| Condition | soft_ot | mean_delta |
|---|---|---|
| success_rate ≥ 0.10 (PATH A) | 0.000 ✗ | 0.000 ✗ |
| best_dist < 5.0 (partial direction) | 16.974 ✗ | 4.114 ✓ |
| infeasible (best_dist ≥ 5.0) | 16.974 ✓ | — |

soft_ot: best_dist=16.974 ≥ 5.0 → **infeasible** (PATH B' inapplicable; B' requires soft_ot
best_dist < 5.0). No amount of beam width or depth will help — the field is anti-contractive
at every gene×start pair.

mean_delta: best_dist=4.114 < 5.0 → the field has **real directionality**. Gate failure
(val margin +0.0214 < threshold +0.030) is the blocker, not field structure.

The applicable path is the one that addresses mean-delta's gate failure while preserving
its contractive structure.

---

## Selected Path

**PATH B: P0B2 — Retrain dynamics on mean-delta pairs with correlation loss
(λ_corr ∈ {0.05, 0.10}) to close the gate (val margin +0.0214 → target +0.030).**

### Rationale

1. **soft-OT is fundamentally infeasible.** fraction_positive=0.000, beam best_dist=16.97,
   ε for 25% success=18.7. No path through soft-OT leads to P0C.

2. **mean-delta has real contractive structure.** fraction_positive=0.826 (close to V1 OT's
   0.955), greedy_dyn_1 reduces distance −2.99 in 3 steps. The field direction is correct.

3. **mean-delta's gate failure is a model-quality issue, not a field issue.** Val margin=+0.0214
   (threshold +0.030). The MLP is not accurately capturing per-gene per-dimension effects.
   Correlation loss (1 − Pearson_d averaged over dims) penalizes exactly this failure mode.

4. **k=8 RL would succeed if gate passes.** From starting dist ~8.5, mean-delta greedy
   achieves ~1 unit/step contraction. At k=8: 8.5 − 8×1.0 ≈ 0.5 < epsilon_p25=3.17.

5. **PATH B' (soft-OT + corr loss) is not warranted.** PATH B' applies when soft-OT has
   partial direction (best_dist < 5.0). At 16.97, soft-OT is not partially directed — it is
   actively anti-contractive. Corr loss cannot rescue a field where ALL genes increase distance.

### Prerequisite for P0C

Retrain dynamics with λ_corr ∈ {0.05, 0.10} on `artifacts_v2/pairs_mean_delta`.
Gate must pass: val margin ≥ +0.030 AND OOD Pearson ≥ 0.40 AND uncertainty spearman ≥ 0.20.
After gate passes, re-probe (beam search) to confirm success_rate > 0 under the new model.
Then proceed to P0C: PPO retrain on the gate-passing mean-delta+corr dynamics.
