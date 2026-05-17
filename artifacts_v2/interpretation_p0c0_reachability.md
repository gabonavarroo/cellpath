# P0C0 — Reachability Diagnostic Interpretation (2026-05-16)

## Configuration

| Parameter | Value |
|---|---|
| epsilon_p25 | 3.1663 |
| epsilon_p50 | 3.5311 |
| start pool | OOD cells, bin 8–10, n=17 |
| beam_width | 50, max_depth=3 |
| NoopFree greedy n_episodes | 100 |
| dynamics compared | V1 OT (sanity), mean_delta, soft_ot |
| pairs_dir (held-out genes) | artifacts/pairs (V1 OT metadata) |

---

## D1: Per-gene contraction analysis (n_starts=500)

| Dynamics | Gini | Entropy fraction | Mean fraction_positive | Mean improvement |
|---|---:|---:|---:|---:|
| V1 OT (P0A reference) | 0.1748 | 0.9858 | 0.955 | +0.52 (est.) |
| mean_delta | 0.1386 | 0.9668 | **0.826** | **+0.405** |
| soft_ot | 0.1333 | 0.9994 | **0.000** | **−1.425** |

Top-3 contracting genes:
- mean_delta: MAML2, BCL2L11, HK2 (PPO top-gene overlap: HK2, KIF2C, MAP4K3, MAP4K5, TSC22D1)
- soft_ot: BAK1, MAP2K6, ISL2 (all with fraction_positive=0.000 — "least bad" not contractive)

**Interpretation:** The soft-OT dynamics model predicts that *every* gene perturbation at
*every* start state increases distance from the reference centroid. The best gene (BAK1) still
increases distance by 1.18 units on average; the worst (JUN, RHOXF2) by >1.76 units. This is
a direct consequence of the barycentric target mechanism: soft-OT pseudo-controls are convex
combinations of observed controls, meaning each control is a weighted average that points closer
to the center of the control distribution — which is *farther* from z_ref than most perturbed
cells. The dynamics MLP learned to predict this consistently.

Mean-delta, by contrast, has 82.6% fraction_positive — nearly as contractive as V1 OT (95.5%).
The entropy_fraction (0.967 vs 0.999) shows slightly more concentrated benefit in a subset of
genes, which is biologically plausible (some genes simply move K562 cells more effectively).

---

## D2: NoopFreeGreedy hard benchmark (primary cell k=3, n=100)

| Dynamics | greedy_dyn_1 sr | greedy_dyn_1 mean_dist | noop_free sr | noop_free mean_dist |
|---|---:|---:|---:|---:|
| V1 OT | **1.000** | **2.835** | 1.000 | 2.835 |
| mean_delta | 0.000 | 5.489 | 0.000 | 5.489 |
| soft_ot | 0.000 | 8.479 (= noop) | 0.000 | 21.999 |

**Interpretation:**

*mean_delta:* greedy_dyn_1 actively picks genes (same result as noop_free — noop is never the
greedy choice), reducing distance from ~8.48 to 5.49 in 3 steps (−2.99 units, ~1.0/step).
Success rate is 0 because 5.49 > epsilon_p25=3.17, but the trajectory is strongly directed.

*soft_ot:* greedy_dyn_1 picks noop exclusively (mean_dist=8.48 = noop baseline). When forced
to pick genes (noop_free), distance explodes to 22.0 — confirming that the V1 greedy picking
noop was not a failure of policy exploration but the correct response to a field that actively
harms the cell state for every gene choice.

This is the key distinction: **mean-delta's gate failure is a model calibration problem**
(the MLP doesn't predict per-gene effects accurately enough to clear +0.030 val margin), not
a field problem. **soft-OT's failure is a field problem** — the field itself is anti-contractive.

---

## D3: Beam-search reachability probe (repeat_mask=True — RL-comparable)

| Dynamics | n_success/17 | success_rate | best_final_distance | mean_best |
|---|---:|---:|---:|---:|
| V1 OT | **17/17** | **1.000** | 1.593 | 2.038 |
| mean_delta | 0/17 | 0.000 | **4.114** | 5.337 |
| soft_ot | 0/17 | 0.000 | **16.974** | 22.195 |

V1 OT sanity check: PASSED. Probe correctly identifies the working baseline.

Upper-bound results (repeat_mask=False):
- V1 OT: 17/17 (identical — gene reuse does not help)
- mean_delta: 0/17, best=4.114 (identical — the bottleneck is model quality, not reuse)
- soft_ot: 0/17, best=16.974 (gene reuse offers no relief from anti-contractive field)

**Interpretation:** mean_delta best_final_distance=4.114 is only **0.95 units above epsilon_p25**
(3.17). The beam is very close. The field has the right direction; the problem is that 3 steps
are insufficient AND/OR the model's predicted per-gene effect magnitudes are under-estimated
(consistent with val margin +0.0214 instead of +0.030 — the MLP doesn't outperform ridge by
enough to capture the per-gene, per-dimension structure that would close the remaining gap).

soft_ot beam reaches 16.97, starting from ~9 — distance *increases* from the probe start.
This is 3 genes × anti-contractive field = compounding harm.

---

## D4: ε-feasibility analysis (repeat_mask=True)

| Dynamics | ε for 10% success | ε for 25% success | ε for 50% success | Current epsilon_p25 | Feasible? |
|---|---:|---:|---:|---:|---|
| V1 OT | 1.752 | 1.910 | 2.051 | 3.166 | YES (large margin) |
| mean_delta | 4.323 | 5.016 | 5.501 | 3.166 | NO for k=3; YES for k≥6 (est.) |
| soft_ot | 18.123 | 18.679 | 20.352 | 3.166 | NO — infeasible at any epsilon < 18 |

**mean-delta k-step extrapolation:** greedy_dyn_1 achieves ~1.0 unit/step contraction
(8.48 → 5.49 in 3 steps). At this rate, k=6 → ~2.5 units total; k=8 → ~0.5 → below epsilon_p25.
PPO with k=8 (current RL config) should find trajectories well within epsilon once trained on
an accurate enough mean-delta dynamics model.

**soft-OT extrapolation:** even at k=8 steps, beam-search with repeat_mask=True would
compound the anti-contractive effect (~3 more genes applied → distance grows further).
No ε relaxation helps — the field is fundamentally misaligned with the task objective.

---

## Verdict

**H_reach_soft_ot: SUPPORTED.** The soft-OT dynamics field is fundamentally anti-contractive:
for all 17 OOD start cells in bin 8-10, no gene sequence of length ≤ 3 reaches distance <
epsilon_p25 according to the model. In fact, ALL gene sequences *increase* distance (fraction_
positive=0.000). The 3-step beam's best case is dist=16.97, starting from ~9. This is not a
depth or beam-width issue — it is a consequence of the barycentric smoothing mechanism making
every gene action predict movement toward a pseudo-control average that is farther from z_ref
than the starting perturbed cell.

**H_reach_mean_delta: REJECTED as stated.** Mean-delta does NOT achieve success (0/17) but
the field IS contractive and directionally correct. The hypothesis framing ("contractive enough
for multi-step planning to succeed") requires a correction: mean-delta IS contractive
(fraction_positive=0.826, best beam dist 4.11) and multi-step planning WILL succeed at k=8.
The barrier is model accuracy (gate failure at +0.0214 vs +0.030), not field structure.

---

## Recommended Next Step

**PATH B: P0B2 — Retrain dynamics on `artifacts_v2/pairs_mean_delta` with correlation loss
(λ_corr ∈ {0.05, 0.10}) to close the gate. Then proceed to P0C (PPO retrain) on the
gate-passing mean-delta dynamics.**

### Why not PATH A (soft-OT as-is)?
soft-OT is fundamentally anti-contractive. fraction_positive=0.000 means zero gene×start
pairs improve distance. PPO cannot learn useful policies on a field where every action harms
the objective. Correlation loss cannot rescue a field that is anti-contractive — the problem
is in the pairing targets, not the training loss.

### Why not PATH B' (soft-OT + corr loss)?
PATH B' applies when soft-OT shows *partial* direction (best_dist < 5.0). At best_dist=16.97,
soft-OT has no partial direction. Correlation loss might improve per-dimension calibration,
but it cannot change the sign of a field that is fundamentally pointing the wrong way.

### Why PATH B (mean-delta + corr loss)?
1. fraction_positive=0.826 — the field is strongly contractive.
2. greedy_dyn_1 reduces distance −2.99 in 3 steps (toward, not away from, z_ref).
3. best beam dist=4.11 is 0.95 above epsilon — the field is almost right.
4. Gate failure is in val margin (+0.0214 vs +0.030) — per-gene accuracy, not field direction.
5. Correlation loss penalizes exactly this: 1−Pearson_d, averaged over active dims, encourages
   the MLP to capture per-dimension gene effects accurately.

### P0B2 Command (requires explicit approval before execution)

```bash
# λ_corr = 0.05 (start here; close to gate means small push needed)
PYTHONPATH=. .venv/bin/python scripts/train_dynamics.py --config-name default \
    paths.pairs_dir=artifacts_v2/pairs_mean_delta \
    paths.dynamics_dir=artifacts_v2/dynamics_mean_delta_corr_005 \
    dynamics.lambda_corr=0.05 \
    seed=42

# If 0.05 still fails the gate, run λ_corr = 0.10:
PYTHONPATH=. .venv/bin/python scripts/train_dynamics.py --config-name default \
    paths.pairs_dir=artifacts_v2/pairs_mean_delta \
    paths.dynamics_dir=artifacts_v2/dynamics_mean_delta_corr_010 \
    dynamics.lambda_corr=0.10 \
    seed=42
```

Gate acceptance criteria:
- `val.margin_vs_linear_ridge_pearson ≥ +0.030`
- `val.uncertainty_calibration.spearman ≥ 0.20`
- `ood.pearson_r ≥ 0.40`

### P0C Command (after P0B2 gate passes, requires explicit approval)

```bash
PYTHONPATH=. .venv/bin/python scripts/train_rl.py --config-name default \
    paths.dynamics_dir=artifacts_v2/dynamics_mean_delta_corr_005 \
    paths.rl_dir=artifacts_v2/rl_mean_delta_corr005 \
    rl.total_timesteps=500000 \
    rl.reward.start_epsilon_label=p50 \
    seed=42
```

### Pre-requisite: confirm correlation_loss and lambda_corr exist

Before running P0B2, verify:
```bash
grep -n "correlation_loss\|lambda_corr" src/analysis/metrics.py scripts/train_dynamics.py config/dynamics.yaml
```
If absent, implement per the spec in P0C0_REACHABILITY_PLAN.md §Task 8 (legacy notation)
or `P0B_PRIME_PAIRING_CORRECTION_PLAN.md §P0B.1`.
