# V3B Phase 4 / Final — Reward-stack 4-seed evaluation summary

> Seeds [42, 0, 1, 7], n=300 episodes/cell, n_cells=7.
> dynamics: artifacts_v2/dynamics_v1ot_ror_corr010 (V2 primary 32D — frozen).
> reward env: biorealistic_fused (so all A-bucket metrics tracked uniformly).
> Greedy oracles are reward-aware under fused objective.

## Final verdict: **`LOCKED_DESIGN_TECHNICAL_ONLY`**

(`LOCKED_DESIGN_TECHNICAL_ONLY` is the expected outcome on V2 dynamics — the controller objective is implemented and evaluable but the field's saturation prevents a planning-advantage headline.)

## Bucket B — reward-independent raw success (4-seed mean ± std)

| cell | PPO_A | PPO_B | PPO_C | PPO_BC | PPO_D | PPO_BCD | greedy_2_F | greedy_5_F | random |
|---|---|---|---|---|---|---|---|---|---|
| k2_bin6-8_splitood | — | — | — | — | — | 0.790±0.060 | 0.910±0.000 | — | 0.125±0.009 |
| k2_bin8-10_splitood | — | — | — | — | — | 0.705±0.000 | 0.695±0.000 | — | 0.056±0.005 |
| k3_bin6-8_splitood | — | — | — | — | — | 0.995±0.010 | 1.000±0.000 | — | 0.344±0.021 |
| k3_bin8-10_splitood | — | — | — | — | — | 1.000±0.000 | 1.000±0.000 | — | 0.280±0.023 |
| k4_bin8-10_splitood | — | — | — | — | — | 1.000±0.000 | 1.000±0.000 | — | 0.481±0.040 |
| k5_bin8-10_splitood | — | — | — | — | — | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.586±0.015 |
| k8_bin8-10_splitood | — | — | — | — | — | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.787±0.029 |

## Bucket A — reward-fit metrics (4-seed mean of PPO_BCD vs key comparators)

Reminder: Bucket A metrics are derived from sources used in the reward (DepMap Chronos for tox/CE; learned dynamics for unc) and represent reward-prior optimisation, not independent biological discovery.

| cell | policy | mean_tox | mean_CE | unc_max | unc_mean | frac_zero_CE |
|---|---|---|---|---|---|---|
| k2_bin6-8_splitood | PPO_BCD | 0.000±0.000 | 0.000±0.000 | 0.362±0.006 | 0.336±0.004 | 1.000±0.000 |
| k2_bin6-8_splitood | PPO_BC | — | — | — | — | — |
| k2_bin6-8_splitood | PPO_D | — | — | — | — | — |
| k2_bin6-8_splitood | PPO_C | — | — | — | — | — |
| k2_bin6-8_splitood | PPO_A | — | — | — | — | — |
| k2_bin6-8_splitood | greedy_dyn_2_fused | 0.000±0.000 | 0.000±0.000 | 0.365±0.000 | 0.340±0.000 | 1.000±0.000 |
| k2_bin6-8_splitood | greedy_dyn_5_fused | — | — | — | — | — |
| k2_bin8-10_splitood | PPO_BCD | 0.000±0.000 | 0.000±0.000 | 0.404±0.003 | 0.376±0.004 | 1.000±0.000 |
| k2_bin8-10_splitood | PPO_BC | — | — | — | — | — |
| k2_bin8-10_splitood | PPO_D | — | — | — | — | — |
| k2_bin8-10_splitood | PPO_C | — | — | — | — | — |
| k2_bin8-10_splitood | PPO_A | — | — | — | — | — |
| k2_bin8-10_splitood | greedy_dyn_2_fused | 0.000±0.000 | 0.000±0.000 | 0.419±0.000 | 0.379±0.000 | 1.000±0.000 |
| k2_bin8-10_splitood | greedy_dyn_5_fused | — | — | — | — | — |
| k3_bin6-8_splitood | PPO_BCD | 0.000±0.000 | 0.000±0.000 | 0.362±0.006 | 0.331±0.003 | 1.000±0.000 |
| k3_bin6-8_splitood | PPO_BC | — | — | — | — | — |
| k3_bin6-8_splitood | PPO_D | — | — | — | — | — |
| k3_bin6-8_splitood | PPO_C | — | — | — | — | — |
| k3_bin6-8_splitood | PPO_A | — | — | — | — | — |
| k3_bin6-8_splitood | greedy_dyn_2_fused | 0.000±0.000 | 0.000±0.000 | 0.366±0.000 | 0.339±0.000 | 1.000±0.000 |
| k3_bin6-8_splitood | greedy_dyn_5_fused | — | — | — | — | — |
| k3_bin8-10_splitood | PPO_BCD | 0.000±0.000 | 0.000±0.000 | 0.404±0.003 | 0.368±0.004 | 1.000±0.000 |
| k3_bin8-10_splitood | PPO_BC | — | — | — | — | — |
| k3_bin8-10_splitood | PPO_D | — | — | — | — | — |
| k3_bin8-10_splitood | PPO_C | — | — | — | — | — |
| k3_bin8-10_splitood | PPO_A | — | — | — | — | — |
| k3_bin8-10_splitood | greedy_dyn_2_fused | 0.000±0.000 | 0.000±0.000 | 0.419±0.000 | 0.369±0.000 | 1.000±0.000 |
| k3_bin8-10_splitood | greedy_dyn_5_fused | — | — | — | — | — |
| k4_bin8-10_splitood | PPO_BCD | 0.000±0.000 | 0.000±0.000 | 0.404±0.003 | 0.368±0.004 | 1.000±0.000 |
| k4_bin8-10_splitood | PPO_BC | — | — | — | — | — |
| k4_bin8-10_splitood | PPO_D | — | — | — | — | — |
| k4_bin8-10_splitood | PPO_C | — | — | — | — | — |
| k4_bin8-10_splitood | PPO_A | — | — | — | — | — |
| k4_bin8-10_splitood | greedy_dyn_2_fused | 0.000±0.000 | 0.000±0.000 | 0.419±0.000 | 0.369±0.000 | 1.000±0.000 |
| k4_bin8-10_splitood | greedy_dyn_5_fused | — | — | — | — | — |
| k5_bin8-10_splitood | PPO_BCD | 0.000±0.000 | 0.000±0.000 | 0.404±0.003 | 0.368±0.004 | 1.000±0.000 |
| k5_bin8-10_splitood | PPO_BC | — | — | — | — | — |
| k5_bin8-10_splitood | PPO_D | — | — | — | — | — |
| k5_bin8-10_splitood | PPO_C | — | — | — | — | — |
| k5_bin8-10_splitood | PPO_A | — | — | — | — | — |
| k5_bin8-10_splitood | greedy_dyn_2_fused | 0.000±0.000 | 0.000±0.000 | 0.419±0.000 | 0.369±0.000 | 1.000±0.000 |
| k5_bin8-10_splitood | greedy_dyn_5_fused | 0.000±0.000 | 0.000±0.000 | 0.426±0.000 | 0.371±0.000 | 1.000±0.000 |
| k8_bin8-10_splitood | PPO_BCD | 0.000±0.000 | 0.000±0.000 | 0.404±0.003 | 0.368±0.004 | 1.000±0.000 |
| k8_bin8-10_splitood | PPO_BC | — | — | — | — | — |
| k8_bin8-10_splitood | PPO_D | — | — | — | — | — |
| k8_bin8-10_splitood | PPO_C | — | — | — | — | — |
| k8_bin8-10_splitood | PPO_A | — | — | — | — | — |
| k8_bin8-10_splitood | greedy_dyn_2_fused | 0.000±0.000 | 0.000±0.000 | 0.419±0.000 | 0.369±0.000 | 1.000±0.000 |
| k8_bin8-10_splitood | greedy_dyn_5_fused | 0.000±0.000 | 0.000±0.000 | 0.426±0.000 | 0.371±0.000 | 1.000±0.000 |

## Paired-by-seed deltas (4-seed 95% CI) — PPO_BCD vs comparators

| cell | vs PPO_A | vs PPO_B | vs PPO_C | vs greedy_5_F | vs random |
|---|---|---|---|---|---|
| k2_bin6-8_splitood | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +0.665 [+0.599,+0.731] ✅ |
| k2_bin8-10_splitood | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +0.649 [+0.644,+0.653] ✅ |
| k3_bin6-8_splitood | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +0.651 [+0.622,+0.681] ✅ |
| k3_bin8-10_splitood | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +0.720 [+0.697,+0.743] ✅ |
| k4_bin8-10_splitood | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +0.519 [+0.479,+0.558] ✅ |
| k5_bin8-10_splitood | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +0.000 [+0.000,+0.000] — | +0.414 [+0.399,+0.428] ✅ |
| k8_bin8-10_splitood | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +nan [+nan,+nan] — | +0.000 [+0.000,+0.000] — | +0.212 [+0.184,+0.241] ✅ |

## Bucket C — held-out biological validation

**Status: pending_no_local_source.** No held-out source not used in the reward is currently loaded in this evaluator. The Phase 2c Replogle K562 essentials check is available (`artifacts_v3/eval_v3b_phase2c/replogle_norman_intersection.json`) and shows the DepMap safety prior does not transfer to the Replogle assay — a Bucket-C finding consistent with the verdict that Variant C is reward-prior optimisation, not independent biological discovery.

