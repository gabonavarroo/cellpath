# Reachability probe summary

epsilon_p25=3.1663, depth=3, beam=50, n_starts=17

| run_key | repeat_mask | n_success | success_rate | best_final_distance | mean_best |
|---|---|---:|---:|---:|---:|
| v1_ot_repeaton | True | 17 | 1.000 | 1.5932 | 2.0378 |
| v1_ot_repeatoff | False | 17 | 1.000 | 1.5932 | 2.0374 |
| mean_delta_repeaton | True | 0 | 0.000 | 4.1139 | 5.3371 |
| mean_delta_repeatoff | False | 0 | 0.000 | 4.1139 | 5.3133 |
| soft_ot_repeaton | True | 0 | 0.000 | 16.9738 | 22.1946 |
| soft_ot_repeatoff | False | 0 | 0.000 | 16.9738 | 22.0074 |

## D4: ε-feasibility (repeat_mask=True runs only)

epsilon_p25 = 3.1663, epsilon_p50 = 3.5311

| dynamics | ε for 10% success | ε for 25% success | ε for 50% success | ε for 75% success |
|---|---:|---:|---:|---:|
| v1_ot | 1.752 | 1.910 | 2.051 | 2.085 |
| mean_delta | 4.323 | 5.016 | 5.501 | 5.792 |
| soft_ot | 18.123 | 18.679 | 20.352 | 24.277 |

Notes:
- v1_ot: all 17 cells succeed under current epsilon_p25 (3.17) with k=3 steps
- mean_delta: 3-step beam needs ε≈5.0 for 25% success; however greedy_dyn_1 achieves
  mean_final_dist=5.49 in k=3 (from start ~8.48), implying k=8 RL would likely reach epsilon_p25
- soft_ot: ε≈18.7 required for 25% success — completely infeasible under any reasonable epsilon
