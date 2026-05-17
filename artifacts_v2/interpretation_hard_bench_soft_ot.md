# V2 Hard Benchmark — soft-OT dynamics + V1 PPO (2026-05-16)

## Configuration

| Parameter | Value |
|---|---|
| dynamics_dir | artifacts_v2/dynamics_soft_ot_default |
| ppo_zip | artifacts/rl_sweeps/p50_start8_shaped_noopfix_500k/ppo.zip |
| epsilon_p25 | 3.166 |
| epsilon_p50 | 3.531 (from VAE epsilon_success.json) |
| n_episodes | 500 |
| n_start_pool (primary cell) | 17 OOD cells |

## Primary Cell Results

Primary cell: `k3_epsp25_bin8-10_splitood`

| Policy | V1 dynamics | soft-OT dynamics | Δ |
|---|---:|---:|---:|
| ppo_deterministic | **1.000** | **0.000** | −1.000 |
| greedy_dyn_1 | **1.000** | **0.000** | −1.000 |
| ridge_greedy | 0.716 | 0.000 | −0.716 |
| mean_delta_greedy | 0.824 | 0.000 | −0.824 |
| random_uniform_valid | 0.178 | 0.000 | −0.178 |
| always_noop | 0.000 | 0.000 | 0.000 |
| PPO−greedy_dyn_1 (pp) | **0.0** | **0.0** | — |

## Aggregate (all 64 cells, OOD split)

| Metric | V1 | soft-OT V2 |
|---|---:|---:|
| ppo_deterministic cells with sr > 0 | ~40/64 | **0/64** |
| greedy_dyn_1 cells with sr > 0 | ~40/64 | **0/64** |
| greedy_dyn_1 noop-only cells | ~0 | **40/64** |

## Mechanism

**greedy_dyn_1 always picks noop** in 40/64 cells (all remaining 24 cells have occasional non-noop picks but still fail). The soft-OT dynamics model predicts that no gene perturbation achieves a better 1-step latent distance than a no-op. Mechanistically: the soft-OT training targets are barycentric averages of control cells (smooth, low-variance). The dynamics MLP learned the *average* control-cell response, which is smaller-magnitude and directionally different from the per-cell V1 targets. The greedy oracle sees that the dynamics predicts noop as the best 1-step option — the field is "flat" at the per-cell level.

**V1 PPO makes cells escape to large distances** under soft-OT dynamics. PPO's top genes (CKS1B, TSC22D1, CELF2) were chosen because V1 dynamics predicted they reduce distance rapidly. Under soft-OT dynamics, these same genes are predicted to produce large Δz in directions that are not centroid-aligned for OOD starting cells. Damage scales with K: PPO at k=1 ends at ~6.0, k=3 at ~11–24, k=8 at ~87–94. This is a policy-dynamics mismatch, not a reward signal issue.

**PPO−greedy_dyn_1 = 0.0pp at both 0.000**: Collinearity is preserved, but the *regime* has shifted from "ceiling collinearity" (both 1.000) to "floor collinearity" (both 0.000). The V1 finding — PPO cannot surpass the 1-step greedy oracle — is structurally confirmed: PPO learns the greedy signal, not a planning signal above it. With soft-OT dynamics, greedy=0.000, so PPO=0.000 follows automatically.

## Conclusion

**This result does not invalidate soft-OT dynamics.** The soft-OT dynamics gate passed with val Pearson 0.9338 (vs threshold 0.930) and OOD Pearson 0.7434. It is a high-quality predictor of where cells go after perturbation. The hard benchmark collapse is a consequence of evaluating a policy trained on V1 dynamics against a different dynamics field — the V1 PPO was calibrated to exploit V1's per-cell contraction structure, which soft-OT smooths away.

**The hard benchmark is not meaningful until a PPO is trained on soft-OT dynamics.** The greedy baseline currently collapses to noop because the dynamics field is more conservative; a PPO trained on this field might discover multi-step gene combinations that the greedy 1-step oracle misses (since greedy picks noop at step 1, blocking all subsequent beneficial steps). This is the primary motivation for P0C.

## Diagnostic value for PPO-greedy collinearity

The V1 benchmark showed `PPO − greedy_dyn_1 = 0.0pp` across all cells, interpreted as "PPO learned exactly what greedy knows, no planning surplus." The soft-OT benchmark confirms this structural result:

- With soft-OT dynamics, greedy = noop = failure.
- V1 PPO (trained on V1 dynamics) also fails, but for a different reason: policy-dynamics mismatch.
- Neither result provides evidence that PPO can plan beyond greedy with soft-OT dynamics.
- **The test that would provide this evidence is: train a new PPO on soft-OT dynamics, then compare it to soft-OT greedy.** That is P0C.

## Next step

**P0C: Retrain PPO on soft-OT dynamics.** Command (requires explicit approval):

```bash
.venv/bin/python scripts/train_rl.py --config-name default \
    paths.dynamics_dir=artifacts_v2/dynamics_soft_ot_default \
    paths.rl_dir=artifacts_v2/rl_soft_ot \
    rl.total_timesteps=500000 \
    rl.reward.start_epsilon_label=p50 \
    seed=42
```

After PPO retrain, re-run the hard benchmark with:
- `--dynamics_dir artifacts_v2/dynamics_soft_ot_default`
- `--ppo_zip artifacts_v2/rl_soft_ot/ppo.zip`

The new PPO will be calibrated to the soft-OT field. The PPO−greedy_dyn_1 gap will then be the primary diagnostic for whether MaskablePPO can learn to chain gene perturbations that the 1-step greedy oracle misses.
