# V3C Phase 4 — Track L / Track N 4-seed escalation summary

**Status:** `NO_STABLE_SIGNAL` — neither Track L nor Track N produces a multi-seed positive at K=2/bin8-10/OOD vs same-field reward-aware `greedy_dyn_2_fused`. Track L's huge anchor lift (4.8×) survives but PPO can only tie greedy with a final-distance regression that fails the Pareto criterion. Track N's seed-42 500k `CANDIDATE_SIGNAL_RAW` is single-seed variance; the 4-seed paired delta CI includes zero at both 500k and 1M.

**Scope:** Trained 3 additional seeds {0, 1, 7} × {500k for Track N, 1M for Track L and N} of PPO_BCD on Track L (`artifacts_v3/dynamics_n64_legacy_ror_corr010`) and Track N (`artifacts_v3/dynamics_n64_nb_ror_corr010`); evaluated each at the canonical 7-cell V3B matrix (n=200 episodes); aggregated 4-seed paired CIs via `aggregate_v3b_phase4.py`. Locked B+C+D reward stack throughout (`λ_tox=0.10, λ_ce=0.05, λ_unc_path=0.05`, freeband `{3, 5, 0.02, 0.10, 1.0}`, `env.max_steps=8`), per-VAE p15 ε (Track L: 3.0193, Track N: 3.1120).

**Frozen tier check:** `git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/` → clean.

**Test suite:** 392 passed / 2 skipped.

---

## §1 — Headline

| Track | Steps | 4-seed K=2/b8-10 PPO_BCD | greedy_dyn_2_fused | Δ (CI95) | Pareto distance? | Verdict |
|---|---|---:|---:|---:|---:|---|
| Track L | 1M | **0.705 ± 0.000** | 0.695 | **+0.010 [+0.010, +0.010]** (excludes 0) | **regresses +0.173** | `NO_STABLE_SIGNAL` (tie but distance regression) |
| Track N | 500k | 0.499 ± 0.052 | 0.495 | +0.004 [−0.047, +0.055] (includes 0) | mild regression | `NO_STABLE_SIGNAL` (seed-42 +0.075 was variance) |
| Track N | 1M | 0.472 ± 0.097 | 0.495 | −0.023 [−0.117, +0.072] (includes 0) | regresses | `NO_STABLE_SIGNAL` (and worse than 500k) |

**Both PPO checkpoints at K=2/bin6-8/OOD lose to same-field greedy_dyn_2 by ~12-14 percentage points (CI excludes 0 in greedy favor) across all 4 seeds.**

---

## §2 — Verdict assignment

Mapping against the user-stated verdict labels (session brief):

| Label | Met? | Reasoning |
|---|---|---|
| `TRACKN_STABLE_POSITIVE` (Track N 4-seed 1M Δ excludes zero with PPO > greedy) | ❌ | 1M CI = [−0.117, +0.072] — includes 0, mean negative. |
| `TRACKN_EARLY_STOP_SIGNAL` (500k 4-seed Δ positive but 1M not) | ❌ | 500k mean = +0.004, CI = [−0.047, +0.055] — does not exclude 0. The seed-42 500k +0.075 was single-seed variance. |
| `TRACKL_PARETO_OR_TIE` (Track L ties greedy AND improves reward/utility axes AND no final-distance regression) | ❌ | Ties on raw (+0.010), small unc_max improvement (−0.015), **but mean_final_distance regresses +0.173** (> Pareto §4 Stage 4 tolerance of 0.10). The Pareto-distance criterion is the binding fail. |
| **`NO_STABLE_SIGNAL`** | **✅** | **Selected.** Track L is a structural tie (zero-variance +0.010) with a distance regression; Track N's 500k signal is variance. Neither field produces a paired Δ vs same-field reward-aware greedy that survives the Pareto-tolerance check. |

---

## §3 — Track L (1M, 4 seeds) — detailed

### 3.1 Raw success per cell (4-seed mean ± std)

| Cell | PPO_BCD | greedy_dyn_2_fused | greedy_dyn_5_fused | random | Δ(PPO − g_2) | CI95 (excludes 0?) |
|---|---:|---:|---:|---:|---:|---|
| K=2/b6-8 | 0.790 ± 0.060 | 0.910 ± 0.000 | — | 0.025 | **−0.120** | [−0.179, −0.061] **PPO loses** |
| **K=2/b8-10** | **0.705 ± 0.000** | **0.695 ± 0.000** | — | 0.010 | **+0.010** | [+0.010, +0.010] **tie (zero-variance)** |
| K=3/b6-8 | 0.978 ± 0.003 | 1.000 | 0.990 ± 0.000 | 0.165 | (saturated) | tie |
| K=3/b8-10 | 0.940 ± 0.000 | 1.000 | 1.000 | 0.075 | (saturated) | PPO loses by 0.060 (saturated greedy) |
| K=4-K=8 | 1.000 | 1.000 | 1.000 | 0.20–0.52 | (saturated) | tie |

**Per-seed Δ at K=2/b8-10:** `[+0.010, +0.010, +0.010, +0.010]` — identical across all 4 seeds. Per-seed Δ at K=2/b6-8: `[−0.085, −0.090, −0.095, −0.210]` — seed 7 is notably worse but the CI still excludes zero in greedy's favor.

The zero-variance behavior at K=2/b8-10 reflects the small OOD start pool at this cell (~20 cells) combined with a deterministic-greedy policy and PPO's converged action distribution: across 200 episodes that cycle through the same starts, both policies produce identical raw success counts. The +0.010 is real but corresponds to **just 2 additional episodes solved** out of 200 — a structural margin, not a discovered controller advantage.

### 3.2 Pareto axes at K=2/bin8-10/OOD (PPO_BCD vs greedy_dyn_2_fused)

| Axis | PPO_BCD (4-seed mean) | greedy_dyn_2_fused | Δ | Pareto pass? |
|---|---:|---:|---:|---|
| success_rate | 0.705 | 0.695 | **+0.010** | within ±0.03 tolerance: tie ✓ |
| mean_final_distance | **3.090 ± 0.047** | 2.917 | **+0.173** | **FAIL** (> 0.10 regression tolerance) |
| mean_tox_path | 0.000 | 0.000 | 0.000 | tie ✓ |
| mean_common_essential_per_ep | 0.000 | 0.000 | 0.000 | tie ✓ |
| mean_unc_path_max | 0.404 | 0.419 | **−0.015** | improved ✓ |

**Pareto verdict:** Three axes tied / improved (success within tolerance + tox/CE tied + unc improved), one critical axis fails (`mean_final_distance` regresses by 0.173 ≫ 0.10 tolerance). Under the V3C plan §4 Stage 4 `CANDIDATE_SIGNAL_PARETO` definition, this is a **Pareto failure**, not a Pareto win.

The final-distance regression is structural: PPO at K=2/b8-10 spends 2 actions but lands at a slightly larger distance from z_ref than greedy_dyn_2 would. PPO succeeds within ε for the same fraction of starts, but its successful trajectories don't terminate as close to z_ref. Under a tightened ε (e.g. p10 = 2.88), this Pareto failure would convert into a raw-success regression — the field is sitting on the very edge of the ε success criterion at this cell.

### 3.3 Anchor lift preserved

V2 anchor 4-seed PPO_BCD K=2/b8-10 = `0.148 ± 0.037` (V3B Phase 4 doc). Track L 4-seed PPO_BCD K=2/b8-10 = `0.705 ± 0.000`. **Lift: +0.557 (4.8×) over anchor.** This validates the Phase 0C Bucket U-B audit prediction (0.560 vs 0.120 reach at p15). The lift is from the 64D representation + RoR_corr010 architecture, not from PPO's planning.

But the lift does NOT compound into a `LOCKED_DESIGN_POSITIVE_SIGNAL` because **same-field reward-aware greedy_dyn_2 also receives the lift** — greedy_dyn_2 jumps from 0.130 (anchor) to 0.695 (Track L). PPO has no additional planning room left.

### 3.4 Phase 1 → Phase 4 consistency

Phase 1 seed-42 K=2/b8-10 = 0.705. Phase 4 4-seed mean = 0.705 ± 0.000. The seed-42 Phase 1 result was representative; multi-seed simply confirms the result is stable (and modestly winning) at a non-saturated cell, but the Pareto distance failure also replicates across seeds.

---

## §4 — Track N (500k and 1M, 4 seeds) — detailed

### 4.1 500k raw success per cell

| Cell | PPO_BCD (4-seed) | greedy_dyn_2_fused | Δ(PPO − g_2) | CI95 (excludes 0?) |
|---|---:|---:|---:|---|
| K=2/b6-8 | 0.759 ± 0.023 | 0.890 | −0.131 | [−0.154, −0.109] **PPO loses** |
| **K=2/b8-10** | **0.499 ± 0.052** | 0.495 | **+0.004** | **[−0.047, +0.055] includes 0** |
| K=3/b6-8 | 0.957 ± 0.004 | 0.995 | −0.038 | saturated greedy |
| K=3/b8-10 | 0.943 ± 0.003 | 1.000 | −0.057 | saturated greedy |
| K=4–K=8 | 1.000 | 1.000 | 0.000 | saturated |

**Per-seed Δ at K=2/b8-10:** `[+0.075, −0.005, −0.050, −0.005]` (seeds 42, 0, 1, 7).

**The seed-42 +0.075 from Phase 1 is a single-seed outlier.** Three of four seeds put PPO at or slightly below greedy. The 4-seed paired CI is `[−0.047, +0.055]`, which **includes zero**. Track N at 500k does not have a stable PPO planning advantage; the Phase 1 `CANDIDATE_SIGNAL_RAW` is single-seed variance.

### 4.2 1M raw success per cell

| Cell | PPO_BCD (4-seed) | greedy_dyn_2_fused | Δ(PPO − g_2) | CI95 (excludes 0?) |
|---|---:|---:|---:|---|
| K=2/b6-8 | 0.749 ± 0.019 | 0.890 | −0.141 | [−0.160, −0.122] **PPO loses** |
| **K=2/b8-10** | **0.472 ± 0.097** | 0.495 | **−0.023** | **[−0.117, +0.072] includes 0** |
| K=3/b6-8 | 0.943 ± 0.009 | 0.995 | −0.052 | saturated greedy |
| K=3/b8-10 | 0.908 ± 0.029 | 1.000 | −0.092 | saturated greedy |
| K=4–K=8 | 1.000 | 1.000 | 0.000 | saturated |

**Per-seed Δ at K=2/b8-10:** `[−0.050, −0.145, +0.030, +0.075]` (seeds 42, 0, 1, 7).

At 1M training, the cross-seed variance is much higher (std 0.097 vs 500k's 0.052), with mean moving slightly below greedy. **1M is no better than 500k** on Track N; the Phase 1 finding that doubling timesteps degraded the seed-42 K=2/b8-10 generalizes across seeds.

### 4.3 500k-vs-1M tension resolved

Phase 1 documented PPO_BCD K=2/b8-10 = 0.570 (500k) → 0.445 (1M) for seed 42 — a 0.125 regression with more training. With 4 seeds:

| | 500k 4-seed mean | 1M 4-seed mean | Δ(1M − 500k) |
|---|---:|---:|---:|
| K=2/b8-10 PPO_BCD | 0.499 | 0.472 | −0.027 |
| K=2/b6-8 PPO_BCD | 0.759 | 0.749 | −0.010 |
| K=3/b6-8 PPO_BCD | 0.957 | 0.943 | −0.014 |
| K=3/b8-10 PPO_BCD | 0.943 | 0.908 | −0.035 |

**Across all non-saturated cells, 1M is consistently slightly worse than 500k on Track N.** This is a real training-horizon non-monotonicity at the per-seed level, but its magnitude (~0.03) is small compared to between-seed variance (~0.05–0.10). The Phase 1 seed-42 0.125 regression was the high-variance tail of this pattern, not a special effect.

---

## §5 — Why neither track yielded a positive signal

The Phase 0C audit's central diagnosis explains both outcomes:

1. **The 64D representation lift (Track L / Track N over anchor at K=2/bin8-10) is a *reachability* lift, not a *planning* lift.** Phase 0C measured `beam_reach@p15 = 0.560` (Track L) vs `0.120` (anchor) at this cell — the 4.7× expansion of the actionable region. But the audit also showed `greedy_dyn_K_distance_success` saturates at K≥3, meaning **greedy itself receives the lift** and PPO has no additional depth to exploit.
2. **The universal-attractor structure (contraction_fraction ≈ 1.0, gene_universality_max ≈ 0.93) caps reward-aware steering.** With near-every (z, g) pair pointing toward z_ref, the fused beam (using B+C+D shaping) and the distance-only beam converge on similar actions — there is no choice for the reward shaping to discriminate.
3. **The Pareto-distance regression at Track L K=2/b8-10 (+0.173) is a sign of the ε-edge.** PPO's policy is *almost* greedy at this cell but commits to actions that land slightly further from z_ref; under the ε success criterion, this still counts as +0.010 raw success but signals the policy is on a different (slightly looser) trajectory bundle. Phase 1 showed the same +0.222 regression at seed 42; Phase 4 confirms this is stable across seeds.

The Phase 0C audit prior estimated *moderate* Phase 2 trigger probability — multi-seed confirms that estimate. The audit's K=2/b8-10 reach gap was promising (0.56 vs 0.12), but the surrounding `gu_max ≈ 0.93` + `cf ≈ 1.0` saturation prevented PPO from amplifying it.

---

## §6 — Cross-reference with Phase 2 (`contraction_aware_v1`)

The Phase 4 conclusion that **PPO has no planning room over greedy on the existing Track L/N dynamics** strengthens the case for Phase 2 dynamics reformulation. The Phase 2 v1 candidate (see `v3c_phase2_contraction_aware_summary.md`) confirmed the mechanism — a regularizer that targets `gene_universality_max` does move the geometry in the predicted direction (gu_max: 0.933 → 0.905) without hurting prediction — but its conservative coefficient regime produced **`PHASE2_DIAGNOSTIC_ONLY`**: the move was too small to unlock new control utility.

**Combined Phase 4 + Phase 2 verdict:** the V3B reward stack is correct, the Track L/N 64D-representation lift is real but capped by the universal-attractor structure, and a conservative contraction-aware regularizer is too mild to break the cap. The next architectural lever is **a more aggressive contraction-aware variant (Phase 2.5)** or **an orthogonal mechanism** (ensemble-disagreement uncertainty, SCANVI / ZINB representation).

---

## §7 — Final recommendation

**Do NOT run reward-coefficient tuning.** Per the V3C plan §6 gating: reward tuning is unlocked only after a robust multi-seed positive or a strong Phase 2 candidate. Neither is in hand. Tuning B+C+D on these dynamics would search a flat landscape.

**Recommended next session:**

1. **Phase 2.5 — more aggressive contraction-aware variants** (`v2_aggressive` with τ=0.60, λ=0.10; optionally `v3_diverse` adding action-diversity floor). Cost: ~5 min train + ~15 min audit per variant. If any variant produces `PHASE2_MODERATE_UTILITY` or better → PPO smoke seed 42.
2. **Phase 3 — ensemble-disagreement dynamics** (V3.fallback.C) as an orthogonal axis. The single-head heteroscedastic uncertainty is *state-dependent but not action-dependent* — V3B Phase 4 finding #2 in PROGRESS.md. An ensemble of 3-5 dynamics models would produce action-discriminating uncertainty that the Variant D reward can actually exploit. Cost: ~30 min for ensemble training + audit.
3. **Phase 4 — SCANVI 32D / ZINB 64D representation reformulations** (V3 stub axis A) if Phase 2.5 + Phase 3 do not unlock signal. This is the slowest path (full VAE retrain) and should be the last resort.

**Stop / pivot conditions:** if Phase 2.5 + Phase 3 both produce `PHASE2_DIAGNOSTIC_ONLY` or worse, recommend pivoting to **either** (a) the SCANVI/ZINB axis, **or** (b) acknowledging the structural ceiling and re-scoping V3 to focus on **explanation** (audit-driven characterization of the universal-attractor failure mode) rather than headline-positive control results.

---

## §8 — Files written

- `artifacts_v3/v3c/rl_final/track_l_4seed_locked/{seed0_1M, seed1_1M, seed7_1M}/{ppo.zip, metadata.json, eval/, rollouts.parquet, action_freq.json, success_curves.png}` — 3 new PPO checkpoints + per-seed evals.
- `artifacts_v3/v3c/rl_final/track_l_4seed_locked/eval/seed42` → symlink to Phase 1 seed-42 eval (reused).
- `artifacts_v3/v3c/rl_final/track_l_4seed_locked/agg/{reward_stack_results.json, reward_stack_results.csv, reward_stack_summary.md}` — 4-seed aggregate.
- `artifacts_v3/v3c/rl_final/track_n_4seed_locked/{seed0_500k, seed0_1M, seed1_500k, seed1_1M, seed7_500k, seed7_1M}/{ppo.zip, metadata.json, eval/, ...}` — 6 new PPO checkpoints + per-seed evals.
- `artifacts_v3/v3c/rl_final/track_n_4seed_locked/eval_500k/seed42` and `eval_1M/seed42` → symlinks to Phase 1 (reused).
- `artifacts_v3/v3c/rl_final/track_n_4seed_locked/agg_500k/{...}` and `agg_1M/{...}` — 4-seed aggregates for both checkpoint horizons.
- `artifacts_v3/v3c/interpretation/v3c_phase4_track_ln_escalation.md` — **this document**.

---

## §9 — Sacred-rule conformance

- ✅ Frozen tiers untouched.
- ✅ All Track A outputs under `artifacts_v3/v3c/rl_final/`.
- ✅ Per-VAE p15 ε (Track L: 3.0193, Track N: 3.1120) — identical to Phase 1.
- ✅ Locked B+C+D coefficients throughout; no reward tuning.
- ✅ Seed 42 evals **reused** from Phase 1 via symlink — no recomputation, no overwriting.
- ✅ PPO vs same-field same-horizon `greedy_dyn_K` as primary comparison; anchor delta as secondary context.
- ✅ Wilson-95 success-rate CIs + paired-by-seed Δ CIs reported.
- ✅ K=5 not used as a hard rejection criterion (Track L/N have saturated K=5 reach in the audit).
