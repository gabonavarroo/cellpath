# V3 Controller Objective Spec — Locked Reward Stack

> **Status**: locked design as of 2026-05-19 (V3B Phase 4). Implementation validated technically
> on V2 primary 32D `RoR_corr010` dynamics. **Not empirically optimal yet** — the V2 dynamics field
> is saturated at K≥4 cells, so a planning-advantage headline is not currently achievable. Future
> dynamics fields must be evaluated with this same locked stack for apples-to-apples comparison.

---

## 1. Purpose

This spec defines the canonical V3 controller objective: the reward stack, environment
configuration, baselines, success criteria, and reporting structure. Subsequent V3C+ work
(new dynamics fields, alternative latents, contraction-regulariser variants, etc.) **must**
evaluate against this locked stack to maintain comparability.

---

## 2. Locked controller architecture

* **Algorithm**: `MaskablePPO` from `sb3-contrib` (no change from V2).
* **Action space**: discrete, `Discrete(n_genes + 1)`. Final index is NO-OP / terminate. Repeat-mask enabled.
* **Observation**: latent vector `z ∈ R^(n_latent)`. No step counter, no mask in obs.
* **Policy network**: `MlpPolicy` with `policy_kwargs.net_arch=[128,128]`, `tanh` activation.
* **PPO hparams**: lr=3e-4, n_steps=1024, batch=256, n_epochs=10, γ=0.99, GAE λ=0.95, clip=0.2,
  ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5. (V2-equivalent.)
* **Total timesteps**: 1 000 000 (V2 primary); 500 000 for smoke/calibration only.
* **Curriculum**: V2 distance-bin curriculum (start_d=4.0 → end_d=10.0 over first 70% of training)
  remains enabled when applicable.
* **Device**: CPU for PPO rollouts; dynamics on CPU; (MPS/CUDA for VAE/dynamics training only).

---

## 3. Available reward modes

All reward modes are mid-episode-zero unless otherwise noted; reward is emitted only at
`terminated` or `truncated`.

| Mode key | Variant | Status | Default? |
|---|---|---|---|
| `absolute_distance` | V1 baseline | V2 byte-identical | NO |
| `delta_distance` | V1 ablation | V2 byte-identical | NO |
| `terminal_only_step_cost` | V2 primary baseline | V2 byte-identical | YES (config/rl.yaml default) |
| `hybrid_delta_terminal` | V2 ablation | V2 byte-identical | NO |
| `safety_aware` | C (V3B Phase 2) | Locked | NO |
| `path_length_freeband` | B (V3B Phase 3) | Locked | NO |
| `safety_path_freeband` | B+C (V3B Phase 4) | Locked | NO |
| `uncertainty_aware` | D (V3B Phase 4) | Locked | NO |
| `biorealistic_fused` | B+C+D (V3B Phase 4) | **Locked — recommended default for FUTURE dynamics** | NO |
| `multi_objective` | Alias of `biorealistic_fused` | — | NO |

### 3.1 Recommended default for FUTURE dynamics fields

`biorealistic_fused` (B+C+D). Formula:

    R_T = success_bonus · 1[is_success]
          − path_penalty(T)                  (Variant B free-band schedule)
          − λ_tox · tox_path                  (Variant C, DepMap K562 Chronos)
          − λ_ce  · common_essential_count    (Variant C, Chronos < −0.5 flag)
          − λ_unc · unc_path_max              (Variant D, learned dynamics' log σ²)

Mid-episode reward = 0 (matches V2 protocol).

### 3.2 Required reward parameters (default values; tune per cell, never per result)

| Parameter | Default | Where |
|---|---|---|
| `reward.success_bonus` | 1.0 | `config/rl.yaml` |
| `reward.freeband.free_steps` | 3 | Anchored at Norman 2019 K=2 measured + 1 plausible step |
| `reward.freeband.mild_until` | 5 | Plausible-extension band end |
| `reward.freeband.mild_beta` | 0.02 | Per-step penalty in [free_steps+1, mild_until] |
| `reward.freeband.heavy_beta` | 0.10 | Per-step penalty beyond mild_until |
| `reward.lambda_tox` | 0.10 | Coefficient on Σ tox_raw(g) for non-NOOP actions |
| `reward.lambda_ce` | 0.05 | Coefficient on common_essential_count |
| `reward.lambda_unc_path` | 0.05 | Coefficient on path-max σ-scalar from dynamics |
| `reward.uncertainty_reduce` | "mean_sigma" | Per-step reduction of dynamics log σ² → scalar |
| `reward.uncertainty_clip_min` | -5.0 | Clamps log σ² for stability |
| `reward.uncertainty_clip_max` | 3.0 | Same |
| `reward.safety_table_path` | `artifacts_v3/v3b_biology/gene_safety.parquet` | Loaded by env when safety variant active |
| `reward.permute_chronos` | false | Null-control switch (V3B Phase 2c) |
| `reward.epsilon_label` / `rl.env.epsilon_override` | p25 (default = 3.1663) / 2.9898 for p15 / 2.8846 for p10 / 2.7362 for p5 | Set explicitly per experiment |

### 3.3 Recommended ablation modes (sequential ablation order)

When evaluating a new dynamics field with this stack, run rewards in this order:

1. **C alone** (`safety_aware`) — verify reward-prior optimisation works (Bucket A).
2. **B alone** (`path_length_freeband`) — verify path-length leverage signal (Bucket B at K≥4).
3. **D alone** (`uncertainty_aware`) — verify σ signal is load-bearing (Bucket A unc_max).
4. **B+C** (`safety_path_freeband`) — first conjunction; no D.
5. **B+C+D** (`biorealistic_fused`) — full stack.

Never start with the full stack — the V2-trap is "I added 4 things and it works, which one
mattered?". Sequential ablation is the contract.

---

## 4. Environment configuration

* **Dynamics model**: locked to V2 primary `artifacts_v2/dynamics_v1ot_ror_corr010/model.pt` until
  a new dynamics is validated.
* **Reference state**: `z_reference_centroid.npy` (unperturbed-K562 NT centroid in scVI latent).
* **Episode max_steps**: 3 (V2 protocol) for short-horizon studies; **8** for path-length variants
  (B / B+C / B+C+D) — necessary to allow the freeband mild band (K=4, 5) to be reachable.
* **Distance metric**: L2.
* **Start state**: `random_perturbation` from `_build_start_pool` (filters by `pert_idx != 0`;
  no gene-split filter — see Phase 2b leakage audit).
* **Start-pool min distance**: 4.0 (curriculum-compatible; raised to 10.0 by curriculum).

### 4.1 Epsilon selection

* **p25** (`3.1663`): V2 reference, always reported for continuity.
* **p15** (`2.9898`): **selected as the V3B Phase 4 calibration epsilon**.
  - At p10, PPO_BCD collapses to 0.000 at K=2/bin8-10/OOD — violates the "do not use p5/p10 if
    severe collapse" rule.
  - p15 preserves all PPOs ≥ 0.13 at every cell while un-saturating K=2 cells.
* **p10** (`2.8846`): Reportable as a stricter reference, NOT for training (collapse risk).
* **p5** (`2.7362`): Never as default; only when explicitly justified.

Always report the exact ε value alongside the percentile label.

---

## 5. Required baselines (must appear in every evaluation table)

| Baseline | Purpose |
|---|---|
| `random_uniform_valid` | Floor — must be clearly lower than any controller. |
| `always_noop` | Sanity — must be ≤ 0.05 success at any cell. |
| `greedy_dyn_1_fused` | Single-step lookahead under the reward objective. |
| `greedy_dyn_2_fused` | V2-equivalent depth-2 oracle, reward-aware. |
| `greedy_dyn_3_fused` | Depth-3 oracle. |
| `greedy_dyn_5_fused` | Depth-5 oracle (only meaningful at K ≥ 5 cells). |
| `greedy_dyn_8_fused` | Depth-8 oracle (only at K = 8 cell). |
| `PPO_A` | V2 primary frozen `rl_v1ot_ror_corr010_terminal_curric_k3_1M_seed42`. |
| `PPO_A_max8` | Future V3C control: same `terminal_only_step_cost` reward as PPO_A, but retrained with the same `max_steps=8` setup as V3B rewards. |
| `PPO_B`, `PPO_C`, `PPO_BC`, `PPO_D`, `PPO_BCD` | V3B Phase 4 retrains (one per reward variant). |

Greedy baselines MUST be evaluated **under the same reward objective** as the PPO being
compared (`greedy_dyn_K_fused` in the table above uses the fused reward). Distance-only greedy
is an OPTIONAL contrast baseline but never the primary comparator.

### 5.1 Future V3C horizon-control baseline

V3C evaluations must include `PPO_A_max8` before interpreting any reward-stack advantage. This
control uses the `terminal_only_step_cost` reward, the same `max_steps=8` training setup as the
V3B reward variants, and the same epsilon/curriculum settings where appropriate.

Purpose: isolate reward semantics from training-horizon effects. Phase 4 found that the
"freeband K=2 advantage" may come from `max_steps=8` and the resulting training distribution,
not from B/C/D reward semantics.

---

## 6. Required reporting — Bucket A / B / C separation (mandatory)

### 6.1 Bucket B — reward-independent control metrics

Report for every (policy, cell, seed):

* `success_rate` (raw, n=300 episodes).
* `mean_steps` and step-distribution histogram.
* `mean_final_distance`.
* Path-length usage: `frac_success_T_le_3`, `frac_success_T_4_or_5`, `frac_success_T_gt_5`.
* Paired-by-seed deltas: `PPO_X − PPO_A`, `PPO_X − greedy_dyn_2_fused`, `PPO_X − greedy_dyn_5_fused`,
  `PPO_X − random` with 4-seed 95 % CIs.

### 6.2 Bucket A — reward-fit metrics (NEVER independent biological validation)

* `mean_tox_path` (Chronos-derived; NOT independent biology).
* `mean_common_essential_per_ep`.
* `fraction_zero_common_essential`.
* `mean_unc_path_max`, `mean_unc_path_mean`.
* `weighted_mean_chronos` from action_freq (when available).

Bucket-A wins are **reward-prior optimization**, not biological discovery. The same reward
sources are in the policy's training reward.

### 6.3 Bucket C — held-out biological validation (optional but recommended)

* Replogle 2022 K562 essential CRISPRi (parsed via Harmonizome — see
  `artifacts_v3/eval_v3b_phase2c/replogle_norman_intersection.json`).
* OGEE v3 / COSMIC CGC / Open Targets — when locally available.

Bucket-C sources must **never overlap** with the reward layer. A Bucket-A "improvement" that
doesn't replicate in Bucket-C is reward-fitting, not generalisation.

---

## 7. Reporting cells (V3B hardness matrix)

| Cell | Notes |
|---|---|
| K=2 / bin 6-8 / OOD | Cleanest non-saturated diagnostic cell at p15. Greedy ~0.65; PPOs show a usable spread. |
| K=2 / bin 8-10 / OOD | Best PPO improvement cell over PPO_A at p15 (+9.1 pp for PPO_B/PPO_D/PPO_BCD), but small pool (17 cells) and no path-length leverage band. |
| K=3 / bin 6-8 / OOD | Saturated at p25; partial spread at p10. |
| K=3 / bin 8-10 / OOD | V2 primary cell (saturated at p25 for V2 dynamics). |
| K=4 / bin 8-10 / OOD | Saturated on V2 dynamics; main test cell for any new dynamics. |
| K=5 / bin 8-10 / OOD | Saturated on V2 dynamics. |
| K=8 / bin 8-10 / OOD | Saturated on V2 dynamics; speculative-band cell. |

When evaluating a new dynamics field, this 7-cell matrix is the canonical test set.

---

## 8. Mandatory caveats (exact wording for V3 reports)

1. **"Not empirically optimal yet."** The locked reward stack is implemented and evaluable
   end-to-end; its empirical performance depends on the underlying dynamics field.
2. **"Validated technically on V2 dynamics."** All reward modes pass unit tests, dispatch
   correctly through env + greedy, and produce numerically-bounded outputs at every cell × seed.
3. **"Future dynamics fields must be evaluated with this same stack."** Otherwise comparisons
   to V3B are meaningless.
4. **"Chronos reward-fit is NOT independent biological validation."** The DepMap K562 Chronos
   labels appear in the reward and in some plausibility-test labels; they cannot be both.
5. **"V2 dynamics is saturated; reward stack is locked as controller design, not as V2 headline."**
   On V2 primary 32D `RoR_corr010`, greedy_dyn_2 saturates at every K≥4 cell at every ε ∈
   {p25, p15, p10, p5}. No reward shaping recovers a planning-advantage headline on this field.

---

## 9. Required interpretation labels

Each new V3 evaluation must conclude with one of:

| Verdict | Meaning |
|---|---|
| `LOCKED_DESIGN_POSITIVE_SIGNAL` | At ≥ 1 non-saturated cell, PPO_BCD − reward-aware-greedy_K with 4-seed CI excluding zero in PPO_BCD's favor. |
| `LOCKED_DESIGN_TECHNICAL_ONLY` | All variants implement & evaluate correctly; no Bucket-B planning advantage. (Expected on V2.) |
| `LOCKED_DESIGN_FAILED_IMPLEMENTATION` | Catastrophic regression vs PPO_A (Δ success > 0.30) on any cell — implementation bug. |

**Do NOT** claim:
* "B+C+D works" without 4-seed CI vs reward-aware greedy.
* "Biological discovery" — never on reward-fit metrics alone.
* "Safer biology independently validated" — only with Bucket-C support.

---

## 10. Code modules (locked file list)

| File | Role |
|---|---|
| `src/rl/biology_rewards.py` | All V3B reward helpers (safety, freeband, uncertainty, fused). |
| `src/rl/reward.py` | `compute_reward` dispatch; preserves V2 modes byte-identical. |
| `src/rl/environment.py` | `CellReprogrammingEnv` with accumulators for tox/CE/unc. |
| `src/rl/baselines.py::GreedyDynamicsBeamPolicy` | Reward-aware beam scoring across all 3 axes. |
| `src/analysis/path_feasibility.py` | Biology layer loader (gene_safety, sl_pairs). |
| `scripts/build_v3b_biology_layer.py` | Phase 0 driver (one-time biology table build). |
| `scripts/train_rl_v3b.py` | Trainer wrapper with `--mode {safety_aware, path_length_freeband, safety_path_freeband, uncertainty_aware, biorealistic_fused}`. |
| `scripts/evaluate_rl_v3b_phase3.py` | Phase 3 evaluator (single PPO, distance/freeband modes). |
| `scripts/evaluate_rl_v3b_phase4.py` | Phase 4 evaluator (multi-PPO under fused env). |
| `scripts/aggregate_v3b_phase4.py` | 4-seed aggregation + verdict derivation. |
| `tests/test_freeband_reward.py`, `tests/test_fused_rewards.py`, `tests/test_biology_rewards.py`, `tests/test_path_feasibility.py` | Unit-test gates that keep V2 byte-identical + V3B reductions verified. |
| `config/rl.yaml` | Single source of truth for reward parameters. |

**No file in `artifacts/`, `artifacts_64/`, `artifacts_v2/`, `artifacts/rl_sweeps/` is modified
by V3B work.** All new outputs live under `artifacts_v3/`.

---

## 11. Next phase (recommended)

**V3C — Dynamics / representation reformulation.**

The reward axis is closed on V2. The bottleneck is the dynamics field's single-step contraction
toward `z_ref`, which makes K≥4 cells saturate at every reasonable controller. Candidates:

1. **V3A Track N safety pre-check** — Track N (64D NB) VAE finished; pairs build → RoR dynamics
   → reachability oracle → greedy-saturation check at p25 / p15. If greedy_dyn_2 < 0.95 at any
   K≥4 cell, re-run Phase 4 reward stack there.
2. **V3.3 / V3.4 — ZINB / SCANVI**. Different latent geometry; possibly less-saturated
   dynamics.
3. **V3.fallback.B — Contraction-regulariser dynamics training**. Add a contraction-rate
   penalty to the dynamics loss to deliberately break the "locally well-conditioned" property
   that causes saturation.

Whichever new dynamics is built, **evaluate with the locked reward stack from this spec.**

---

## 12. Version history

* **v1.0 (2026-05-19)** — Initial lock. Verdict `LOCKED_DESIGN_TECHNICAL_ONLY` on V2 primary
  RoR_corr010 32D dynamics at ε = p15. All 11 reward modes implemented and unit-tested
  (356 passed / 2 skipped). 12 PPOs trained at 1M timesteps; 4-seed × 7-cell evaluation matrix
  complete. Phase 4 final report at
  `artifacts_v3/interpretation/v3b_reward_stack_lock.md`.
