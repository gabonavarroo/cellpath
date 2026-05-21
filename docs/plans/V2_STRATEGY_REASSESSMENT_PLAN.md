# V2_STRATEGY_REASSESSMENT_PLAN.md — strategy plan (do not implement yet)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` only when this
> strategy is converted to an implementation phase. The current document is a **strategy
> reassessment**, not an implementation plan. When implementation begins, Task 1 will commit a
> verbatim copy of this file to `/Users/gabo/Developer/ITAM/IA/cellpath/V2_STRATEGY_REASSESSMENT_PLAN.md`.

---

## 1. Context

CellPath V2 has produced enough evidence to require a strategy reassessment before any further
training. The chain of phases (P0A → P0B′ → P0B″ → P0C0 → P0B2) has converged on a clear
diagnostic finding that the original Phase-2 dynamics gate is **necessary but not sufficient**
for end-to-end RL success, and that two of the three V2 dynamics fields trained have failed
end-to-end testing for orthogonal reasons:

* The supervised gate metric (`val margin_vs_linear_ridge_pearson ≥ +0.030`) is **decorrelated
  from RL controllability**: soft-OT passed it cleanly (+0.0413) and is control-hostile (beam
  best_dist = 16.97, fraction_positive = 0.000); V1 OT failed it (+0.0074) and is fully
  controllable (beam 17/17, PPO success 1.000 at the V2-hard primary cell).
* Mean-delta dynamics is directionally correct (fraction_positive = 0.826) but does **not** reach
  ε under k = 3 (beam best_dist = 4.11 vs ε_p25 = 3.17). Correlation loss ∈ {0.05, 0.10, 0.30}
  pushed the val margin from +0.0214 to +0.0232 — monotonic, saturating, and never reaching
  +0.030. The remaining bottleneck is OOD dim-11 (ridge Pearson = 0.310 across all λ; MLP
  Pearson = 0.06 across all λ — the dim-11 ceiling is a property of the pairing, not the loss).

The user's directive: choose the path most likely to produce a **strong, honest, end-to-end V2
result**, not the path that strictly optimizes the supervised gate first. Sacred rules remain
in force (no VAE retrain, no gate-threshold lowering, no destructive writes to V1 artifacts,
all V2 outputs under `artifacts_v2/`).

---

## 2. Evidence synthesis (verified from artifacts and gate.json files)

### 2.1 Dynamics inventory snapshot

| Dynamics | Val margin | OOD margin | OOD unc. Spearman | Beam 17 OOD (k=3) | Best dist | fraction_positive |
|---|---:|---:|---:|---:|---:|---:|
| `artifacts/dynamics` (V1 OT) | +0.0074 ❌ | +0.0401 ✅ | 0.204 ✅ | **17/17 ✅** | **1.59 ✅** | **~0.955** |
| `…/dynamics_mean_delta_default` | +0.0214 ❌ | +0.1095* ✅ | ~0.20 | 0/17 ❌ | 4.11 (1.0 step est.) | 0.826 |
| `…/dynamics_mean_delta_corr_030` | +0.0232 ❌ | +0.1137 ✅ | 0.181 ❌ | 0/17 ❌ | 4.09 | 0.826 |
| `…/dynamics_soft_ot_default` | **+0.0413 ✅** | +0.0026 ❌ | 0.256 ✅ | 0/17 ❌ | **16.97 ❌** | **0.000 ❌** |

\* OOD ridge baseline collapses for mean-delta (ridge Pearson = 0.271), inflating the MLP–ridge
margin; informative but not diagnostic of control.

### 2.2 Two-axis taxonomy

| Axis | V1 OT | mean_delta | soft_ot |
|---|---|---|---|
| **Supervised (val gate)** | Fail (+0.007) | Fail (+0.023) | **Pass (+0.041)** |
| **Reachability (beam k=3)** | **Pass (17/17, 1.59)** | Fail (0/17, 4.11) | Fail (0/17, 16.97) |
| **PPO transfer (V1 PPO)** | Success 1.000 (primary cell) | Success 0.000 | Success 0.000 |

**The two axes are independent.** The supervised gate measures *predictive accuracy beyond a
ridge baseline*; reachability measures *whether the predicted field has the geometry needed for
a planner to reduce distance*. The two are not the same problem. **V2's reportable result must
acknowledge this and be designed around it.**

### 2.3 Implications

1. The dim-11 OOD ceiling (ridge = 0.310, MLP = 0.06 across mean-delta variants) is a
   pairing-bound limit. No loss shaping on mean-delta pairs will close it.
2. Soft-OT's barycentric target is anti-contractive by construction: every pseudo-control is a
   weighted average of observed controls, so per-cell predictions point toward the control-cloud
   center, which is *farther* from `z_ref` than the perturbed start for OOD cells in bin 8–10.
   No loss shaping repairs that.
3. V1 OT is the only dynamics that is *both* directionally correct *and* gate-improvable. Its
   only failure is one supervised metric on the *in-distribution* val split (where the ridge
   baseline is unusually strong because gene one-hots are seen at training time).
4. The V1 hard-bench results show PPO det = 1.000 ≈ greedy_dyn_1 = 1.000 at the primary cell.
   The headroom for PPO to demonstrate non-trivial multi-step planning beyond greedy is **at
   harder cells** (smaller K, smaller ε, larger start distance) and **under a reward shape that
   does not collapse to "any contractive action wins"**.

---

## 3. Answers to the 6 strategic questions

### Q1. Should we prioritize A (gate-close V1 OT), B (PPO retrain on V1 OT), C (small PPO sweep), D (hybrid loss), or E (something else)?

**Recommendation: a sequenced hybrid — D-narrow + B + A in that order, encoded as a single
phase (P0D below).** Specifically:

* **First** (cheap, ~2 h): run a PPO-side hard-evaluation extension on V1 OT to map the *true*
  PPO-vs-greedy gap across the entire (K, ε, distance-bin, gene-split) lattice and identify
  cells where the existing V1 PPO is *not* saturated at 1.000. This is purely diagnostic and
  re-uses `scripts/evaluate_rl_hard.py` already in the repo. It tells us whether retraining is
  worth the time.
* **Second** (medium, ~8 h): two parallel-conceptually, sequential-in-practice tracks on V1 OT.
  * **Track A — gate-honest dynamics:** train V1 OT + residual-over-ridge (the specific
    architectural change from V2_RESEARCH_PLAN.md §7.B.2). RoR forces the MLP to learn only
    what ridge cannot, which is the cleanest gate-closer for the V1 OT field. Optionally
    layer correlation loss on top (`λ_corr ∈ {0.0, 0.05, 0.10}`). Three runs.
  * **Track B — PPO/reward refinement on V1 OT:** implement *delta-distance* and
    *terminal-only-with-step-cost* reward modes plus a distance-bin curriculum, retrain PPO
    at 500k–1M timesteps on V1 OT dynamics, evaluate on the V2 hard benchmark.
* **Third** (conditional): if Track A produces a gate-passing **and** still-reachable dynamics,
  retrain the winning PPO config on the new dynamics and compare. If not, V1 OT remains the
  primary V2 dynamics and the V2 write-up documents the gate-control decoupling as a finding.

This sequencing is justified because: (a) Q1-A alone leaves the end-to-end story unimproved if
RoR fails the gate; (b) Q1-B alone leaves the gate failure unresolved; (c) Q1-C is too narrow
because PPO retraining on the failing fields (mean-delta, soft-OT) is wasted compute. The
hybrid uses each track to inform the other.

### Q2. Is it useful to RL-evaluate dynamics that fail the gate?

**Yes, and we have already proven it.** The reachability probe (`scripts/probe_reachability.py`)
plus the focused hard benchmark (`scripts/evaluate_rl_hard.py`) on mean-delta and soft-OT
revealed that:

* Frozen-V1-PPO transfer to a new dynamics field tests **whether the new field has the same
  one-step-greedy geometry V1 PPO was trained against**. It does NOT test whether the new
  dynamics could support a freshly-trained policy. V1 PPO is essentially a one-step greedy
  oracle (V2_RESEARCH_PLAN.md §4.6); transferring it to a control-hostile field (soft-OT)
  trivially fails; transferring it to a contractive-but-undershoot field (mean-delta) gives a
  trajectory ~5.4 final distance, not 0% by accident.
* Retraining PPO on a non-gate-passing field tells us whether **the field is RL-learnable in
  principle**, independent of supervised accuracy. We have NOT yet run this experiment on any
  V2 dynamics. For mean-delta, beam best_dist = 4.11 > ε_p25 means a fresh PPO will fail at
  k = 3 regardless of training quality, so retraining there is pointless. For V1 OT, fresh
  PPO retraining is the right experiment to push the PPO-vs-greedy gap beyond 0.

**Honest reporting rule:** any RL evaluation under a gate-failing dynamics must be reported
with the gate status next to it. Do not silently report success rates from a model that did
not pass the gate.

### Q3. Should we run a small PPO retraining sweep now? Per-dynamics reasoning:

| Dynamics | Retrain PPO now? | Reason |
|---|---|---|
| **V1 OT** (`artifacts/dynamics`) | **YES — primary** | Reachable + V1 PPO already strong at primary cell; retraining tests reward/curriculum changes against a field that supports them. The end-to-end metric will move. |
| **mean_delta_default** | **NO** | Beam best_dist = 4.11 > ε_p25 = 3.17 at k = 3. No reward shape closes a geometry gap. Maybe at k = 8, but k = 8 is V1's setting and not the V2 hard primary cell. |
| **mean_delta_corr_030** | **NO** | Identical geometry to baseline (beam best_dist = 4.09); the diff is supervised, not control. |
| **soft_ot_default** | **NO — diagnostic only** | Anti-contractive (fraction_positive = 0.000). A fresh PPO cannot find a contractive policy in a field where no contractive policy exists. We could run a single short PPO smoke as a negative control to confirm "PPO fails on a gate-passing-but-control-hostile field" in writing, but this is presentation-only and 0.5 h max. |

### Q4. Should reward / curriculum changes be introduced now?

**Yes. Specifically: delta-distance reward, terminal success bonus, and a distance-bin
curriculum, all on V1 OT dynamics.** Rationale and analysis per mode:

The current reward `R_t = −d_{t+1} − 0.05·1[a ≠ NOOP]` has three known failure modes
(V2_RESEARCH_PLAN.md §4.2–4.3): (1) every contractive action receives positive shaped reward,
which makes random ≈ PPO under contractive dynamics; (2) it never directly rewards reaching
the goal, so PPO has no incentive to take fewer steps; (3) it does not discourage early NO-OP
because NO-OP is reward-neutral on sparsity.

**Reward mode analysis (all on V1 OT, K = 3 primary):**

| Mode | Objective | Why it might help | Failure mode | Smallest decisive test |
|---|---|---|---|---|
| **abs_distance** (current) | `R_t = −d_{t+1} − λ·1[a≠NOOP]` | None — this is the baseline | Random ≈ PPO under contraction | Already measured |
| **delta_distance** | `R_t = (d_t − d_{t+1}) − λ·1[a≠NOOP]` | Rewards *progress* per step, scale-invariant across start distances; harder for a random policy to match because it must consistently move closer | Bias toward small contractive steps even when a larger one is available; could discourage exploration of long-horizon planning | 200k timesteps at K = 3, V1 OT, primary hard cell. Compare PPO−random and PPO−greedy_dyn_1 to abs_distance baseline. |
| **terminal_only_step_cost** | `R_t = 0` mid-episode; `R_T = 1[d<ε] − β·t` at terminal | Forces the agent to find sequences that *land within ε*; step cost incentivizes short trajectories; no shaping confounds | High variance, slow learning — may require 1–2M timesteps before signal emerges; especially fragile when β is mis-tuned | 500k timesteps, β = 0.05; if 500k shows trend, scale to 1M. |
| **hybrid_delta_plus_terminal** | `R_t = α(d_t − d_{t+1}) − λ·1[a≠NOOP]`; `R_T += B·1[d<ε]` | Combines delta shaping (faster learning) with terminal bonus (correct objective at termination) | Tuning two hyperparams; may overcomplicate the narrative | 200k timesteps, α = 1, B = 1 ("unit weights"). |

**NO-OP / min_steps ablation.** Currently NO-OP terminates immediately and is reward-neutral
on sparsity. A `min_steps_before_noop` flag would force the agent to use the planning budget
when distance > ε. Worth testing but secondary to reward-mode changes.

**Curriculum (start-distance schedule).** The current min_start_distance = 8 puts every episode
in bin 8–10. A curriculum that starts at bin 4–6 (easier) and progressively raises the floor
to 10–12 over training is the standard fix for slow-learning sparse rewards. Implementation is
cheap: the env already supports `min_start_distance` as a config knob; we add a callback that
mutates it during training. Alternative: train at K = 8 first, then fine-tune at K = 3.

**Test plan (small, decisive):**
* `abs_distance` (control) vs `delta_distance` (treatment) at K = 3, 200k timesteps, V1 OT, no
  curriculum. If `delta_distance` wins, scale; if not, try `terminal_only_step_cost`.
* If both lose, that's a reportable finding: V1 PPO already saturates the V1 OT dynamics at the
  primary cell, and there's no headroom from reward changes alone.

### Q5. Should we continue trying to pass the gate?

**Yes, but the right experiment is residual-over-ridge on V1 OT pairs, not more correlation
loss on mean-delta.** Rationale per candidate:

| Approach | Likely to improve end-to-end? | Likely to close gate? |
|---|---|---|
| **Residual-over-ridge on V1 OT** (P0B.2 in original V2 plan) | Indirect — gate-closes the controllable field, makes V2 report honest | **Plausible** — forces MLP to predict only ridge residuals; the ridge baseline is *already* the gate's reference, so any non-zero residual signal counts. Has highest prior of all gate-closers because the math directly targets the gate metric. |
| **Correlation loss on V1 OT pairs alone** | Indirect | Plausible — P0B.1 in the original V2 plan but never run; we did it on mean-delta which has a different ceiling. |
| **Control-aware dynamics loss** (contraction regulariser) | Possibly direct, but changes the field and could break reachability | Low certainty; not standard; introduces a new hyperparam space; defer. |
| **Hybrid pairing (OT-train + mean-delta-OOD)** | Possibly | Low — adds complexity, unlikely to fix dim-11 OOD given the mean-delta dim-11 ceiling is intrinsic. |
| **Per-dim loss weighting (upweight dim-11)** | Specifically targets the OOD bottleneck | On V1 OT: untested. On mean-delta: dim-11 has a ridge-defined ceiling at 0.310 which weighting alone cannot raise. Worth adding as a side-knob to the RoR run. |

**Choice: residual-over-ridge on V1 OT (+ optional correlation loss) is the primary gate-closer
attempt.** Mathematically: `μ = ridge(z, gene) + mlp(z, gene_emb)`; ridge is frozen at the
ridge baseline; MLP is trained on the residual. Cost: ~30 min × 3 runs. Risk: residuals are
mostly OT pairing noise → MLP converges to ~0 → margin unchanged. Even in failure, the
result is informative (it caps the achievable margin on OT pairs, completing the diagnostic
trilogy).

### Q6. What is the next concrete implementation phase?

See §5 below (the **P0D phase definition**).

---

## 4. Comparative analysis of candidate phases

| Candidate | Pros | Cons | Compute | Verdict |
|---|---|---|---|---|
| **C1 — P0B3 controllable gate-closing (RoR on V1 OT)** | Targets the right field (V1 OT). Mathematically aligned with the gate metric. Honest gate-closer. | If residual is mostly noise (P0A noise ratio = 0.89), MLP→0 and margin stays low. | ~2 h | Strong but partial — does not by itself improve end-to-end. |
| **C2 — P0C-lite PPO/reward sweep on V1 OT** | Tests the end-to-end story directly. Cheap. Independent of gate. Has the highest "expected end-to-end metric movement" of any candidate. | Doesn't close the gate — V2 report still has to explain the gate decoupling. | ~6 h (3 small PPO runs) | Strong — but better paired with C1 for the full V2 story. |
| **C3 — Hybrid (C1 + C2)** | Both gate and end-to-end are addressed. Decoupling becomes a *finding* with both arms experimentally supported. | More moving parts; needs careful sequencing. | ~8–10 h | **RECOMMENDED.** |
| **C4 — Dynamics objective redesign (contraction regulariser)** | Could fix both gate and control simultaneously. | New hyperparam space; risk of breaking reachability; not justified by current evidence; high failure-mode opacity. | unknown | Defer to V3. |
| **C5 — Promote V1 OT to V2 primary unchanged + PPO refinement only** | Cleanest possible report; minimum compute. | Leaves the gate-failure as a known limitation; reduces V2 novelty. | ~4 h | Acceptable fallback if C1's RoR run fails. |
| **C6 — VAE retraining / re-tuning** | Could address the underlying latent geometry. | Sacred rule #1; everything downstream invalidates; large recompute; no evidence the latent is the bottleneck (ε is fine, separation ratio fine). | days | **Rejected.** |

**Ranked recommendation:**
1. **C3 (P0D Hybrid)** — primary.
2. **C5 (PPO refinement on V1 OT only)** — fallback if C1's RoR fails.
3. **C2 (PPO sweep across dynamics)** — diagnostic complement, narrowed to V1 OT.
4. **C1 (RoR alone)** — partial; only useful inside C3.
5. **C4 (contraction regulariser)** — V3 work.
6. **C6 (VAE)** — rejected.

---

## 5. Recommended next phase — **P0D: V1 OT Hardening (dual-track, dynamics + RL)**

### 5.1 Objective

Produce a defensible end-to-end V2 result on `artifacts/dynamics` (V1 OT). Two co-equal goals:

1. **Track A — gate honesty:** Try once and decisively to close the V1 OT supervised gate via
   residual-over-ridge architecture plus optional correlation loss on V1 OT pairs (not
   mean-delta).
2. **Track B — RL learnability:** Identify whether reward-mode and curriculum changes lift
   PPO above one-step greedy on the V2 hard primary cell and on harder cells (K = 1, K = 2,
   bin 10–12).

Whichever track yields the cleaner end-to-end metric is the V2 headline. If both yield
improvements, both are V2 contributions.

### 5.2 Rationale

* V1 OT is the only verified controllable field (beam 17/17, V1 PPO success 1.000 at primary
  cell).
* RoR is the architectural intervention specifically aligned with the gate metric and was
  pre-registered in the original V2 plan but never executed.
* PPO/reward changes are the lowest-cost, highest-information experiments because they test
  the actual reportable metric.
* Mean-delta and soft-OT are diagnostically complete; further training on either is not
  justified.

### 5.3 Hypotheses (preregistered)

* **H_RoR_gate:** V1 OT + residual-over-ridge (best of `λ_corr ∈ {0.0, 0.05, 0.10}`) closes the
  val gate margin to ≥ +0.030 without reducing OOD Pearson below 0.40 or unc Spearman below
  0.20, and without breaking beam reachability (≥ 13/17 at k=3).
* **H_delta_reward:** Retrained PPO on V1 OT with `reward_mode=delta_distance`,
  total_timesteps=500k, K=3, raises **PPO−random** at the primary hard cell by ≥ +0.10 over
  the V1 PPO baseline at the same cell, with PPO−greedy_dyn_1 ≥ 0.
* **H_curriculum:** A distance-bin curriculum (start 4–6, escalate to 10–12 over training)
  reduces PPO success-rate variance across distance bins by ≥ 30 % vs the no-curriculum run.
* **H_gate_vs_control:** Whichever track wins, the V2 report concludes that *gate ≥ +0.030
  is neither necessary nor sufficient for hard-benchmark success*; the evidence is the
  combined V1 OT (gate-fail, control-pass) and soft-OT (gate-pass, control-fail) results,
  with the P0D outcome as the third anchor.

### 5.4 Experiment matrix

Each cell is a single run; outputs under `artifacts_v2/`. V1 artifacts are never modified.

#### Track A — dynamics

| # | Dynamics dir | Pairs source | `use_residual_over_ridge` | `use_state_linear_skip` | `lambda_corr` | Notes |
|---|---|---|:---:|:---:|:---:|---|
| A1 | `artifacts_v2/dynamics_v1ot_ror` | `artifacts/pairs` | true | false | 0.0 | RoR only |
| A2 | `artifacts_v2/dynamics_v1ot_ror_corr005` | `artifacts/pairs` | true | false | 0.05 | RoR + light corr |
| A3 | `artifacts_v2/dynamics_v1ot_ror_corr010` | `artifacts/pairs` | true | false | 0.10 | RoR + medium corr |

Acceptance per run: `gate.json` written; `gate_diagnostics.json` written; beam reachability probe
on the run with `repeat_mask=True` (n=17 OOD bin 8–10) yields success_rate ≥ 13/17 and
best_final_distance ≤ 2.5.

#### Track B — RL (PPO retraining on V1 OT dynamics)

All runs reuse `artifacts/dynamics` and `artifacts/vae`. K = 3 unless stated.
total_timesteps = 200k for first pass; winning variant scales to 500k.

| # | Reward mode | Curriculum | total_timesteps | rl_dir | Purpose |
|---|---|---|---:|---|---|
| B1 | `abs_distance` (control) | none | 200k | `artifacts_v2/rl_v1ot_abs_k3_200k` | Baseline for fair comparison (V1 PPO was at K=10, p50, 500k) |
| B2 | `delta_distance` | none | 200k | `artifacts_v2/rl_v1ot_delta_k3_200k` | Direct test of H_delta_reward |
| B3 | `terminal_only_step_cost` (β=0.05) | none | 500k | `artifacts_v2/rl_v1ot_terminal_k3_500k` | Direct test of terminal-only learnability |
| B4 | best of {B2, B3} | bin 4–6 → 8–10 over training | 500k | `artifacts_v2/rl_v1ot_<best>_curriculum_k3_500k` | Test H_curriculum |
| B5 | best of {B2, B3, B4} | as in winner | 1M | `artifacts_v2/rl_v1ot_<best>_scaled_1M` | Scale only the winner |

Acceptance per run: `ppo.zip` written; tensorboard logs available; final-eval `summary.json`
on the V2 hard primary cell written via `scripts/evaluate_rl_hard.py`.

#### Track C — Optional conditional (only if Track A produces a gate-passing **and** reachable dynamics)

* C1: retrain the winning Track-B reward/curriculum config on the winning Track-A dynamics
  for 500k timesteps. Compare PPO−greedy_dyn_1 to the V1-OT-dynamics version.

### 5.5 Files to create / modify

**New code (with TDD where the change introduces logic):**

| File | Change | Test file |
|---|---|---|
| `src/models/dynamics.py` | Add `use_residual_over_ridge: bool` flag; when true, register three buffers (`ridge_W_z`, `ridge_W_gene`, `ridge_b`) and add `ridge_pred = z @ W_z + W_gene[gene_idx] + b` to `mu`. Mutual exclusion with `use_state_linear_skip`. | extend `tests/test_dynamics.py` with `TestResidualOverRidge` |
| `scripts/train_dynamics.py` | When `cfg.dynamics.use_residual_over_ridge=true`, fit `_fit_ridge_baseline` once on train pairs, split into z-block and gene-block, convert to torch tensors, assign to model buffers before optimisation. Write `ridge_baseline.npz` into the run dir for audit. | covered by smoke run |
| `config/dynamics.yaml` | Add `use_residual_over_ridge: false` (default off); document the mutual-exclusion with `use_state_linear_skip`. | n/a |
| `src/rl/reward.py` | Add `reward_mode` parameter to `compute_reward`. Supported modes: `"absolute_distance"` (default), `"delta_distance"`, `"terminal_only_step_cost"`. For `delta_distance`, caller passes `prev_distance`; for `terminal_only_step_cost`, dense reward is 0 and terminal reward is `1·success − β·step_count`. | extend `tests/test_reward.py` (or create) with one test per mode |
| `src/rl/environment.py` | Plumb `prev_distance` through `step()` so `compute_reward` can use it; thread the new reward mode through env config. | extend `tests/test_environment.py` with reward-mode tests |
| `config/rl.yaml` | Add `reward.mode` and `reward.beta_step_cost` (default 0.05). Keep `lambda_sparse` separate so existing runs are bit-stable when `mode=absolute_distance`. | n/a |
| `src/rl/curriculum.py` (NEW) | Stable-baselines3 callback that mutates `env.min_start_distance` according to a schedule (linear or step). Read schedule from `cfg.rl.train.curriculum`. | new test file `tests/test_curriculum.py` |
| `scripts/train_rl.py` | Wire the new callback when `cfg.rl.train.curriculum.enabled=true`. | covered by smoke |

**Files not to modify:**

* `src/analysis/metrics.py` gate logic (single source of truth) — gate evaluation must remain
  unchanged so V1↔V2 comparisons are valid.
* `config/dynamics.yaml` `dynamics.gate.*` thresholds.
* `artifacts/`, `artifacts_64/`, `artifacts/rl_sweeps/` (V1 frozen).
* `artifacts/vae/*` (sacred rule #1).

### 5.6 Commands (representative)

Track A:
```bash
PYTHONPATH=. .venv/bin/python scripts/train_dynamics.py --config-name default \
  paths.pairs_dir=artifacts/pairs \
  paths.dynamics_dir=artifacts_v2/dynamics_v1ot_ror \
  dynamics.use_residual_over_ridge=true dynamics.use_state_linear_skip=false \
  dynamics.lambda_corr=0.0 seed=42 +force=true

# Followed by reachability probe + gate breakdown on each Track-A run:
PYTHONPATH=. .venv/bin/python scripts/probe_reachability.py \
  --dynamics_dirs v1ot_ror:artifacts_v2/dynamics_v1ot_ror \
  --vae_dir artifacts/vae --pairs_dir artifacts/pairs \
  --out artifacts_v2/reachability_probe_v1ot_ror \
  --epsilon 3.1662898064 --distance_bin 8-10 --held_out_genes_only \
  --max_depth 3 --beam_width 50 --n_genes 105 --device cpu
```

Track B:
```bash
# B2 — delta_distance, K=3, 200k
PYTHONPATH=. .venv/bin/python scripts/train_rl.py --config-name default \
  paths.dynamics_dir=artifacts/dynamics \
  paths.rl_dir=artifacts_v2/rl_v1ot_delta_k3_200k \
  rl.env.max_steps=3 rl.env.epsilon_override=3.1662898064 \
  rl.env.min_start_distance=8.0 \
  rl.reward.mode=delta_distance rl.reward.lambda_sparse=0.05 \
  rl.ppo.total_timesteps=200000 \
  rl.train.skip_gate=true \
  seed=42
```

(`rl.train.skip_gate=true` is required because V1 OT did not pass the gate; this is logged as
a P0 warning per the existing CLAUDE.md §3 rule #9. Documentation in PROGRESS.md must record
the override and reason.)

Hard-bench eval per Track-B run:
```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_rl_hard.py \
  --vae_dir artifacts/vae \
  --dynamics_dir artifacts/dynamics \
  --ppo_zip artifacts_v2/rl_v1ot_delta_k3_200k/ppo.zip \
  --out_dir artifacts_v2/eval_v1ot_delta_k3_200k \
  --k_values 1 2 3 --epsilon_values p25 p50 \
  --distance_bins 6-8 8-10 10-12 \
  --held_out_genes_only true,false --n_episodes 200 \
  --baselines random_uniform_valid,always_noop,greedy_dyn_1,greedy_dyn_1_noop_free,ridge_greedy
```

### 5.7 Expected runtime (Apple Silicon MPS; halve on CUDA)

| Track / phase | Wall-clock |
|---|---|
| Code + tests for `use_residual_over_ridge` (Track A) | ~2 h dev |
| Track A — 3 dynamics trainings + diagnostics | ~3 h |
| Code + tests for new reward modes + curriculum callback | ~3 h dev |
| Track B — 3 PPO trainings @ 200k, 1 @ 500k, 1 @ 1M | ~12 h |
| Track C (conditional) — 1 PPO @ 500k | ~3 h |
| Interpretation + PROGRESS update | ~1 h |
| **Total (excluding Track C)** | **~22 h** |

This is the largest V2 phase to date. It can be split into two sessions: (1) Track A end-to-end
+ reward-mode code/tests + B1, B2; (2) the rest.

### 5.8 Primary evaluation metrics (V2 hard benchmark, per AGENTS.md / V2 §9)

For every Track-B PPO checkpoint, on the **primary hard cell** (K=3, ε=p25, bin 8–10, OOD):

* success_rate (Wilson 95 % CI)
* mean_final_distance
* mean_steps
* action diversity (entropy of `action_freq`)
* top-10 actions
* **PPO − random_uniform_valid (Δ pp, 95 % CI)** — primary delta
* **PPO − greedy_dyn_1 (Δ pp, 95 % CI)** — secondary delta (evidence of multi-step planning)
* PPO − always_noop (sanity)
* `weighted_action_freq_chronos_spearman` (biology plausibility, reported but not gating)

Reachability probe consistency: for every Track-A dynamics, re-run the probe and report
`success_rate`, `best_final_distance`, and the `repeat_mask=True` vs `False` gap.

### 5.9 Acceptance criteria

**Track A succeeds if any A-run satisfies all four:**

1. `gate.json["passed"] == true` (val margin ≥ +0.030 and unc Spearman ≥ 0.20).
2. OOD Pearson ≥ 0.40.
3. Beam-search reachability ≥ 13/17 at k = 3 with best_final_distance ≤ 2.5.
4. Per-gene action-contraction `fraction_positive ≥ 0.90` (preserve V1's controllability).

**Track B succeeds if any B-run satisfies:**

1. Primary-cell PPO success_rate ≥ V1 PPO baseline + 0 pp (i.e. no regression).
2. **AND** (PPO − random) ≥ +0.50 pp at primary cell **OR** (PPO − greedy_dyn_1) ≥ 0 pp at K=2
   or bin 10–12 (evidence of generalization beyond V1).

**Track C (conditional):** only triggered if Track A succeeds. Acceptance: end-to-end
primary-cell success on the new dynamics ≥ V1 OT baseline.

### 5.10 Rollback criteria

* **Track A — all three runs fail acceptance:** declare V1 OT gate-closure infeasible under
  ridge-residual architecture; V2 report concludes the gate failure is OT-pairing-noise-bound
  (the P0A finding becomes a V2 limitation). Proceed with Track B alone.
* **Track B — all three reward modes regress vs V1 PPO at the primary cell:** declare PPO/reward
  changes do not help at K = 3 (V1 PPO is saturated there); narrow the headline to the harder
  cells (K = 1, bin 10–12) where V1 PPO was not 1.000. Do not deceptively re-report the K = 10
  number.
* **Curriculum (B4) increases variance instead of decreasing it:** retain the no-curriculum
  winner (B2 or B3).
* **Any training NaN or evaluator crash:** record, do not retry with hyperparam tweaks here —
  that is V3 scope.

### 5.11 Interpretation template (`artifacts_v2/interpretation_p0d_v1ot_hardening.md`)

```markdown
# P0D — V1 OT Hardening Interpretation (YYYY-MM-DD)

## Track A — Dynamics (V1 OT + residual-over-ridge)
| Run | λ_corr | Val margin | OOD Pearson | Unc Spearman | Beam 17/n | Best dist | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 (RoR) | 0.0 | … | … | … | … | … | … |
| A2 (RoR+corr 0.05) | 0.05 | … | … | … | … | … | … |
| A3 (RoR+corr 0.10) | 0.10 | … | … | … | … | … | … |

H_RoR_gate: <supported|partially|rejected>. Why: ….

## Track B — RL/reward (V1 OT dynamics)
| Run | Reward | TS | Curr. | Primary success | PPO−rand | PPO−greedy | mean_steps | top-3 actions |
|---|---|---:|---|---:|---:|---:|---:|---|
| V1 baseline | abs | 500k | no | 1.000 | … | 0.000 | 2.28 | CKS1B/TSC22D1/CELF2 |
| B1 (abs k=3) | abs | 200k | no | … | … | … | … | … |
| B2 (delta) | delta | 200k | no | … | … | … | … | … |
| B3 (terminal) | terminal | 500k | no | … | … | … | … | … |
| B4 (best + curric.) | … | 500k | yes | … | … | … | … | … |
| B5 (winner @1M) | … | 1M | … | … | … | … | … | … |

H_delta_reward: <supported|rejected>. Why: ….
H_curriculum: <supported|rejected>. Why: ….

## H_gate_vs_control
Combined evidence from P0B″, P0B2, P0C0, P0D:
- soft-OT: gate PASS / control FAIL.
- V1 OT: gate FAIL / control PASS.
- mean-delta: gate FAIL / control directional / RL FAIL.
- V1 OT + RoR: <gate result> / <control result>.

Verdict: <gate is/isn't sufficient> for end-to-end success. Recommended V2 reporting:
explicitly decouple the two axes.

## V2 headline recommendation
Primary dynamics for V2 report: <V1 OT or Track-A winner>.
Primary PPO config for V2 report: <V1 baseline or Track-B winner>.
Primary headline number(s) (with caveats): ….

## Next step
- Promote <selected dynamics + PPO> to V2 primary.
- If Track A succeeded: run Track C to combine.
- Defer to V3: <listed items>.
```

### 5.12 Sacred-rule conformance check

* No VAE retraining; uses `artifacts/vae` read-only.
* No edits to `src/utils/device.py` callers; all device handling inherited.
* No new `random.seed` / `torch.manual_seed` outside `src/utils/seeding.py`.
* No inline metric definitions; gate metric, reward, distance, correlation all live in their
  canonical modules.
* No path hardcoding; all Hydra overrides.
* No V1-artifact modification; `git status -- artifacts/ artifacts_64/ artifacts/rl_sweeps/`
  must be clean at the end of every run.
* No gate-threshold lowering; the gate logic and thresholds are unchanged.
* `train_rl.py` skip_gate is set true for V1 OT runs (acknowledged P0 warning, logged in
  PROGRESS.md alongside the rationale that V1 OT is the verified controllable field).
* No knockout / CRISPRi action-space changes.

---

## 6. What should NOT be implemented yet

* No further correlation-loss sweeps on mean-delta (saturated at +0.0232 → +0.0030 of gap).
* No additional λ values on soft-OT (anti-contractive field; loss shaping does not change
  geometry).
* No control-aware loss / contraction regulariser (defer to V3 unless P0D both tracks fail).
* No VAE retraining (sacred rule; no evidence the latent is the bottleneck).
* No CRISPRi / knockout action space (sacred rule).
* No K = 8 retraining (V1 setting; not the V2 hard primary).
* No external healthy reference (sacred rule on therapeutic-reprogramming claims).
* No gate-threshold lowering (explicit constraint).

---

## 7. Final ranked recommendation

1. **Implement P0D (C3 hybrid) — Track A (RoR on V1 OT) + Track B (reward/curriculum on V1 OT
   PPO).** This is the recommended next phase.
2. If forced to pick one track only: **Track B first.** It directly addresses the reportable
   end-to-end metric; Track A's outcome doesn't change which dynamics is controllable.
3. Track C (combine winners) is conditional on Track A success.
4. Defer all other candidates (C4, C5 as fallback, C6) per §6.

---

## 8. Concise implementation prompt for P0D

> You are implementing CellPath V2 P0D — V1 OT Hardening (dual-track).
>
> Constraints (verbatim, do not violate):
> - Do not retrain VAE.
> - Do not modify V1 artifacts under `artifacts/`, `artifacts_64/`, `artifacts/rl_sweeps/`.
> - All new outputs under `artifacts_v2/`.
> - Do not lower the dynamics gate thresholds.
> - PPO retraining on V1 OT requires `rl.train.skip_gate=true` and a PROGRESS.md note.
> - Do not force-add large artifact directories to git.
> - Stop after writing `artifacts_v2/interpretation_p0d_v1ot_hardening.md`. Do not promote the
>   winner to V2 primary yet — that's a separate session.
>
> Execution order:
> 1. Track A code + tests: add `use_residual_over_ridge` to `src/models/dynamics.py`,
>    wire ridge buffer fit-and-assign in `scripts/train_dynamics.py`, add config flag,
>    write `TestResidualOverRidge` unit tests (mutual exclusion with `use_state_linear_skip`,
>    forward-pass shape, ridge buffer values after fit). Run pytest.
> 2. Track A training: A1, A2, A3 (Residual-over-ridge with λ_corr ∈ {0.0, 0.05, 0.10}).
> 3. Per Track-A run: gate.json read, gate breakdown, beam-search reachability probe.
> 4. Track B code + tests: add `reward.mode` to `config/rl.yaml`, implement `delta_distance`
>    and `terminal_only_step_cost` in `src/rl/reward.py`, thread `prev_distance` through
>    `src/rl/environment.py`, add `src/rl/curriculum.py` SB3 callback, write tests for each.
>    Run pytest.
> 5. Track B training (in order, stop early if B1 and B2 both regress: that is itself a
>    finding): B1 (abs control), B2 (delta), B3 (terminal), B4 (best + curriculum), B5
>    (winner scaled to 1M).
> 6. Track C (conditional, only if any Track-A run passes Acceptance §5.9): retrain winning
>    PPO config on winning Track-A dynamics.
> 7. Write `artifacts_v2/interpretation_p0d_v1ot_hardening.md` per §5.11 template.
> 8. Update `PROGRESS.md` with one new session entry.
> 9. Run full `pytest -q` and `git status -- artifacts/ artifacts_64/ artifacts/rl_sweeps/`.
>
> Final response must report: code changed, tests passed, dynamics runs and exact gate metrics
> per run, PPO runs and exact primary-cell metrics per run, the H_RoR_gate / H_delta_reward /
> H_curriculum / H_gate_vs_control verdicts, the recommended V2 primary (dynamics + PPO), and
> confirmation that no VAE / V1 artifacts / gate thresholds were changed.
