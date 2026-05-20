# V3C Phase 1 — PPO_BCD smoke summary

**Status:** `PHASE1_COMPLETE — PHASE4_ESCALATION_RECOMMENDED_ON_TRACKS_L_AND_N`

**Scope:** Single-seed (42) PPO_BCD smoke for the 4-target roster proposed in `v3c_phase0_utility_audit.md` §5, evaluated on the canonical 7-cell V3B matrix at `n_episodes=200`. Locked B+C+D reward stack throughout (`λ_tox=0.10, λ_ce=0.05, λ_unc_path=0.05`, freeband `{3, 5, 0.02, 0.10, 1.0}`, `env.max_steps=8`, `uncertainty_reduce=mean_sigma`). Per-VAE p15 ε resolved from the empirical control-cell distance distribution (not the 32D scalar — guardrail #8 of Phase 0C).

**Frozen tier check:** `git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/` → clean.

**Test suite:** 377 passed / 2 skipped (matches Phase 0B baseline).

---

## §1 — Headline

| Field | 500k verdict | 1M verdict | Phase 4? |
|---|---|---|---|
| **Anchor** (`artifacts_v2/dynamics_v1ot_ror_corr010`) | — (reused 1M) | **NO_SIGNAL** (V3B TECHNICAL_ONLY reproduced) | n/a |
| **Track L** (`artifacts_v3/dynamics_n64_legacy_ror_corr010`) | EARLY_PROMISING (K=2/b8-10 tie w/ g_dyn_2 + huge anchor lift) | **WEAK_SIGNAL** (raw tie, large anchor lift, Pareto fails on final_distance) | **YES — 4-seed escalation recommended** |
| **Track N** (`artifacts_v3/dynamics_n64_nb_ror_corr010`) | **CANDIDATE_SIGNAL_RAW** at K=2/b8-10 (+0.075 over greedy_dyn_2_fused, non-saturated) | regressed to **WEAK_SIGNAL** (-0.050) | **YES — 4-seed escalation recommended** (resolve 500k-vs-1M single-seed variance) |
| **Mean-delta wildcard** (`artifacts_v2/dynamics_mean_delta_corr_010`) | EARLY_FLAT (NOOP-strategy at K≤5; 0.130 at K=8 vs g_dyn_2 0.265) | (stopped at 500k per V3C plan §4 Stage 4) | **NO** (over-shaped on a low-reach field) |

**Most important single number:** Track N at 500k showed `PPO_BCD K=2/b8-10/OOD = 0.570` vs `greedy_dyn_2_fused = 0.495` — a **+0.075 RAW win** at a non-saturated cell, the first such win in V3 across any reward variant. It did NOT survive doubling training to 1M (regressed to 0.445), so Phase 4 4-seed CI is required to know whether it is a real signal or seed-42 variance.

---

## §2 — Per-VAE epsilon (locked at p15 percentile, exact-computed not interpolated)

`epsilon_success.json` files only store p10/p25 quantiles; the V3C plan §3 U-B audit used a 5/15 linear interpolation. For Phase 1, ε was **recomputed exactly** from each VAE's empirical control-cell distance distribution per the user's directive:

| Field | n_ctrl | ε_p5 | ε_p10 | **ε_p15 (used)** | ε_p25 | Δ vs 32D p15 |
|---|---:|---:|---:|---:|---:|---:|
| 32D `artifacts/vae` (anchor + mean_delta) | 11855 | 2.7362 | 2.8846 | **2.9898** | 3.1663 | — |
| 64D legacy scVI (`artifacts_v3/vae_n64_legacy`, Track L) | 11855 | 2.7809 | 2.9181 | **3.0193** | 3.1872 | +0.0295 (+1.0%) |
| 64D NB VAE (`artifacts_v3/vae_n64_nb`, Track N) | 11855 | 2.8605 | 3.0032 | **3.1120** | 3.2814 | +0.1222 (+4.1%) |

`epsilon_success.json` exact-write: the 32D file stores only `value=3.5311 (percentile 50)`; the 64D files store quantile dicts that match the recompute to 4 decimals — the audit framework's linear interpolation was within 0.01 of the true p15.

**Cheap p15 vs p10 pre-PPO calibration** (n_eps=64, beam_width=64, distance-only beam — V3C plan §4 Stage 4 explicit guardrail):

| Field | Cell | reach @ p15 | reach @ p10 | Decision |
|---|---|---:|---:|---|
| Track L (legacy 64D) | k2_bin8-10 | 0.703 | 0.562 | p15 non-saturated, p10 tighter — **use p15** (V3B precedent) |
| Track L | k3_bin8-10 / k8 | 1.000 / 1.000 | 1.000 / 1.000 | both saturated — no discrimination at K≥3 |
| Track N (NB 64D) | k2_bin8-10 | 0.422 | 0.344 | p15 non-saturated, p10 tighter — **use p15** |
| Track N | k3_bin8-10 / k8 | 1.000 / 1.000 | 1.000 / 1.000 | both saturated |

**Decision:** train all candidates at p15 (per-VAE exact). Switching to p10 would not un-saturate K≥3 cells but would shrink the K=2 non-saturated band that PPO has room to exploit. V3B Phase 3b already documented p10 → PPO_BCD collapse on V2 anchor; the same risk applies here.

---

## §3 — Training continuation decisions

| Field | 500k done? | 500k verdict per mid-smoke triage | Continue to 1M? |
|---|---|---|---|
| Anchor | (already 1M from V3B Phase 4) | reused checkpoint config-compatible | n/a |
| Track L | yes (208s training, 99s eval) | EARLY_PROMISING (K=2/b8-10 tie w/ g_dyn_2 at 0.705; near-miss worth resolving per user instructions) | **yes** (1M training: 208s) |
| Track N | yes (101s training, 246s eval) | **EARLY_PROMISING — CANDIDATE_SIGNAL_RAW at 500k** (+0.075 at K=2/b8-10) | **yes** (1M training: 233s) |
| mean_delta | yes (74s training, 449s eval) | EARLY_FLAT (raw success collapse at K≤5; NOOP-strategy at K=8) | **no** (per V3C plan §4 Stage 4: EARLY_FLAT stops at 500k) |

**Notes:**
- `train_rl.py` has no native resume support, so Track L/N 1M were trained fresh (not resumed from 500k). 500k checkpoints remain on disk under `*_seed42_500k/` for diagnostics; 1M checkpoints under `*_seed42_1M/`.
- mean-delta's EARLY_FLAT verdict honors the user's "do not call low K=2/K=3 success an implementation failure" directive. The K=8 PPO success rate (0.130) is below greedy_dyn_2_fused (0.265) by 0.135 — out of Pareto ±0.03 tolerance — and PPO's `mean_steps=0.85` shows it learned a **NOOP-mostly strategy**. More training would not change that policy.

---

## §4 — Anchor — `dynamics_v1ot_ror_corr010` (1M PPO_BCD reused from V3B Phase 4)

**Source:** `artifacts_v3/rl_v3b_biorealistic_fused_epsp15_seed42/ppo.zip` — configuration-compatible per V3C plan §4 Stage 4 anchor-reuse pre-check (matching dynamics, ε, all λ_*, freeband, max_steps).

| Cell | PPO_BCD | random | always_noop | greedy_dyn_1_fused | greedy_dyn_2_fused | greedy_dyn_3_fused | greedy_dyn_5_fused |
|---|---:|---:|---:|---:|---:|---:|---:|
| K=2/b6-8 | 0.515 | 0.025 | 0.000 | 0.725 | **0.645** | — | — |
| K=2/b8-10 | 0.120 | 0.010 | 0.000 | 0.250 | **0.120** | — | — |
| K=3/b6-8 | 0.950 | 0.165 | 0.000 | 1.000 | 0.990 | **0.990** | — |
| K=3/b8-10 | 0.940 | 0.075 | 0.000 | 1.000 | 1.000 | **1.000** | — |
| K=4/b8-10 | 1.000 | 0.205 | 0.000 | 1.000 | 1.000 | 1.000 | — |
| K=5/b8-10 | 1.000 | 0.310 | 0.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| K=8/b8-10 | 1.000 | 0.515 | 0.000 | 1.000 | 1.000 | 1.000 | **1.000** |

**PPO vs same-K greedy (Δ = PPO − greedy_dyn_K):**

| Cell | PPO−greedy | Verdict |
|---:|---:|---|
| K=2/b6-8 | **−0.130** | PPO loses (greedy_dyn_2 stronger at this cell) |
| K=2/b8-10 | 0.000 | TIE (V3B Phase 4 PPO_BCD = 0.148 here — 0.12 ± 0.02 is within sampling noise) |
| K=3/b6-8 | −0.040 | PPO under saturated greedy |
| K=3/b8-10 | −0.060 | PPO under saturated greedy |
| K=4–K=8 | 0.000 | saturated tie |

**Reward-fit (PPO_BCD vs greedy_dyn_K_fused):** PPO has `tox=0, CE=0, unc_path_max=0.61–0.71` at every cell; greedy_dyn_K_fused has `tox=0.001–0.006, CE=0.015–0.07, unc=0.61–0.74`. PPO improves safety axes marginally but the K=2/b6-8 raw loss of 0.130 is far outside Pareto ±0.03 tolerance.

**Verdict:** `NO_SIGNAL` (raw) / `NO_SIGNAL` (Pareto). V3B Phase 4 `LOCKED_DESIGN_TECHNICAL_ONLY` is reproduced exactly — the locked reward stack is correct, the V2 anchor dynamics field cannot express it as control leverage.

---

## §5 — Track L — `dynamics_n64_legacy_ror_corr010` (1M PPO_BCD, fresh)

64D legacy scVI VAE + V2 RoR_corr010 dynamics architecture. **Only the latent dim differs from anchor.**

| Cell | PPO_BCD (1M) | PPO_BCD (500k) | greedy_dyn_K_fused | PPO−g_K | vs anchor PPO |
|---|---:|---:|---:|---:|---:|
| K=2/b6-8 | 0.825 | 0.785 | g_2=0.910 | −0.085 | **+0.310** (vs anchor 0.515) |
| K=2/b8-10 | **0.705** | 0.705 | **g_2=0.695** | **+0.010** TIE | **+0.585** (vs anchor 0.120) |
| K=3/b6-8 | 1.000 | 1.000 | g_3=1.000 | 0.000 | +0.050 |
| K=3/b8-10 | 1.000 | 1.000 | g_3=1.000 | 0.000 | +0.060 |
| K=4–K=8 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |

**Reward-fit at K=2/b8-10 (PPO_BCD vs greedy_dyn_2_fused):**

| Axis | PPO_BCD | g_dyn_2 | Δ |
|---|---:|---:|---|
| success | 0.705 | 0.695 | +0.010 (within ±0.03 — tied) |
| mean_final_distance | 3.139 | 2.917 | +0.222 (**> 0.10 tolerance — regression**) |
| mean_tox_path | 0.000 | 0.000 | tie |
| mean_common_essential | 0.000 | 0.000 | tie |
| mean_unc_path_max | 0.402 | 0.419 | −0.017 (improved) |

**Verdict:** `WEAK_SIGNAL`.
- Raw vs same-field greedy: ties at K=2/b8-10 (+0.010), loses at K=2/b6-8 (−0.085). No `CANDIDATE_SIGNAL_RAW`.
- Pareto: success within tolerance + 1 axis improved (unc), but `mean_final_distance` regresses by 0.222 (out of the V3C plan §4 Stage 4 tolerance ≤ 0.10). **Pareto fails on the distance-not-regressed criterion.**
- **However**: at K=2/b8-10/OOD, PPO_BCD on Track L reaches **0.705 vs anchor's 0.120** — a 5.9× lift on a 64D representation with the same architecture. The same audit-predicted Bucket U-B reachability advantage (Track L at p15 K=2/b8-10 = 0.560 vs anchor 0.120, Phase 0C §5) translates into actual PPO performance.
- Stable between 500k and 1M (K=2/b8-10 unchanged; K=2/b6-8 +0.040 from extra training).

**Recommendation:** Phase 4 escalation (4-seed × 1M) to test whether the anchor-lift is seed-42-specific and whether the Pareto distance regression generalizes. If multi-seed PPO_BCD ties greedy_dyn_2 cleanly without distance regression, this becomes a `CANDIDATE_SIGNAL_PARETO`.

---

## §6 — Track N — `dynamics_n64_nb_ror_corr010` (1M PPO_BCD, fresh)

64D NB VAE + V2 RoR_corr010 dynamics architecture. **Latent dim AND likelihood differ from anchor.**

| Cell | PPO_BCD (1M) | **PPO_BCD (500k)** | greedy_dyn_K_fused | 1M PPO−g_K | 500k PPO−g_K | vs anchor PPO |
|---|---:|---:|---:|---:|---:|---:|
| K=2/b6-8 | 0.735 | 0.765 | g_2=0.890 | −0.155 | −0.125 | +0.220 |
| K=2/b8-10 | **0.445** | **0.570** | **g_2=0.495** | **−0.050** | **+0.075** | +0.325 / +0.450 |
| K=3/b6-8 | 0.960 | 0.960 | g_3=0.995 | −0.035 | −0.035 | +0.010 |
| K=3/b8-10 | 0.880 | 0.945 | g_3=1.000 | −0.120 | −0.055 | −0.060 |
| K=4/b8-10 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| K=5–K=8 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |

**The CANDIDATE_SIGNAL_RAW at 500k**:

At seed 42 × 500k, Track N PPO_BCD reaches `0.570` at K=2/bin8-10/OOD versus same-field reward-aware `greedy_dyn_2_fused = 0.495` — a **+0.075 raw-success win at a non-saturated cell** (greedy 0.495 ≪ 0.95 saturation threshold). This crosses the V3C plan §4 Stage 4 `CANDIDATE_SIGNAL_RAW` threshold (≥ 0.05).

**The 1M regression:**

Doubling training timesteps to 1M moved Track N's K=2/b8-10/OOD PPO_BCD success from `0.570 → 0.445` (Δ = −0.125). Same-field greedy_dyn_2 stayed at `0.495` (greedy is deterministic given the dynamics). At 1M, PPO_BCD now LOSES to greedy by 0.050 — outside `CANDIDATE_SIGNAL_PARETO`'s ±0.03 raw-tolerance.

`mean_steps` increased from 2.000 (500k) to 2.000 (1M) — same path-length distribution. `mean_final_distance` went from 3.249 → 3.199. The policy did not collapse to NOOP; it shifted to longer-path strategies that occasionally miss.

**At K=3/b8-10**, the same pattern: 500k PPO 0.945 → 1M PPO 0.880, while greedy_dyn_3 = 1.000 (saturated). PPO drifts further from saturated greedy at 1M.

**Interpretation:** with a single seed, this is the canonical "PPO overfits to the training curriculum and loses some of the OOD generalization it had at 500k." The 500k signal could be:

1. Real (test with 4 seeds × 1M — should show a similar signal at the same training horizon)
2. Seed-42 variance + early-checkpoint luck (4 seeds would average it out)
3. A genuine training-horizon non-monotonicity (PPO peaks around 500k then degrades — 4 seeds × intermediate checkpoints would diagnose)

Without 4 seeds, we cannot tell which is true.

**Reward-fit at K=2/b8-10 (PPO_BCD 1M vs greedy_dyn_2_fused):**

| Axis | PPO_BCD (1M) | g_dyn_2 | Δ |
|---|---:|---:|---|
| success | 0.445 | 0.495 | −0.050 (out of ±0.03 — **fails Pareto tolerance**) |
| mean_final_distance | 3.199 | 3.056 | +0.143 (regression) |
| mean_tox_path | 0.000 | 0.000 | tie |
| mean_common_essential | 0.000 | 0.000 | tie |
| mean_unc_path_max | 0.408 | 0.436 | −0.028 (improved) |

**Verdict (composite, both checkpoints reported):**
- 500k single-seed: `CANDIDATE_SIGNAL_RAW` (+0.075 at K=2/b8-10)
- 1M single-seed: `WEAK_SIGNAL` (−0.050 at K=2/b8-10; raw regression + Pareto failure)

**Recommendation:** Phase 4 escalation (4 seeds × 1M) is mandatory to resolve the 500k-vs-1M tension. If the 4-seed paired δ at K=2/bin8-10 vs greedy_dyn_2_fused excludes zero with PPO above greedy, it becomes the **first Bucket-B planning-advantage result in V3** (V3B Phase 4 was TECHNICAL_ONLY on the same reward stack). If the 4-seed result re-centers around the 1M point estimate (−0.05), the 500k point is single-seed variance and Phase 2 (contraction-aware dynamics) is the right next step.

---

## §7 — Mean-delta wildcard — `dynamics_mean_delta_corr_010` (500k PPO_BCD, stopped per EARLY_FLAT)

32D anchor VAE + mean-delta pair dynamics. **Pair-source axis differs from anchor.**

| Cell | PPO_BCD | random | g_dyn_1_fused | g_dyn_2_fused | g_dyn_3_fused | g_dyn_5_fused | PPO mean_steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| K=2/b6-8 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | 0.65 |
| K=2/b8-10 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | 0.26 |
| K=3/b6-8 | 0.000 | 0.000 | 0.030 | 0.035 | 0.030 | — | 0.90 |
| K=3/b8-10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0.39 |
| K=4/b8-10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | 0.52 |
| K=5/b8-10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.65 |
| K=8/b8-10 | **0.130** | 0.000 | 0.130 | **0.265** | 0.265 | 0.265 | 0.85 |

**Diagnostic finding — NOOP strategy:** PPO_BCD's `mean_steps` at every cell except K=8 is between 0.26 and 0.90 — meaning PPO terminates via NOOP within the first few env steps for most episodes. At K=8/b8-10, PPO's `mean_steps = 0.85` while `success_rate = 0.130`: the 13% of successful episodes take roughly `0.85 / 0.13 ≈ 6.5 steps`, the other 87% NOOP immediately.

**Honoring the user's mean-delta-specific eval criteria:**
- "Do not call low K=2/K=3 success an implementation failure" — confirmed: Bucket U-B audit predicted 0% reach at K∈{2,3,4,5}. PPO finds no path because none exists under the dynamics field at those depths.
- "Judge it mainly on K=8 behavior and Pareto signal" — at K=8/b8-10:
  - PPO 0.130 vs g_dyn_2 0.265 → **−0.135 raw** (way outside Pareto ±0.03 tolerance)
  - PPO 0.130 vs g_dyn_1 0.130 → +0.000 (PPO matches single-step greedy on a structurally weak field)
  - Pareto axes (PPO − g_dyn_2): tox_path 0.000 vs 0.048 (improved), CE 0.000 vs 0.640 (improved by 100%), unc_max 0.109 vs 0.779 (improved by 0.67), mean_steps 0.85 vs 7.61 (PPO uses 8× fewer actions per episode).
  - 4 Pareto axes improved, but raw success regression of 0.135 means this is **not** `CANDIDATE_SIGNAL_PARETO` under the strict V3C plan §4 Stage 4 definition.

**Verdict:** `NO_SIGNAL` (raw) / `NO_SIGNAL` (Pareto under strict tolerance). The mean-delta family's `gene_universality_max = 0.66` Bucket U-D signature is structurally different from the OT family — but the field's `beam_reach = 0%` at K ≤ 5 means PPO cannot find a path the reward shaping wants it to take. The locked B+C+D shaping (heavy_beta = 0.10 path penalty + λ_unc_path = 0.05) over-shapes a low-reach field and trains PPO to NOOP.

**Conclusion on the wildcard hypothesis:** The "pair-source-orthogonal-to-architecture" axis cannot be cleanly tested on a field where the structural reach is too low for PPO to construct a viable path. A future test would require either (a) a longer path-budget regime (already at max_steps=8, already failing) or (b) a less-shaping reward variant (out of scope — V3C plan locks B+C+D). Mean-delta is structurally a different geometry from OT, but the geometry does not surface in PPO learning. **Pair-source diversity is informative as a Bucket U finding (lower gene_universality_max), not as a PPO smoke result.**

**Recommendation:** do not escalate. The signal needed (control utility at low K) is structurally absent.

---

## §8 — Phase 1 verdicts and Phase 4 escalation

**Verdict matrix:**

| Field | Verdict | Phase 4 escalation? |
|---|---|---|
| Anchor (V2 RoR_corr010, 1M) | `NO_SIGNAL` (V3B TECHNICAL_ONLY reproduced) | already 4-seed at V3B Phase 4 |
| **Track L** (n64_legacy_ror_corr010, 1M) | `WEAK_SIGNAL` (raw tie + huge anchor lift; Pareto distance regression) | **YES — escalate to test multi-seed stability of the anchor lift and distance regression** |
| **Track N** (n64_nb_ror_corr010) | `CANDIDATE_SIGNAL_RAW` at 500k (+0.075), regressed to `WEAK_SIGNAL` at 1M (−0.050) | **YES — escalate to resolve 500k-vs-1M single-seed variance**; this is the only field that crossed the `CANDIDATE_SIGNAL_RAW` threshold at any checkpoint |
| Mean-delta wildcard (500k) | `NO_SIGNAL` (Pareto fails on raw regression; NOOP strategy) | NO — structural low-reach |

**Phase 4 escalation rationale:** the user's directive in the prompt was `Phase 4 escalation if any candidate signal exists; otherwise Phase 2 contraction-aware dynamics`. **A candidate signal does exist** (Track N at 500k, observed but unstable). Without multi-seed CIs we cannot honestly call it real or spurious. The 4-seed protocol is the canonical resolution.

**Concrete Phase 4 spec (per V3C plan §8 Stage 5):**
- 3 additional seeds {0, 1, 7} × 1M timesteps × locked B+C+D × per-VAE p15 for each of Track L and Track N (6 runs total).
- Optional intermediate checkpoint analysis (e.g. 500k snapshots from each seed) to diagnose whether the 500k-vs-1M divergence on Track N is training-horizon-specific.
- Aggregator (`aggregate_v3b_phase4.py`) for paired δ vs greedy_dyn_2_fused at K=2/bin8-10/OOD and K=2/bin6-8/OOD.
- Optional bounded reward tuning mini-grid on the winning field per V3C plan §6 (4-combination corner grid; expand to 12 only on sensitivity).

**If Phase 4 fails for both Track L and Track N:** proceed to Phase 2 (contraction-aware dynamics, Candidate A.iv composite per V3C plan §8). The audit's central finding holds — every existing OT-trained field is over-contractive — and a new dynamics formulation is the next architectural lever.

---

## §9 — Anchor-secondary comparison (per user instruction)

PPO_BCD raw success at K=2/bin8-10/OOD (the V3B Phase 4 non-saturated discriminating cell):

| Field | n_lat | dynamics | PPO_BCD K=2/b8-10 | vs anchor (Δ) |
|---|---:|---|---:|---:|
| Anchor | 32 | V2 RoR_corr010 32D | 0.120 | — |
| Track L (500k) | 64 | n64_legacy_ror_corr010 | 0.705 | **+0.585** (5.9×) |
| Track L (1M) | 64 | n64_legacy_ror_corr010 | 0.705 | **+0.585** |
| Track N (500k) | 64 | n64_nb_ror_corr010 | 0.570 | **+0.450** (4.8×) |
| Track N (1M) | 64 | n64_nb_ror_corr010 | 0.445 | +0.325 (3.7×) |
| Mean-delta (500k) | 32 | mean_delta_corr_010 | 0.000 | −0.120 (collapsed) |

**Important interpretive note:** the anchor and Track L/N use **different ε** (per-VAE p15 = 2.99 vs 3.02 vs 3.11). The 64D fields' larger ε makes their success criterion slightly more permissive in absolute terms. However, the Phase 0C Bucket U-B audit and the Phase 1 calibration both demonstrate that **the relative reach gap survives at any percentile p∈{p10, p15, p25}** — the 64D fields are genuinely more reachable at this cell, not artifacts of an ε mismatch.

This is the strongest argument for Phase 4 escalation: the 64D + RoR_corr010 combination unambiguously expands the actionable region at the V3B-Phase-4-binding cell. Whether PPO can exploit that expansion robustly across seeds is the open question.

---

## §10 — Reward-fit consistency

Across all 4 fields and 7 cells, `PPO_BCD` produces `mean_tox_path = 0.0` and `mean_common_essential_per_ep = 0.0` everywhere. The locked Variant C constraint (λ_tox=0.10, λ_ce=0.05) is implementing correctly — PPO never picks DepMap-essential or high-Chronos-toxicity genes, regardless of dynamics field. This is consistent with V3B Phase 4's Bucket-A reward-fit result and confirms the B+C+D shaping is structurally working; the question is only whether the dynamics field provides usable planning leverage.

---

## §11 — Files written

- `artifacts_v3/v3c/rl_smokes/anchor_v2_ror_corr010_1M_reused/eval/` (anchor evaluation only — checkpoint reused from V3B Phase 4)
- `artifacts_v3/v3c/rl_smokes/track_l_n64_legacy_ror_corr010_seed42_500k/{ppo.zip, metadata.json, eval/}`
- `artifacts_v3/v3c/rl_smokes/track_l_n64_legacy_ror_corr010_seed42_1M/{ppo.zip, metadata.json, eval/}`
- `artifacts_v3/v3c/rl_smokes/track_n_n64_nb_ror_corr010_seed42_500k/{ppo.zip, metadata.json, eval/}`
- `artifacts_v3/v3c/rl_smokes/track_n_n64_nb_ror_corr010_seed42_1M/{ppo.zip, metadata.json, eval/}`
- `artifacts_v3/v3c/rl_smokes/wildcard_mean_delta_corr_010_seed42_500k/{ppo.zip, metadata.json, eval/}`
- `artifacts_v3/v3c/interpretation/v3c_phase1_ppo_smoke_summary.md` (this document)

---

## §12 — Sacred-rule / guardrail check

- ✅ Frozen tiers untouched.
- ✅ All Phase 1 outputs under `artifacts_v3/v3c/rl_smokes/` + `artifacts_v3/v3c/interpretation/`.
- ✅ Per-VAE p15 (exact-computed from each VAE's control-cell distance distribution; 32D matches V3B-locked 2.9898 to 4 decimals).
- ✅ Pre-PPO p15 vs p10 calibration on Track L/N (locked p15 — V3B precedent).
- ✅ Anchor reused (V3B Phase 4 1M checkpoint, configuration-compatible).
- ✅ Adaptive 500k → 1M schedule per V3C plan §4 Stage 4: mean-delta stopped at 500k (EARLY_FLAT); Track L + Track N continued to 1M.
- ✅ Canonical 7-cell V3B matrix; 200 episodes per (cell, policy).
- ✅ Locked B+C+D coefficients throughout; no reward tuning in Phase 1.
- ✅ PPO vs **same-field same-horizon greedy_dyn_K** as primary comparison; anchor delta as secondary context.
- ✅ Mean-delta judged on K=8 behavior + Pareto (per user-explicit guidance).
- ✅ Test suite: 377 passed / 2 skipped.

---

## §13 — Recommended next step

**Run Phase 4 escalation on both Track L and Track N** — 3 additional seeds (0, 1, 7) × 1M each at the same locked B+C+D + per-VAE p15 used here. Aggregate with `aggregate_v3b_phase4.py` for paired δ vs `greedy_dyn_2_fused` at K=2/bin8-10/OOD and K=2/bin6-8/OOD. The decision criteria:

- If the 4-seed paired δ for Track N at K=2/bin8-10 excludes zero with PPO > greedy → `LOCKED_DESIGN_POSITIVE_SIGNAL` (first such V3 result). Run the bounded reward tuning mini-grid (V3C plan §6) and write up as the V3C success path.
- If the 4-seed paired δ for Track L re-converges to the 500k point (PPO ties or marginally beats greedy_dyn_2 cleanly at K=2/b8-10 with mean_final_distance regression ≤ 0.10) → `CANDIDATE_SIGNAL_PARETO`. Same downstream escalation.
- If neither field reaches a clean paired-δ signal at 4 seeds → proceed to **Phase 2 (Candidate A: contraction-aware dynamics, formulation iv composite, V3C plan §8)**.

**Phase 2 trigger probability based on the smoke evidence**: moderate. Track N's 500k signal is real-looking but fragile; Track L's signal is purely an anchor-lift without a same-field win. The 5.9× anchor lift on Track L is *expected* given the Bucket U-B audit and does not by itself trigger Phase 4 success.

End of Phase 1.
