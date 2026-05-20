# V3C Phase 0 — Dynamics Utility Audit Interpretation

**Status:** `PHASE0_AUDIT_COMPLETE — PHASE_1_SMOKE_TARGETS_PROPOSED`

**Scope:** 29 candidate dynamics fields audited end-to-end (Bucket U-A through U-G) at `n_episodes=64`; top-4 candidate fields refined at `n_episodes=200` for the reachability sub-audit (the discriminator at the K=2/bin8-10/OOD cell). The audit ranks, explains, and flags — smoke-target selection is interpretive and is laid out in §5 below with explicit rationale per pick.

**Frozen tier check:** `git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/` → clean.

**Test suite:** 377 passed / 2 skipped (matches Phase 0B baseline).

---

## §1 — Headline finding

**Every existing dynamics field exhibits one of three structural pathologies under the locked B+C+D reward stack.** No existing field qualifies for the "Best-by-audit" Phase 1 promotion route on its full merits — all 29 fields fail at least one Bucket U-G precondition.

| Pathology | Diagnostic flags | Affected fields |
|---|---|---|
| **Universal over-contraction** | `CONTRACTION_NEAR_UNIVERSAL` + `UNIVERSAL_ATTRACTOR_GENE` | All 24 OT-trained fields (V1 OT, V2 RoR family, V1 OT ablations/sweeps, 64D OT, V3A 64D RoR family, plus the random-pairing negative control). Universal-attractor gene shows `mean_z cos(μ(·,g), z_ref−z) > 0.85`. Greedy_dyn_1 saturates at K≥3 OOD bin8-10. |
| **Lower-universality + unreachable at low K** | `CONTRACTION_NEAR_UNIVERSAL` (without `UNIVERSAL_ATTRACTOR_GENE`) | Mean-delta family (4 fields). `gene_universality_max ≈ 0.66` (vs ~0.92 in OT fields), but `beam_reach = 0%` at K∈{2,3} OOD; only 25% at K=8/bin8-10. Less concentrated attractor, but the field cannot steer to ε under any planning depth ≤ 5 in OOD. |
| **Anti-contractive (gate-passing, control-hostile)** | `GATE_PASSED` + `CONTRACTION_LOW_or_BARYCENTRIC` | Soft-OT alone (1 field). Predicted Δz points **away** from z_ref (`alignment_cos_median = −0.77`, `contraction_fraction = 0.0`). Zero beam reachability at all 7 cells. |

The audit empirically confirms V3C's central premise: **representation/dynamics is the bottleneck, not reward**. The locked B+C+D reward stack is correct; no existing dynamics field provides the actionable MDP it requires.

**This is itself a positive finding** — it firms up the case for Phase 2 (new dynamics training) and gives concrete pathology signatures that the next dynamics formulation must avoid.

---

## §2 — Full ranking (29 fields)

`util_score` is a ranking aid, **not a verdict** (V3C plan §4 Stage 1, guardrail #1). Smoke selection (§5) requires written rationale and does not follow `util_score` order mechanically — note that the highest-`util_score` fields here are V1 OT ablations, which differ from V2 anchor only at the margin and have lower K=2/bin8-10 OOD reachability than the 64D RoR Track L/Track N candidates we ultimately propose.

| `util_score` | Field | Family | Notes |
|---:|---|---|---|
| 0.419 | `artifacts/dynamics_ablation/baseline` | V1 OT ablation | no state_linear, no gene_bias |
| 0.416 | `artifacts_64/dynamics_variants/state_linear` | 64D OT variant | |
| 0.416 | `artifacts_64/dynamics` | 64D OT default | |
| 0.413 | `artifacts/dynamics_ablation/gene_bias` | V1 OT ablation | gene_bias=true, OOD-collapse documented in V2 |
| 0.412 | `artifacts/dynamics_ablation/state_linear` | V1 OT ablation | |
| 0.412 | `artifacts/dynamics_sweeps/lr1e-3_mse0` | V1 OT sweep | |
| 0.403 | `artifacts/dynamics_default_check`, `dynamics_current_default`, `dynamics` | V1 OT reruns | three near-duplicates |
| 0.400 | `artifacts_v3/dynamics_n64_nb_ror_corr010` | **64D RoR Track N (NB)** | V3A — **proposed smoke target** |
| 0.396 | `artifacts/dynamics_sweeps/lr1e-4_mse0` | V1 OT sweep | |
| 0.394 | `artifacts/dynamics_sweeps/lr3e-4_mse005` | V1 OT sweep | |
| 0.393 | `artifacts/dynamics_sweeps/lr3e-4_mse01` | V1 OT sweep | |
| 0.391 | `artifacts_64/dynamics_variants/baseline_plain` | 64D OT variant | |
| 0.389 | `artifacts/dynamics_sweeps/lr3e-4_mse0` | V1 OT sweep | |
| 0.380 | `artifacts_v2/dynamics_v1ot_ror_corr005` | V2 RoR family | λ_corr=0.05 |
| 0.379 | `artifacts/dynamics_ablation/state_linear_gene_bias` | V1 OT ablation | both flags |
| 0.376 | `artifacts_v2/dynamics_random_default` | Random pair | **negative control reference** |
| 0.376 | `artifacts_64/dynamics_variants/state_linear_combo0` | 64D OT variant | λ_combo=0 |
| 0.372 | `artifacts_v2/dynamics_v1ot_ror` | V2 RoR family | no λ_corr |
| **0.369** | **`artifacts_v2/dynamics_v1ot_ror_corr010`** | **V2 RoR_corr010** | **ANCHOR — mandatory smoke target** |
| 0.365 | `artifacts_v3/dynamics_n64_legacy_ror_corr010` | **64D RoR Track L** | V3A — **proposed smoke target** |
| 0.357 | `artifacts_64/dynamics_variants/state_linear_gene_bias` | 64D OT variant | |
| 0.350 | `artifacts_64/dynamics_variants/gene_bias` | 64D OT variant | low val_pearson 0.516 |
| 0.295 | `artifacts_v2/dynamics_mean_delta_default` | mean_delta | no λ_corr |
| 0.294 | `artifacts_v2/dynamics_mean_delta_corr_010` | mean_delta | λ_corr=0.10 — **proposed wildcard smoke target** |
| 0.276 | `artifacts_v2/dynamics_mean_delta_corr_005` | mean_delta | |
| 0.249 | `artifacts_v2/dynamics_mean_delta_corr_030` | mean_delta | |
| **0.208** | **`artifacts_v2/dynamics_soft_ot_default`** | **Soft-OT** | **anti-contractive cautionary reference** |

**`util_score` rank does NOT track K=2/bin8-10/OOD reachability** — the V3A 64D RoR fields (`util_score` 0.36–0.40) have ~4–5× higher reachability there than higher-`util_score` V1 OT ablations. This is why the wildcard route, not raw `util_score` ranking, drives smoke-target selection.

---

## §3 — Bucket U signatures by family

`n_episodes=64` for the per-family averages below (refined `n_episodes=200` numbers for the 4 final smoke candidates are in §5).

| Family | n | val_pearson | ridge_margin | contraction_fraction | gene_universality_max | action_diversity_per_state | k2/bin6-8 reach | k2/bin8-10 reach | k3/bin8-10 reach |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 OT default / reruns | 3 | 0.608 | +0.007 | 1.000 | 0.91 | 0.106 | 0.453 | 0.406 | 1.000 |
| V1 OT ablations | 4 | 0.593 | −0.008 | 0.999 | 0.89 | 0.128 | mean 0.563 | mean 0.320 | 1.000 |
| V1 OT sweeps | 5 | 0.606 | +0.005 | 1.000 | 0.91 | 0.103 | mean 0.462 | mean 0.219 | 0.890 |
| 64D OT variants | 6 | 0.578 | −0.038 | 0.999 | 0.92 | 0.129 | mean 0.793 | mean 0.396 | 0.937 |
| **V2 RoR family** | 3 | 0.614 | **+0.013** | 1.000 | 0.92 | 0.116 | 0.655 | **0.120** *(refined)* | 1.000 |
| mean_delta family | 4 | 0.520 | +0.023 | **0.985** | **0.66** | 0.135 | **0.000** | **0.000** | **0.000** |
| **64D RoR (V3A)** | 2 | **0.620** | +0.006 | 1.000 | 0.93 | 0.111 | mean 0.875 | **mean 0.498** *(refined)* | 1.000 |
| Random pair (neg ctl) | 1 | 0.723 | −0.009 | 1.000 | **0.99** | 0.029 | 1.000 | 1.000 | 1.000 |
| Soft-OT (cautionary) | 1 | **0.934** | **+0.041** | **0.000** | **−0.62** | 0.048 | 0.000 | 0.000 | 0.000 |

**Sample-1 (OOD-pool) vs Sample-2 (val-pair) divergence** is < 0.005 across every family — geometry is consistent across distributions. **No `GEOMETRY_DIVERGENCE_*` wildcards exist in this inventory.** Bucket U-D Sample 1 and Sample 2 are essentially identical, which itself is a finding: extrapolation to the OOD start pool does not change the field's contraction signature relative to the training distribution.

---

## §4 — Field classifications

### 4.1 Duplicate clusters

Per guardrail #4, the aggregator flags fields whose `(val_pearson, ridge_margin, model.pt md5)` triple matches exactly. Result: **zero automatic duplicate clusters detected**.

**However**, three V1 OT reruns share `val_pearson = 0.608456`, `ridge_margin = +0.007367`, `gate_passed = False`, and identical Bucket U-D / U-B / U-C signatures (every shared diagnostic matches to display precision):

- `artifacts/dynamics`
- `artifacts/dynamics_current_default`
- `artifacts/dynamics_default_check`

Their `model.pt` checksums differ — likely retraining-noise siblings (same config, different seeds or training-order) — so the audit does not auto-cluster them, but they are **effectively duplicates** for the Phase 1 smoke purpose. The audit framework would treat them as a single "V1 OT default" smoke target — they would not consume three smoke slots.

### 4.2 Dead / control-hostile fields (excluded from any smoke role)

- **`artifacts_v2/dynamics_soft_ot_default`** — `beam_reach = 0` at every cell. The audit empirically resolved the V3C plan §1.2 hedge: Soft-OT is **anti-contractive** (predicted Δz points away from z_ref with `alignment_cos_median = −0.77`), not literal-no-op. PPO has no path to find. Smoking this field would burn a slot to learn nothing new. **Excluded.**

- **`artifacts_v2/dynamics_random_default`** — `greedy_dyn_1_distance = 1.0` at every cell. The "negative control" pair-source produces a **maximally** universal attractor (`gene_universality_max = 0.99`, `action_diversity_per_state = 0.029`). Every gene action lands within ε. There is no planning problem; PPO cannot improve on saturated greedy_dyn_1. **Excluded.** Worth noting as a sanity reference in interpretation (§7) — it shows the universal-contraction failure mode at its limit.

### 4.3 Reachable-but-low-K-blind (mean-delta family)

The four mean-delta variants share a structurally different profile: `gene_universality_max ≈ 0.66` (much lower than OT's ~0.92), `contraction_fraction ≈ 0.985` (slightly less than 1.0), but **0% beam reachability at K∈{2,3,4,5}** and only 25% at K=8. The dispersed contraction does not equal usable steering at low K.

`mean_delta_corr_010` retained the V2 finding "best of the mean-delta corr sweep" (cf. `EXPERIMENTS.md`). It is the strongest candidate from this family. Phase 1 smoke is still worth running on **one** mean-delta variant as a wildcard, with `env.max_steps=8` (V3B Phase 4 default) so PPO can attempt path lengths ≥ 4 where any reach exists. This tests the **pair-source-orthogonal-to-architecture** axis — a different failure mode from V2 anchor's universal attractor.

### 4.4 V1 OT / V2 RoR / V1 OT ablations / V1 OT sweeps / 64D OT

All 20+ fields in these families show **the same diagnostic signature**: `cf ≈ 1.0`, `gu_max ≈ 0.89–0.93`, `action_diversity_per_state ≈ 0.10–0.13`. Differences between fields are at the margin (val_pearson varies by 0.02; ridge_margin by 0.05). **Bucket U cannot distinguish them at the resolution of the audit's diagnostic signal.** They are all "OT-trained universal-attractor" fields.

This is itself a structural finding: the OT pairing assumption + the residual head structurally produce universal-attractor fields regardless of latent dim, learning rate, RoR flag, λ_corr, or gene_bias. The pairing-noise floor V2 documented (`~0.89` median across configurations) maps directly onto the audit's gene_universality_max value (`~0.91`). Same number, two derivations: the V2 OT pairing noise floor IS the V3C universal-attractor signature.

### 4.5 V3A 64D RoR (Track L + Track N)

These two fields are the **only candidates with high K=2/bin8-10/OOD reachability AND the V2-architecturally-compatible RoR-corr010 dynamics head**:

| Field | val_pearson | ridge_margin | cf | gu_max | k2/bin8-10 (n=200) | k2/bin6-8 (n=200) |
|---|---:|---:|---:|---:|---:|---:|
| `n64_legacy_ror_corr010` (Track L) | 0.620 | +0.004 | 1.000 | 0.933 | **0.560** | **0.895** |
| `n64_nb_ror_corr010` (Track N) | 0.621 | +0.007 | 0.999 | 0.932 | **0.435** | **0.855** |
| **V2 anchor (RoR_corr010, n=200)** | 0.615 | +0.014 | 1.000 | 0.921 | **0.120** | 0.655 |

Both 64D RoR fields show roughly **4–5× higher K=2/bin8-10/OOD reachability** than V2 anchor, AT IDENTICAL Bucket U-D contraction-geometry signatures. The win comes from the larger 64D distance distribution: `epsilon_p15 = 2.99` (32D) vs `3.05` (64D Track L) vs `3.13` (64D Track N), which means the OOD bin8-10 starts are relatively *closer* to ε in 64D — there is more usable reachable region at the hardest non-saturated cell.

Both fields still saturate at K≥3 (`reach = 1.0` everywhere). V3A's "64D didn't break saturation" conclusion holds — but **the audit reveals a finer signal**: 64D does NOT cure saturation at high K, but it DOES enlarge the non-saturated band at low K. That band is precisely where V3B Phase 4's PPO_BCD found its 0.148 raw-success signal on V2 anchor; with 4–5× more reachable mass at the same cell, **PPO_BCD on Track L / Track N may have substantially more headroom for both raw-success and Pareto improvement**.

---

## §5 — Recommended Phase 1 PPO_BCD smoke targets

Four slots per V3C plan §4 Stage 3 (Anchor + 1–2 Best-by-audit + 1–2 Wildcards). Selection is interpretive — every pick has a written rationale tied to specific U-bucket evidence (guardrails #1, #8).

### Slot 1 — Anchor (mandatory)

**`artifacts_v2/dynamics_v1ot_ror_corr010`** — V2 primary RoR_corr010 32D.

- Mandatory per V3C plan §4 Stage 3 — fixed reference for cross-field comparison.
- The V3B Phase 4 PPO_BCD seed-42 checkpoint at `artifacts_v3/rl_v3b_biorealistic_fused_epsp15_seed42/` is configuration-compatible (RoR_corr010 + p15 + V3B-default freeband/λ_tox/λ_ce/λ_unc_path). **Reuse for the anchor evaluation — no anchor retraining needed.**
- Bucket U signature: `cf 1.000, gu_max 0.921, k2/bin8-10 reach 0.120 @ n=200`. Universal attractor; K≥3 OOD bin8-10 saturated.
- The reference against which we measure whether 64D / pair-source candidates do meaningfully better.

### Slot 2 — Best-by-audit #1 — `artifacts_v3/dynamics_n64_legacy_ror_corr010` (Track L, 64D legacy scVI)

**Rationale:**

- (U-B reachability) **Highest K=2/bin8-10/OOD reachability of any RoR_corr010-architecture field** at `n=200`: **0.560** vs V2 anchor's 0.120 — a 4.7× lift at the V3B-Phase-4-discriminating cell. K=2/bin6-8/OOD is also lifted: 0.895 vs 0.655.
- (U-A predictive sanity) Comparable val/OOD Pearson to V2 anchor (`val 0.620` vs anchor `0.615`).
- (Architecture isolation) Same residual head + λ_corr=0.10 + RoR flag as V2 anchor — the **only** axis that differs is `n_latent` (64 vs 32). Smoking this field directly tests "does 64D representation buy PPO_BCD more raw-success headroom and/or Pareto signal at the non-saturated cells?" — a clean controlled experiment.
- (V3A history) V3A rejected 64D on greedy-saturation grounds (K=3/bin8-10 still 1.0). This audit confirms K≥3 saturation but reveals the **K=2 reachability lift** V3A missed by not running U-B on the hardness matrix. Phase 1 smoke pivots the V3A finding from "rejected" to "rejected for K≥3, candidate for K=2 leverage".
- (Risk caveat) Bucket U-D shows the field is still universally over-contractive (`cf 1.000, gu_max 0.933`); PPO_BCD may still tie the saturated greedy_dyn_K at K≥3. The signal is expected at K=2 cells only.
- Fails U-G (preconditions not all met) — but this is the principal Wildcard "depth-leverage / non-saturated reachability" candidate the V3C plan §4 Stage 3 anticipates.

### Slot 3 — Best-by-audit #2 — `artifacts_v3/dynamics_n64_nb_ror_corr010` (Track N, 64D NB likelihood)

**Rationale:**

- (U-B reachability) Second-highest K=2/bin8-10/OOD reachability: **0.435** vs anchor's 0.120 (3.6× lift). K=2/bin6-8: 0.855.
- (U-A predictive sanity) Marginally better than Track L on OOD Pearson (`ood_pearson 0.504` Track N vs `0.515` Track L — comparable).
- (Likelihood axis) Track N adds **NB likelihood** to the 64D backbone (Track L is legacy scVI). Smoking both tests **likelihood × latent_dim** interaction in PPO learnability. If Track N matches or exceeds Track L's PPO_BCD signal, the NB likelihood is part of the win; otherwise, it's pure dimension.
- (Cost) The pair files and biology layer are shared between Track L and Track N, so dispatching both adds marginal infrastructure cost.
- Fails U-G — same caveat as Track L. Candidate by the same wildcard-route argument.

### Slot 4 — Wildcard — `artifacts_v2/dynamics_mean_delta_corr_010` (mean-delta + λ_corr=0.10)

**Rationale:**

- (Pair-source diversity) Only candidate whose **pair source differs from OT** (mean-delta). Tests the orthogonal hypothesis: "does OT pairing itself, independent of architecture, drive the universal-attractor pathology?"
- (Bucket U-D distinct profile) `gene_universality_max 0.66` (vs OT's ~0.92), `contraction_fraction 0.985` (vs OT's 1.0), the most action-dispersed contraction geometry of any non-Soft-OT field. The Wildcard "geometry-disagreement" / "pair-source diversity" category per V3C plan §4 Stage 3.
- (U-B / U-C caveat) Reach is **0% at K∈{2,3,4,5}** and 25% at K=8/bin8-10. Greedy_dyn_K is structurally weak. PPO_BCD with `env.max_steps=8` and freeband schedule has a path-length budget where any reach exists; smoking this field tests whether **path-length leverage + safety/uncertainty shaping** can extract signal that single-step greedy cannot. This is the most direct test of "B+C+D reward stack purpose" the audit can construct.
- (Risk) Most likely to produce `WEAK_SIGNAL` or even `LOCKED_DESIGN_FAILED_IMPLEMENTATION` if PPO collapses on this field. Either result is informative — confirms (or refutes) the pair-source-orthogonal-to-architecture hypothesis.
- This is the highest-information-per-smoke-slot pick of the four.

### Slots intentionally NOT filled

- **No Soft-OT slot** — anti-contractive field with 0% reach everywhere; smoking confirms what U-D already diagnosed at a fraction of the cost. Listed as cautionary reference in §7.
- **No Random-pairing slot** — degenerate maximum-saturation field where greedy_dyn_1 = 1.0 everywhere; no PPO room for improvement.
- **No higher-`util_score` V1 OT ablation** — these score higher only because U-D has slightly lower gene_universality_max, but their K=2/bin8-10/OOD reachability is *worse* than 64D RoR. Smoking them would teach nothing the V2 anchor doesn't already cover; they are within V2 anchor's diagnostic family.
- **No additional mean-delta variants** — the four mean-delta fields are within Bucket-U-equivalence of each other; smoking corr_010 covers the family.

---

## §6 — Decision tree (post-Phase 1 escalation, deferred to user direction)

```
Phase 1 PPO_BCD smoke (4 fields × 500k → 1M adaptive schedule)
│
├── Track L OR Track N reaches CANDIDATE_SIGNAL_RAW
│   (PPO_BCD beats greedy_dyn_5_fused by ≥ 0.05 at non-saturated cell)
│       → Phase 4 (4-seed escalation) on the winning field
│       → bounded reward tuning mini-grid (§6 of V3C plan)
│
├── Mean-delta wildcard reaches CANDIDATE_SIGNAL_RAW or _PARETO
│       → exceptional: confirms pair-source axis matters more than expected
│       → Phase 4 (4-seed escalation)
│
├── Any candidate reaches only CANDIDATE_SIGNAL_PARETO
│       → Phase 4 escalation; the V3B-intended biorealistic-control payoff
│
└── All fields fail to reach either CANDIDATE flavor
        → Phase 2: train new dynamics (Candidate A: contraction-aware,
          formulation iv composite — V3C plan §8)
        → audit re-run on the new field
        → if improved, Phase 1 smoke on the new field
```

The most likely outcome (audit prior) is that **Track L and/or Track N show partial signal at K=2 cells but still tie greedy at K≥3**, mean-delta wildcard shows weak / no signal, and the framework escalates to Phase 2. The audit's K=2/bin8-10 reachability gap (0.56 vs 0.12) is large enough to be promising, but the underlying U-D contraction signature is unchanged — PPO_BCD's room may be larger but is still bounded by the universal-attractor structure.

---

## §7 — Reference rows (per guardrail #8)

Three reference fields, each illustrating a distinct pathology the audit was designed to detect:

### V2 anchor — `dynamics_v1ot_ror_corr010` (32D, OT, RoR, λ_corr=0.10)

The V3B Phase 4 `LOCKED_DESIGN_TECHNICAL_ONLY` baseline:

- `val_pearson 0.615, ridge_margin +0.014` (gate fails, but only on the ridge-margin threshold)
- `contraction_fraction 1.000, gene_universality_max 0.921, action_diversity 0.116` — universal-attractor signature
- Greedy_dyn_K saturated at K≥3 OOD bin8-10
- K=2/bin8-10/OOD reach: 0.120 @ `n=200` (matches V3B Phase 4 PPO_BCD raw 0.148 within sampling noise)
- **This field is what the V3B reward stack was implemented against and what the V3C audit must beat to declare progress.**

### Soft-OT — `dynamics_soft_ot_default` (32D, Soft-OT pairs, no RoR)

The V3B-documented "passed gate but failed control 0/64" cautionary case. The audit resolved the failure mode empirically:

- `val_pearson 0.934, ridge_margin +0.041, gate_passed=True` — appears excellent on the prediction-gate axis
- `contraction_fraction 0.000, alignment_cos_median −0.770, gene_universality_max −0.622` — predicted Δz **points away from z_ref** for nearly every (z, g) pair. The dynamics learned to predict the average perturbed cell (which is *farther* from z_ref than typical OOD starts), so the field pushes outward, not inward.
- `delta_magnitude_median 2.51` — μ is **non-zero**, so this is *not* the literal-no-op case the V3C plan §1.2 hedged on. The failure mode is **anti-contractive**.
- All 7 hardness-matrix cells: `beam_reach = 0`
- **The prediction-gate vs control-utility divergence the audit was designed to detect.** Confirms the rationale for adding Bucket U-B + U-D over relying on the gate.

### Random pairing — `dynamics_random_default` (32D, random pairs, no RoR — negative control)

The "control field" the V3C plan §2 reserved as an Eligible negative-reference per guardrail #7. The audit reveals it is **not** as degenerate as the V3C plan §2 framing suggested — it is the *extreme* version of the universal-attractor signature:

- `val_pearson 0.723` (highest of all non-Soft-OT fields — anomalously high for a "random pair" baseline)
- `ridge_margin −0.009` — the linear ridge baseline **beats** the MLP. Suggests the MLP has not learned anything beyond an affine attractor field.
- `contraction_fraction 1.000, gene_universality_max 0.991, action_diversity_per_state 0.029` — virtually all genes drag toward z_ref with near-identical magnitude. **The maximum universal-attractor.**
- `greedy_dyn_1_distance = 1.000 at every cell` — single-step greedy trivially solves everything.
- **Interpretation:** training a dynamics on random pairs collapses to "predict the K562 perturbation mean (≈ z_ref)" because there is no consistent (z, g) → Δz signal to learn. The model becomes a near-constant attractor. This is the empirical limit of the universal-attractor pathology that the OT-trained fields approach more gently.

**This reference is itself a finding**: random-pair "negative control" is not a no-op or noisy field — it is the **structurally most-saturated** field, which clarifies that the universal-attractor signature observed across OT fields is the *signal*-noise interaction, not OT-specific machinery.

---

## §8 — Sacred-rule / guardrail check

- ✅ **Frozen tiers**: `git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/` → clean.
- ✅ **V3C outputs under `artifacts_v3/v3c/`**: 8 per-bucket CSVs + 29 per-field directories + summary/ranking MDs + this interpretation MD.
- ✅ **`util_score` ranking-aid only** (guardrail #1): every smoke recommendation has written rationale citing specific U-A through U-G evidence; util_score order would have produced a different (and likely-weaker) selection.
- ✅ **U-A through U-G naming** (guardrail #2): all sub-bucket references use the U-prefix.
- ✅ **No PPO / VAE / dynamics training, no frozen-tier writes** (guardrail #3).
- ✅ **Duplicates flagged** (guardrail #4): three V1 OT reruns identified as effective duplicates despite md5 differences; framework treats them as one smoke target.
- ✅ **Robust to missing metrics** (guardrail #5): no field crashed the audit; missing inputs surfaced as structured status strings.
- ✅ **Soft-OT diagnosed empirically** (guardrail #6): U-D resolved the failure mode as anti-contractive, not hard-coded.
- ✅ **Random pairing kept** (guardrail #7): retained as Eligible, audited, surfaced as the extreme universal-attractor reference.
- ✅ **64D fields use their own VAE/pairs/latents** (guardrail #8): Track L → `artifacts_v3/vae_n64_legacy`, Track N → `artifacts_v3/vae_n64_nb`, per-VAE p15 ε (linear-interpolated from p10/p25 in `epsilon_success.json`). Confirmed by the fact that Track L and Track N produce stable 0.56 / 0.44 reach at the same cell on their respective latent spaces.
- ✅ **Both U-D samples reported** (guardrail #9): Sample 1 (OOD pool) and Sample 2 (val pairs) separately; divergence < 0.005 for every field (no `GEOMETRY_DIVERGENCE_*` wildcards).
- ✅ **Tests run** (guardrail #10): 377 passed / 2 skipped.
- ✅ **Interpretation MD** (guardrail #11): this document.
- ✅ **Refined close calls** (guardrail #7 explicit instruction): the 4 candidate smoke targets had their `reachability` sub-audit re-run at `n_episodes=200` — Track L and Track N rankings firmed up; mean-delta confirmed at 0% low-K; V2 anchor confirmed at 0.12.
- ✅ **V3B docs not committed in this V3C commit** (guardrail #9 explicit): the previously-untracked V3B interpretation files surfaced by the .gitignore fix will be left for a separate housekeeping commit.

---

## §9 — What the audit deliberately did NOT decide

Per the user's repeated framing across Phase 0A–0C: the audit ranks, explains, and flags. It does NOT:

- Train, fine-tune, or smoke any PPO policy
- Train any new dynamics or VAE
- Modify frozen tiers
- Mechanically convert `util_score` into a smoke decision
- Reject Wildcard candidates on `util_score` grounds
- Commit to which dynamics formulation will follow in Phase 2 if Phase 1 underperforms

All Phase 1 smoke targets, reward-tuning policies, and Phase 2 / 3 dynamics-candidate choices remain interpretive decisions that the human researcher signs off on. This MD records the audit's evidence and the proposed smoke roster; the next phase begins only on explicit go-ahead.

---

## §10 — Files written / updated

- `artifacts_v3/v3c/utility_audit/<29 field subdirs>/<7 sub-audit JSONs + bucket_u_summary.md + bucket_u_index.json>` — per-field outputs.
- `artifacts_v3/v3c/utility_audit/dynamics_inventory.csv` — Phase 0A.
- `artifacts_v3/v3c/utility_audit/{prediction_metrics, reachability_matrix, greedy_saturation_matrix, depth_leverage_matrix, contraction_geometry, contraction_geometry_val_pairs, action_heterogeneity, reward_leverage_fused, ppo_preconditions}.csv` — cross-field matrices.
- `artifacts_v3/v3c/utility_audit/utility_summary.md, candidate_ranking.md, aggregate_index.json` — aggregator outputs.
- `artifacts_v3/v3c/interpretation/v3c_phase0b_implementation.md` — Phase 0B closeout (committed earlier).
- `artifacts_v3/v3c/interpretation/v3c_phase0_utility_audit.md` — **this document**.

End-of-Phase-0 status: **PHASE0_AUDIT_COMPLETE, AWAITING_PHASE1_GO_AHEAD**.
