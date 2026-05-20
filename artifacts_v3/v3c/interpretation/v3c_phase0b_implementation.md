# V3C Phase 0B — utility audit implementation + smoke

**Status:** `PHASE0B_IMPLEMENTATION_COMPLETE` (driver + aggregator built and TDD-tested; 2-field smoke confirms the audit produces actionable diagnostic output. Full Phase 0C audit on all 29 inventory fields is the next step under the user's direction.)

**Scope:** Implementation only. Smoke-budget allocation, candidate ranking, and PPO_BCD smoke targets remain interpretive Phase 0D work.

---

## What Phase 0B delivered

### Library — `src/analysis/dynamics_utility.py`

Pure-function metric implementations for Bucket U sub-buckets U-D, U-E, U-F, and the composite `util_score`:

- `compute_contraction_geometry(dynamics, z_starts, z_ref, n_genes, sample_label)` — alignment cosine, contraction fraction, action diversity per state, state diversity per action, gene universality max/Gini, null gene fraction, delta magnitude quantiles.
- `compute_action_heterogeneity(...)` — Shannon entropy, Gini, top-K freq, path diversity, distance-vs-fused overlap.
- `compute_reward_leverage(cell_id, rollouts_distance, rollouts_fused)` — delta tabulation across raw success / final distance / T / tox / CE / unc, with explicit `pareto_signal` flag matching V3C plan §4 Stage 4 `CANDIDATE_SIGNAL_PARETO`.
- `compute_norman_combo_consistency(plans, measured_combos, check_ordered)` — Norman 2019 measured-combo overlap realism diagnostic; gracefully returns `status="no_combo_data"` when combo file is unavailable (guardrail #5).
- `compute_utility_score(buckets, allow_missing)` — composite weighted score, docstring explicitly marks it "ranking aid, not a verdict" (guardrail #1).
- `contraction_divergence(sample1, sample2)` — flags fields whose contraction fraction diverges > 0.30 between OOD pool and val pair distributions (guardrail #9: both samples reported separately).

### Driver — `scripts/audit_dynamics_utility_v3c.py`

Per-field sub-audit runner with subcommands `prediction`, `reachability`, `greedy`, `contraction`, `heterogeneity`, `reward_leverage`, `preconditions`, `all`. Properties:

- Idempotent (skips on-disk JSONs unless `--force`).
- Per-field path resolution honors guardrail #8: 64D dynamics use `artifacts_v3/vae_n64_legacy` / `artifacts_v3/vae_n64_nb` latents + per-VAE p15 ε (interpolated from `epsilon_success.json`'s p10/p25). 32D fields use `artifacts/vae` and the V3B-locked ε = 2.9898.
- Robust to missing inputs (guardrail #5): missing gate.json → `status: "missing_gate_json"`; missing combo_pairs.npz → `status: "no_combo_data"`; missing biology layer → safety arrays default to neutral (λ-inactive).
- Reuses `src.analysis.gate_breakdown.load_dynamics_model`, `src.rl.baselines.GreedyDynamicsBeamPolicy`, `src.rl.environment.CellReprogrammingEnv`, `src.rl.biology_rewards.build_safety_arrays`. No duplicated logic.
- Drives the canonical 7-cell V3B matrix per `V3_CONTROLLER_OBJECTIVE_SPEC.md`.

### Aggregator — `scripts/aggregate_v3c_utility_audit.py`

Cross-field rollup. Writes:

- One CSV per bucket (`prediction_metrics.csv`, `reachability_matrix.csv`, `greedy_saturation_matrix.csv`, `contraction_geometry.csv`, `contraction_geometry_val_pairs.csv`, `action_heterogeneity.csv`, `reward_leverage_fused.csv`, `ppo_preconditions.csv`).
- `utility_summary.md` — ranking by `util_score` (ranking aid) with explicit "not a verdict" disclaimer.
- `candidate_ranking.md` — partitions into U-G-pass (Best-by-audit candidates) and U-G-fail-but-flagged (Wildcard candidates) with per-field flags: `CONTRACTION_NEAR_UNIVERSAL`, `CONTRACTION_LOW_or_BARYCENTRIC`, `UNIVERSAL_ATTRACTOR_GENE`, `GEOMETRY_DIVERGENCE_X.XX`, `GATE_PASSED`, `U-G_ALL_PASS`, `PREDICTION_PATHOLOGICAL`, `DUPLICATE_OF[...]`.
- Duplicate detection (guardrail #4): groups fields whose `(val_pearson, ridge_margin, model.pt md5)` match exactly; flags so they don't consume separate smoke slots.

### Tests — `tests/test_dynamics_utility.py`

21 unit tests on synthetic dynamics with known geometry:

- `NoOpDynamics`, `UniversalAttractorDynamics`, `ActionDiscriminatingDynamics` fixtures.
- TDD discipline: every test was written failing first, then implementation made it pass.
- Verifies: contraction geometry shapes/sign/quantile statistics; Shannon entropy bounds; path diversity counting; Pareto signal verdict logic; Norman combo overlap on ordered/unordered/empty/missing-data cases; util_score sums to 1.0; util_score returns `None` when buckets missing unless `allow_missing=True`; docstring contains "ranking aid" and "not a verdict".

---

## Smoke findings (2 fields audited)

Both smokes ran with `n_episodes=24` per (cell, policy) for speed. Not the full Phase 0C numbers — but enough to verify the audit produces signal-bearing diagnostics.

### V2 anchor — `artifacts_v2/dynamics_v1ot_ror_corr010` (32D, RoR, λ_corr=0.10)

| Bucket | Result | Flag |
|---|---|---|
| U-A val/OOD Pearson | 0.6146 / 0.5163 | `pre_a = PASS` |
| U-B reachability (K=2/bin8-10) | **0.042** | `pre_b = FAIL` |
| U-C K=3/4/5/8 distance success | **1.000 everywhere** | `pre_c = FAIL (saturated)` |
| U-D contraction_fraction (OOD pool) | **0.9995** | `CONTRACTION_NEAR_UNIVERSAL` |
| U-D gene_universality_max | **0.92** | `UNIVERSAL_ATTRACTOR_GENE` |
| U-D action_diversity_per_state | 0.116 | `pre_d = FAIL` |
| U-D divergence (OOD vs val) | 0.0006 | (no divergence — geometry consistent) |
| U-G all_preconditions_pass | False (4/7 fail) | `util_score = 0.369` |

**Interpretation:** the V3B-documented saturation pattern reproduces exactly. Nearly every (z, g) pair contracts toward z_ref; one gene dominates as a universal attractor; greedy fully solves K ≥ 3 OOD bin8-10. This is the field whose reward stack `LOCKED_DESIGN_TECHNICAL_ONLY` outcome motivated V3C.

### Cautionary reference — `artifacts_v2/dynamics_soft_ot_default` (32D, Soft-OT pairs)

| Bucket | Result | Flag |
|---|---|---|
| U-A val/OOD Pearson | **0.9338 / 0.7434**, gate_passed=True, ridge_margin=+0.041 | `GATE_PASSED` |
| U-B reachability (all 7 cells) | **0.000 everywhere** | `pre_b = FAIL` |
| U-D contraction_fraction (OOD pool) | **0.000** | `CONTRACTION_LOW_or_BARYCENTRIC` |
| U-D alignment_cos_median | **−0.770** | (motion points AWAY from z_ref) |
| U-D gene_universality_max | **−0.62** | (even the "best" gene anti-contracts on average) |
| U-D delta_magnitude_median | **2.51** | (μ is NOT ≈ 0 — the field is not literal-no-op) |
| U-G all_preconditions_pass | False (6/7 fail; only U-A passes) | `util_score = 0.208` |

**Interpretation — new finding:** V3B documented Soft-OT as "passed gate but failed control (0/64)" and the V3C plan §1.2 hedged on the mechanism ("μ ≈ 0, μ approximately independent of g, or μ concentrated on a single attractor direction"). **Bucket U-D resolves the hedge empirically**: Soft-OT is **anti-contractive**, not no-op. The dynamics has non-zero predicted Δz (median ‖μ‖ = 2.5, comparable to V2 anchor's 2.8), but every gene predicts motion *away* from z_ref (median alignment cosine = −0.77). Soft-OT pairs train the model to predict the average perturbed cell, which is *farther* from the control centroid than typical OOD starts — so the learned vector field pushes outward.

This is precisely the failure mode Bucket U-D was designed to detect that the prediction gate cannot see.

### Diagnostic value of the smoke

Two fields, two different rejection signatures, both surfaced by automatic flags:

- **V2 anchor**: `CONTRACTION_NEAR_UNIVERSAL` + `UNIVERSAL_ATTRACTOR_GENE` → the saturation case
- **Soft-OT**: `GATE_PASSED` + `CONTRACTION_LOW_or_BARYCENTRIC` → the anti-contractive case (gate-passing-but-control-hostile)

A human reading `candidate_ranking.md` can immediately see why neither field is a viable Best-by-audit candidate, and which Wildcard sub-class each falls into. The plan §4 Stage 3 promotion route would be:

- V2 anchor: forced Phase 1 smoke (it is the fixed Anchor).
- Soft-OT: candidate for the "Soft-OT-vs-control disagreement wildcard" slot, but the audit confirms U-B = 0 across all cells — there is no path for PPO to find. Likely NOT promoted; the audit's diagnosis is the answer.

These outcomes match V3B's empirical record. The Phase 0B implementation is producing the right kind of signal.

---

## Sacred rules / guardrails check

- **Frozen tiers untouched**: `git status -- artifacts/ artifacts_64/ artifacts_v2/ artifacts/rl_sweeps/` → clean (verified post-commit).
- **V3C outputs under `artifacts_v3/v3c/`**: yes; sub-audit JSONs, CSVs, MD files all under `artifacts_v3/v3c/utility_audit/` or `artifacts_v3/v3c/interpretation/`.
- **Guardrail #1**: `util_score` docstring + summary MD explicitly mark "ranking aid only, not a verdict"; smoke-target selection requires written rationale.
- **Guardrail #2**: All V3C sub-buckets named U-A through U-G; legacy Bucket A/B/C/D references explicitly labeled `V3B-legacy`.
- **Guardrail #3**: No PPO smoking, no dynamics training, no VAE training, no frozen-tier writes.
- **Guardrail #4**: Aggregator implements (val_pearson, ridge_margin, model.pt md5) duplicate detection; flags clusters with `DUPLICATE_OF[...]` annotation. No duplicates in the 2-field smoke set; full Phase 0C should detect the `artifacts/dynamics` / `dynamics_current_default` / `dynamics_default_check` cluster the inventory identified.
- **Guardrail #5**: Sub-audits emit `status: "..."` strings on missing inputs rather than crashing (verified: missing combo_pairs → `no_combo_data`; missing gate.json → `missing_gate_json`).
- **Guardrail #6**: Soft-OT was diagnosed empirically by U-D (anti-contractive), not by hard-coded prior conclusion.
- **Guardrail #7**: Random-pairing field kept as Eligible in inventory; will be audited as negative control in Phase 0C.
- **Guardrail #8**: Per-VAE p15 ε resolution (interpolated for 64D fields); per-field VAE / pairs / latents lookup. Verified by reading the env construction code path; full validation comes in Phase 0C when 64D fields are audited.
- **Guardrail #9**: Bucket U-D reports both OOD-pool (primary) and val-pairs (secondary) samples separately, with divergence flag.
- **Guardrail #10**: `tests/test_dynamics_utility.py` → 21 passing. Full suite `PYTHONPATH=. .venv/bin/pytest -q` → 377 passing / 2 skipped (was 356 / 2 baseline; +21 utility tests added).
- **Guardrail #11**: This document.

---

## Next steps (Phase 0C, deferred to user direction)

1. Run `audit_dynamics_utility_v3c.py all --field-id <id> --n-episodes 64` on each of the 27 Eligible inventory fields (V2 anchor + soft_ot already done). Expect ~2–4 hours total wallclock on a single machine; trivially parallel.
2. Re-run `aggregate_v3c_utility_audit.py`. Inspect the duplicate clusters, U-D divergence flags, and U-F Norman-combo numbers.
3. Phase 0D — write `artifacts_v3/v3c/interpretation/v3c_phase0_utility_audit.md` with the four pre-committed Phase 1 smoke targets (1 Anchor + 1–2 Best-by-audit + 1–2 Wildcards), each with written rationale citing specific U-A–U-G evidence. **No PPO smoke before that doc lands.**

The audit infrastructure is ready. Time to point it at the rest of the inventory.
