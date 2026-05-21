# V3C candidate ranking and wildcard flags

_Per-field flags surface unusual signatures that the human researcher_
_should weigh when picking smoke targets. The audit ranks/explains/flags;_
_smoke-target selection is interpretive and requires written rationale._

## Fields with U-G all-pass (Best-by-audit candidates)

_(no fields pass all 7 U-G preconditions — wildcard route applies to top candidates)_

## Fields below U-G (Wildcard candidates — see flags for promotion rationale)

- **0.547** `artifacts_v3/v3c/dynamics_candidates/contraction_aware_v4_combo` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.536** `artifacts_v3/v3c/dynamics_candidates/contraction_aware_v3_diverse` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.419** `artifacts/dynamics_ablation/baseline` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.416** `artifacts_64/dynamics_variants/state_linear` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.416** `artifacts_64/dynamics` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.413** `artifacts/dynamics_ablation/gene_bias` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.412** `artifacts/dynamics_ablation/state_linear` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.412** `artifacts/dynamics_sweeps/lr1e-3_mse0` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.403** `artifacts/dynamics_default_check` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.403** `artifacts/dynamics_current_default` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.403** `artifacts/dynamics` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.400** `artifacts_v3/dynamics_n64_nb_ror_corr010` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.398** `artifacts_v3/v3c/dynamics_candidates/contraction_aware_v2_aggressive` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.396** `artifacts/dynamics_sweeps/lr1e-4_mse0` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.394** `artifacts/dynamics_sweeps/lr3e-4_mse005` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.393** `artifacts/dynamics_sweeps/lr3e-4_mse01` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.391** `artifacts_64/dynamics_variants/baseline_plain` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.389** `artifacts/dynamics_sweeps/lr3e-4_mse0` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.380** `artifacts_v2/dynamics_v1ot_ror_corr005` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.379** `artifacts/dynamics_ablation/state_linear_gene_bias` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.376** `artifacts_v2/dynamics_random_default` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.376** `artifacts_64/dynamics_variants/state_linear_combo0` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.372** `artifacts_v2/dynamics_v1ot_ror` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.370** `artifacts_v3/v3c/dynamics_candidates/contraction_aware_v1` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.369** `artifacts_v2/dynamics_v1ot_ror_corr010` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.365** `artifacts_v3/dynamics_n64_legacy_ror_corr010` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.357** `artifacts_64/dynamics_variants/state_linear_gene_bias` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.350** `artifacts_64/dynamics_variants/gene_bias` — CONTRACTION_NEAR_UNIVERSAL, UNIVERSAL_ATTRACTOR_GENE
- **0.295** `artifacts_v2/dynamics_mean_delta_default` — CONTRACTION_NEAR_UNIVERSAL
- **0.294** `artifacts_v2/dynamics_mean_delta_corr_010` — CONTRACTION_NEAR_UNIVERSAL
- **0.276** `artifacts_v2/dynamics_mean_delta_corr_005` — CONTRACTION_NEAR_UNIVERSAL
- **0.249** `artifacts_v2/dynamics_mean_delta_corr_030` — CONTRACTION_NEAR_UNIVERSAL
- **0.208** `artifacts_v2/dynamics_soft_ot_default` — GATE_PASSED, CONTRACTION_LOW_or_BARYCENTRIC

## Anchor

`artifacts_v2/dynamics_v1ot_ror_corr010` is **always** in the Phase 1
smoke roster regardless of `util_score` (V3C plan §4 Stage 3).

## Reminder

- `util_score` is a ranking aid. **Not** a verdict.
- Smoke-target selection requires written qualitative rationale.
- Duplicate clusters consume **one** smoke slot, not many.
- 64D fields use their own latent/pair files (guardrail #8) and a
  per-VAE p15 ε; their reachability is not directly comparable to 32D.

