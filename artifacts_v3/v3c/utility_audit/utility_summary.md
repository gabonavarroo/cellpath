# V3C dynamics utility audit summary

_Aggregator output. `util_score` is a **ranking aid only** — it ranks
the inventory for triage; it never decides which fields get PPO smoke._
_Smoke-target selection requires written qualitative rationale citing
specific Bucket U-A through U-G evidence (V3C plan §4 Stage 3 /
guardrails #1, #8)._

## Fields by `util_score` (ranking aid)

| Field | n_lat | pair | RoR | λ_corr | util_score | audit_class | duplicate_of |
|---|---:|---|:-:|---:|---:|---|---|
| `artifacts_v3/v3c/dynamics_candidates/contraction_aware_v4_combo` | 64 | ot_n64_contraction_aware | ✓ | 0.50 | 0.5471 | Eligible-V3C-P2.5 |  |
| `artifacts_v3/v3c/dynamics_candidates/contraction_aware_v3_diverse` | 64 | ot_n64_contraction_aware | ✓ | 0.50 | 0.5355 | Eligible-V3C-P2.5 |  |
| `artifacts/dynamics_ablation/baseline` | 32 | ot |  | 0.00 | 0.4188 | Eligible-conditional |  |
| `artifacts_64/dynamics_variants/state_linear` | 64 | ot |  | 0.00 | 0.4155 | Eligible-conditional |  |
| `artifacts_64/dynamics` | 64 | ot |  | 0.00 | 0.4155 | Eligible |  |
| `artifacts/dynamics_ablation/gene_bias` | 32 | ot |  | 0.00 | 0.4128 | Eligible-conditional |  |
| `artifacts/dynamics_ablation/state_linear` | 32 | ot |  | 0.00 | 0.4116 | Eligible-conditional |  |
| `artifacts/dynamics_sweeps/lr1e-3_mse0` | 32 | ot |  | 0.00 | 0.4116 | Eligible-conditional |  |
| `artifacts/dynamics_default_check` | 32 | ot |  | 0.00 | 0.4032 | Eligible |  |
| `artifacts/dynamics_current_default` | 32 | ot |  | 0.00 | 0.4032 | Eligible |  |
| `artifacts/dynamics` | 32 | ot |  | 0.00 | 0.4032 | Eligible |  |
| `artifacts_v3/dynamics_n64_nb_ror_corr010` | 64 | ot_nb | ✓ | 0.10 | 0.4004 | Eligible |  |
| `artifacts_v3/v3c/dynamics_candidates/contraction_aware_v2_aggressive` | 64 | ot_n64_contraction_aware | ✓ | 0.50 | 0.3978 | Eligible-V3C-P2.5 |  |
| `artifacts/dynamics_sweeps/lr1e-4_mse0` | 32 | ot |  | 0.00 | 0.3963 | Eligible-conditional |  |
| `artifacts/dynamics_sweeps/lr3e-4_mse005` | 32 | ot |  | 0.00 | 0.3936 | Eligible-conditional |  |
| `artifacts/dynamics_sweeps/lr3e-4_mse01` | 32 | ot |  | 0.00 | 0.3928 | Eligible-conditional |  |
| `artifacts_64/dynamics_variants/baseline_plain` | 64 | ot |  | 0.00 | 0.3913 | Eligible-conditional |  |
| `artifacts/dynamics_sweeps/lr3e-4_mse0` | 32 | ot |  | 0.00 | 0.3889 | Eligible-conditional |  |
| `artifacts_v2/dynamics_v1ot_ror_corr005` | 32 | ot | ✓ | 0.05 | 0.3797 | Eligible |  |
| `artifacts/dynamics_ablation/state_linear_gene_bias` | 32 | ot |  | 0.00 | 0.3792 | Eligible-conditional |  |
| `artifacts_v2/dynamics_random_default` | 32 | random |  | 0.00 | 0.3762 | Eligible |  |
| `artifacts_64/dynamics_variants/state_linear_combo0` | 64 | ot |  | 0.00 | 0.3757 | Eligible-conditional |  |
| `artifacts_v2/dynamics_v1ot_ror` | 32 | ot | ✓ | 0.00 | 0.3724 | Eligible |  |
| `artifacts_v3/v3c/dynamics_candidates/contraction_aware_v1` | 64 | ot_n64_contraction_aware | ✓ | 0.10 | 0.3696 | Eligible-V3C-P2 |  |
| `artifacts_v2/dynamics_v1ot_ror_corr010` | 32 | ot | ✓ | 0.10 | 0.3691 | Anchor |  |
| `artifacts_v3/dynamics_n64_legacy_ror_corr010` | 64 | ot | ✓ | 0.10 | 0.3652 | Eligible |  |
| `artifacts_64/dynamics_variants/state_linear_gene_bias` | 64 | ot |  | 0.00 | 0.3566 | Eligible-conditional |  |
| `artifacts_64/dynamics_variants/gene_bias` | 64 | ot |  | 0.00 | 0.3496 | Eligible-conditional |  |
| `artifacts_v2/dynamics_mean_delta_default` | 32 | mean_delta |  | 0.00 | 0.2952 | Eligible |  |
| `artifacts_v2/dynamics_mean_delta_corr_010` | 32 | mean_delta |  | 0.10 | 0.2944 | Eligible |  |
| `artifacts_v2/dynamics_mean_delta_corr_005` | 32 | mean_delta |  | 0.05 | 0.2763 | Eligible |  |
| `artifacts_v2/dynamics_mean_delta_corr_030` | 32 | mean_delta |  | 0.30 | 0.2486 | Eligible |  |
| `artifacts_v2/dynamics_soft_ot_default` | 32 | soft_ot |  | 0.00 | 0.2075 | Eligible |  |

## Duplicate clusters (guardrail #4)

(none detected)

## Next-step protocol (V3C plan §4 Stage 3)

Selection of smoke targets is interpretive, not algorithmic. Each of
the **four** smoke slots (Anchor + 1–2 Best-by-audit + 1–2 Wildcards)
must be documented in `artifacts_v3/v3c/interpretation/v3c_phase0_utility_audit.md`
with written rationale citing specific U-bucket evidence. A high
`util_score` alone is not sufficient rationale.

