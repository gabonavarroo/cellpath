# Proposal 1 — Aggregated Comparison

## 1. Pairing noise (per-gene residual var / total var of Δz; lower = better)

| Method | median | mean | p25 | p75 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 OT eps=0.05 (existing) | 0.8935 | 0.8666 | 0.8127 | 0.9486 | 0.9815 |
| OT eps=0.02 (NEW) | – | – | – | – | – |
| OT eps=0.01 (NEW) | – | – | – | – | – |
| soft_ot eps=0.05 (existing) | 0.7829 | 0.7606 | 0.6484 | 0.8789 | 0.9649 |
| soft_ot eps=0.05 (verify) | – | – | – | – | – |
| soft_ot eps=0.01 (NEW) | – | – | – | – | – |
| mean_delta (existing) | 0.8493 | 0.8100 | 0.7147 | 0.9200 | 0.9771 |
| random (existing) | 0.9495 | 0.9230 | 0.8837 | 0.9747 | 0.9904 |

## 2. Dynamics (RoR + corr0.10 on each pair set; gate passes if val MLP−ridge Pearson margin ≥ 0.030)

| Pair set | gate | val Pearson | ridge | margin | OOD Pearson | OOD margin | epochs | arch |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V1 OT eps=0.05 + RoR (NEW, baseline) | ✗ | 0.6199 | 0.6079 | +0.0120 | 0.5224 | +0.0768 | 81 | RoR |
| V2 soft_ot eps=0.05 + RoR (NEW, ⭐ winner) | ✓ | 0.9400 | 0.8925 | +0.0475 | 0.8334 | +0.0926 | 61 | RoR |
| V2 mean_delta + RoR (NEW, abandoned — memory) | ✗ | 0.0000 | 0.0000 | +0.0000 | 0.0000 | +0.0000 | None | ? |

## 3. Reference dynamics already on disk (read-only, for comparison)

| Reference | gate | val Pearson | OOD Pearson | val margin | n_latent | arch |
| --- | :---: | ---: | ---: | ---: | ---: | --- |
| artifacts/dynamics  (V1, state_linear_skip, OT eps=0.05) | ✗ | 0.6176 | 0.4954 | +0.0097 | 32 | state_linear |
| artifacts_v2_experiments/dynamics_soft_ot_default  (V2 soft_ot, state_linear_skip) | ✓ | 0.9338 | 0.7434 | +0.0413 | 32 | state_linear |
| artifacts_v2_experiments/dynamics_mean_delta_corr_010  (V2 mean_delta, RoR+corr0.10) | ✗ | 0.5200 | 0.3836 | +0.0227 | 32 | state_linear |
| artifacts_64/dynamics  (64D legacy, state_linear_skip) | ✗ | 0.5965 | 0.3686 | -0.0191 | 64 | state_linear |

