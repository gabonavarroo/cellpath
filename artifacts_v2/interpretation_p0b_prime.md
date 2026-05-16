# P0B′ Interpretation — 2026-05-16

## Inputs

| Run | Pairs dir | Dynamics dir | n_train pairs | pairing_noise_median |
|---|---|---|---:|---:|
| V1 OT (baseline) | artifacts/pairs | artifacts/dynamics | 38958 | 0.8935 |
| mean_delta | artifacts_v2/pairs_mean_delta | artifacts_v2/dynamics_mean_delta_default | 38958 | 0.8493 |
| random | artifacts_v2/pairs_random | artifacts_v2/dynamics_random_default | 38958 | 0.9495 |

V1 VAE (`artifacts/vae`) and V1-default dynamics architecture (`use_state_linear_skip=true`, `selection_metric=gate_margin`, `lr=1e-4`, `max_epochs=300`, `patience=35`, `seed=42`) held bit-identical across the three runs. Only `pairing.method` differs.

## Gate (val pairs)

| Run | mlp_pearson | ridge_pearson | mlp_minus_ridge | unc_spearman | passed? |
|---|---:|---:|---:|---:|---|
| V1 OT | 0.5640 | 0.5566 | **+0.0074** | 0.2490 | NO |
| mean_delta | 0.5186 | 0.4973 | **+0.0214** | 0.2214 | NO |
| random | 0.7227 | 0.7320 | **−0.0094** | 0.3121 | NO (ridge wins) |

Threshold: `+0.030`. None of the three runs clears the val ridge-margin gate. Random pairs flip the sign (ridge beats MLP on random) — expected negative-control behavior. Mean-delta produces a ~2.9× gain over OT.

## OOD (report-only)

| Run | mlp_pearson | ridge_pearson | mlp_minus_ridge | dim11_val_margin | dim11_ood_margin |
|---|---:|---:|---:|---:|---:|
| V1 OT | 0.4900 | 0.4500 | +0.0401 | −0.1239 | −0.4331 |
| mean_delta | 0.3833 | 0.2712 | **+0.1121** | **−0.0630** | **−0.2533** |
| random | 0.6383 | 0.6789 | −0.0406 | −0.0887 | −0.4826 |

OOD MLP-minus-ridge margin nearly triples on mean-delta vs V1 OT (`+0.040 → +0.112`). Dim 11 — the dominant negative dim in V1 — improves on both val (×2) and OOD (×1.7) but remains the worst dim in absolute terms. Mean-delta OOD MLP Pearson drops below the `0.40` secondary check (`0.383 < 0.40`), but ridge drops even more so the margin grows.

## Verdict

- **Outcome band: PARTIAL** (val margin `+0.0214` ∈ `[+0.015, +0.030)`).
- **H_pair_primary**: **partially supported**. Switching to mean-delta closed about 47 % of the gap to `+0.030` (`+0.0074 → +0.0214`, target `+0.0300`). Pairing-noise median dropped only `0.044` (target was `≥ 0.10`); within-gene Δz variance is therefore dominated by *biological* cell-to-cell heterogeneity, not pure pair-assignment noise. The pairing fix is real but only partially closes the gate.
- **H_pair_neg_control**: **distinct**. Random val margin `−0.0094` vs. mean-delta `+0.0214` — a `+0.031` separation, monotonic in pairing-noise ratio (`random > OT > mean_delta`). The gate metric is responsive to pair-assignment information content.
- **Dim 11 specifically**: V1 val `−0.124` → mean-delta `−0.063` (≈ ×2 improvement); V1 OOD `−0.433` → mean-delta `−0.253` (≈ ×1.7 improvement). Still the dominant negative dimension. This is consistent with dim 11 carrying a residual that ridge models cheaply (per-gene mean ⊕ global linear) but the MLP overfits to noise on.
- **Pairing-noise drop**: V1 `0.8935` → mean-delta `0.8493` (`Δ = −0.044`). p25 / p75: `0.813 / 0.949` → `0.715 / 0.920`.
- **Hard requirements (acceptance criteria 1–7)**: all satisfied. Schemas pass, both dynamics output trees present, comparator emitted, V1 artifacts byte-identical (SHA-verified fixture + clean `git status` on `artifacts/` and `artifacts_64/`), no gate threshold modified, no model or metrics code changed.
- **Secondary rollback signals**: none triggered. Random did *not* pass (it would have been a stop-and-escalate). Mean-delta did not show the HURT band. No training NaN'd. The OOD MLP Pearson dipping to `0.383` is a soft flag worth recording but is offset by the much larger ridge drop.

## Next step (mechanically by §"Final decision rules")

**Plan a P0B″** combining soft-OT (replace the OT `argmax` step with the soft-expectation `T[:, j].T @ z_ctrl` in `pair_ot`) AND/OR correlation loss (`λ_corr ∈ {0.05, 0.10, 0.30}`) trained on `artifacts_v2/pairs_mean_delta`. Keep architecture pinned. Run one config first (soft-OT alone with `λ_corr=0`) and gate-check before broadening the sweep.

Rationale: PARTIAL band + a strong OOD margin gain + monotonic negative-control behavior = the V2 architecture is sound on cleaner targets, but mean-delta alone leaves ~0.01 of margin on the table. Soft-OT and correlation loss are the two natural next levers that share the same architectural commitments. Do not advance to P0C yet (gate not cleared). Do not retrain VAE (P0A.4 showed the noise is mostly within-gene biology, not VAE geometry).
