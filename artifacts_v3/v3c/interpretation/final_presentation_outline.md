# CellPath — Final presentation outline (8–10 slides)

> Audience: research-savvy reviewers; tight 20–30 min slot. Emphasize framework + honest verdict. Do NOT claim biological discovery. Reuse figures from `artifacts_v3/v3c/figures/`.

---

## Slide 1 — Problem and ambition

- **Question**: can a learned policy steer cancer cells back toward an unperturbed baseline in latent space, using CRISPRa perturbations as actions?
- **System**: scVI latent + learned dynamics + MaskablePPO acting on `Discrete(n_genes+1)`.
- **Dataset**: Norman 2019 K562 CRISPRa Perturb-seq (105 genes, ~12K control + ~80K perturbed cells).
- **Caveat (front-loaded)**: in-silico latent-space steering, NOT therapeutic reprogramming. Target = unperturbed-K562 NT centroid, not a healthy cell state.
- *Figure: `pipeline_overview.png` (top half).*

## Slide 2 — Pipeline

- Norman → scVI VAE (32D or 64D) → OT-paired (z_ctrl, z_pert) → residual dynamics MLP f̂(z, g) = z + μ + log σ² → MaskablePPO with locked B+C+D reward → 7-cell hardness evaluation with 4-seed CIs.
- All paths in `config/paths.yaml`; all artifacts under `artifacts_v3/v3c/`.
- *Figure: `pipeline_overview.png` (full).*

## Slide 3 — Reward stack (V3B lock)

- `R_T = bonus·1[success] − freeband(T) − λ_tox·tox_path − λ_ce·CE_count − λ_unc·unc_path_max`.
- λ_tox=0.10, λ_ce=0.05, λ_unc=0.05; freeband `{free=3, mild_until=5, mild_β=0.02, heavy_β=0.10}`; max_steps=8; ε=p15.
- **Sequential ablation contract**: C → B → D → B+C → B+C+D. Never start with full stack.
- *Figure: `reward_stack_bucketA.png` — PPO_BCD wins all 3 Bucket-A axes on V2.*

## Slide 4 — Why greedy was hard to beat

- V3B verdict `LOCKED_DESIGN_TECHNICAL_ONLY` on V2 primary: stack implements correctly (392 tests pass), but no PPO − greedy_dyn_5 CI excludes zero.
- Mechanism: V2 dynamics is saturated at K≥4 — greedy_dyn_K already reaches 1.0 success, leaving no room for PPO planning.
- The reward axis is closed on V2 dynamics. The next lever is the dynamics field itself.

## Slide 5 — Dynamics utility audit

- New Bucket-U framework (U-A through U-G): prediction sanity, beam reachability, greedy saturation, contraction geometry, action heterogeneity, reward leverage, PPO preconditions.
- 29 dynamics fields audited end-to-end at n=64 (refined to n=200 for top-4 candidates).
- `util_score` is a ranking aid only — smoke selection requires written rationale.
- *Figure: `util_vs_reach_scatter.png` — util_score doesn't track reach.*

## Slide 6 — Three structural pathologies

| Pathology | Affected | Diagnostic |
|---|---|---|
| Universal over-contraction | 24 OT fields | cf≈1.0, gu_max≈0.92 |
| Lower-universality + unreachable at low K | mean-Δ family | gu_max≈0.66, reach=0 at K≤5 |
| Anti-contractive (gate-passing, control-hostile) | Soft-OT | cf=0, align_med=−0.77 |

The V1-OT / RoR pairing-noise floor (≈0.89) IS the universal-attractor signature. **Representation is the bottleneck, not reward.**
- *Figure: `dynamics_pathology_summary.png` and `contraction_geometry_comparison.png`.*

## Slide 7 — Final push: Track L / Track N + Phase 2 / 2.5

- Track L (64D legacy) at K=2/bin8-10/OOD: **4.8× anchor lift** (PPO_BCD 0.705 vs V2 anchor 0.148).
- 4-seed CI: PPO ties greedy (+0.010) but `mean_final_distance` regresses +0.173 → Pareto fail.
- Phase 2 v1 (conservative, τ=0.80): geometry moved (gu_max 0.933 → 0.905) but no control utility lift.
- Phase 2.5: TODO_FILL — v2_aggressive (τ=0.60), v3_diverse (+λ_ad>0), v4_combo.
- *Figure: `phase4_track_ln_results.png` and `phase2_5_geometry_move.png`.*

## Slide 8 — What failed and why

- No `LOCKED_DESIGN_POSITIVE_SIGNAL` emerged because every dynamics field surveyed is **either saturated (PPO has no room) or unreachable (no plan exists)**.
- The OT pairing-noise floor of ~0.89 mean alignment is the structural cause; the universal-attractor pattern is robust to architecture (RoR, λ_corr, n_latent, gene_bias).
- Single-head heteroscedastic σ is state-dependent but not action-discriminating → Variant D cannot easily steer policy within a fixed start.
- *Figure: `training_error_curves.png` (shows training is converged, not under-fit).*

## Slide 9 — Future work

- **Action-dependent uncertainty via ensemble disagreement** (V3.fallback.C): 3–5 dynamics models, different seeds → ensemble σ that Variant D can actually exploit.
- **Stronger contraction regularization at moderate τ ∈ {0.65, 0.70, 0.75}** — Phase 2.5 showed τ=0.80 too weak, τ=0.60 too strong.
- **Representation reformulation**: SCANVI 32D (semi-supervised) or ZINB 64D (count likelihood).
- **PPO early-stopping**: Track N showed 500k → 1M non-monotonicity; checkpoint every 100k and select on held-out cell.
- **Held-out biology**: Replogle 2022 K562 essential CRISPRi + OGEE v3 essentiality flags as orthogonal validation.

## Slide 10 — Takeaway

- The **reward axis is closed** (V3B locked, technically validated end-to-end). The next architectural lever is the dynamics field.
- The **dynamics-utility audit framework** is the lasting V3C contribution: it discriminates control utility from prediction quality on 29 fields, surfaces three concrete pathology signatures, and tells us exactly what a useful future dynamics field must avoid.
- The **honest verdict**: no `LOCKED_DESIGN_POSITIVE_SIGNAL` from existing fields or first Phase 2 candidates, but a deep reproducible framework and a clear bottleneck diagnosis. Champion is selected per Stage 3 (see `final_champion_selection.md`); the runnable pipeline (`make final-v3c`) reproduces the result end-to-end.

---

## Backup slides (if questions)

- **Locked reward parameters**: full table from `V3_CONTROLLER_OBJECTIVE_SPEC.md` §3.2.
- **Per-VAE p15 ε**: 32D=2.9898, Track L=3.0193, Track N=3.1120.
- **7-cell hardness matrix** (V3B): K=2/b6-8, K=2/b8-10, K=3/b6-8, K=3/b8-10, K=4/b8-10, K=5/b8-10, K=8/b8-10 (all OOD).
- **Why p15 over p10**: p10 caused PPO_BCD to collapse to 0.000 at K=2/b8-10/OOD on V2 dynamics.
- **Why not SCANVI/ZINB in V3C**: out of compute / time; flagged as future work.
