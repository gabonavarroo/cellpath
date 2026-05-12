# PROGRESS.md

> Living state file. Update at the **end** of every work session. Format documented in
> CLAUDE.md §8. The current state is always the **top** session entry; older entries stay
> below in reverse chronological order.

---

## Session 2026-05-11-2125 (agent: lead-architect)

**Phase:** 0 — Scaffold complete; both agents unblocked to start Phase 1.

**Status:** All 12 documentation + scaffold deliverables produced. Scientific realism audit
applied; original spec corrected on 5 points (CRISPRa-only action space, `z_reference_centroid`
naming, OT pseudo-pairing + uncertainty-aware dynamics, validation gate, DepMap enrichment as
plausibility test). User-confirmed scope:

- Action space: CRISPRa-only, ~106 single genes + NO-OP. Knockout disabled (future work).
- Reference state: unperturbed K562 NT centroid. No external healthy dataset in v1.
- ε_success: data-driven, 90th percentile of `||z_ctrl − z_ref||` distribution.
- Pairing: OT default, random + mean-delta fallbacks.
- Validation: primary gate on held-out cells; OOD report on held-out genes (non-gating).
- scVI save/load: official `model.save() / SCVI.load()` API only.
- Raw counts preserved in `adata.layers["counts"]` for the NB likelihood.

### Metrics (initial state — all unset)

| Component | Target | Current | Status |
| --- | --- | --- | --- |
| VAE — ELBO converged | early stop or epochs > 200 with negative-ELBO trend | — | not started |
| VAE — Silhouette (perturbation) | ≥ 0.05 | — | not started |
| VAE — ε_success | 0.1 < value < 10 (sanity) | — | not started |
| Dynamics — primary gate | passed | — | not started |
| Dynamics — uncertainty calibration | Spearman ≥ 0.20 | — | not started |
| Dynamics — OOD R² | reported (non-gating) | — | not started |
| RL — gymnasium env_checker | pass | — | not started |
| RL — final success rate | ≥ 30% on in-distribution starts | — | not started |
| RL — mean steps per success | ≤ 5 (stretch) | — | not started |
| DepMap — at least one FDR q < 0.05 | yes | — | not started |

### Blockers

- None at scaffold time. Both agents may start Phase 1 immediately.

### Next session priorities

1. **[Agent A]** `make setup` on Mac; confirm Norman download via `make data`; complete
   `src/data/preprocess.py` end-to-end.
2. **[Agent B]** Implement `generate_mock_pairs` skeleton if Agent A is slow; start dynamics
   model construction and forward pass.
3. **[Both]** Sanity-check the integration test (`tests/test_integration.py::TestHydraConfig`)
   in the new venv to confirm Hydra composition works.

---

## Phase-by-phase deliverable checklist (rolls up PHASES.md)

> Engineers tick items as they complete. Each tick should be accompanied by a metric or
> artifact reference in the session entry above.

### Phase 0 — Day 0
- [x] Repo skeleton (this scaffold).
- [x] ARCHITECTURE.md, CLAUDE.md, AGENTS.md, PHASES.md, DATA.md, EXPERIMENTS.md written.
- [x] All Hydra configs present and composable.
- [x] All `src/` stubs with full docstrings and `NotImplementedError`.
- [x] Two utility modules implemented (`device.py`, `seeding.py`).
- [x] Tests collect and pass (with most marked `xfail` until agents implement).
- [x] Notebooks 01/02/03 scaffolded.
- [ ] `make setup` validated on both engineers' machines.
- [ ] `make data` validated (Norman + DepMap download).
- [ ] `generate_mock_pairs` (Agent A Day 0 deliverable) implemented.
- [ ] First commit + push.

### Phase 1 — Days 1–3 (Data + VAE  ||  Dynamics architecture)
- [ ] [A] `src.data.download` real path implemented.
- [ ] [A] `src.data.preprocess.run_preprocessing` end-to-end.
- [ ] [A] `src.models.vae.train_vae` produces all four Contract-1 artifacts.
- [ ] [A] ELBO converges; silhouette ≥ 0.05.
- [ ] [B] `PerturbationDynamicsModel.forward` implemented; shape tests pass (remove xfail).
- [ ] [B] `heteroscedastic_nll` + `composition_loss` implemented.
- [ ] [B] Dynamics smoke train on mock pairs; loss decreases.

### Phase 2 — Days 4–6 (Latent validation  ||  Dynamics training + gate)
- [ ] [A] OT pairing implemented; `build_pairs` writes all four .npz files.
- [ ] [A] `src.analysis.latent_space.analyze_latent_quality` produces UMAP + silhouette + ARI.
- [ ] [B] `dynamics_validation_gate` in `metrics.py` implemented.
- [ ] [B] Primary gate **passes** on real data; `gate.json.passed=True`.
- [ ] [B] OOD metrics reported.

### Phase 3 — Days 7–9 (Analysis  ||  RL env + PPO)
- [ ] [A] `src.analysis.depmap_validation.run_depmap_enrichment` implemented.
- [ ] [A] Trajectory rendering implemented.
- [ ] [B] `CellReprogrammingEnv` implemented; gymnasium env_checker passes.
- [ ] [B] NO-OP success semantics correct (tests in `tests/test_environment.py` flipped on).
- [ ] [B] MaskablePPO training runs without crash; success-rate curve trends up.

### Phase 4 — Days 10–12 (Integration)
- [ ] Joint: `make pipeline` runs end-to-end on cluster.
- [ ] Joint: `tests/test_integration.py::test_mock_pipeline` xfail flipped off.
- [ ] [A] Pairing-method ablation complete.
- [ ] [B] λ_sparse ablation complete.

### Phase 5 — Days 13–14 (DepMap + presentation)
- [ ] [A] DepMap enrichment table with ≥ 1 q < 0.05 finding.
- [ ] [A+B] `scripts/visualize.py` reproduces every defense figure.
- [ ] [A+B] README.md updated with final metric values.
- [ ] Defense rehearsal complete.

---

## Session log (newest first)

_(empty — sessions append here as the project progresses)_
