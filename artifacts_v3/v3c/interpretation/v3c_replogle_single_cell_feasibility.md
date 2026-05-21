# V3C Replogle Single-Cell Compatibility — Feasibility Analysis (Stage 5)

**Date:** 2026-05-21
**Scope:** Decide whether a Replogle 2022 single-cell h5ad could be used as an
*external start-state test* for the final V3C champion (CRISPRa-trained Track L 64D VAE
+ contraction_aware_v2_aggressive dynamics + PPO_BCD seed 42 500k), *without* retraining.

**Verdict: NOT FEASIBLE as an external dynamics validation. The CRISPRa-vs-CRISPRi
modality mismatch is disqualifying. Even the simpler "encode Replogle control cells into
the Track L latent" probe is risky and out of scope for V3C.** This document records why.

No single-cell probe is run; no Replogle h5ad is downloaded. The Stage 2 download covers
only the Harmonizome gene-set mirror (a 5 MB knowledge-graph artifact), not the
~100 GB raw Replogle Perturb-seq matrix.

---

## 1. Question framing

If a processed Replogle h5ad were on disk, three increasingly ambitious uses are
conceivable:

| Probe | What it would test | What it would need |
|---|---|---|
| (P1) Encode Replogle *control* cells into the Track L VAE | Whether Norman control geometry generalises to a sibling K562 NT-guide population (smallest feasibility probe). | Same gene panel; HVG overlap; same library-size + log1p normalisation. |
| (P2) Encode Replogle *perturbed* cells, project onto the trained dynamics field | Whether Replogle CRISPRi perturbations move cells along the same latent vector field that Norman CRISPRa perturbations do — i.e. the dynamics-model generalises. | Same VAE; gene panel; perturbation-direction conversion (KD ↔ KA); explicit batch handling. |
| (P3) Use Replogle perturbed cells as *start states* and roll out PPO_BCD champion to evaluate held-out planning utility | Whether the controller learned on Norman generalises to a *biologically different* starting cell distribution. | All of (P2), plus alignment of the action space (CRISPRa target = CRISPRi target?), plus a defensible interpretation of CRISPRa-on-already-knocked-down-gene. |

The spec asks specifically about (P2)/(P3). The blocker is biological / methodological,
not engineering.

## 2. The disqualifying issues

### 2.1 Perturbation modality is inverted

* **Norman 2019**: dCas9-SunTag-VP64 **CRISPRa** (gain-of-function). Norman's
  `(z_ctrl, gene, z_pert)` triples encode `f(z, gene)` as the latent displacement caused
  by *over-expression* of a gene.
* **Replogle 2022**: dCas9-KRAB **CRISPRi** (loss-of-function). Replogle's perturbations
  encode the latent displacement caused by *knockdown* of a gene.

The CellPath dynamics MLP `f(z, gene) → Δz` was trained exclusively on Norman CRISPRa
(`adata.obs.nperts ∈ {0, 1, 2}` from `data/processed/norman_hvg.h5ad`), and the
RL environment exposes `Discrete(105_single_gene_CRISPRa + 1_NO_OP)` (per
`src/rl/environment.py`, `config/rl.yaml::action_space.enable_knockout: false`,
sacred rule §10). The action embedding has no direction covariate (per `DATA.md` §7,
this is documented as future work).

Applying Norman-CRISPRa dynamics to Replogle-CRISPRi cells would conflate:
- The cell-state distribution (which may legitimately generalise across CRISPR systems).
- The action semantics (which **do not** — `f(z, gene)` will predict an
  over-expression Δz when the cell was actually shifted by a knock-down).

Any "successful" rollout would be a coincidence of the two opposite Δz directions,
not validation of dynamics generalisation.

### 2.2 Action universe overlap is sparse

The CRISPRa action universe (Norman 105 single-gene targets) was chosen for its
combinatorial coverage, not for K562 essentiality. Only **6 of 105 Norman genes**
overlap with the Replogle K562-essential panel (Stage 3 gene-set construction:
FOXL2, KIF18B, NCL, PLK4, SET, STIL). The Replogle-essential ∩ Norman set is too
small (4 Replogle-only after subtracting DepMap-overlap) to define a meaningful
restricted action space, and the disjoint 99 Norman actions have no Replogle
counterpart.

Even if we restricted to the 6 overlapping genes, the CRISPRa direction would still
not flip semantics for free.

### 2.3 Gene-panel and HVG mismatch

* The Track L VAE was trained on the Norman 2000-HVG panel (Seurat v3 selection on
  Norman counts; `src/data/preprocess.py` step 2.4).
* Replogle's published 10x v3 K562 dataset has different cell counts, depth, and
  capture biases. Its HVG panel selected by the *same* procedure on Replogle counts
  would differ.
* A naïve `scvi.model.SCVI.load(path, adata)` call expects the *exact* gene panel
  and order. Re-aligning Replogle to Norman's panel means subsetting to the
  intersection (likely smaller than 2000 HVGs), padding missing genes with zeros, or
  doing transfer learning (`transfer_anndata_setup`). Each of those choices
  *changes* the VAE inference and adds new failure modes — none of which can be
  audited within a "no-retraining" V3C scope.

### 2.4 Batch / technical effects

Replogle and Norman were sequenced years apart, on different runs, with different
guide-library designs. Without batch correction (Harmony / scVI's `batch_key` /
SCANVI reference-mapping), latent coordinates from the two datasets are not on the
same scale. CellPath's success metric is L2 distance to `z_reference_centroid`;
unresolved batch shift would systematically bias that distance.

CellPath has **no batch infrastructure currently** — the VAE is trained on a single
batch (Norman). Adding batch correction is squarely "representation reformulation"
(closeout §11) and out of V3C scope.

## 3. What the gene-set audit can validly say

Stage 4 (action-overlap audit) avoids all four issues above by operating purely on
*action-frequency vectors* against an external gene-essentiality panel. It does not
require encoding Replogle cells into the Norman latent, does not assume modality
equivalence, and does not need the Replogle h5ad on disk. Therefore the gene-set
audit produces a valid Bucket-C *action-side* probe of the controller, even though
a Bucket-C *state-side* probe is out of reach.

This is the same scope that V3B Phase 2c held: gene-set audit only, no Replogle
cells encoded.

## 4. Concrete decision

* **No Replogle h5ad download.** The Stage 2 script intentionally only fetches the
  Harmonizome gene-set artifacts.
* **No P1/P2/P3 probe** in this session. Even (P1) — encoding Replogle controls into
  the Track L VAE — requires per-cell normalisation, HVG alignment, and a batch
  story we do not have time to validate without risking false-positive claims.
* The Stage 4 audit is the only Bucket-C validity claim we make about the final
  V3C champion in this session.

## 5. What would be required to revisit this

For a future session that *could* run (P2)/(P3) defensibly:

1. **Action-space extension** — register a CRISPRi dataset in `config/paths.yaml`
   (the existing `paths.replogle_crispri_h5ad` placeholder is for exactly this
   future), lift the `NotImplementedError` guard at
   `config/rl.yaml::action_space.enable_knockout`, and add a direction covariate
   to `src/models/dynamics.py::forward` (per `DATA.md` §7 "future work").
2. **Joint dynamics training** — retrain dynamics on the union of Norman CRISPRa
   pairs and Replogle CRISPRi pairs with the direction covariate. This is a V4
   work item.
3. **Batch-aware VAE** — either retrain scVI with `batch_key=dataset_id` or run
   SCANVI reference mapping. Either is incompatible with the V2 frozen artifacts.
4. **Healthy-reference centroid** — the current `z_reference_centroid` is the
   unperturbed-K562 NT centroid, not a cross-dataset reference. A combined centroid
   would have to be re-derived per the new latent.

All four are out of scope for V3C per the sacred rules (no VAE retrain, no dynamics
retrain, no PPO retrain in this session).

## 6. Pointers

- `DATA.md` §7 — "Future work: CRISPRi / knockout integration" (the canonical roadmap entry).
- `config/paths.yaml::replogle_crispri_h5ad` — forward-only placeholder path.
- `config/rl.yaml::action_space.enable_knockout` — guard flag (default false).
- `CLAUDE.md` §3 sacred-rule #10 — explicit prohibition on enabling the knockout
  action space without a registered CRISPRi dataset.
- `artifacts_v3/v3c/interpretation/v3c_final_closeout.md` §11 future-work bullets.
