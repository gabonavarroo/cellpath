# V3C Replogle Held-out Availability Check (Stage 1)

**Date:** 2026-05-21
**Scope:** Inventory whether the Replogle 2022 K562 essential CRISPRi held-out source is
present locally, and classify the kind of artifact we have, before deciding whether to
re-download for the final V3C champion audit.

---

## 1. Locations inspected

| Location | What was looked for | Result |
|---|---|---|
| `data/raw/` | `replogle*` h5ad / signature dumps | **Absent.** Only `norman_2019.h5ad` and `depmap_chronos.csv`. |
| `data/raw/replogle/` | Raw Harmonizome / figshare dump | **Absent.** Directory does not exist. |
| `data/processed/` | Processed Replogle gene-set or h5ad | **Absent.** Only `norman_hvg.h5ad` and `depmap_k562_chronos.parquet`. |
| `data/processed/replogle/` | Persisted gene-set CSV/JSON | **Absent.** |
| `artifacts_v3/eval_v3b_phase2c/` | Phase 2c Bucket-C outputs | **Present (gene-set residue).** `replogle_norman_intersection.json`, `replogle_heldout_per_seed.csv`, `replogle_heldout_paired_deltas.json`, `replogle_heldout_summary.md`. |
| `artifacts_v3/interpretation/` | Phase 2c interpretation | **Present.** `v3b_phase2c_seed_escalation.md` documents source URL + parsing. |
| `config/paths.yaml` | Registered Replogle path | **Forward-only.** `paths.replogle_crispri_h5ad = ${paths.data_raw}/replogle_2022_crispri.h5ad` exists as a not-yet-downloaded placeholder for CRISPRi/knockout action-space extension; not the held-out essentiality source itself. |
| `/tmp/replogle_harm/` | Phase 2c session scratch | **Absent.** Phase 2c interpretation states source data was persisted only in `/tmp` (not committed). |

## 2. What we have on disk (verbatim)

From `artifacts_v3/eval_v3b_phase2c/replogle_norman_intersection.json`:

* **Source:** Harmonizome mirror — *Replogle et al., Cell, 2022 K562 Essential Perturb-seq Gene Perturbation Signatures*.
* **URL:** `https://maayanlab.cloud/Harmonizome/dataset/Replogle+et+al.,+Cell,+2022+K562+Essential+Perturb-seq+Gene+Perturbation+Signatures`.
* **n_replogle_essential_genes_unique:** 1875 (only the count survives; the full Replogle essential gene list is *not* on disk).
* **Replogle ∩ Norman 105:** 6 genes — FOXL2, KIF18B, NCL, PLK4, SET, STIL.
* **DepMap ∩ Norman 105:** 5 genes — CBFA2T3, HK2, PLK4, PTPN1, STIL.
* **Replogle-only ∩ Norman 105 (the Bucket-C clean set):** 4 genes — FOXL2, KIF18B, NCL, SET.

Phase 2c interpretation also references the three Harmonizome files that were parsed at that time:
`gene_attribute_matrix.txt.gz`, `attribute_list_entries.txt.gz`, `gene_list_terms.txt.gz`.
These files are **no longer on disk** (their working copies lived at `/tmp/replogle_harm/`).

## 3. Classification

Per the four buckets in the spec:

| Bucket | Match? |
|---|---|
| 1. Unavailable | No — gene-set residue persists. |
| **2. Gene-set / signature only** | **Yes — this is the current state.** |
| 3. Processed Replogle single-cell h5ad | No. |
| 4. Raw but not processed | No. |

**Availability verdict: `gene-set/signature only (Phase 2c residue, source not persisted)`.**

## 4. What this enables, what it does not

### Sufficient for (without re-download)

* The **post-hoc action-overlap audit** for the final V3C champion at Norman-105 resolution.
  All four Replogle-only essentials (FOXL2, KIF18B, NCL, SET) are already enumerated; the
  V3C action universe is Norman 105; therefore every "fraction of selected actions on
  Replogle-only essentials" metric is computable directly from the existing
  `summary.json::action_freq` payloads.
* The reproduction of the Phase 2c headline counts (`Replogle ∩ Norman = 6`,
  `Replogle-only essential = 4`) — those counts are already stored.

### Not sufficient for (without re-download)

* **Independent re-verification** that the Harmonizome-parsed gene list still produces the
  same 1875-genes / 6-intersect / 4-only counts (the original source files are gone; only
  the *result* of parsing them is on disk). We cannot detect upstream drift without a fresh
  download.
* Any **single-cell** compatibility / encoding probe (Stage 5 feasibility analysis): this
  requires a processed Replogle Perturb-seq h5ad, not a gene set.
* Any **biological-discovery** claim that would extend beyond what Phase 2c already covered.

## 5. Decision for Stage 2

Per the spec, Stage 2 attempts a fresh download regardless, because:

1. We need at minimum a re-derivation step we can re-run later (reproducibility).
2. Even a partial re-download (the three Harmonizome files) lets us regenerate the gene-set
   artifacts deterministically and persist them under version control (`data/processed/replogle/`).
3. Phase 2c committed only the *intersection* JSON. The full Replogle essential gene list
   (~1875) was never persisted and is needed for any future essentiality-aware analysis.

If Stage 2 succeeds, Stage 3 reconciles the new counts against the Phase 2c numbers (6
intersect, 4 Replogle-only). If Stage 2 fails (network unreachable, upstream changes), the
audit downgrades gracefully to using the cached intersection JSON, and Stage 6 flags the
limitation.

---

## 6. Pointers

- `artifacts_v3/eval_v3b_phase2c/replogle_norman_intersection.json` — cached gene-set
  intersection (the single source of truth right now).
- `artifacts_v3/eval_v3b_phase2c/replogle_heldout_summary.md` — Phase 2c Bucket-C verdict
  (`HELDOUT_INCONCLUSIVE_NO_GENERALIZATION_DETECTED` for safety-aware PPO_C on V2 dynamics).
- `artifacts_v3/interpretation/v3b_phase2c_seed_escalation.md` — Phase 2c rationale, parsing
  method, three Harmonizome files used.
- `config/paths.yaml::replogle_crispri_h5ad` — placeholder path for a *different* future
  Replogle artifact (single-cell CRISPRi h5ad for knockout action extension), not the held-out
  essentiality source.
