# DATA.md — datasets, preprocessing, pairing, and DepMap

This file documents the complete data flow from raw GEO downloads to the artifacts consumed by
the dynamics model and RL environment. Every preprocessing step is justified biologically.

---

## 1. Norman et al. 2019 (GSE133344) — primary dataset

**Citation.** Norman TM, Horlbeck MA, Replogle JM, Ge AY, Xu A, Jost M, Gilbert LA, Weissman JS.
"Exploring genetic interaction manifolds constructed from rich single-cell phenotypes."
*Science* 365, 786–793 (2019). doi:10.1126/science.aax4438.

**What it is.** A Perturb-seq dataset of K562 chronic myeloid leukemia cells transduced with a
dCas9-SunTag-VP64 CRISPRa system targeting ~287 distinct guide constructs. Cells are profiled
by 10x Chromium v3 single-cell RNA-seq. Effective contents:

| Quantity | Value |
|---|---|
| Total cells (post-QC, published) | ~111,255 |
| Total genes profiled | ~19,018 |
| Distinct single-gene CRISPRa targets | ~106 |
| Distinct dual-gene combinations | ~131 |
| Non-targeting (NT) guide cells (controls) | ~10,000 |
| Modality | **CRISPR activation only (gain-of-function)** — no knockouts |

**Why this dataset.** It is the largest publicly available Perturb-seq study that includes
substantial dual-gene combinations, which lets us *test composition* — does our dynamics model
correctly compose two single-gene effects into the joint effect we actually observed?

**Modality caveat.** Because Norman is CRISPRa-only, our action space is gain-of-function
*activation* only. CellPath does not support knockouts in v1; see `config/rl.yaml::action_space.
enable_knockout` (default `false`).

### 1.1 Download instructions

The primary path uses `pertpy`, which caches a cleaned `.h5ad` locally:

```bash
# scripts/download_data.sh runs this for you:
python - <<'PY'
import pertpy as pt
from pathlib import Path
adata = pt.dt.norman_2019()
out = Path("data/raw/norman_2019.h5ad")
out.parent.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(out)
print(f"Wrote {out} | shape {adata.shape}")
PY
```

If `pertpy` fails (cache unreachable, version drift), fall back to **scperturb.org**:

```bash
# Fallback: scperturb-curated h5ad
curl -L -o data/raw/norman_2019.h5ad \
    "https://zenodo.org/records/10044268/files/NormanWeissman2019_filtered.h5ad?download=1"
sha256sum data/raw/norman_2019.h5ad
```

If both fail, the last-resort path is the raw GEO MTX archive:

```bash
# Last resort: GEO raw (much slower; requires processing the MTX into AnnData)
mkdir -p data/raw/geo
cd data/raw/geo
wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344_RAW.tar"
tar -xvf GSE133344_RAW.tar
# Then load via scanpy.read_mtx + per-sample stitching (see src/data/download.py for the path).
```

### 1.2 File sizes & checksums

| File | Size (approx) | sha256 (pertpy 0.7 snapshot) |
|---|---|---|
| `data/raw/norman_2019.h5ad` (pertpy) | ~3.5 GB | pinned in `scripts/download_data.sh` |
| `data/raw/norman_2019.h5ad` (scperturb) | ~2.8 GB | pinned in `scripts/download_data.sh` |
| GEO raw tarball | ~6 GB | (very volatile — not pinned) |

Checksums are verified by `scripts/download_data.sh` after each download and the script fails
loudly if they don't match. If they don't match, *do not silently update the checksum* — open
an issue and investigate. A checksum mismatch usually means upstream re-released the file.

---

## 2. Preprocessing pipeline — what each step does and why

The full pipeline lives in `src/data/preprocess.py`. Below is the biological justification for
each step. **The single most important rule:** raw integer counts must be preserved in
`adata.layers["counts"]` throughout. scVI's NB likelihood requires integer counts as targets
(see Concept 1 in ARCHITECTURE.md).

### 2.1 `sc.pp.filter_cells(adata, min_counts=cfg.data.min_counts)` (default 500)

**What.** Drops cells with fewer than 500 total UMIs across all genes.

**Why biologically.** A cell with <500 UMIs is almost certainly: (a) an empty droplet that
captured ambient RNA, (b) a dying cell whose transcripts have leaked, or (c) a doublet caught
in QC. Norman's published QC already excludes most of these; this is a defensive filter.

### 2.2 `sc.pp.filter_genes(adata, min_cells=cfg.data.min_cells)` (default 10)

**What.** Drops genes detected in fewer than 10 cells.

**Why biologically.** Genes detected in only a handful of cells contribute no statistical
power and add noise to HVG selection. The scVI NB likelihood becomes unstable on near-empty
columns.

### 2.3 Preserve raw counts BEFORE normalization

```python
adata.layers["counts"] = adata.X.copy()   # integer UMIs
```

**Why.** scVI's NB likelihood expects integer counts. If we run `normalize_total` and `log1p`
in-place on `adata.X`, the NB likelihood becomes ill-defined and training silently produces
nonsense (the loss decreases but the model is fitting log-normalized continuous values with
a discrete likelihood). We use `layers["counts"]` for scVI input and `adata.X` for visualization.

### 2.4 `sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000, layer="counts")`

**What.** Computes HVGs using Seurat v3's variance-stabilizing transformation directly on the
raw counts layer. Marks 2000 most variable genes.

**Why biologically.** Most genes are housekeeping or constitutive; their expression varies
little across cells and contributes no signal to perturbation discrimination. The 2000 HVGs
contain >95% of the perturbation-relevant variance in Norman (validated in the original paper).
**Why `flavor="seurat_v3"`:** it operates on raw counts (correct for our pipeline), whereas
`flavor="seurat"` (default) operates on log-normalized data (introduces the same circular bias
as item 2.3).

### 2.5 `sc.pp.normalize_total(adata, target_sum=1e4)` + `sc.pp.log1p(adata)`

**What.** Library-size normalization to 10,000 UMIs per cell, then `log(1 + x)`.

**Why biologically.** Normalization controls for the large per-cell variation in capture
efficiency (some cells captured 5x more transcripts than others). `log1p` stabilizes variance
across the dynamic range of expression. **This output is used for visualization (UMAP/PCA),
HVG selection if `seurat_v3` is not used, and any downstream cross-cell distance metric — but
NOT for scVI training input.** scVI uses `adata.layers["counts"]`.

### 2.6 HVG subsetting

```python
adata = adata[:, adata.var["highly_variable"]].copy()
```

**What.** Drops all non-HVG columns. Final shape: `(N_cells, 2000)`.

**Why.** scVI scales as O(N × G); 2000 vs 19018 is a 9.5× speedup with negligible signal loss.

### 2.7 Perturbation encoding

```python
unique_perturbations = sorted(adata.obs["perturbation"].unique())
ctrl_label = "control" if "control" in unique_perturbations else "ctrl"
encoder = {ctrl_label: 0}
for i, p in enumerate(x for x in unique_perturbations if x != ctrl_label):
    encoder[p] = i + 1
adata.obs["perturbation_idx"] = adata.obs["perturbation"].map(encoder).astype("int32")
```

**Why.** The RL action space is `Discrete(N_genes + 1)`. `perturbation_idx = 0` is the control;
`1..N_genes` are the single-gene CRISPRa targets; the NO-OP action gets index `N_genes` at
runtime. Combinatorial perturbations have their own encoding (see §3).

### 2.8 Final saved file

```
data/processed/norman_hvg.h5ad
    .X                 : float32 (N, 2000)    — log1p-normalized HVG expression
    .layers["counts"]  : int32   (N, 2000)    — raw UMIs (scVI input)
    .obs               : DataFrame with perturbation (str), perturbation_idx (int32),
                         total_counts, pct_counts_mt, ...
    .var               : DataFrame with gene_symbol (str), highly_variable (bool), ...
    .uns               : {"perturbation_encoder": {<label>: <int>}, "noop_idx": <int>}
```

This is the artifact Agent A produces in Phase 1 Day 2. All downstream consumers (the scVI
trainer, the latent-space analysis, the pairing builder) read from it.

---

## 3. Combinatorial perturbations

Norman's ~131 dual-gene combinations are a unique feature of the dataset. We use them in two
ways:

1. **Training signal for the dynamics model (80% of combos).** We construct `combo_pairs.npz`
   where each record is `(z_ctrl, gene_idx_a, gene_idx_b, z_pert_ab)`. The dynamics model is
   asked to predict `z_pert_ab` by *composing* its single-gene predictions:
   `z_after_a = z + f(z, a).mu; z_after_ab = z_after_a + f(z_after_a, b).mu`. Composition loss
   is added to the per-cell NLL.

2. **Held-out evaluation (20% of combos).** A randomly selected 20% of dual perturbations are
   never seen at training time, in any form. They are used purely to answer: *does the agent's
   sequential application of single-gene actions match the empirical dual-perturbation
   distribution?* This is the cleanest test we have for compositionality.

The 20% held-out combos are listed in `metadata.json::held_out_combos` after the pairing build.

---

## 4. Pairing methodology — the cross-sectional honesty section

**This is the most important caveat in the whole project.** Perturb-seq is cross-sectional:
we never observe the same cell pre- and post-perturbation. Instead, we observe populations:

- ~10,000 control cells with NT guides — call their latent distribution `p_ctrl(z)`.
- For each perturbation `p`, ~200–2,000 perturbed cells — call their latent distribution `p_p(z)`.

The dynamics model `f(z, p)` must be trained on `(z_in, p, z_out)` triples, but no such triple
exists in the raw data. We *construct* triples via pseudo-pairing. Whichever method we choose,
we are encoding an assumption about the perturbation effect's structure.

### 4.1 Entropic optimal transport (default)

For each perturbation `p`:
- Cost matrix `C_ij = ||z_ctrl_i - z_pert_j||_2`, then `C := C / median(C)` for numerical stability.
- Sinkhorn divergence with entropic regularization `ε = 0.05`, `numItermax = 500`.
- The resulting transport plan `T_p` is a coupling between the two empirical distributions.
- We sample a hard pairing once per epoch by taking, for each `i`, the `j` with the largest
  `T_ij` (greedy) or by sampling from `T_i,:` proportionally (stochastic). Default: greedy.

**Assumption it encodes.** Each perturbed cell "came from" the control cell closest to it in
latent space — a *minimum-displacement* assumption. This is good when perturbations make small
moves, which is empirically the case for most Norman single-gene CRISPRa.

**Limitations.** OT can produce degenerate transport plans if the cost matrix is poorly scaled.
We monitor `||T||_∞` (max entry); if it concentrates on a single column, we re-run with larger
`ε`. If three retries fail, we fall back to `mean_delta`.

### 4.2 Random within-perturbation pairing (fallback / ablation)

Simply sample a random control cell for each perturbed cell. Encodes the assumption that the
control population is exchangeable.

**When to use.** Sanity baseline; when OT is too slow during dev; when investigating whether
the OT advantage is real (compare R² of dynamics trained on OT pairs vs random pairs).

### 4.3 Mean-delta pseudo-pairing (fallback / ablation)

Compute the population-mean displacement `Δp = mean(z_pert) − mean(z_ctrl)` for perturbation
`p`. For each perturbed cell `z_pert_j`, the paired control is the *closest* control cell to
`z_pert_j − Δp` in `p_ctrl`.

**When to use.** Fast, deterministic, and a strong baseline. Often within 5% of OT R² on real
Norman data.

### 4.4 Splits

| Split | Source | Held-out axis | Purpose |
|---|---|---|---|
| `train_pairs.npz` | 80% perturbations × 90% cells | — | Train the dynamics model |
| `val_pairs.npz` | 80% perturbations × 10% cells | held-out cells | **Primary validation gate** (across-cell generalization) |
| `ood_pairs.npz` | 20% perturbations × 100% cells | held-out genes | **OOD report** (across-gene generalization, NOT gating) |
| `combo_pairs.npz` (train) | 80% of combos | — | Composition-loss term |
| `combo_pairs.npz` (held-out) | 20% of combos | held-out combos | Final composition evaluation in Phase 5 |

---

## 5. DepMap K562 essentiality data

**Citation.** Dempster JM et al. "Chronos: a cell population dynamics model of CRISPR experiments
that improves inference of gene fitness effects." *Genome Biology* 22, 343 (2021).

**Source.** https://depmap.org/portal/download/ — current public release.

**What we use.** The K562 Chronos score: per-gene fitness change when that gene is knocked out
in K562. Note this is **loss-of-function**; Norman is **gain-of-function**. The two modalities
are not equivalent (see ARCHITECTURE.md Concept 6).

### 5.1 Download

```bash
# scripts/download_data.sh:
DEPMAP_VERSION="24Q2"  # update via Hydra config when DepMap releases a new version
curl -L -o data/raw/depmap_chronos.csv \
    "https://depmap.org/portal/api/download/files/CRISPRGeneEffect.csv?release=${DEPMAP_VERSION}"
```

The file is a wide matrix with rows = cell lines and columns = genes. We extract the K562 row
into a long format on first use.

### 5.2 Schema after processing

```
data/processed/depmap_k562_chronos.parquet
    gene_symbol  : str (canonical HGNC symbol)
    chronos      : float32 (Chronos fitness score; more negative = more essential)
    is_essential : bool (chronos < -0.5, conventional threshold)
```

### 5.3 Enrichment tests (in `src/analysis/depmap_validation.py`)

For an RL action frequency vector `freq[gene]`:

1. **Hypergeometric (top-K test).** Sort genes by frequency, take top K (default 20). Test
   overlap with: (a) DepMap K562 essentials, (b) MSigDB Hallmark Hematopoiesis, (c) leukemia
   driver panel. Reports raw p, FDR-adjusted q (Benjamini-Hochberg).
2. **GSEA preranked.** Use frequency as ranking signal; gene set as Chronos-defined essentials.
   Reports ES, NES, FDR.
3. **Null comparison.** Repeat both tests on (a) random gene sets matched to RL action set
   size; (b) random gene sets matched to RL action *expression-level distribution* (controls
   for HVG bias). Reports z-score of observed enrichment vs null.

**Honesty.** A positive enrichment is *biological plausibility of selected genes*. It is not:
proof of reprogramming, evidence of causality, or therapeutic validity. ARCHITECTURE.md
Concept 6 details this. Any thesis claim derived from DepMap enrichment must respect this
scope.

---

## 6. Pointers

- `src/data/download.py` — implements both pertpy and GEO fallback.
- `src/data/preprocess.py` — implements the steps in §2.
- `src/data/perturbation_pairs.py` — implements the pairing strategies in §4.
- `src/analysis/depmap_validation.py` — implements the enrichment tests in §5.3.
- `config/paths.yaml` — single source of truth for every path mentioned in this document.

## 7. Future work (NOT IN v1)

These are explicitly out of scope for the 14-day project but noted here so future contributors
know where the extensions belong:

- **CRISPRi / knockout integration.** Add Replogle 2022 (Cell) K562 genome-wide CRISPRi data
  alongside Norman. Action space doubles to `Discrete(2 · N_genes + 1)`; gene embedding is
  shared but with a `direction` covariate (1 = activation, −1 = repression). Touches:
  `src/data/preprocess.py` (multi-dataset loader), `src/models/dynamics.py` (direction
  covariate), `src/rl/environment.py` (action decoder).
- **External healthy reference dataset.** Add a CD34+ hematopoietic stem/progenitor cell
  scRNA-seq dataset (e.g. Granja et al. 2019 *Nature Biotechnology*) and align its latent
  space with Norman via scVI's `transfer_anndata_setup` or scANVI's reference mapping. This
  would let `z_reference_centroid` point to a non-leukemic cell state — at which point the
  word "reprogramming" becomes a defensible claim. Currently the v1 target is the unperturbed-
  K562 NT centroid; do not call this "normal" or "healthy" in code or docs.
- **Combinatorial action space.** Allow the RL agent to apply two genes in one step (matching
  Norman's combo modality directly). Touches `src/rl/environment.py::action_space` and
  `src/models/dynamics.py::forward` (joint encoding).
