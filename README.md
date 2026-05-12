# CellPath

> **In-silico cell-state steering** via a learned scVI latent geometry, a residual
> perturbation-dynamics surrogate, and a MaskablePPO reinforcement-learning agent that
> sequences CRISPRa interventions to drive K562 leukemia cells toward an unperturbed
> reference centroid.

This is **not** a demonstration of therapeutic cancer reprogramming. CellPath learns to
reverse perturbation drift in scVI latent space over a CRISPRa surrogate environment built
from Norman et al. 2019 (GSE133344) Perturb-seq data. The v1 reward target is the
unperturbed-K562 NT-guide centroid; "normal" or "healthy" cell states are explicitly out of
scope (see ARCHITECTURE.md, Concept 7 and DATA.md §7).

---

## System diagram

```
  Norman 2019 Perturb-seq (K562 + CRISPRa)
            │
            ▼
   preprocess  ──► HVG selection, raw counts preserved in adata.layers["counts"]
            │
            ▼
        scVI VAE (NB likelihood, 32-dim latent)
            │              │
   z_reference_centroid   epsilon_success (data-driven, 90th percentile)
            │              │
            └───►  OT pseudo-pairing  ◄───  Norman perturbed populations
                           │
                           ▼
       Residual heteroscedastic dynamics MLP  (μ, log σ²)
                           │
                  validation gate (primary + OOD report)
                           │
                           ▼
         MaskablePPO over CellReprogrammingEnv
                           │
                           ▼
                    rollouts.parquet
                           │
                           ▼
       DepMap enrichment + trajectory analysis
```

See `ARCHITECTURE.md` for the detailed system diagram, all seven concept explanations, and
the design-decisions log.

---

## Quick start (3 commands)

```bash
make setup           # creates .venv with uv, installs all dependencies
make data            # downloads Norman 2019 + DepMap K562
make pipeline        # data → vae → pairs → dynamics → rl → evaluate
```

The pipeline is idempotent: each step skips if its artifact already exists. Pass `--force <step>`
or `make nuke` to re-run.

For dev runs, use `--dry-run`:

```bash
python -m src.pipeline run --config-name default --dry-run
```

---

## Installation

Recommended: native `uv` venv with Python 3.11 (for Apple Silicon MPS support).

```bash
# Mac (MPS) / Linux dev
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Cluster (CUDA):

```bash
make docker-cuda
docker run --gpus all -v $(pwd):/workspace cellpath:cuda make pipeline
```

CI / smoke (CPU):

```bash
make docker-cpu
docker compose --profile cpu up smoke
```

---

## Datasets

| Dataset | Used for | Source | Notes |
|---|---|---|---|
| Norman et al. 2019 (GSE133344) | scVI training, perturbation pairing, dynamics, RL env | `pertpy.dt.norman_2019()` primary; scperturb / GEO fallback | K562 CRISPRa Perturb-seq; ~111k cells × ~19k genes; ~106 single-gene targets + ~131 dual-gene combos |
| DepMap Chronos | Plausibility enrichment of RL-selected genes | DepMap public release (default `24Q2`) | CRISPR knockout fitness, K562 cell line. Used for biological-plausibility enrichment only, NOT therapeutic validation |

See `DATA.md` for the full preprocessing pipeline and the cross-sectional honesty section.

### Citations

- Norman TM et al. "Exploring genetic interaction manifolds constructed from rich single-cell
  phenotypes." *Science* 365, 786–793 (2019). doi:10.1126/science.aax4438
- Lopez R et al. "Deep generative modeling for single-cell transcriptomics." *Nature Methods*
  15, 1053–1058 (2018). doi:10.1038/s41592-018-0229-2
- Bunne C et al. "Learning single-cell perturbation responses using neural optimal transport."
  *Nature Methods* 20, 1759–1768 (2023). doi:10.1038/s41592-023-01969-x
- Dempster JM et al. "Chronos: a cell population dynamics model of CRISPR experiments." *Genome
  Biology* 22, 343 (2021). doi:10.1186/s13059-021-02540-7

---

## Results

> Placeholder — populated by `scripts/evaluate.py` after a full pipeline run.

| Metric | Component | Target | Current |
|---|---|---|---|
| ELBO converged | VAE | — | — |
| Silhouette on perturbation | VAE latent | ≥ 0.05 | — |
| Primary validation gate | Dynamics | passed | — |
| Held-out cell R² | Dynamics | > per-gene-mean-Δ | — |
| Uncertainty calibration (Spearman) | Dynamics | ≥ 0.20 | — |
| OOD held-out gene R² | Dynamics | reported | — |
| Final success rate | RL | ≥ 30% on in-distribution | — |
| Mean interventions per success | RL | ≤ 5 (stretch) | — |
| At least one DepMap q < 0.05 | DepMap | yes | — |

---

## Repository layout

See `CLAUDE.md` for the full directory tree and sacred rules. High level:

```
src/{data,models,rl,analysis,utils}    # implementation
config/                                # Hydra configs (single source of truth for paths)
tests/                                 # pytest (CI runs without real data)
scripts/                               # entry points (--config-name, --dry-run)
notebooks/                             # visualization-only templates
artifacts/                             # runtime outputs (gitignored)
```

Documentation:

- `ARCHITECTURE.md` — system diagram + 7 concept explanations + decision log
- `CLAUDE.md` — agent context, sacred rules, env setup
- `AGENTS.md` — Agent A + B missions, interface contracts, conflict zones
- `PHASES.md` — 14-day master plan
- `DATA.md` — preprocessing biology, OT pairing methodology, DepMap schema
- `EXPERIMENTS.md` — hyperparameters + ablation matrix
- `PROGRESS.md` — living state file (updated end of each session)

---

## License

MIT. See `LICENSE`.

---

## How to cite

```bibtex
@misc{cellpath2026,
  title  = {CellPath: in-silico cell-state steering over a CRISPRa surrogate environment},
  author = {CellPath Team},
  year   = {2026},
  note   = {Master's thesis project. Code: https://github.com/gabonavarroo/cellpath},
}
```
