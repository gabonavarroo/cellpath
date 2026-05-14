"""Analysis modules + the single source of truth for all metrics.

Module ownership
----------------
- :mod:`src.analysis.metrics`           : SINGLE SOURCE OF TRUTH; shared (both agents add here).
- :mod:`src.analysis.model_selection`   : Agent B. Checkpoint recommendation logic.
- :mod:`src.analysis.latent_space`      : Agent A. UMAP, silhouette, ARI.
- :mod:`src.analysis.trajectory`        : Agent A owns plumbing; Agent B contributes data.
- :mod:`src.analysis.depmap_validation` : Agent A. Hypergeometric + GSEA + null comparison.

CLAUDE.md sacred rule #4: every metric, every loss component, every enrichment statistic lives
here. Notebooks visualize; they do not define new metrics.
"""
