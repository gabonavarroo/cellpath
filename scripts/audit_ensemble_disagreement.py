"""V3C Phase 3 — Ensemble-disagreement audit.

Loads N dynamics models from the same VAE/pair source, evaluates:
1. Prediction sanity per member (val Pearson preserved).
2. Across-member disagreement of μ(z, g) — does it vary by ACTION g for fixed state z?
   This is the missing axis the V3B single-head heteroscedastic σ couldn't capture.
3. Greedy distance vs greedy fused divergence under ensemble-mean dynamics.

Output: artifacts_v3/v3c/utility_audit/ensemble_track_l/disagreement_audit.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.gate_breakdown import load_dynamics_model

LOG = logging.getLogger("v3c.ensemble")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _z_ref(vae_dir: Path) -> np.ndarray:
    return np.load(vae_dir / "z_reference_centroid.npy").astype(np.float32)


def _ood_pool(vae_dir: Path, n_sample: int = 500, seed: int = 42) -> np.ndarray:
    """Sample OOD perturbed cells from bin 6-10 distances."""
    adata = ad.read_h5ad(vae_dir / "latents.h5ad")
    z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    z_ref = _z_ref(vae_dir)
    pert_idx = np.asarray(adata.obs["perturbation_idx"].values)
    mask = pert_idx != 0
    d = np.linalg.norm(z - z_ref, axis=1)
    mask &= (d >= 6.0) & (d < 10.0)
    z_filt = z[mask]
    rng = np.random.default_rng(seed)
    n_sample = min(n_sample, z_filt.shape[0])
    idx = rng.choice(z_filt.shape[0], size=n_sample, replace=False)
    return z_filt[idx]


def _predict_all(model, z_starts: np.ndarray, n_genes: int) -> np.ndarray:
    """For each (z, g), return μ predictions. Shape (n_starts, n_genes, n_latent)."""
    z_t = torch.from_numpy(z_starts)
    out = np.zeros((z_starts.shape[0], n_genes, z_starts.shape[1]), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for g in range(1, n_genes + 1):  # 1-indexed
            g_t = torch.full((z_starts.shape[0],), int(g), dtype=torch.long)
            _, mu, _ = model(z_t, g_t)
            out[:, g - 1, :] = mu.cpu().numpy()
    return out


def main(ensemble_dirs: list[str], vae_dir: str, out_path: str) -> None:
    vae_p = REPO_ROOT / vae_dir
    out_p = REPO_ROOT / out_path
    out_p.parent.mkdir(parents=True, exist_ok=True)

    z_starts = _ood_pool(vae_p, n_sample=300, seed=42)
    LOG.info("OOD start pool: %d", z_starts.shape[0])

    # n_genes from any member's config
    cfg = json.loads((REPO_ROOT / ensemble_dirs[0] / "config.json").read_text())
    n_genes = int(cfg["n_genes"])
    n_latent = int(cfg["n_latent"])
    LOG.info("n_genes=%d, n_latent=%d", n_genes, n_latent)

    # Load all members and compute μ
    members: list[np.ndarray] = []
    member_info: list[dict] = []
    for d in ensemble_dirs:
        m = load_dynamics_model(REPO_ROOT / d)
        mu = _predict_all(m, z_starts, n_genes)  # (N, G, D)
        members.append(mu)
        gate = json.loads((REPO_ROOT / d / "gate.json").read_text())
        prim = gate.get("primary", {})
        ood = gate.get("ood", {}) or {}
        member_info.append({
            "dir": d,
            "val_pearson": float(prim.get("pearson_r", 0.0)),
            "ood_pearson": float(ood.get("pearson_r", 0.0)),
        })
        LOG.info("loaded %s (val=%.3f, ood=%.3f)", d, member_info[-1]["val_pearson"], member_info[-1]["ood_pearson"])

    stack = np.stack(members, axis=0)  # (M, N, G, D)
    M, N, G, D = stack.shape

    # Per-(state, gene) disagreement: σ across members
    member_std_per_state_gene = stack.std(axis=0, ddof=1)  # (N, G, D) → reduce over D
    sigma_sg = member_std_per_state_gene.mean(axis=-1)     # (N, G) — mean across dims

    # Aggregate: average disagreement per gene (across states), per state (across genes)
    sigma_per_gene = sigma_sg.mean(axis=0)   # (G,) — mean across states
    sigma_per_state = sigma_sg.mean(axis=1)  # (N,) — mean across genes
    overall_sigma = float(sigma_sg.mean())

    # Action-dependence: variance of sigma across genes per state — if nonzero, σ depends on g
    sigma_action_var_per_state = sigma_sg.var(axis=1, ddof=1)  # (N,)
    action_dependence_score = float(sigma_action_var_per_state.mean())

    # Per-state σ range — gives a sense of best vs worst action
    sigma_action_range_per_state = (sigma_sg.max(axis=1) - sigma_sg.min(axis=1))  # (N,)
    mean_action_range = float(sigma_action_range_per_state.mean())

    # Ensemble-mean μ and compare to per-member μ (for ensemble-as-mean dynamics)
    mean_mu = stack.mean(axis=0)  # (N, G, D)

    # Cosine alignment with z_ref - z (per (state, gene)) for ensemble mean
    z_ref = _z_ref(vae_p)
    target_dir = z_ref[None, :] - z_starts  # (N, D)
    tnorm = np.linalg.norm(target_dir, axis=-1, keepdims=True) + 1e-12
    target_unit = target_dir / tnorm

    # Per gene: pick best-action mean μ alignment with target
    mean_mu_norm = np.linalg.norm(mean_mu, axis=-1, keepdims=True) + 1e-12  # (N, G, 1)
    mean_mu_unit = mean_mu / mean_mu_norm  # (N, G, D)
    align = (mean_mu_unit * target_unit[:, None, :]).sum(axis=-1)  # (N, G)
    align_median = float(np.median(align))
    align_max_per_state = align.max(axis=1)  # (N,)
    align_top_gene_per_state = np.argmax(align, axis=1)  # (N,) — best-aligned gene per state

    # Diversity of "best gene" across states (entropy)
    top_gene_counts = np.bincount(align_top_gene_per_state, minlength=G)
    nonzero = top_gene_counts > 0
    p = top_gene_counts[nonzero] / top_gene_counts.sum()
    top_gene_entropy = float(-np.sum(p * np.log(p + 1e-12)))
    max_entropy = float(np.log(G))
    norm_entropy = top_gene_entropy / max_entropy

    # Action-dependent vs state-dependent uncertainty (the V3B finding to check)
    # σ_action_avg = mean over states of sigma_sg's action-direction-var divided by overall
    if overall_sigma > 1e-12:
        action_to_overall = action_dependence_score / (overall_sigma ** 2 + 1e-12)
    else:
        action_to_overall = 0.0

    out = {
        "n_members": M,
        "n_states": N,
        "n_genes": G,
        "n_latent": D,
        "members": member_info,
        "overall_disagreement_sigma": overall_sigma,
        "action_dependence_score_per_state_var_of_sigma": action_dependence_score,
        "action_dependence_normalized": action_to_overall,
        "mean_action_sigma_range_per_state": mean_action_range,
        "sigma_per_gene_top10": sorted(zip(sigma_per_gene.tolist(), range(G)),
                                       reverse=True)[:10],
        "ensemble_mean_alignment_median": align_median,
        "ensemble_mean_top_gene_entropy_per_state": top_gene_entropy,
        "ensemble_mean_top_gene_entropy_normalized": norm_entropy,
        "top_gene_indices_distribution": {
            int(idx): int(count) for idx, count in enumerate(top_gene_counts) if count > 0
        },
        "interpretation": {
            "verdict_inputs": {
                "prediction_preserved": all(m["val_pearson"] >= 0.55 for m in member_info),
                "action_dependent_disagreement": action_dependence_score > 1e-3,
                "norm_top_gene_entropy_above_05": norm_entropy >= 0.5,
            },
        },
    }

    out_p.write_text(json.dumps(out, indent=2))
    LOG.info("wrote %s", out_p)
    LOG.info("overall σ disagreement: %.4f", overall_sigma)
    LOG.info("action-dependence score (var of σ across genes per state, averaged): %.6f",
             action_dependence_score)
    LOG.info("normalized top-gene entropy: %.3f (1.0 = uniform across genes)", norm_entropy)


if __name__ == "__main__":
    main(
        ensemble_dirs=[
            "artifacts_v3/dynamics_n64_legacy_ror_corr010",
            "artifacts_v3/v3c/dynamics_ensemble/track_l_clone_seed0",
            "artifacts_v3/v3c/dynamics_ensemble/track_l_clone_seed1",
        ],
        vae_dir="artifacts_v3/vae_n64_legacy",
        out_path="artifacts_v3/v3c/utility_audit/ensemble_track_l/disagreement_audit.json",
    )
