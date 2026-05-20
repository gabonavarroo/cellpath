"""V3C dynamics utility audit driver (Phase 0B).

Per-field sub-audit runner. Subcommands write JSON under
``artifacts_v3/v3c/utility_audit/<field_id>/``:

    prediction      → U-A   (reads existing gate.json / val_metrics.json)
    reachability    → U-B   (beam reachability on canonical 7-cell V3B matrix)
    greedy          → U-C   (greedy_dyn_K success + depth leverage)
    contraction     → U-D   (geometry on OOD pool + val pairs samples)
    heterogeneity   → U-E   (depends on greedy rollouts)
    reward_leverage → U-F   (depends on greedy rollouts + Norman combo realism)
    preconditions   → U-G   (composite of U-A through U-F)
    all             → run every subcommand for one field

Read-only on frozen tiers. Writes only under ``artifacts_v3/v3c/``.
Idempotent: existing JSON outputs are skipped unless ``--force``.
Robust to missing inputs: per guardrail #5, a sub-audit that cannot
compute due to missing files emits a warning and writes a ``status``
field rather than crashing.

See V3C_DYNAMICS_UTILITY_AUDIT_AND_REFORMULATION_PLAN.md §3, §11 Phase 0B.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl
import torch

from src.analysis.dynamics_utility import (
    compute_action_heterogeneity,
    compute_contraction_geometry,
    compute_norman_combo_consistency,
    compute_reward_leverage,
    compute_utility_score,
    contraction_divergence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_REL = "artifacts_v3/v3c/utility_audit/dynamics_inventory.csv"
OUTPUT_ROOT_REL = "artifacts_v3/v3c/utility_audit"
BIOLOGY_DIR_REL = "artifacts_v3/v3b_biology"

LOG = logging.getLogger("v3c.audit")

# Canonical 7-cell V3B hardness matrix per V3_CONTROLLER_OBJECTIVE_SPEC.md
CANONICAL_CELLS: tuple[dict[str, Any], ...] = (
    {"id": "k2_bin6-8_splitood",  "K": 2, "bin": (6.0, 8.0),  "ood": True},
    {"id": "k2_bin8-10_splitood", "K": 2, "bin": (8.0, 10.0), "ood": True},
    {"id": "k3_bin6-8_splitood",  "K": 3, "bin": (6.0, 8.0),  "ood": True},
    {"id": "k3_bin8-10_splitood", "K": 3, "bin": (8.0, 10.0), "ood": True},
    {"id": "k4_bin8-10_splitood", "K": 4, "bin": (8.0, 10.0), "ood": True},
    {"id": "k5_bin8-10_splitood", "K": 5, "bin": (8.0, 10.0), "ood": True},
    {"id": "k8_bin8-10_splitood", "K": 8, "bin": (8.0, 10.0), "ood": True},
)

# V3B-locked freeband schedule and λ defaults
LOCKED_FREEBAND: dict[str, float] = {
    "free_steps": 3.0, "mild_until": 5.0, "mild_beta": 0.02, "heavy_beta": 0.10,
    "success_bonus": 1.0,
}
LOCKED_LAMBDA_TOX = 0.10
LOCKED_LAMBDA_CE = 0.05
LOCKED_LAMBDA_UNC_PATH = 0.05
LOCKED_EPSILON_PERCENTILE = 15
LOCKED_EPSILON_32D = 2.9898  # V3B Phase 4 lock for 32D fields
UNCERTAINTY_CLIP_MIN = -5.0
UNCERTAINTY_CLIP_MAX = 3.0


# ---------------------------------------------------------------------------
# Path / file resolution (guardrail #8 — 64D dynamics never run on 32D starts)
# ---------------------------------------------------------------------------


def _vae_dir_for(field_path: str, n_latent: int) -> Path:
    """Resolve the VAE dir whose latent geometry matches a given dynamics field."""
    if "artifacts_v3/dynamics_n64_legacy" in field_path:
        return REPO_ROOT / "artifacts_v3/vae_n64_legacy"
    if "artifacts_v3/dynamics_n64_nb" in field_path:
        return REPO_ROOT / "artifacts_v3/vae_n64_nb"
    # V3C Phase 2 — contraction-aware candidates use Track L's 64D legacy VAE by default.
    if "artifacts_v3/v3c/dynamics_candidates/contraction_aware" in field_path:
        return REPO_ROOT / "artifacts_v3/vae_n64_legacy"
    if field_path.startswith("artifacts_64/"):
        return REPO_ROOT / "artifacts_64/vae"
    # Default 32D — both artifacts/ and artifacts_v2/ dynamics use artifacts/vae
    if n_latent == 32:
        return REPO_ROOT / "artifacts/vae"
    raise RuntimeError(f"Unable to resolve VAE for {field_path} at n_latent={n_latent}")


def _pairs_dir_for(field_path: str) -> Path:
    """Map a dynamics field to the pair-set it was trained on."""
    if "dynamics_soft_ot" in field_path:
        return REPO_ROOT / "artifacts_v2/pairs_soft_ot"
    if "dynamics_mean_delta" in field_path:
        return REPO_ROOT / "artifacts_v2/pairs_mean_delta"
    if "dynamics_random" in field_path:
        return REPO_ROOT / "artifacts_v2/pairs_random"
    if "dynamics_n64_legacy" in field_path:
        return REPO_ROOT / "artifacts_v3/pairs_n64_legacy"
    if "dynamics_n64_nb" in field_path:
        return REPO_ROOT / "artifacts_v3/pairs_n64_nb"
    # V3C Phase 2 — contraction-aware candidates trained on Track L's 64D legacy pairs.
    if "artifacts_v3/v3c/dynamics_candidates/contraction_aware" in field_path:
        return REPO_ROOT / "artifacts_v3/pairs_n64_legacy"
    if field_path.startswith("artifacts_64/"):
        return REPO_ROOT / "artifacts_64/pairs"
    # V1 OT pairs (artifacts/dynamics*, artifacts_v2/dynamics_v1ot_*)
    return REPO_ROOT / "artifacts/pairs"


def _epsilon_for(vae_dir: Path) -> float:
    """Return the per-VAE p15 epsilon (interpolated p10/p25 if not stored)."""
    eps_path = vae_dir / "epsilon_success.json"
    if not eps_path.exists():
        return float(LOCKED_EPSILON_32D)
    payload = json.loads(eps_path.read_text())
    if "p15" in payload:
        return float(payload["p15"])
    if "p10" in payload and "p25" in payload:
        # Linear interpolation: p15 = p10 + (p25 - p10) * (15 - 10) / (25 - 10)
        return float(payload["p10"]) + float(payload["p25"] - payload["p10"]) * (5.0 / 15.0)
    # Fallback to the locked 32D value (this matches V3B for artifacts/vae)
    return float(LOCKED_EPSILON_32D)


def _load_held_out_genes(pairs_dir: Path) -> list[str]:
    meta = pairs_dir / "metadata.json"
    if not meta.exists():
        return []
    payload = json.loads(meta.read_text())
    return list(payload.get("held_out_genes", []) or [])


def _load_z_ref(vae_dir: Path) -> np.ndarray:
    return np.load(vae_dir / "z_reference_centroid.npy").astype(np.float32)


def _load_start_pool(
    vae_dir: Path, *, distance_bin: tuple[float, float], held_out_genes: list[str]
) -> np.ndarray:
    """Filter latents to OOD perturbed cells in a distance bin (mirrors phase4 evaluator)."""
    import anndata as ad
    lo, hi = float(distance_bin[0]), float(distance_bin[1])
    adata = ad.read_h5ad(vae_dir / "latents.h5ad")
    z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    z_ref = _load_z_ref(vae_dir)
    pert_idx = np.asarray(adata.obs["perturbation_idx"].values)
    mask = pert_idx != 0
    pert = np.asarray(adata.obs["perturbation"].astype(str).values)
    mask &= np.isin(pert, np.asarray(held_out_genes, dtype=object))
    d = np.linalg.norm(z - z_ref, axis=1)
    mask &= (d >= lo) & (d < hi)
    return z[mask].astype(np.float32)


# ---------------------------------------------------------------------------
# Greedy rollout helper — drives a real env with `_run_policy`-style stats.
# ---------------------------------------------------------------------------


def _gene_lookup_from_vocab(vae_dir: Path) -> dict[int, str]:
    vocab_path = vae_dir / "gene_vocab.json"
    if not vocab_path.exists():
        return {}
    payload = json.loads(vocab_path.read_text())
    symbols = payload.get("symbols") or payload.get("genes") or []
    return {i: str(s) for i, s in enumerate(symbols)}


def _load_biology_arrays(n_genes: int, biology_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load (tox_per_action, essential_per_action) from biology layer.

    Returns (None, None) if the biology layer is missing — the audit
    proceeds with safety-neutral defaults (λ_tox / λ_ce inactive).
    """
    safety_path = biology_dir / "gene_safety.parquet"
    if not safety_path.exists():
        return None, None
    from src.rl.biology_rewards import build_safety_arrays
    df = pl.read_parquet(safety_path)
    tox, ess = build_safety_arrays(df, n_genes=n_genes, permute_chronos=False, seed=42)
    return tox, ess


def _make_env(
    *, dynamics_model, z_ref: np.ndarray, epsilon: float, n_genes: int,
    max_steps: int, start_pool: np.ndarray, seed: int,
    safety_tox: np.ndarray | None, safety_ess: np.ndarray | None,
    reward_mode: str,
    lambda_tox: float, lambda_ce: float, lambda_unc_path: float,
):
    """Build a CellReprogrammingEnv configured for one cell's hardness."""
    from src.rl.environment import CellReprogrammingEnv
    return CellReprogrammingEnv(
        dynamics_model=dynamics_model,
        z_reference_centroid=z_ref,
        epsilon_success=float(epsilon),
        n_genes=int(n_genes),
        max_steps=int(max_steps),
        lambda_sparse=0.0,
        lambda_unc=0.0,
        repeat_mask=True,
        start_state_strategy="random_perturbation",
        distance_metric="l2",
        start_pool_latents=start_pool,
        success_bonus=0.0,
        failure_penalty=0.0,
        seed=seed,
        reward_mode=reward_mode,
        beta_step_cost=0.05,
        safety_tox_per_action=safety_tox,
        safety_essential_per_action=safety_ess,
        lambda_tox=float(lambda_tox),
        lambda_ce=float(lambda_ce),
        free_steps=int(LOCKED_FREEBAND["free_steps"]),
        mild_until=int(LOCKED_FREEBAND["mild_until"]),
        mild_beta=float(LOCKED_FREEBAND["mild_beta"]),
        heavy_beta=float(LOCKED_FREEBAND["heavy_beta"]),
        freeband_success_bonus=float(LOCKED_FREEBAND["success_bonus"]),
        lambda_unc_path=float(lambda_unc_path),
        uncertainty_reduce="mean_sigma",
        uncertainty_clip_min=float(UNCERTAINTY_CLIP_MIN),
        uncertainty_clip_max=float(UNCERTAINTY_CLIP_MAX),
    )


def _rollout_one_policy(
    env, policy, *, n_episodes: int, n_genes: int
) -> dict[str, Any]:
    """Rollout a policy and return per-episode + aggregate statistics.

    Adds ``first_action_per_episode`` and ``two_step_path_per_episode``
    relative to the phase4 ``_run_policy`` — needed for U-E / U-F.
    """
    successes = 0
    steps_list, final_distances, rewards = [], [], []
    tox_paths, ce_counts, unc_maxes, unc_means = [], [], [], []
    success_steps: list[int] = []
    action_freq: dict[int, int] = {}
    first_actions: list[int] = []
    paths_full: list[tuple[int, ...]] = []
    paths_two_step: list[tuple[int, int]] = []
    success_records: list[dict[str, Any]] = []

    for ep in range(int(n_episodes)):
        obs, info = env.reset(seed=ep)
        terminated, truncated = False, False
        ep_reward = 0.0
        terminal_success = False
        last_info = info
        ep_actions: list[int] = []
        while not (terminated or truncated):
            action = int(policy.select_action(obs, info["action_mask"], info))
            ep_actions.append(action)
            action_freq[action] = action_freq.get(action, 0) + 1
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            terminal_success = bool(info.get("success", False))
            last_info = info

        is_success = bool(terminal_success and terminated)
        successes += int(is_success)
        T_final = int(last_info.get("step", 0))
        steps_list.append(T_final)
        final_distances.append(float(last_info.get("distance", np.nan)))
        rewards.append(ep_reward)
        tox_paths.append(float(last_info.get("tox_path", 0.0)))
        ce_counts.append(int(last_info.get("common_essential_count", 0)))
        unc_maxes.append(float(last_info.get("unc_path_max", 0.0)))
        unc_means.append(float(last_info.get("unc_path_mean", 0.0)))
        if is_success:
            success_steps.append(T_final)

        # First action (0-indexed in env → store 1-indexed gene id; n_genes = NOOP)
        if ep_actions:
            first_actions.append(ep_actions[0] + 1)  # NOOP becomes n_genes+1
        else:
            first_actions.append(n_genes + 1)
        # Two-step path: first two non-NOOP actions (1-indexed gene ids)
        non_noop = [a + 1 for a in ep_actions if a != n_genes]
        if len(non_noop) >= 2:
            paths_two_step.append((non_noop[0], non_noop[1]))
        paths_full.append(tuple(a + 1 for a in ep_actions))

        if is_success and len(non_noop) >= 2:
            success_records.append({
                "episode": ep,
                "path": tuple(non_noop),
                "T": T_final,
            })

    n = int(n_episodes)
    mean_T_success = (
        float(np.mean(success_steps)) if success_steps else float("nan")
    )

    return {
        "n_episodes": n,
        "successes": successes,
        "success_rate": float(successes / max(n, 1)),
        "mean_steps": float(np.mean(steps_list)) if steps_list else float("nan"),
        "mean_final_distance": float(np.nanmean(final_distances)) if final_distances else float("nan"),
        "mean_T_at_success": mean_T_success,
        "mean_tox_path": float(np.mean(tox_paths)),
        "mean_common_essential_count": float(np.mean(ce_counts)),
        "mean_unc_path_max": float(np.mean(unc_maxes)),
        "mean_unc_path_mean": float(np.mean(unc_means)),
        "first_actions": first_actions,
        "paths_two_step": paths_two_step,
        "paths_full": paths_full,
        "action_freq": dict(sorted(action_freq.items(), key=lambda kv: -kv[1])),
        "success_records": success_records,
    }


def _make_greedy(
    *,
    dynamics_model, z_ref: np.ndarray, n_genes: int, depth: int,
    safety_tox: np.ndarray | None, safety_ess: np.ndarray | None,
    lambda_tox: float, lambda_ce: float, lambda_unc_path: float,
    epsilon: float,
):
    from src.rl.baselines import GreedyDynamicsBeamPolicy
    return GreedyDynamicsBeamPolicy(
        dynamics=dynamics_model,
        n_genes=int(n_genes),
        z_ref=z_ref,
        noop_idx=int(n_genes),
        depth=int(depth),
        beam_width=20,
        safety_tox_per_action=safety_tox,
        safety_essential_per_action=safety_ess,
        lambda_tox=float(lambda_tox),
        lambda_ce=float(lambda_ce),
        freeband_schedule={
            "free_steps": int(LOCKED_FREEBAND["free_steps"]),
            "mild_until": int(LOCKED_FREEBAND["mild_until"]),
            "mild_beta": float(LOCKED_FREEBAND["mild_beta"]),
            "heavy_beta": float(LOCKED_FREEBAND["heavy_beta"]),
            "success_bonus": float(LOCKED_FREEBAND["success_bonus"]),
        },
        success_epsilon=float(epsilon),
        lambda_unc_path=float(lambda_unc_path),
        uncertainty_reduce="mean_sigma",
        uncertainty_clip_min=float(UNCERTAINTY_CLIP_MIN),
        uncertainty_clip_max=float(UNCERTAINTY_CLIP_MAX),
    )


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


class FieldContext:
    """Resolved per-field paths + model handles, shared across sub-audits."""

    def __init__(self, *, row: dict[str, Any]):
        self.row = row
        self.field_id = str(row["field_id"])
        self.field_path = str(row["path"])
        self.n_latent = int(row["n_latent"]) if row.get("n_latent") is not None else 0
        if self.n_latent <= 0:
            raise RuntimeError(f"{self.field_id}: n_latent undefined")
        self.dynamics_dir = REPO_ROOT / self.field_path
        self.vae_dir = _vae_dir_for(self.field_path, self.n_latent)
        self.pairs_dir = _pairs_dir_for(self.field_path)
        self.biology_dir = REPO_ROOT / BIOLOGY_DIR_REL
        self.z_ref = _load_z_ref(self.vae_dir)
        self.epsilon_p15 = _epsilon_for(self.vae_dir)
        self.held_out_genes = _load_held_out_genes(self.pairs_dir)
        # Dynamics model loading deferred to first use.
        self._model = None

    @property
    def n_genes(self) -> int:
        # Read from config; fall back to gene_vocab cardinality
        cfg_path = self.dynamics_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            n = cfg.get("n_genes")
            if n is not None:
                return int(n)
        vocab = _gene_lookup_from_vocab(self.vae_dir)
        return len(vocab) if vocab else 105

    @property
    def model(self):
        if self._model is None:
            from src.analysis.gate_breakdown import load_dynamics_model
            self._model = load_dynamics_model(self.dynamics_dir)
        return self._model


def sub_prediction(ctx: FieldContext) -> dict[str, Any]:
    """U-A — read existing gate.json / val_metrics / ood_metrics.

    Apply the prediction-pathological flag (NOT a hard rejection) per
    V3C plan §3 U-A: low val_pearson is a warning, not exclusion from
    geometry/reachability.
    """
    gate_path = ctx.dynamics_dir / "gate.json"
    if not gate_path.exists():
        return {"status": "missing_gate_json", "field_id": ctx.field_id}
    gate = json.loads(gate_path.read_text())
    primary = gate.get("primary") or {}
    ood = gate.get("ood") or {}
    margin_checks = primary.get("margin_checks") or {}

    val_pearson = float(primary.get("pearson_r", 0.0))
    ood_pearson = float(ood.get("pearson_r", 0.0)) if ood else None
    ridge_margin = (margin_checks.get("margin_vs_linear_ridge_pearson") or {}).get("value")
    unc_spearman = (gate.get("uncertainty_calibration") or {}).get("spearman")

    # Per-dim Pearson distribution from gate_diagnostics.json (when available)
    per_dim_path = ctx.dynamics_dir / "gate_diagnostics.json"
    per_dim_median: float | None = None
    per_dim_p10: float | None = None
    if per_dim_path.exists():
        try:
            diag = json.loads(per_dim_path.read_text())
            per_dim = diag.get("per_dim_pearson")
            if per_dim is not None:
                arr = np.asarray(per_dim, dtype=np.float64)
                per_dim_median = float(np.median(arr))
                per_dim_p10 = float(np.quantile(arr, 0.10))
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    pathological = (
        val_pearson < 0.10 or (per_dim_median is not None and per_dim_median < 0.05)
    )

    return {
        "status": "ok",
        "field_id": ctx.field_id,
        "val_pearson": val_pearson,
        "ood_pearson": ood_pearson,
        "ridge_margin": ridge_margin,
        "uncertainty_spearman": unc_spearman,
        "per_dim_pearson_median": per_dim_median,
        "per_dim_pearson_p10": per_dim_p10,
        "gate_passed": bool(gate.get("passed", False)),
        "prediction_pathological_flag": bool(pathological),
    }


def sub_contraction(ctx: FieldContext) -> dict[str, Any]:
    """U-D on both samples (OOD pool + val pairs), report each + divergence."""
    n_genes = ctx.n_genes

    # Sample 1 — OOD pool (primary for control utility)
    sample1: dict[str, Any] | None = None
    try:
        pool_lo = _load_start_pool(ctx.vae_dir, distance_bin=(8.0, 10.0), held_out_genes=ctx.held_out_genes)
        pool_mid = _load_start_pool(ctx.vae_dir, distance_bin=(6.0, 8.0), held_out_genes=ctx.held_out_genes)
        pool = np.concatenate([pool_lo, pool_mid], axis=0)
        if pool.shape[0] == 0:
            sample1 = {"status": "empty_pool"}
        else:
            n_sample = min(1500, pool.shape[0])
            rng = np.random.default_rng(42)
            idx = rng.choice(pool.shape[0], size=n_sample, replace=False)
            sample1 = compute_contraction_geometry(
                dynamics=ctx.model,
                z_starts=pool[idx],
                z_ref=ctx.z_ref,
                n_genes=n_genes,
                sample_label="ood_pool",
            )
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("U-D OOD pool failed for %s: %s", ctx.field_id, exc)
        sample1 = {"status": f"error: {type(exc).__name__}: {exc}"}

    # Sample 2 — held-out validation pairs (secondary, prediction-gate comparison)
    sample2: dict[str, Any] | None = None
    try:
        # Combine val + OOD pair endpoints (z_ctrl); they are control latents
        # paired against held-out gene perturbations.
        pair_files = [
            ctx.pairs_dir / "val_pairs.npz",
            ctx.pairs_dir / "ood_pairs.npz",
        ]
        z_ctrls = []
        gene_ids = []
        for fp in pair_files:
            if fp.exists():
                payload = np.load(fp)
                z_ctrls.append(payload["z_ctrl"])
                gene_ids.append(payload["gene_idx"])
        if not z_ctrls:
            sample2 = {"status": "no_val_pairs"}
        else:
            z_ctrl = np.concatenate(z_ctrls, axis=0).astype(np.float32)
            gene_idx = np.concatenate(gene_ids, axis=0).astype(np.int64)
            # Sample down for tractability
            n_sample = min(2000, z_ctrl.shape[0])
            rng = np.random.default_rng(43)
            idx = rng.choice(z_ctrl.shape[0], size=n_sample, replace=False)
            # Use the unique gene set as the sub-set to keep cost bounded.
            unique_genes = np.unique(gene_idx)
            sample2 = compute_contraction_geometry(
                dynamics=ctx.model,
                z_starts=z_ctrl[idx],
                z_ref=ctx.z_ref,
                n_genes=n_genes,
                gene_idx_subset=unique_genes,
                sample_label="val_pairs",
            )
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("U-D val-pairs failed for %s: %s", ctx.field_id, exc)
        sample2 = {"status": f"error: {type(exc).__name__}: {exc}"}

    divergence = None
    if isinstance(sample1, dict) and isinstance(sample2, dict) \
            and "contraction_fraction" in sample1 and "contraction_fraction" in sample2:
        divergence = contraction_divergence(sample1, sample2)

    return {
        "status": "ok",
        "field_id": ctx.field_id,
        "sample1_ood_pool": sample1,
        "sample2_val_pairs": sample2,
        "divergence": divergence,
    }


def sub_reachability(ctx: FieldContext, *, n_episodes: int = 64) -> dict[str, Any]:
    """U-B — beam reachability at K∈{2,3,4,5,8} per cell with deep beam (width 64)."""
    n_genes = ctx.n_genes
    safety_tox, safety_ess = _load_biology_arrays(n_genes, ctx.biology_dir)
    cells_out: list[dict[str, Any]] = []
    for cell in CANONICAL_CELLS:
        cell_id = cell["id"]
        K = int(cell["K"])
        try:
            pool = _load_start_pool(ctx.vae_dir, distance_bin=cell["bin"], held_out_genes=ctx.held_out_genes)
            if pool.shape[0] < 4:
                cells_out.append({"cell_id": cell_id, "status": "insufficient_pool", "n_pool": int(pool.shape[0])})
                continue
            env = _make_env(
                dynamics_model=ctx.model, z_ref=ctx.z_ref, epsilon=ctx.epsilon_p15,
                n_genes=n_genes, max_steps=K, start_pool=pool, seed=42,
                safety_tox=safety_tox, safety_ess=safety_ess,
                reward_mode="biorealistic_fused",
                lambda_tox=0.0, lambda_ce=0.0, lambda_unc_path=0.0,  # distance-only beam
            )
            from src.rl.baselines import GreedyDynamicsBeamPolicy
            beam = GreedyDynamicsBeamPolicy(
                dynamics=ctx.model, n_genes=n_genes, z_ref=ctx.z_ref,
                noop_idx=n_genes, depth=K, beam_width=64,
                # distance-only — no reward terms in the score
            )
            roll = _rollout_one_policy(env, beam, n_episodes=n_episodes, n_genes=n_genes)
            cells_out.append({
                "cell_id": cell_id, "status": "ok", "K": K,
                "beam_reach_at_K_p15": roll["success_rate"],
                "n_pool": int(pool.shape[0]), "n_episodes": int(n_episodes),
            })
        except Exception as exc:                                   # noqa: BLE001
            LOG.warning("U-B %s cell %s failed: %s", ctx.field_id, cell_id, exc)
            cells_out.append({"cell_id": cell_id, "status": f"error: {exc}"})
    return {"status": "ok", "field_id": ctx.field_id, "epsilon": ctx.epsilon_p15, "cells": cells_out}


def sub_greedy(ctx: FieldContext, *, n_episodes: int = 64) -> dict[str, Any]:
    """U-C — greedy_dyn_K for K∈{1,2,3,5,8} under distance-only AND fused.

    Stores per-episode action records so U-E and U-F can re-use them
    without re-rolling.
    """
    n_genes = ctx.n_genes
    safety_tox, safety_ess = _load_biology_arrays(n_genes, ctx.biology_dir)
    cells_out: list[dict[str, Any]] = []
    for cell in CANONICAL_CELLS:
        cell_id = cell["id"]
        cell_K = int(cell["K"])
        try:
            pool = _load_start_pool(ctx.vae_dir, distance_bin=cell["bin"], held_out_genes=ctx.held_out_genes)
            if pool.shape[0] < 4:
                cells_out.append({"cell_id": cell_id, "status": "insufficient_pool"})
                continue
            cell_record: dict[str, Any] = {
                "cell_id": cell_id, "K": cell_K, "n_pool": int(pool.shape[0]),
                "status": "ok", "depths": {},
            }
            for depth in (1, 2, 3, 5, 8):
                if depth > cell_K:
                    continue
                for mode, lam_tox, lam_ce, lam_unc in (
                    ("distance",     0.0, 0.0, 0.0),
                    ("fused",        LOCKED_LAMBDA_TOX, LOCKED_LAMBDA_CE, LOCKED_LAMBDA_UNC_PATH),
                ):
                    env = _make_env(
                        dynamics_model=ctx.model, z_ref=ctx.z_ref, epsilon=ctx.epsilon_p15,
                        n_genes=n_genes, max_steps=cell_K, start_pool=pool, seed=42,
                        safety_tox=safety_tox, safety_ess=safety_ess,
                        reward_mode="biorealistic_fused",
                        lambda_tox=lam_tox, lambda_ce=lam_ce, lambda_unc_path=lam_unc,
                    )
                    policy = _make_greedy(
                        dynamics_model=ctx.model, z_ref=ctx.z_ref, n_genes=n_genes,
                        depth=depth, safety_tox=safety_tox, safety_ess=safety_ess,
                        lambda_tox=lam_tox, lambda_ce=lam_ce, lambda_unc_path=lam_unc,
                        epsilon=ctx.epsilon_p15,
                    )
                    roll = _rollout_one_policy(env, policy, n_episodes=n_episodes, n_genes=n_genes)
                    cell_record["depths"].setdefault(str(depth), {})[mode] = roll
            # Depth leverage (distance only) for U-C composite
            d_dist = {int(k): v.get("distance", {}).get("success_rate")
                      for k, v in cell_record["depths"].items() if v.get("distance")}
            cell_record["depth_leverage"] = {
                "g1_dist": d_dist.get(1), "g2_dist": d_dist.get(2),
                "g3_dist": d_dist.get(3), "g5_dist": d_dist.get(5),
                "g8_dist": d_dist.get(8),
                "g2_minus_g1": _maybe_sub(d_dist.get(2), d_dist.get(1)),
                "g3_minus_g2": _maybe_sub(d_dist.get(3), d_dist.get(2)),
                "g5_minus_g3": _maybe_sub(d_dist.get(5), d_dist.get(3)),
                "g8_minus_g5": _maybe_sub(d_dist.get(8), d_dist.get(5)),
                "cumulative_K_max_minus_K1": _maybe_sub(
                    max((v for v in d_dist.values() if v is not None), default=None),
                    d_dist.get(1),
                ),
            }
            cells_out.append(cell_record)
        except Exception as exc:                                   # noqa: BLE001
            LOG.warning("U-C %s cell %s failed: %s", ctx.field_id, cell_id, exc)
            cells_out.append({"cell_id": cell_id, "status": f"error: {exc}"})
    # Top-line: max cumulative depth leverage across cells
    cum_leverages = [
        c["depth_leverage"]["cumulative_K_max_minus_K1"]
        for c in cells_out
        if c.get("status") == "ok" and c.get("depth_leverage", {}).get("cumulative_K_max_minus_K1") is not None
    ]
    return {
        "status": "ok", "field_id": ctx.field_id,
        "epsilon": ctx.epsilon_p15, "n_episodes": int(n_episodes),
        "cells": cells_out,
        "max_cumulative_depth_leverage": float(max(cum_leverages)) if cum_leverages else None,
    }


def _maybe_sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def sub_heterogeneity(ctx: FieldContext, greedy_payload: dict[str, Any]) -> dict[str, Any]:
    """U-E — derived from the greedy rollouts at depth 1 + 2 + 3."""
    out_cells: list[dict[str, Any]] = []
    n_genes = ctx.n_genes
    for cell in greedy_payload.get("cells", []):
        if cell.get("status") != "ok":
            out_cells.append({"cell_id": cell["cell_id"], "status": cell.get("status")})
            continue
        depths = cell.get("depths", {})
        d1 = depths.get("1", {})
        d2 = depths.get("2", {})
        d3 = depths.get("3", {})

        first_dist = d1.get("distance", {}).get("first_actions")
        first_fused = d1.get("fused", {}).get("first_actions")
        plans_d2 = d2.get("distance", {}).get("paths_two_step", [])
        plans_d3 = d3.get("distance", {}).get("paths_full", [])
        if first_dist is None or first_fused is None:
            out_cells.append({"cell_id": cell["cell_id"], "status": "missing_depth1_rollouts"})
            continue
        out_cells.append({
            "cell_id": cell["cell_id"],
            **compute_action_heterogeneity(
                n_genes=n_genes,
                first_actions_distance=np.asarray(first_dist, dtype=np.int64),
                first_actions_fused=np.asarray(first_fused, dtype=np.int64),
                beam_plans_depth2_distance=plans_d2,
                beam_plans_depth3_distance=plans_d3 if plans_d3 else plans_d2,
            ),
        })
    return {"status": "ok", "field_id": ctx.field_id, "cells": out_cells}


def sub_reward_leverage(ctx: FieldContext, greedy_payload: dict[str, Any]) -> dict[str, Any]:
    """U-F — distance-vs-fused tabulation, plus optional Norman-combo realism."""
    out_cells: list[dict[str, Any]] = []
    for cell in greedy_payload.get("cells", []):
        if cell.get("status") != "ok":
            out_cells.append({"cell_id": cell["cell_id"], "status": cell.get("status")})
            continue
        # Use the depth equal to cell K for canonical leverage tabulation
        K_str = str(int(cell["K"]))
        depths = cell.get("depths", {})
        bucket = depths.get(K_str) or depths.get("3") or depths.get("2") or {}
        if "distance" not in bucket or "fused" not in bucket:
            out_cells.append({"cell_id": cell["cell_id"], "status": "missing_distance_or_fused"})
            continue
        out_cells.append({
            "cell_id": cell["cell_id"],
            **compute_reward_leverage(
                cell_id=cell["cell_id"],
                rollouts_distance=bucket["distance"],
                rollouts_fused=bucket["fused"],
            ),
        })

    # Norman measured-combo realism — runs only when ctx field has access to
    # 32D combo data with the correct latent dim
    norman = _norman_combo_diagnostic(ctx, greedy_payload)

    return {
        "status": "ok",
        "field_id": ctx.field_id,
        "cells": out_cells,
        "norman_combo_realism": norman,
    }


def _norman_combo_diagnostic(ctx: FieldContext, greedy_payload: dict[str, Any]) -> dict[str, Any]:
    combo_path = REPO_ROOT / "artifacts/pairs/combo_pairs.npz"
    if not combo_path.exists() or ctx.n_latent != 32:
        return compute_norman_combo_consistency(plans=[], measured_combos=None, check_ordered=True)
    try:
        combo = np.load(combo_path)
        measured = {
            "gene_idx_a": combo["gene_idx_a"],
            "gene_idx_b": combo["gene_idx_b"],
            "z_ctrl":     combo["z_ctrl"],
            "z_pert_ab":  combo["z_pert_ab"],
        }
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("Failed to read combo_pairs for %s: %s", ctx.field_id, exc)
        return {"status": "combo_load_failed", "error": str(exc)}

    # Collect (path, z_predicted_post) from successful 2-step trajectories
    # under fused greedy at depth ≥ 2.
    plans: list[dict[str, Any]] = []
    for cell in greedy_payload.get("cells", []):
        if cell.get("status") != "ok":
            continue
        bucket = cell.get("depths", {}).get("2", {}).get("fused")
        if not bucket:
            continue
        # Compute predicted post-2-step latent for each successful record.
        # NOTE: we approximate by re-running the dynamics on the recorded path
        # from the canonical z_ctrl (combo set's own start). This is correct
        # under the assumption that Norman combos are evaluated at their own
        # z_ctrl distribution — which the comparison requires.
        for rec in bucket.get("success_records", []):
            path = rec["path"]
            if len(path) >= 2:
                # Average over a small sample of measured combo z_ctrls to get
                # an empirical control latent. The diagnostic only compares
                # *direction* (cosine), so this is robust to z_ctrl variability.
                z0 = measured["z_ctrl"].mean(axis=0).astype(np.float32)
                z_pred = _rollout_two_steps(ctx.model, z0, path[0], path[1])
                plans.append({"path": (path[0], path[1]), "z_predicted_post": z_pred})

    return compute_norman_combo_consistency(
        plans=plans, measured_combos=measured, check_ordered=False,
    )


def _rollout_two_steps(model, z0: np.ndarray, g1: int, g2: int) -> np.ndarray:
    """Forward two steps under (g1, g2) with 1-indexed gene IDs."""
    z = torch.as_tensor(z0, dtype=torch.float32).unsqueeze(0)
    g = torch.as_tensor([int(g1)], dtype=torch.long)
    with torch.no_grad():
        z_next, _, _ = model(z, g)
        g = torch.as_tensor([int(g2)], dtype=torch.long)
        z_next2, _, _ = model(z_next, g)
    return z_next2.squeeze(0).cpu().numpy().astype(np.float32)


def sub_preconditions(
    *, prediction: dict[str, Any], reachability: dict[str, Any], greedy: dict[str, Any],
    contraction: dict[str, Any], heterogeneity: dict[str, Any], reward_leverage: dict[str, Any],
) -> dict[str, Any]:
    """U-G — eight-criterion check per V3C plan §3 U-G."""
    pre_a = (
        prediction.get("status") == "ok"
        and float(prediction.get("val_pearson") or 0.0) >= 0.40
        and float(prediction.get("ood_pearson") or 0.0) >= 0.20
        and float(prediction.get("uncertainty_spearman") or 0.0) >= 0.10
    )
    # U-B: beam_reach_at_K=2/bin8-10/OOD ≥ 0.10
    pre_b = False
    for cell in reachability.get("cells", []):
        if cell.get("cell_id") == "k2_bin8-10_splitood" and cell.get("status") == "ok":
            pre_b = float(cell.get("beam_reach_at_K_p15") or 0.0) >= 0.10
            break

    # U-C: ≥3 cells with greedy_dyn_5 ∈ [0.10, 0.95] AND cumulative depth leverage ≥0.05 somewhere K≥4
    cells_in_band = 0
    has_depth_leverage = False
    for cell in greedy.get("cells", []):
        if cell.get("status") != "ok":
            continue
        d5 = cell.get("depths", {}).get("5", {}).get("distance", {})
        d5_sr = d5.get("success_rate")
        if d5_sr is not None and 0.10 <= d5_sr <= 0.95:
            cells_in_band += 1
        cum = cell.get("depth_leverage", {}).get("cumulative_K_max_minus_K1")
        if cum is not None and cum >= 0.05 and int(cell.get("K", 0)) >= 4:
            has_depth_leverage = True
    pre_c = cells_in_band >= 3 and has_depth_leverage

    # U-D: contraction_fraction in [0.30, 0.95] AND action_diversity_per_state ≥ 0.05
    s1 = (contraction.get("sample1_ood_pool") or {})
    cf = s1.get("contraction_fraction")
    div = s1.get("action_diversity_per_state")
    pre_d = (
        cf is not None and div is not None
        and 0.30 <= float(cf) <= 0.95
        and float(div) >= 0.05
    )

    # U-E: first_action_entropy_fused ≥ 0.3 * log(n_genes) AND path_diversity_depth2 ≥ 0.10
    pre_e_passes = []
    for cell in heterogeneity.get("cells", []):
        if cell.get("status") and cell.get("status") != "ok":
            # heterogeneity cells reuse OK signal but other-error
            continue
        h = cell.get("first_action_entropy_fused")
        h_max = cell.get("first_action_entropy_max_nats")
        pd2 = cell.get("path_diversity_depth2_distance")
        if h is None or h_max is None or pd2 is None:
            continue
        pre_e_passes.append(h >= 0.30 * h_max and pd2 >= 0.10)
    pre_e = bool(pre_e_passes) and any(pre_e_passes)

    # U-F: distance_vs_fused_first_action_overlap ≤ 0.95 anywhere
    pre_f = False
    for cell in heterogeneity.get("cells", []):
        ov = cell.get("distance_vs_fused_first_action_overlap")
        if ov is not None and ov <= 0.95:
            pre_f = True
            break

    # Pre 7/8: no-op collapse low AND random clearly below planner
    # (Inferred from greedy: success ≥ 0.10 at any K=2/bin8-10/OOD greedy with random ≤ that - 0.10.
    #  We approximate by checking greedy_dyn_2 success is ≥ 0.10 at any bin8-10 cell.)
    g2_signal = False
    for cell in greedy.get("cells", []):
        if cell.get("status") != "ok":
            continue
        d2 = cell.get("depths", {}).get("2", {}).get("distance", {})
        if d2.get("success_rate") is not None and d2["success_rate"] >= 0.10:
            g2_signal = True
            break
    pre_g = g2_signal  # both 7 & 8 collapsed: greedy is non-degenerate

    all_pass = all((pre_a, pre_b, pre_c, pre_d, pre_e, pre_f, pre_g))

    return {
        "status": "ok",
        "u_a_predictive_sanity": bool(pre_a),
        "u_b_reachable": bool(pre_b),
        "u_c_depth_leverage_and_non_saturated": bool(pre_c),
        "u_d_geometry_non_degenerate": bool(pre_d),
        "u_e_action_heterogeneous": bool(pre_e),
        "u_f_reward_leverage_present": bool(pre_f),
        "u_g_planner_provides_leverage": bool(pre_g),
        "all_preconditions_pass": bool(all_pass),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_inventory_row(field_id: str) -> dict[str, Any]:
    inv = pl.read_csv(REPO_ROOT / INVENTORY_REL)
    matches = inv.filter(pl.col("field_id") == field_id)
    if matches.height == 0:
        raise SystemExit(f"field_id '{field_id}' not in {INVENTORY_REL}")
    return matches.to_dicts()[0]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Cannot serialize {type(obj)}")


def _load_or_run(
    out_path: Path, runner, *, force: bool,
) -> dict[str, Any]:
    if out_path.exists() and not force:
        LOG.info("reusing %s", out_path)
        return json.loads(out_path.read_text())
    payload = runner()
    _write_json(out_path, payload)
    return payload


def _write_summary_md(field_dir: Path, field_id: str, payload: dict[str, dict[str, Any]]) -> None:
    lines = [
        f"# Bucket U summary — {field_id}",
        "",
        "Generated by `scripts/audit_dynamics_utility_v3c.py`. `util_score` is a",
        "ranking aid only; smoke-target selection requires written rationale (V3C",
        "plan §4 Stage 3 / guardrail #1).",
        "",
        "## Per-sub-bucket status",
        "",
    ]
    for key in ("u_a", "u_b", "u_c", "u_d", "u_e", "u_f", "u_g"):
        status = (payload.get(key) or {}).get("status", "missing")
        lines.append(f"- **{key.upper()}**: {status}")
    util = payload.get("util_score")
    lines += ["", f"**util_score** (ranking aid): {util}"]
    field_dir.joinpath("bucket_u_summary.md").write_text("\n".join(lines) + "\n")


def cmd_run(args) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    row = _load_inventory_row(args.field_id)
    if not row.get("eligible", True):
        LOG.warning("field %s is not eligible per inventory", args.field_id)
    try:
        ctx = FieldContext(row=row)
    except Exception as exc:                                       # noqa: BLE001
        LOG.error("FieldContext init failed for %s: %s", args.field_id, exc)
        return 2

    field_dir = REPO_ROOT / OUTPUT_ROOT_REL / args.field_id
    field_dir.mkdir(parents=True, exist_ok=True)

    sub = args.subcommand
    force = bool(args.force)

    payload_index: dict[str, Any] = {}

    def _run_sub(name: str, fn) -> dict[str, Any]:
        return _load_or_run(field_dir / f"{name}.json", fn, force=force)

    if sub in ("prediction", "all"):
        payload_index["u_a"] = _run_sub("prediction_metrics", lambda: sub_prediction(ctx))
    if sub in ("contraction", "all"):
        payload_index["u_d"] = _run_sub("contraction_geometry", lambda: sub_contraction(ctx))
    if sub in ("reachability", "all"):
        payload_index["u_b"] = _run_sub(
            "reachability",
            lambda: sub_reachability(ctx, n_episodes=args.n_episodes),
        )
    if sub in ("greedy", "heterogeneity", "reward_leverage", "preconditions", "all"):
        greedy_payload = _run_sub(
            "greedy_saturation",
            lambda: sub_greedy(ctx, n_episodes=args.n_episodes),
        )
        payload_index["u_c"] = greedy_payload
        if sub in ("heterogeneity", "preconditions", "all"):
            payload_index["u_e"] = _run_sub(
                "action_heterogeneity",
                lambda: sub_heterogeneity(ctx, greedy_payload),
            )
        if sub in ("reward_leverage", "preconditions", "all"):
            payload_index["u_f"] = _run_sub(
                "reward_leverage_fused",
                lambda: sub_reward_leverage(ctx, greedy_payload),
            )
    if sub in ("preconditions", "all"):
        prereqs = {
            "prediction": payload_index.get("u_a") or {},
            "reachability": payload_index.get("u_b") or {},
            "greedy": payload_index.get("u_c") or {},
            "contraction": payload_index.get("u_d") or {},
            "heterogeneity": payload_index.get("u_e") or {},
            "reward_leverage": payload_index.get("u_f") or {},
        }
        payload_index["u_g"] = _run_sub(
            "ppo_preconditions",
            lambda: sub_preconditions(**prereqs),
        )

    # util_score (allow_missing=True so partial sub-audits still report).
    if sub == "all":
        buckets_for_score = _buckets_for_score(payload_index)
        util = compute_utility_score(buckets_for_score, allow_missing=True)
        payload_index["util_score"] = util
        _write_summary_md(field_dir, args.field_id, payload_index)
        (field_dir / "bucket_u_index.json").write_text(
            json.dumps(
                {"field_id": args.field_id, "util_score": util,
                 "sub_audits": list(payload_index.keys())},
                indent=2,
                default=_json_default,
            )
        )

    return 0


def _buckets_for_score(payload_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten heterogeneous sub-bucket payloads into util_score-compatible inputs."""
    out: dict[str, dict[str, Any]] = {}
    pred = payload_index.get("u_a") or {}
    if pred.get("status") == "ok":
        out["u_a"] = {"val_pearson": pred.get("val_pearson", 0.0)}
    reach = payload_index.get("u_b") or {}
    if reach.get("cells"):
        by_id = {c["cell_id"]: c for c in reach["cells"] if c.get("status") == "ok"}
        out["u_b"] = {
            "beam_reach_at_K4_bin8_10_p15": (by_id.get("k4_bin8-10_splitood") or {}).get("beam_reach_at_K_p15"),
            "beam_reach_at_K5_bin8_10_p15": (by_id.get("k5_bin8-10_splitood") or {}).get("beam_reach_at_K_p15"),
            "beam_reach_at_K8_bin8_10_p15": (by_id.get("k8_bin8-10_splitood") or {}).get("beam_reach_at_K_p15"),
        }
    greedy = payload_index.get("u_c") or {}
    if greedy.get("max_cumulative_depth_leverage") is not None:
        out["u_c"] = {"cumulative_depth_leverage": greedy.get("max_cumulative_depth_leverage", 0.0)}
    contraction = payload_index.get("u_d") or {}
    s1 = (contraction.get("sample1_ood_pool") or {})
    if s1.get("contraction_fraction") is not None:
        out["u_d"] = {
            "contraction_fraction": s1.get("contraction_fraction", 0.0),
            "gene_universality_max": s1.get("gene_universality_max", 1.0),
        }
    het = payload_index.get("u_e") or {}
    if het.get("cells"):
        primary_cell = next(
            (c for c in het["cells"] if c.get("cell_id") == "k3_bin8-10_splitood" and c.get("first_action_entropy_fused") is not None),
            None,
        )
        if primary_cell:
            out["u_e"] = {
                "first_action_entropy_fused": primary_cell["first_action_entropy_fused"],
                "first_action_entropy_max_nats": primary_cell["first_action_entropy_max_nats"],
            }
    rew = payload_index.get("u_f") or {}
    if rew.get("cells"):
        # Pull a representative cell's overlap (smaller = better)
        ov_vals = []
        # Need to call heterogeneity since reward_leverage doesn't include overlap directly
        if het.get("cells"):
            for c in het["cells"]:
                ov = c.get("distance_vs_fused_first_action_overlap")
                if ov is not None:
                    ov_vals.append(ov)
        if ov_vals:
            out["u_f"] = {"distance_vs_fused_first_action_overlap": float(np.median(ov_vals))}
    g = payload_index.get("u_g") or {}
    out["u_g"] = {"all_preconditions_pass": bool(g.get("all_preconditions_pass", False))}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V3C dynamics utility audit driver (Phase 0B)")
    parser.add_argument("subcommand",
                        choices=["prediction", "reachability", "greedy", "contraction",
                                 "heterogeneity", "reward_leverage", "preconditions", "all"])
    parser.add_argument("--field-id", required=True, help="inventory field_id (e.g. artifacts_v2__dynamics_v1ot_ror_corr010)")
    parser.add_argument("--n-episodes", type=int, default=64, help="rollouts per (cell, policy) for U-B and U-C")
    parser.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
