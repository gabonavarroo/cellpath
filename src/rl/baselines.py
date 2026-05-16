"""Read-only policy baselines for the V2 hard RL benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _valid_actions(mask: np.ndarray) -> np.ndarray:
    valid = np.where(np.asarray(mask, dtype=bool))[0]
    return valid.astype(np.int64)


class RandomUniformValidPolicy:
    name = "random_uniform_valid"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        valid = _valid_actions(mask)
        if len(valid) == 0:
            return int(len(mask) - 1)
        return int(self.rng.choice(valid))


class AlwaysNoopPolicy:
    name = "always_noop"

    def __init__(self, noop_idx: int) -> None:
        self.noop_idx = int(noop_idx)

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        return self.noop_idx


class GreedyDynamicsPolicy:
    name = "greedy_dyn_1"

    def __init__(self, dynamics: Any, *, n_genes: int, z_ref: np.ndarray, noop_idx: int) -> None:
        self.dynamics = dynamics
        self.n_genes = int(n_genes)
        self.z_ref = np.asarray(z_ref, dtype=np.float32)
        self.noop_idx = int(noop_idx)

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        import torch

        z = np.asarray(z, dtype=np.float32)
        valid = _valid_actions(mask)
        if len(valid) == 0:
            return self.noop_idx
        gene_actions = np.array([a for a in valid if 0 <= a < self.n_genes], dtype=np.int64)
        candidates: dict[int, np.ndarray] = {}
        if self.noop_idx in valid:
            candidates[self.noop_idx] = z
        if len(gene_actions):
            z_batch = np.repeat(z[None, :], len(gene_actions), axis=0)
            gene_idx = gene_actions + 1
            with torch.no_grad():
                out = self.dynamics(
                    torch.from_numpy(z_batch).float(),
                    torch.from_numpy(gene_idx.astype(np.int64)).long(),
                )
            z_next = out[0] if isinstance(out, tuple) else out
            if isinstance(z_next, torch.Tensor):
                z_next_np = z_next.detach().cpu().numpy().astype(np.float32)
            else:
                z_next_np = np.asarray(z_next, dtype=np.float32)
            for action, zn in zip(gene_actions, z_next_np, strict=True):
                candidates[int(action)] = zn
        return _argmin_distance(candidates, self.z_ref)


class NoopFreeGreedyPolicy:
    """Greedy 1-step dynamics policy that never picks noop.

    Identical to GreedyDynamicsPolicy but excludes noop_idx from candidates.
    Probes whether the dynamics field is navigable when the agent is forced
    to apply gene perturbations at every step (no early termination).
    Falls back to noop only if ALL gene actions are masked.
    """
    name = "greedy_dyn_1_noop_free"

    def __init__(self, dynamics: Any, *, n_genes: int, z_ref: np.ndarray, noop_idx: int) -> None:
        self.dynamics = dynamics
        self.n_genes = int(n_genes)
        self.z_ref = np.asarray(z_ref, dtype=np.float32)
        self.noop_idx = int(noop_idx)

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        import torch

        z = np.asarray(z, dtype=np.float32)
        valid = _valid_actions(mask)
        gene_actions = np.array([a for a in valid if 0 <= a < self.n_genes], dtype=np.int64)
        if len(gene_actions) == 0:
            return self.noop_idx  # only safe fallback when all genes are masked
        candidates: dict[int, np.ndarray] = {}
        z_batch = np.repeat(z[None, :], len(gene_actions), axis=0)
        gene_idx = gene_actions + 1  # 0-indexed action → 1-indexed gene_idx
        with torch.no_grad():
            out = self.dynamics(
                torch.from_numpy(z_batch).float(),
                torch.from_numpy(gene_idx.astype(np.int64)).long(),
            )
        z_next = out[0] if isinstance(out, tuple) else out
        if isinstance(z_next, torch.Tensor):
            z_next_np = z_next.detach().cpu().numpy().astype(np.float32)
        else:
            z_next_np = np.asarray(z_next, dtype=np.float32)
        for action, zn in zip(gene_actions, z_next_np, strict=True):
            candidates[int(action)] = zn
        return _argmin_distance(candidates, self.z_ref)


class RidgeGreedyPolicy:
    name = "ridge_greedy"

    def __init__(self, W_z: np.ndarray, W_gene: np.ndarray, b: np.ndarray, *, z_ref: np.ndarray, noop_idx: int) -> None:
        self.W_z = np.asarray(W_z, dtype=np.float32)
        self.W_gene = np.asarray(W_gene, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)
        self.z_ref = np.asarray(z_ref, dtype=np.float32)
        self.noop_idx = int(noop_idx)

    @classmethod
    def from_npz(cls, path: str | Path, *, z_ref: np.ndarray, noop_idx: int) -> "RidgeGreedyPolicy":
        with np.load(path) as data:
            return cls(data["W_z"], data["W_gene"], data["b"], z_ref=z_ref, noop_idx=noop_idx)

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        z = np.asarray(z, dtype=np.float32)
        valid = _valid_actions(mask)
        candidates: dict[int, np.ndarray] = {}
        for action in valid:
            action = int(action)
            if action == self.noop_idx:
                candidates[action] = z
            elif 0 <= action < len(self.W_gene):
                delta = z @ self.W_z + self.W_gene[action] + self.b
                candidates[action] = z + delta
        if not candidates:
            return self.noop_idx
        return _argmin_distance(candidates, self.z_ref)


class MeanDeltaGreedyPolicy:
    name = "mean_delta_greedy"

    def __init__(self, mean_delta_table: np.ndarray, *, z_ref: np.ndarray, noop_idx: int) -> None:
        self.mean_delta_table = np.asarray(mean_delta_table, dtype=np.float32)
        self.z_ref = np.asarray(z_ref, dtype=np.float32)
        self.noop_idx = int(noop_idx)

    def select_action(self, z: np.ndarray, mask: np.ndarray, info: dict[str, Any]) -> int:
        z = np.asarray(z, dtype=np.float32)
        valid = _valid_actions(mask)
        candidates: dict[int, np.ndarray] = {}
        for action in valid:
            action = int(action)
            if action == self.noop_idx:
                candidates[action] = z
            elif 0 <= action < len(self.mean_delta_table):
                candidates[action] = z + self.mean_delta_table[action]
        if not candidates:
            return self.noop_idx
        return _argmin_distance(candidates, self.z_ref)


def _argmin_distance(candidates: dict[int, np.ndarray], z_ref: np.ndarray) -> int:
    best_action = min(
        candidates,
        key=lambda action: (
            float(np.linalg.norm(np.asarray(candidates[action], dtype=np.float32) - z_ref)),
            int(action),
        ),
    )
    return int(best_action)
