"""Residual heteroscedastic dynamics MLP.

Owner: Agent B. See ARCHITECTURE.md Concept 3.

Architecture
------------
::

    z (B, 32)   ─┐
                 ├─►  cat  ──► input_proj ──► residual blocks ──► (mu, log_var)
    gene_idx (B,)┘                                                  each (B, 32)

    z_next = z + mu                                  (residual head)
    loss   = heteroscedastic Gaussian NLL on Δz       (training)
    σ²     = exp(log_var.clamp(min, max))             (clamped output)

The residual head matches the small ``||Δz|| ≪ ||z||`` magnitude observed in Norman; the
heteroscedastic head provides per-step uncertainty used both for the validation gate and as an
optional exploration penalty in the RL reward.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class _ResidualBlock(nn.Module):
    """Single residual block: LayerNorm → Linear → SiLU → Dropout → Linear + skip."""

    def __init__(self, n_hidden: int, dropout: float = 0.1, use_layernorm: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_hidden) if use_layernorm else nn.Identity()
        self.linear1 = nn.Linear(n_hidden, n_hidden)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x + residual


class PerturbationDynamicsModel(nn.Module):
    """f_θ(z, gene_idx) → (z_next, μ, log σ²) with residual head + heteroscedastic outputs.

    Parameters
    ----------
    n_latent
        Latent dim (matches ``cfg.vae.n_latent``, default 32).
    n_genes
        Number of perturbable genes (matches ``gene_vocab["n_genes"]``). The embedding table is
        sized ``n_genes + 1`` so index 0 is a reserved placeholder (ctrl) and indices 1..n_genes
        are the actual gene embeddings, matching the 1-indexed ``gene_idx`` in Contract 2 pairs.
    d_emb
        Gene-embedding dimension (default 64).
    n_hidden
        MLP hidden width (default 256).
    n_layers
        Number of residual blocks in the trunk (default 3).
    log_var_min
        Lower clamp on predicted log σ² (default -5.0).
    log_var_max
        Upper clamp (default 3.0).

    Examples
    --------
    >>> model = PerturbationDynamicsModel(n_latent=32, n_genes=100)
    >>> z = torch.randn(8, 32)
    >>> g = torch.randint(1, 101, (8,))  # 1-indexed gene indices
    >>> z_next, mu, log_var = model(z, g)
    >>> z_next.shape, mu.shape, log_var.shape
    (torch.Size([8, 32]), torch.Size([8, 32]), torch.Size([8, 32]))
    """

    def __init__(
        self,
        n_latent: int = 32,
        n_genes: int = 100,
        d_emb: int = 64,
        n_hidden: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "silu",
        log_var_min: float = -5.0,
        log_var_max: float = 3.0,
        log_var_init_bias: float = -2.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.n_latent = n_latent
        self.n_genes = n_genes
        self.d_emb = d_emb
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        _act_map = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}
        if activation not in _act_map:
            raise ValueError(f"Unknown activation {activation!r}. Choose from {list(_act_map)}.")
        act_cls = _act_map[activation]

        # Gene embedding: index 0 = ctrl placeholder (unused in training);
        # indices 1..n_genes correspond to the 1-indexed gene_idx in Contract 2 pairs.
        self.gene_embedding = nn.Embedding(n_genes + 1, d_emb)

        # Input projection: (n_latent + d_emb) → n_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(n_latent + d_emb, n_hidden),
            act_cls(),
        )

        # Residual trunk: n_layers blocks at n_hidden
        self.trunk = nn.Sequential(*[
            _ResidualBlock(n_hidden, dropout=dropout, use_layernorm=use_layernorm)
            for _ in range(n_layers)
        ])

        # Output heads
        self.head_mu = nn.Linear(n_hidden, n_latent)
        self.head_log_var = nn.Linear(n_hidden, n_latent)

        # Initialise log_var bias to avoid early overconfidence
        nn.init.constant_(self.head_log_var.bias, log_var_init_bias)

    def forward(
        self,
        z: torch.Tensor,
        gene_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict (z_next, μ, log σ²) for a batch of (latent, gene_idx) pairs.

        Parameters
        ----------
        z
            Shape ``(B, n_latent)``.
        gene_idx
            Shape ``(B,)`` int. Range ``[1, n_genes]`` during training (1-indexed); the
            environment converts 0-indexed RL actions to 1-indexed gene_idx before calling.

        Returns
        -------
        z_next : torch.Tensor
            Shape ``(B, n_latent)``. Equals ``z + μ`` (residual).
        mu : torch.Tensor
            Predicted Δz mean, shape ``(B, n_latent)``.
        log_var : torch.Tensor
            Predicted per-dim log σ² (clamped to ``[log_var_min, log_var_max]``),
            shape ``(B, n_latent)``.
        """
        emb = self.gene_embedding(gene_idx)            # (B, d_emb)
        h = torch.cat([z, emb], dim=-1)                # (B, n_latent + d_emb)
        h = self.input_proj(h)                         # (B, n_hidden)
        h = self.trunk(h)                              # (B, n_hidden)
        mu = self.head_mu(h)                           # (B, n_latent)
        log_var = self.head_log_var(h).clamp(
            self.log_var_min, self.log_var_max
        )                                              # (B, n_latent)
        z_next = z + mu                                # residual connection
        return z_next, mu, log_var


def heteroscedastic_nll(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target_delta: torch.Tensor,
    log_var_reg: float = 0.01,
) -> torch.Tensor:
    """Per-dim heteroscedastic Gaussian NLL on Δz (with optional log_var L2 penalty).

    NLL_i = 0.5 * exp(-log_var_i) * (target_delta_i - mu_i)^2 + 0.5 * log_var_i

    Parameters
    ----------
    mu
        Predicted mean Δz, shape ``(B, n_latent)``.
    log_var
        Predicted log σ², shape ``(B, n_latent)``.
    target_delta
        Empirical Δz = z_pert - z_ctrl, shape ``(B, n_latent)``.
    log_var_reg
        L2 penalty on ``log_var`` to discourage extremes. Default 0.01.

    Returns
    -------
    torch.Tensor
        Scalar mean NLL across the batch and latent dims.
    """
    precision = torch.exp(-log_var)
    per_element = 0.5 * precision * (target_delta - mu).pow(2) + 0.5 * log_var
    loss = per_element.mean()
    if log_var_reg > 0.0:
        loss = loss + log_var_reg * log_var.pow(2).mean()
    return loss


def composition_loss(
    model: PerturbationDynamicsModel,
    z_ctrl: torch.Tensor,
    gene_idx_a: torch.Tensor,
    gene_idx_b: torch.Tensor,
    z_pert_ab: torch.Tensor,
) -> torch.Tensor:
    """Sequential-composition loss for combinatorial perturbations.

    Predicts z after applying ``gene_idx_a`` then ``gene_idx_b`` to ``z_ctrl``, compares to
    the empirical ``z_pert_ab`` of the dual-guide cells. This is the main signal that the
    dynamics model has learned *composable* perturbation effects.

    Parameters
    ----------
    model
        The dynamics model.
    z_ctrl
        Shape ``(B, n_latent)``.
    gene_idx_a, gene_idx_b
        Each shape ``(B,)``.
    z_pert_ab
        Shape ``(B, n_latent)``.

    Returns
    -------
    torch.Tensor
        Scalar MSE between predicted and empirical post-double-perturbation latent.

    Notes
    -----
    ``gene_idx_a`` and ``gene_idx_b`` must be ``LongTensor`` (1-indexed per Contract 2).
    Gradients flow through both forward passes; do **not** detach ``z_next_a`` so that
    composability is the actual training signal.
    """
    z_next_a, _, _ = model(z_ctrl, gene_idx_a)
    z_next_ab, _, _ = model(z_next_a, gene_idx_b)
    return F.mse_loss(z_next_ab, z_pert_ab)
