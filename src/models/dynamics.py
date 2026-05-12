"""Residual heteroscedastic dynamics MLP.

Owner: Agent B. See ARCHITECTURE.md Concept 3.

Architecture
------------
::

    z (B, 32)   ─┐
                 ├─►  cat  ──► residual MLP  ──► (mu, log_var)
    gene_idx (B,)┘                                  each (B, 32)

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
from torch import nn


class PerturbationDynamicsModel(nn.Module):
    """f_θ(z, gene_idx) → (z_next, μ, log σ²) with residual head + heteroscedastic outputs.

    Parameters
    ----------
    n_latent
        Latent dim (matches ``cfg.vae.n_latent``, default 32).
    n_genes
        Number of perturbable genes (matches ``gene_vocab["n_genes"]``). The embedding table is
        sized ``n_genes + 1`` so the final index can encode NO-OP at inference time, though
        NO-OP is not used during dynamics training.
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
    >>> g = torch.randint(0, 100, (8,))
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
        # Subclasses / future agent B implementation should populate:
        #   self.gene_embedding : nn.Embedding(n_genes + 1, d_emb)
        #   self.trunk          : residual MLP (n_latent + d_emb -> n_hidden, n_layers blocks)
        #   self.head_mu        : Linear(n_hidden, n_latent)
        #   self.head_log_var   : Linear(n_hidden, n_latent), bias init = log_var_init_bias

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
            Shape ``(B,)`` int. In ``[0, n_genes]``; ``n_genes`` is NO-OP.

        Returns
        -------
        z_next : torch.Tensor
            Shape ``(B, n_latent)``. Equals ``z + μ`` (residual).
        mu : torch.Tensor
            Predicted Δz mean, shape ``(B, n_latent)``.
        log_var : torch.Tensor
            Predicted per-dim log σ² (clamped to ``[log_var_min, log_var_max]``),
            shape ``(B, n_latent)``.

        Raises
        ------
        NotImplementedError
            Agent B: implement the trunk + heads per ARCHITECTURE.md Concept 3.
        """
        raise NotImplementedError(
            "Agent B: implement forward pass. Trunk = residual MLP with SiLU + LayerNorm. "
            "Heads = Linear(n_hidden -> n_latent) for mu and Linear(n_hidden -> n_latent) for log_var. "
            "Clamp log_var to [log_var_min, log_var_max]. Return (z + mu, mu, log_var)."
        )


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

    Raises
    ------
    NotImplementedError
        Agent B: implement the formula above.
    """
    raise NotImplementedError(
        "Agent B: heteroscedastic Gaussian NLL with the formula in the docstring."
    )


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

    Raises
    ------
    NotImplementedError
        Agent B: implement ``z_after_a = model(z_ctrl, a).z_next`` then
        ``z_after_ab = model(z_after_a, b).z_next`` and MSE against ``z_pert_ab``.
    """
    raise NotImplementedError(
        "Agent B: chained forward + MSE. Used in DATA.md §3 / PHASES.md Phase 1 dynamics training."
    )
