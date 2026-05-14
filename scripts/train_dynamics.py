"""scripts/train_dynamics.py — dynamics-model training entry point.

Owner: Agent B. See AGENTS.md §2 Phase 1 + Phase 2.

Usage
-----
::

    python scripts/train_dynamics.py --config-name default
    python scripts/train_dynamics.py dynamics.lambda_combo=0.0
    python scripts/train_dynamics.py +dry_run=true          # validate config + paths, exit 0
    python scripts/train_dynamics.py +force=true            # retrain even if model.pt exists
    python scripts/train_dynamics.py --multirun dynamics.n_layers=2,3,4

Workflow
--------
1.  Compose Hydra config.
2.  ``src.utils.seeding.set_seed(cfg.seed)``.
3.  Log device summary.
4.  Check for an existing checkpoint at ``cfg.paths.dynamics_model``; if present and
    ``force=false``, skip training (prints a warning if ``n_genes`` changed vs saved config).
    Training is skipped but the gate (Step 10) still runs if val pairs exist.
5.  Load Contract 2 pairs from ``cfg.paths.pairs_{train,val,combo}``. If the train file is
    missing, fall back to :func:`src.data.perturbation_pairs.generate_mock_pairs`.
6.  Infer ``n_genes`` from ``gene_vocab.json`` (real Norman path) or ``metadata.json`` (mock).
    Cross-check ``n_latent`` against ``cfg.vae.n_latent``.
7.  Build :class:`src.models.dynamics.PerturbationDynamicsModel` from ``cfg.dynamics``.
8.  Build ``DataLoader``\\s; train with AdamW + optional cosine LR scheduler.
    Loss = heteroscedastic NLL + ``lambda_combo`` × composition MSE (when combo pairs exist).
    Early stopping on val NLL; best checkpoint saved atomically.
9.  Write ``artifacts/dynamics/model.pt`` (best state dict) + ``config.json``.
10. Run dynamics validation gate: load best checkpoint, predict on val + OOD splits,
    call :func:`src.analysis.metrics.dynamics_validation_gate`, write ``gate.json``,
    ``val_metrics.json``, and ``ood_metrics.json``. Return exit code 1 if the val gate fails.
    OOD is report-only and does not gate. Missing val pairs are a hard error (exit 1).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Generator

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def _iter_infinite(loader: DataLoader) -> Generator:
    """Yield batches from *loader* indefinitely, restarting after each epoch."""
    while True:
        yield from loader


def _npz_to_tensors(path: Path, keys: list[str]) -> tuple[torch.Tensor, ...]:
    """Load selected arrays from a ``.npz`` file as CPU tensors.

    Integer arrays (int8/16/32) are upcast to ``int64`` as required by
    ``nn.Embedding``. Float arrays are kept as-is (float32).
    """
    data = np.load(path)
    tensors: list[torch.Tensor] = []
    for k in keys:
        arr = data[k].copy()
        t = torch.from_numpy(arr)
        if t.dtype in (torch.int8, torch.int16, torch.int32):
            t = t.long()
        tensors.append(t)
    return tuple(tensors)


def _build_loaders(
    train_path: Path,
    val_path: Path,
    combo_path: Path | None,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Return (train_loader, val_loader, combo_loader | None)."""

    def _loader(path: Path, keys: list[str], shuffle: bool) -> DataLoader:
        tensors = _npz_to_tensors(path, keys)
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    train_loader = _loader(train_path, ["z_ctrl", "gene_idx", "z_pert"], shuffle=True)
    val_loader   = _loader(val_path,   ["z_ctrl", "gene_idx", "z_pert"], shuffle=False)
    combo_loader = (
        _loader(
            combo_path,
            ["z_ctrl", "gene_idx_a", "gene_idx_b", "z_pert_ab"],
            shuffle=True,
        )
        if (combo_path is not None and combo_path.exists())
        else None
    )
    return train_loader, val_loader, combo_loader


def _resolve_n_genes(cfg: DictConfig, train_path: Path) -> int:
    """Infer ``n_genes`` with a three-level fallback chain.

    1. ``gene_vocab.json`` — written by Agent A after VAE training (real Norman path).
    2. ``pairs/metadata.json`` — written by ``generate_mock_pairs`` (mock path).
    3. ``int(train_pairs['gene_idx'].max())`` — last-resort defensive inference.
    """
    vocab_path = Path(cfg.paths.vae_gene_vocab_json)
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text())
        n = int(vocab["n_genes"])
        log.info("n_genes=%d  [source: gene_vocab.json]", n)
        return n

    meta_path = Path(cfg.paths.pairs_metadata)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if "n_genes" in meta:
            n = int(meta["n_genes"])
            log.info("n_genes=%d  [source: pairs/metadata.json]", n)
            return n

    data = np.load(train_path)
    n = int(data["gene_idx"].max())
    log.warning(
        "n_genes=%d inferred from gene_idx.max() — neither gene_vocab.json nor "
        "pairs/metadata.json found; verify this matches the intended action-space size.",
        n,
    )
    return n


def _save_state_dict(model: nn.Module, path: Path) -> None:
    """Atomically write ``model.state_dict()`` to *path* via a tmp-then-rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(str(tmp), str(path))
    log.debug("Checkpoint saved → %s", path)


def _predict_split(
    model: nn.Module,
    z_ctrl: torch.Tensor,
    gene_idx: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run dynamics in eval mode; return (z_pert_pred, log_var) as float32 numpy arrays.

    MPS-safe: each mini-batch is moved to *device* and predictions are moved back to CPU
    before concatenation. Stays in float32 throughout — MPS does not support float64.

    z_pert_pred = z_ctrl + mu  (residual head, same as z_next from the model).
    log_var is returned raw; the gate computes exp(log_var) internally.
    """
    pred_parts: list[np.ndarray] = []
    lv_parts:   list[np.ndarray] = []
    n = z_ctrl.shape[0]
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            zc = z_ctrl[start : start + batch_size].to(device)
            gi = gene_idx[start : start + batch_size].to(device)
            _, mu, lv = model(zc, gi)
            # z_pert_pred = z_ctrl + mu (explicit residual; mirrors the math in the gate)
            pred_parts.append((zc + mu).detach().cpu().numpy().astype(np.float32))
            lv_parts.append(lv.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(pred_parts, axis=0), np.concatenate(lv_parts, axis=0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> int:
    """Hydra-driven main. See module docstring for the 10-step workflow."""
    # Deferred src.* imports so Hydra config composition never touches the model.
    from src.data.perturbation_pairs import generate_mock_pairs
    from src.models.dynamics import (
        PerturbationDynamicsModel,
        composition_loss,
        heteroscedastic_nll,
    )
    from src.utils.device import device_summary, get_device
    from src.utils.seeding import set_seed

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    set_seed(int(cfg.seed))
    device = get_device()
    log.info(device_summary())

    # ------------------------------------------------------------------
    # Dry-run: validate config resolution, then exit
    # ------------------------------------------------------------------
    if cfg.get("dry_run", False):
        log.info(
            "DRY RUN — dynamics config: d_emb=%d  n_hidden=%d  n_layers=%d  "
            "lr=%.1e  lambda_combo=%.3f",
            cfg.dynamics.d_emb,
            cfg.dynamics.n_hidden,
            cfg.dynamics.n_layers,
            cfg.dynamics.lr,
            cfg.dynamics.lambda_combo,
        )
        log.info("DRY RUN — output dir: %s", cfg.paths.dynamics_dir)
        return 0

    # ------------------------------------------------------------------
    # Step 4 (pair loading) — mock fallback if real pairs are absent
    # ------------------------------------------------------------------
    train_path  = Path(cfg.paths.pairs_train)
    val_path    = Path(cfg.paths.pairs_val)
    combo_path  = Path(cfg.paths.pairs_combo)
    pairs_dir   = Path(cfg.paths.pairs_dir)

    if not train_path.exists():
        log.info(
            "Pair files not found at %s — generating mock pairs "
            "(n=10_000, n_genes=100, n_latent=%d, seed=%d).",
            pairs_dir,
            int(cfg.vae.n_latent),
            int(cfg.seed),
        )
        generate_mock_pairs(
            n=10_000,
            n_genes=100,
            n_latent=int(cfg.vae.n_latent),
            n_combo=1_000,
            seed=int(cfg.seed),
            out_dir=pairs_dir,
        )
        log.info("Mock pairs written to %s", pairs_dir)

    # ------------------------------------------------------------------
    # Steps 5+6 — infer n_genes; cross-check n_latent vs config
    # ------------------------------------------------------------------
    n_genes = _resolve_n_genes(cfg, train_path)

    train_np       = np.load(train_path)
    n_latent_data  = int(train_np["z_ctrl"].shape[1])
    n_latent_cfg   = int(cfg.vae.n_latent)
    if n_latent_data != n_latent_cfg:
        raise ValueError(
            f"Contract violation: train_pairs z_ctrl has n_latent={n_latent_data} "
            f"but cfg.vae.n_latent={n_latent_cfg}. "
            "Delete artifacts/pairs/ or update the config to match."
        )
    n_latent = n_latent_data
    log.info("n_latent=%d (verified against cfg.vae.n_latent)", n_latent)

    # ------------------------------------------------------------------
    # Checkpoint check — skip retraining if a compatible model.pt exists
    # ------------------------------------------------------------------
    dynamics_dir = Path(cfg.paths.dynamics_dir)
    dynamics_dir.mkdir(parents=True, exist_ok=True)
    model_path   = Path(cfg.paths.dynamics_model)

    skip_training = False
    if model_path.exists() and not cfg.get("force", False):
        saved_cfg_path = Path(cfg.paths.dynamics_config)
        mismatches: list[str] = []
        if saved_cfg_path.exists():
            saved = json.loads(saved_cfg_path.read_text())
            if saved.get("n_genes") != n_genes:
                mismatches.append(
                    f"n_genes: saved={saved.get('n_genes')}  current={n_genes}"
                )
            if saved.get("n_latent") != n_latent:
                mismatches.append(
                    f"n_latent: saved={saved.get('n_latent')}  current={n_latent}"
                )
            meta_path = Path(cfg.paths.pairs_metadata)
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                current_source = "mock" if meta.get("pairing_method") == "mock" else "real"
                if saved.get("trained_on") != current_source:
                    mismatches.append(
                        f"trained_on: saved={saved.get('trained_on')}  current={current_source}"
                    )
            # Architecture-ablation flags: a saved checkpoint that lacks these keys
            # is treated as both-False (the legacy baseline) so existing checkpoints
            # continue to load when flags stay off.
            current_state_skip = bool(cfg.dynamics.get("use_state_linear_skip", False))
            current_gene_bias  = bool(cfg.dynamics.get("use_gene_delta_bias",   False))
            if bool(saved.get("use_state_linear_skip", False)) != current_state_skip:
                mismatches.append(
                    f"use_state_linear_skip: saved="
                    f"{bool(saved.get('use_state_linear_skip', False))}  current={current_state_skip}"
                )
            if bool(saved.get("use_gene_delta_bias", False)) != current_gene_bias:
                mismatches.append(
                    f"use_gene_delta_bias: saved="
                    f"{bool(saved.get('use_gene_delta_bias', False))}  current={current_gene_bias}"
                )
        if mismatches:
            raise RuntimeError(
                f"Checkpoint at {model_path} is incompatible with current data:\n"
                + "\n".join(f"  • {m}" for m in mismatches)
                + "\nRerun with +force=true to retrain from scratch."
            )
        log.info(
            "Checkpoint found at %s — config matches, skipping training. "
            "Pass +force=true to retrain. Proceeding to validation gate (Step 10).",
            model_path,
        )
        skip_training = True

    # ------------------------------------------------------------------
    # Step 7 — build model (always, whether training or loading checkpoint)
    # ------------------------------------------------------------------
    use_state_linear_skip = bool(cfg.dynamics.get("use_state_linear_skip", False))
    use_gene_delta_bias   = bool(cfg.dynamics.get("use_gene_delta_bias",   False))
    model = PerturbationDynamicsModel(
        n_latent              = n_latent,
        n_genes               = n_genes,
        d_emb                 = int(cfg.dynamics.d_emb),
        n_hidden              = int(cfg.dynamics.n_hidden),
        n_layers              = int(cfg.dynamics.n_layers),
        dropout               = float(cfg.dynamics.dropout),
        activation            = str(cfg.dynamics.activation),
        log_var_min           = float(cfg.dynamics.log_var_min),
        log_var_max           = float(cfg.dynamics.log_var_max),
        log_var_init_bias     = float(cfg.dynamics.log_var_init_bias),
        use_layernorm         = bool(cfg.dynamics.use_layernorm),
        use_state_linear_skip = use_state_linear_skip,
        use_gene_delta_bias   = use_gene_delta_bias,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "PerturbationDynamicsModel built | n_latent=%d | n_genes=%d | "
        "params=%d | device=%s",
        n_latent, n_genes, n_params, device,
    )

    # ------------------------------------------------------------------
    # Steps 8–9 — train + write config (skipped when checkpoint reused)
    # ------------------------------------------------------------------
    if not skip_training:
        # ---- 8a: DataLoaders ----
        pin_memory = (device.type == "cuda")
        train_loader, val_loader, combo_loader = _build_loaders(
            train_path  = train_path,
            val_path    = val_path,
            combo_path  = combo_path,
            batch_size  = int(cfg.dynamics.batch_size),
            num_workers = int(cfg.dynamics.num_workers),
            pin_memory  = pin_memory,
        )
        log.info(
            "DataLoaders ready | train=%d batches | val=%d batches | combo=%s",
            len(train_loader),
            len(val_loader),
            f"{len(combo_loader)} batches" if combo_loader is not None else "none",
        )

        # ---- 8b: optimizer + LR scheduler ----
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = float(cfg.dynamics.lr),
            weight_decay = float(cfg.dynamics.weight_decay),
        )

        scheduler_name = str(cfg.dynamics.scheduler).lower()
        if scheduler_name == "cosine":
            scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max   = int(cfg.dynamics.max_epochs),
                    eta_min = float(cfg.dynamics.lr) * 1e-2,
                )
            )
            log.info(
                "CosineAnnealingLR: %.2e → %.2e over %d epochs",
                float(cfg.dynamics.lr),
                float(cfg.dynamics.lr) * 1e-2,
                int(cfg.dynamics.max_epochs),
            )
        else:
            scheduler = None

        # ---- 8c: training loop ----
        lambda_combo      = float(cfg.dynamics.lambda_combo)
        lambda_lv_reg     = float(cfg.dynamics.lambda_log_var_reg)
        max_epochs        = int(cfg.dynamics.max_epochs)
        patience          = int(cfg.dynamics.early_stop_patience)

        combo_iter        = _iter_infinite(combo_loader) if combo_loader is not None else None
        best_val_nll      = float("inf")
        best_epoch        = 0
        patience_counter  = 0
        epochs_run        = 0
        final_val_nll     = float("inf")

        log.info(
            "Training start | max_epochs=%d | patience=%d | "
            "lambda_combo=%.3f | lambda_lv_reg=%.4f",
            max_epochs, patience, lambda_combo, lambda_lv_reg,
        )

        for epoch in range(max_epochs):
            # ---- train ----
            model.train()
            epoch_nll   = 0.0
            epoch_combo = 0.0
            n_train     = 0

            for z_c, g, z_p in train_loader:
                z_c = z_c.to(device)
                g   = g.to(device)
                z_p = z_p.to(device)

                _, mu, lv        = model(z_c, g)
                target_delta     = z_p - z_c
                loss_nll         = heteroscedastic_nll(mu, lv, target_delta, log_var_reg=lambda_lv_reg)
                loss             = loss_nll
                batch_combo_loss = 0.0

                if combo_iter is not None and lambda_combo > 0.0:
                    z_cc, g_a, g_b, z_ab = next(combo_iter)
                    z_cc = z_cc.to(device)
                    g_a  = g_a.to(device)
                    g_b  = g_b.to(device)
                    z_ab = z_ab.to(device)
                    loss_c           = composition_loss(model, z_cc, g_a, g_b, z_ab)
                    loss             = loss + lambda_combo * loss_c
                    batch_combo_loss = loss_c.item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_nll   += loss_nll.item()
                epoch_combo += batch_combo_loss
                n_train     += 1

            if scheduler is not None:
                scheduler.step()

            # ---- validate ----
            model.eval()
            val_nll_sum = 0.0
            n_val       = 0
            with torch.no_grad():
                for z_c, g, z_p in val_loader:
                    z_c = z_c.to(device)
                    g   = g.to(device)
                    z_p = z_p.to(device)
                    _, mu_v, lv_v = model(z_c, g)
                    val_nll_sum  += heteroscedastic_nll(
                        mu_v, lv_v, z_p - z_c, log_var_reg=0.0
                    ).item()
                    n_val += 1

            avg_nll   = epoch_nll   / max(n_train, 1)
            avg_combo = epoch_combo / max(n_train, 1)
            avg_val   = val_nll_sum / max(n_val,   1)
            avg_total = avg_nll + lambda_combo * avg_combo
            epochs_run    = epoch + 1
            final_val_nll = avg_val

            log.info(
                "epoch %03d/%03d | loss=%.4f  nll=%.4f  combo=%.4f | "
                "val_nll=%.4f  best=%.4f | lr=%.2e",
                epochs_run, max_epochs,
                avg_total, avg_nll, avg_combo,
                avg_val, best_val_nll,
                optimizer.param_groups[0]["lr"],
            )

            if avg_val < best_val_nll:
                best_val_nll     = avg_val
                best_epoch       = epochs_run
                patience_counter = 0
                _save_state_dict(model, model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log.info(
                        "Early stopping at epoch %d (no val improvement for %d epochs). "
                        "Best epoch=%d  val_nll=%.4f.",
                        epochs_run, patience, best_epoch, best_val_nll,
                    )
                    break

        # Safety fallback: if val_nll never improved (e.g. NaN on first epoch)
        if not model_path.exists():
            log.warning("No best checkpoint was written; saving current model state.")
            _save_state_dict(model, model_path)

        log.info(
            "Training complete | epochs_run=%d | best_epoch=%d | best_val_nll=%.4f",
            epochs_run, best_epoch, best_val_nll,
        )

        # ---- Step 9: write config.json ----
        meta_path  = Path(cfg.paths.pairs_metadata)
        trained_on = "unknown"
        if meta_path.exists():
            meta       = json.loads(meta_path.read_text())
            trained_on = "mock" if meta.get("pairing_method") == "mock" else "real"

        config_record: dict = {
            "n_latent"             : n_latent,
            "n_genes"              : n_genes,
            "d_emb"                : int(cfg.dynamics.d_emb),
            "n_hidden"             : int(cfg.dynamics.n_hidden),
            "n_layers"             : int(cfg.dynamics.n_layers),
            "dropout"              : float(cfg.dynamics.dropout),
            "activation"           : str(cfg.dynamics.activation),
            "log_var_min"          : float(cfg.dynamics.log_var_min),
            "log_var_max"          : float(cfg.dynamics.log_var_max),
            "use_layernorm"        : bool(cfg.dynamics.use_layernorm),
            "use_state_linear_skip": use_state_linear_skip,
            "use_gene_delta_bias"  : use_gene_delta_bias,
            "trained_on"           : trained_on,
            "epochs_run"           : epochs_run,
            "best_epoch"           : best_epoch,
            "best_val_nll"         : best_val_nll,
            "final_val_nll"        : final_val_nll,
            "lambda_combo"         : lambda_combo,
            "lambda_log_var_reg"   : lambda_lv_reg,
            "seed"                 : int(cfg.seed),
        }
        config_path = Path(cfg.paths.dynamics_config)
        config_path.write_text(json.dumps(config_record, indent=2))
        log.info("config.json written → %s", config_path)

    else:
        # Checkpoint reused: write a minimal config.json only if one doesn't exist yet.
        config_path = Path(cfg.paths.dynamics_config)
        if not config_path.exists():
            meta_path  = Path(cfg.paths.pairs_metadata)
            trained_on = "unknown"
            if meta_path.exists():
                meta       = json.loads(meta_path.read_text())
                trained_on = "mock" if meta.get("pairing_method") == "mock" else "real"
            minimal: dict = {
                "n_latent"             : n_latent,
                "n_genes"              : n_genes,
                "d_emb"                : int(cfg.dynamics.d_emb),
                "n_hidden"             : int(cfg.dynamics.n_hidden),
                "n_layers"             : int(cfg.dynamics.n_layers),
                "dropout"              : float(cfg.dynamics.dropout),
                "activation"           : str(cfg.dynamics.activation),
                "log_var_min"          : float(cfg.dynamics.log_var_min),
                "log_var_max"          : float(cfg.dynamics.log_var_max),
                "use_layernorm"        : bool(cfg.dynamics.use_layernorm),
                "use_state_linear_skip": use_state_linear_skip,
                "use_gene_delta_bias"  : use_gene_delta_bias,
                "trained_on"           : trained_on,
                "epochs_run"           : 0,
                "best_epoch"           : 0,
                "best_val_nll"         : None,
                "final_val_nll"        : None,
                "checkpoint_reused"    : True,
                "seed"                 : int(cfg.seed),
            }
            config_path.write_text(json.dumps(minimal, indent=2))
            log.info("config.json (checkpoint-reuse stub) written → %s", config_path)

    # ------------------------------------------------------------------
    # Step 10 — Phase 2 validation gate (always runs)
    # ------------------------------------------------------------------
    from src.analysis.metrics import dynamics_validation_gate

    val_path_p = Path(cfg.paths.pairs_val)
    if not val_path_p.exists():
        log.error(
            "Validation pairs not found at %s — cannot run the dynamics validation gate. "
            "Run `make pairs` or `generate_mock_pairs` so val_pairs.npz exists. "
            "RL training is blocked until gate.json is written with passed=true.",
            val_path_p,
        )
        return 1

    # Reload best checkpoint into the already-built model.
    # map_location="cpu" avoids the MPS deserialiser path; load_state_dict then copies
    # tensors into the model already resident on `device`.
    log.info("Loading best checkpoint from %s for gate evaluation.", model_path)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Load pair splits for baseline fitting and prediction.
    train_z_ctrl, train_g_idx, train_z_pert = _npz_to_tensors(
        train_path, ["z_ctrl", "gene_idx", "z_pert"]
    )
    val_z_ctrl, val_g_idx, val_z_pert = _npz_to_tensors(
        val_path_p, ["z_ctrl", "gene_idx", "z_pert"]
    )

    batch_size_gate = int(cfg.dynamics.batch_size)
    val_z_pert_pred, val_log_var = _predict_split(
        model, val_z_ctrl, val_g_idx, device=device, batch_size=batch_size_gate,
    )

    baselines_train_data = {
        "z_ctrl":   train_z_ctrl.numpy(),
        "gene_idx": train_g_idx.numpy(),
        "z_pert":   train_z_pert.numpy(),
    }

    log.info("Running validation gate on %d val samples ...", len(val_z_ctrl))
    val_out = dynamics_validation_gate(
        z_ctrl          = val_z_ctrl.numpy(),
        gene_idx        = val_g_idx.numpy(),
        z_pert_true     = val_z_pert.numpy(),
        z_pert_pred_mlp = val_z_pert_pred,
        log_var_pred    = val_log_var,
        cfg_gate        = cfg.dynamics.gate,
        baselines_train_data = baselines_train_data,
    )
    log.info(
        "Val gate: passed=%s | R²=%.4f | Pearson=%.4f | Spearman=%.4f",
        val_out["passed"],
        val_out["primary"]["r2"],
        val_out["primary"]["pearson_r"],
        val_out["uncertainty_calibration"]["spearman"],
    )

    # OOD split — report-only, does not affect gate.json["passed"]
    ood_path  = Path(cfg.paths.pairs_ood)
    ood_out: dict | None = None
    ood_z_ctrl_np:      np.ndarray | None = None
    ood_g_idx_np:       np.ndarray | None = None
    ood_z_pert_np:      np.ndarray | None = None
    ood_z_pert_pred_np: np.ndarray | None = None
    if ood_path.exists():
        ood_z_ctrl, ood_g_idx, ood_z_pert = _npz_to_tensors(
            ood_path, ["z_ctrl", "gene_idx", "z_pert"]
        )
        if len(ood_z_ctrl) > 0:
            log.info("Running OOD report on %d OOD samples ...", len(ood_z_ctrl))
            ood_z_pert_pred, ood_log_var = _predict_split(
                model, ood_z_ctrl, ood_g_idx, device=device, batch_size=batch_size_gate,
            )
            ood_z_ctrl_np      = ood_z_ctrl.numpy()
            ood_g_idx_np       = ood_g_idx.numpy()
            ood_z_pert_np      = ood_z_pert.numpy()
            ood_z_pert_pred_np = ood_z_pert_pred
            ood_out = dynamics_validation_gate(
                z_ctrl          = ood_z_ctrl_np,
                gene_idx        = ood_g_idx_np,
                z_pert_true     = ood_z_pert_np,
                z_pert_pred_mlp = ood_z_pert_pred_np,
                log_var_pred    = ood_log_var,
                cfg_gate        = cfg.dynamics.gate,
                baselines_train_data = baselines_train_data,
            )
            log.info(
                "OOD report (non-gating): R²=%.4f | Pearson=%.4f | Spearman=%.4f",
                ood_out["primary"]["r2"],
                ood_out["primary"]["pearson_r"],
                ood_out["uncertainty_calibration"]["spearman"],
            )
    else:
        log.info(
            "OOD pairs not found at %s — skipping OOD report (non-gating). "
            "This is expected when running on mock pairs.",
            ood_path,
        )

    # Write gate.json — only val outcome contributes to 'passed'
    gate_record: dict = {
        "passed"                  : val_out["passed"],
        "primary"                 : val_out["primary"],
        "ood"                     : ood_out["primary"] if ood_out is not None else None,
        "uncertainty_calibration" : val_out["uncertainty_calibration"],
        "uncertainty_calibration_ood": (
            ood_out["uncertainty_calibration"] if ood_out is not None else None
        ),
        "margins_used"            : val_out["margins_used"],
    }
    gate_path = Path(cfg.paths.dynamics_gate)
    gate_path.write_text(json.dumps(gate_record, indent=2))
    Path(cfg.paths.dynamics_val_metrics).write_text(json.dumps(val_out, indent=2))
    if ood_out is not None:
        Path(cfg.paths.dynamics_ood_metrics).write_text(json.dumps(ood_out, indent=2))

    log.info("gate.json written (passed=%s) → %s", gate_record["passed"], gate_path)

    # ------------------------------------------------------------------
    # Diagnostics — per-dim + per-gene MLP-vs-ridge breakdown for debugging
    # which dims / genes the MLP loses on. Uses the same ridge baseline helper
    # as the gate, so numbers are directly comparable.
    # ------------------------------------------------------------------
    from src.analysis.metrics import gate_diagnostics

    diagnostics = gate_diagnostics(
        z_ctrl_train         = train_z_ctrl.numpy(),
        gene_idx_train       = train_g_idx.numpy(),
        z_pert_train         = train_z_pert.numpy(),
        z_ctrl_val           = val_z_ctrl.numpy(),
        gene_idx_val         = val_g_idx.numpy(),
        z_pert_val           = val_z_pert.numpy(),
        z_pert_pred_mlp_val  = val_z_pert_pred,
        z_ctrl_ood           = ood_z_ctrl_np,
        gene_idx_ood         = ood_g_idx_np,
        z_pert_ood           = ood_z_pert_np,
        z_pert_pred_mlp_ood  = ood_z_pert_pred_np,
    )
    diag_path = Path(cfg.paths.dynamics_diagnostics)
    diag_path.write_text(json.dumps(diagnostics, indent=2))
    log.info(
        "gate_diagnostics.json written | val mlp-ridge Pearson=%+.4f | "
        "ood mlp-ridge Pearson=%s | %s",
        diagnostics["overall"]["val"]["mlp_minus_ridge_pearson"],
        (
            f"{diagnostics['overall']['ood']['mlp_minus_ridge_pearson']:+.4f}"
            if diagnostics["overall"]["ood"] is not None else "n/a"
        ),
        diag_path,
    )

    if not val_out["passed"]:
        log.error(
            "Dynamics validation gate FAILED on the val split. "
            "RL training is blocked until the gate passes. "
            "Check margin_checks in val_metrics.json for which baselines the MLP lost to."
        )
        return 1

    log.info("Dynamics validation gate PASSED. RL training is unblocked.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
