"""Checkpoint save/load helpers.

Owner: shared.

**scVI must be saved/loaded via the official API** — ``model.save()`` /
``SCVI.load()`` — not via raw ``torch.save(state_dict)``. The official API
bundles the AnnData registry alongside the state dict; manual state-dict saves
silently break when the AnnData schema differs (CLAUDE.md sacred rule #1,
ARCHITECTURE.md D14).

Plain PyTorch checkpointing (dynamics model, PPO sidecar state) goes through
:func:`save_torch_checkpoint` / :func:`load_torch_checkpoint`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_scvi_model(model: Any, path: str | Path, *, overwrite: bool = True) -> Path:
    """Save a scVI model using the official API.

    Writes ``model.pt``, ``attr.pkl``, and ``var_names.csv`` into ``path``.

    Parameters
    ----------
    model
        A trained ``scvi.model.SCVI`` instance.
    path
        Directory to write into.
    overwrite
        Replace an existing directory.

    Returns
    -------
    Path
        Directory that was written.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save(str(path), overwrite=overwrite)

    if not (path / "model.pt").exists():
        present = {f.name for f in path.iterdir() if f.is_file()}
        raise RuntimeError(
            f"scVI save appears incomplete — model.pt missing. "
            f"Present: {present}. Check scvi-tools version compatibility."
        )
    return path


def load_scvi_model(path: str | Path, adata: Any) -> Any:
    """Load a scVI model via the official API.

    Parameters
    ----------
    path
        Directory written by :func:`save_scvi_model`.
    adata
        AnnData with the same schema as the training data.

    Returns
    -------
    scvi.model.SCVI
        Loaded model, ready for ``get_latent_representation()``.
    """
    import scvi

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"scVI model directory not found: {path}. "
            "Run training first or check the path in config/paths.yaml."
        )
    if not (path / "model.pt").exists():
        raise FileNotFoundError(
            f"Incomplete scVI checkpoint at {path} — model.pt not found."
        )
    return scvi.model.SCVI.load(str(path), adata=adata)


def save_torch_checkpoint(state: dict[str, Any], path: str | Path) -> Path:
    """Atomically save a plain torch dict checkpoint.

    Parameters
    ----------
    state
        Dict with at minimum ``model_state_dict`` and ``epoch`` keys.
    path
        Destination file. Parent directories are created if missing.

    Returns
    -------
    Path
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_torch_checkpoint(
    path: str | Path,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Load a plain torch dict checkpoint.

    Parameters
    ----------
    path
        Path to the saved checkpoint.
    map_location
        Forwarded to :func:`torch.load`. Pass ``get_device()`` for in-place
        restoration on the correct accelerator.

    Returns
    -------
    dict
        State dict as written by :func:`save_torch_checkpoint`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(
            f"Expected a dict checkpoint, got {type(state)}. "
            "Was it saved with save_torch_checkpoint()?"
        )
    return state
