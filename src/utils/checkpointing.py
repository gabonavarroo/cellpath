"""Checkpoint save/load helpers.

Owner: shared.

Why this module exists
----------------------
The scVI model uses an official ``model.save() / SCVI.load()`` API that bundles the PyTorch
state dict alongside AnnData registry metadata. **Manual ``state_dict`` saves are forbidden**
(see CLAUDE.md sacred rule #1 and ARCHITECTURE.md D14). All scVI save/load goes through
:func:`save_scvi_model` / :func:`load_scvi_model` so this contract is enforced in one place.

Plain PyTorch checkpointing (used by the dynamics model and PPO) goes through
:func:`save_torch_checkpoint` / :func:`load_torch_checkpoint`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_scvi_model(model: Any, path: str | Path, *, overwrite: bool = True) -> Path:
    """Save a scVI model using the official API.

    Parameters
    ----------
    model
        An ``scvi.model.SCVI`` (or compatible) instance.
    path
        Directory to write into. scVI writes ``model.pt``, ``attr.pkl``, and
        ``var_names.csv`` inside this directory.
    overwrite
        If True, replace an existing directory.

    Returns
    -------
    Path
        The directory that was written.

    Raises
    ------
    NotImplementedError
        Agent A: implement using ``model.save(str(path), overwrite=overwrite)``.
        Verify the directory contains the three expected files and return the path.
    """
    raise NotImplementedError(
        "Agent A: implement scVI save via official `model.save()` API. "
        "Must produce model.pt + attr.pkl + var_names.csv in the target directory. "
        "DO NOT use raw torch.save(state_dict) — see CLAUDE.md sacred rule #1."
    )


def load_scvi_model(path: str | Path, adata: Any) -> Any:
    """Load a scVI model via the official API.

    Parameters
    ----------
    path
        Directory containing a previously saved scVI model.
    adata
        AnnData that the model was originally trained on (schema-compatible).

    Returns
    -------
    scvi.model.SCVI
        Loaded model.

    Raises
    ------
    NotImplementedError
        Agent A: implement using ``scvi.model.SCVI.load(str(path), adata)``.
    FileNotFoundError
        If the model directory or any of the required files is missing.
    """
    raise NotImplementedError(
        "Agent A: implement scVI load via official `SCVI.load(path, adata)`. "
        "Schema mismatch between saved model and provided adata must raise loudly."
    )


def save_torch_checkpoint(
    state: dict[str, Any],
    path: str | Path,
) -> Path:
    """Save a plain torch dict checkpoint (used for dynamics, PPO sidecar state).

    Parameters
    ----------
    state
        Dict containing ``model_state_dict``, ``optimizer_state_dict``, ``epoch``,
        and any user metadata.
    path
        Filesystem path. Parent dirs are created if missing.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    NotImplementedError
        Agent B: implement with ``torch.save`` to a tmp file, then atomic rename.
    """
    raise NotImplementedError(
        "Agent B: torch.save(state, tmp) → os.replace(tmp, path) for atomic write."
    )


def load_torch_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    """Load a plain torch dict checkpoint.

    Parameters
    ----------
    path
        Path to the saved checkpoint.
    map_location
        Forwarded to :func:`torch.load`. Use the device returned by
        :func:`src.utils.device.get_device` for in-place restoration.

    Returns
    -------
    dict
        State dict as written by :func:`save_torch_checkpoint`.

    Raises
    ------
    NotImplementedError
        Agent B: implement with ``torch.load`` + schema sanity check.
    """
    raise NotImplementedError(
        "Agent B: torch.load with map_location; verify required keys present."
    )
