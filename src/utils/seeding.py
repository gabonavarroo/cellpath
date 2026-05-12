"""Global RNG seeding — the ONLY module that may call ``*.seed()``.

Owner: shared (Agents A + B coordinate on changes).

Every entry point calls :func:`set_seed(cfg.seed)` exactly once at start. Downstream code may
read from these RNGs but must not re-seed.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy, PyTorch (CPU + CUDA), and ``PYTHONHASHSEED``.

    Parameters
    ----------
    seed
        Integer seed. Use the same value across the pipeline for reproducibility.
    deterministic_torch
        If True, configure ``torch.use_deterministic_algorithms(True)`` and disable cudnn
        benchmark. Adds runtime cost; use only for full reproducibility runs.

    Returns
    -------
    None

    Examples
    --------
    >>> set_seed(42)
    >>> import torch
    >>> torch.rand(1).item() == torch.rand(1).item()
    False
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
